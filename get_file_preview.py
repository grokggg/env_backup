"""文件预览与解析工具集。

提供两类能力：
1. MCP 工具调用 — 通过 Polaris 服务发现连接 MAS Tool MCP 服务，
   支持 parse_file / image2text / video_parse_v2。
2. 文件内容解析与消息块构建 — 将上传文件解析为纯文本，
   并格式化为 [user_doc N] / [user_image N] 消息块供 agent 使用。
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

from asyncio import Queue
from pathlib import Path

import httpcore
import httpx

from fastmcp import Client
from fastmcp.client.transports import SSETransport

from nexus.extensions.agents.yb_one_agent.multimodal import (
    ImageRejectedError,
    build_image_part,
    describe_image_info,
)


# ─────────────────────── Polaris 服务发现 ───────────────────────

# Polaris 服务发现的连接/总超时（秒）。无超时的同步 requests.post 会在节点不可达时
# 直接阻塞事件循环线程（表现为 eventloop_monitor 的 STALLED 告警），故显式限时。
_POLARIS_TIMEOUT = httpx.Timeout(5.0, connect=3.0)


async def _get_polaris_ip_port(service: str, namespace: str) -> str:
    """通过 Polaris 服务发现，随机返回一个健康节点的 host:port。"""
    import random

    url = 'http://polaris-discover.woa.com:8080/v1/Discover'
    data = {
        'type': 1,
        'service': {'name': service, 'namespace': namespace},
    }
    async with httpx.AsyncClient(timeout=_POLARIS_TIMEOUT) as client:
        resp = await client.post(url, json=data)
        resp.raise_for_status()
        payload = resp.json()
    instances = [f'{ins["host"]}:{ins["port"]}' for ins in payload['instances'] if ins['weight'] > 0]
    return random.choice(instances)


async def _create_mcp_client() -> Client:
    """创建连接到 MAS Tool MCP 服务的 fastmcp Client。"""
    namespace = 'Test'
    polaris = 'trpc.mas_tool.mcp_tool_service.mcp'
    ip_port = await _get_polaris_ip_port(polaris, namespace)
    url = f'http://{ip_port}/sse'
    headers = {
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json',
    }
    return Client(SSETransport(url=url, headers=headers))


# ─────────────────────── 共享 Client 连接池 ───────────────────────


class ClientPool:
    """异步 Client 对象池，避免每次调用都重新建连。

    Pool 中存储 fastmcp Client 对象（Python 对象复用），但每次
    call_tool 仍会走一次 SSE 握手 → 调用 → 断开的完整流程，
    Pool 的实际作用是限制并发数。
    """

    def __init__(self, create_func, pool_size: int = 4):
        self._create_func = create_func
        self._pool_size = pool_size
        self._pool = Queue(maxsize=pool_size)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            print(f'正在初始化连接池，大小：{self._pool_size}')
            for i in range(self._pool_size):
                client = await self._create_func()
                await self._pool.put(client)
                print(f'  ✓ 创建第 {i + 1}/{self._pool_size} 个 client')
            self._initialized = True
            print('连接池初始化完成')

    async def acquire(self) -> Client:
        if not self._initialized:
            await self.initialize()
        return await self._pool.get()

    async def release(self, client: Client) -> None:
        await self._pool.put(client)


# 所有 MCP 工具共享同一个连接池（它们都路由到同一服务）
_shared_pool: ClientPool | None = None
_pool_lock = asyncio.Lock()


async def _get_shared_pool(pool_size: int = 4) -> ClientPool:
    global _shared_pool
    if _shared_pool is None:
        async with _pool_lock:
            if _shared_pool is None:
                _shared_pool = ClientPool(_create_mcp_client, pool_size=pool_size)
    assert _shared_pool is not None
    return _shared_pool


# ─────────────────────── MCP 工具调用 ───────────────────────

_SUPPORTED_TOOLS = {'parse_file', 'image2text', 'video_parse_v2'}
_MAX_RETRIES = 3


async def call_mcp_tool(tool_name: str, parameters: dict):
    """调用指定的 MCP 工具，内置重试与连接池复用。

    Args:
        tool_name:  MCP 工具名称，可选：parse_file / image2text / video_parse_v2。
        parameters: 工具入参 dict。

    Returns:
        工具返回的结构化数据（已从外层 payload 中提取）。
    """
    if tool_name not in _SUPPORTED_TOOLS:
        raise ValueError(f'不支持的 tool：{tool_name!r}，可选：{sorted(_SUPPORTED_TOOLS)}')

    pool = await _get_shared_pool(pool_size=4)
    client = await pool.acquire()
    try:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with client:
                    result = await client.call_tool_mcp(tool_name, parameters)
                break
            except (asyncio.TimeoutError, httpcore.ConnectTimeout, httpx.ConnectTimeout) as e:
                print(f'[call_mcp_tool] 第 {attempt}/{_MAX_RETRIES} 次：网络超时/建连失败：{e}')
                if attempt == _MAX_RETRIES:
                    raise
                await asyncio.sleep(1)
            except httpx.HTTPStatusError as e:
                print(f'[call_mcp_tool] 第 {attempt}/{_MAX_RETRIES} 次：HTTP 错误：{e}，重新获取节点…')
                if attempt == _MAX_RETRIES:
                    raise
                client = await _create_mcp_client()
                await asyncio.sleep(1)
            except Exception as e:
                # MCP 协议层错误（如 "tool not found"）：可能是节点未就绪/工具未注册，
                # 换节点重试。
                print(f'[call_mcp_tool] 第 {attempt}/{_MAX_RETRIES} 次：MCP 调用异常：{e}，重新获取节点…')
                if attempt == _MAX_RETRIES:
                    raise
                client = await _create_mcp_client()
                await asyncio.sleep(2)
    finally:
        await pool.release(client)

    payload = json.loads(result.content[0].text)  # type: ignore[possibly-missing-attribute]

    if tool_name == 'parse_file':
        return payload['content']['parse_info']
    if tool_name in ('image2text', 'video_parse_v2'):
        return payload['content']

    return payload


# ─────────────────────── COS 上传 ───────────────────────


def _get_cos_client():
    """从环境变量读取凭证，构造 CosS3Client。

    使用腾讯云 COS 内网专线域名 cos-internal.{region}.tencentcos.cn，
    与 MCP parse_file 服务处于同一 VPC，可正常访问，且不触发 SSRF 防护。

    必须设置的环境变量：
        COS_SECRET_ID   — 腾讯云 SecretId
        COS_SECRET_KEY  — 腾讯云 SecretKey
        COS_REGION      — 存储桶所在地域，如 ap-guangzhou
        COS_BUCKET      — 存储桶名称，如 my-bucket-1234567890
    """
    from qcloud_cos import CosConfig, CosS3Client

    secret_id = os.environ['COS_SECRET_ID']
    secret_key = os.environ['COS_SECRET_KEY']
    region = os.environ['COS_REGION']

    endpoint = f'cos-internal.{region}.tencentcos.cn'
    config = CosConfig(
        Region=region,
        SecretId=secret_id,
        SecretKey=secret_key,
        Endpoint=endpoint,
    )
    return CosS3Client(config)


def _cos_upload_and_presign(local_path: str, expires: int = 900) -> str:
    """将本地文件上传到 COS，返回有效期为 expires 秒的预签名 URL。

    COS Key 格式：nexus-preview/<uuid>/<原始文件名>
    上传后文件不自动删除，建议在 COS 控制台配置生命周期规则统一清理。
    """
    client = _get_cos_client()
    bucket = os.environ['COS_BUCKET']

    path = Path(local_path)
    cos_key = f'nexus-preview/{uuid.uuid4().hex}/{path.name}'

    with open(local_path, 'rb') as f:
        client.put_object(Bucket=bucket, Body=f, Key=cos_key)

    return client.get_presigned_url(
        Method='GET',
        Bucket=bucket,
        Key=cos_key,
        Expired=expires,
    )


def _cos_upload_bytes_and_presign(filename: str, raw: bytes, expires: int = 900) -> str:
    """将内存中的文件字节直接上传到 COS，返回有效期为 expires 秒的预签名 URL。

    与 _cos_upload_and_presign 相同逻辑，但接受 bytes 而非本地路径，避免落盘。
    COS Key 格式：nexus-preview/<uuid>/<原始文件名>
    """
    client = _get_cos_client()
    bucket = os.environ['COS_BUCKET']

    cos_key = f'nexus-preview/{uuid.uuid4().hex}/{Path(filename).name}'
    client.put_object(Bucket=bucket, Body=raw, Key=cos_key)

    return client.get_presigned_url(
        Method='GET',
        Bucket=bucket,
        Key=cos_key,
        Expired=expires,
    )


_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif'}


def _build_parse_file_arguments(filename: str, url: str) -> dict:
    """根据文件后缀构造 parse_file 的入参 dict。"""
    suffix = Path(filename).suffix.lower()
    parse_type = 'image_parse' if suffix in _IMAGE_EXTS else 'file_parse'

    arguments: dict = {
        'url': url,
        'parse_type': parse_type,
        'options': {
            'no_xml': True,
            'gen_summary': True,
        },
        'param': {
            'biz_id': 'im',
        },
    }
    if parse_type == 'file_parse':
        arguments['file_parse_param'] = {'disable_youtu': False, 'return_mode': 0}

    return arguments


async def get_file_preview(local_path: str) -> dict:
    """上传本地文件到腾讯 COS，然后通过 MCP parse_file 工具解析，返回 parse_info。

    Args:
        local_path: 本地文件路径（绝对或相对路径均可）。

    Returns:
        parse_info dict，与直接调用 call_mcp_tool('parse_file', ...) 的返回值相同。

    所需环境变量：COS_SECRET_ID, COS_SECRET_KEY, COS_REGION, COS_BUCKET
    """
    loop = asyncio.get_running_loop()
    url = await loop.run_in_executor(None, _cos_upload_and_presign, local_path)
    print(f'[get_file_preview] 已上传，预签名 URL（15min 有效）：{url}')
    return await call_mcp_tool('parse_file', _build_parse_file_arguments(local_path, url))


async def get_file_preview_from_bytes(filename: str, raw: bytes) -> dict:
    """将内存中的文件字节上传到 COS，通过 MCP parse_file 工具解析，返回 parse_info。

    与 get_file_preview 逻辑相同，但直接接受 bytes，无需先落盘。
    适用于已在内存中持有文件内容的场景（如 Web 上传的 UploadFile）。

    Args:
        filename: 原始文件名（用于确定后缀和 COS Key）。
        raw:      文件原始字节。

    Returns:
        parse_info dict。

    所需环境变量：COS_SECRET_ID, COS_SECRET_KEY, COS_REGION, COS_BUCKET
    """
    loop = asyncio.get_running_loop()
    url = await loop.run_in_executor(None, _cos_upload_bytes_and_presign, filename, raw)
    print(f'[get_file_preview_from_bytes] 已上传 {filename}，预签名 URL（15min 有效）：{url}')
    return await call_mcp_tool('parse_file', _build_parse_file_arguments(filename, url))


# ─────────────────────── 文件内容解析 ───────────────────────

UPLOAD_SUPPORTED = {
    '.txt',
    '.md',
    '.csv',
    '.json',
    '.yaml',
    '.yml',
    '.xml',
    '.html',
    '.pdf',
    '.doc',
    '.docx',
    '.pptx',
    '.xlsx',
    '.xls',
}
IMAGE_SUPPORTED = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
UPLOAD_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


# ─────────────────────── 消息块构建 ───────────────────────

_DOC_CONTENT_MAX_CHARS = 15_000


def build_doc_block(index: int, title: str, sandbox_path: str, content: str | None, error: str | None) -> str:
    """将文档信息格式化为 [user_doc N ...] 消息块，供 agent 读取。

    文档内容超过 _DOC_CONTENT_MAX_CHARS 个字符时，截取前 _DOC_CONTENT_MAX_CHARS 个字符
    并在末尾追加 ...<truncated>。
    """
    lines = [f'[user_doc {index} begin]', '', f'[user_doc_title] {title}', f'[user_doc_path] {sandbox_path}']
    if error:
        lines += ['', f'[error message] {error}']
    else:
        body = content or ''
        if len(body) > _DOC_CONTENT_MAX_CHARS:
            body = body[:_DOC_CONTENT_MAX_CHARS] + '...<truncated>'
        lines += ['', '[user_doc_content begin]', '', body, '', '[user_doc_content end]']
    lines += ['', f'[user_doc {index} end]']
    return '\n'.join(lines)


def build_image_block(
    index: int,
    path: str,
    raw: bytes | None = None,
    error: str | None = None,
) -> list[dict]:
    """将图片格式化为 [user_image N ...] 消息块，返回 content part 列表。

    格式：
        [user_image N begin]
        [user_image_path] <沙箱内图片路径>
        [user_image_binary] <原生 image part>
        [user_image_info] <尺寸/体积/格式>
        [user_image N end]

    ``[user_image_binary]`` 之后紧跟一个 ``image_url`` part，图片像素直接进入模型
    视野，不再经由 caption 中转。``raw`` 缺失或图片无法安全投递时，该位置降级为
    一行文本说明——绝不返回半坏的 image part，否则整段 history 每轮重发都会被
    API 拒绝。
    """
    head = f'[user_image {index} begin]\n[user_image_path] {path}\n[user_image_binary] '
    tail = f'\n[user_image {index} end]'

    if error:
        return [{'type': 'text', 'text': f'{head}<unavailable: {error}>{tail}'}]
    if not isinstance(raw, bytes | bytearray) or not raw:
        return [{'type': 'text', 'text': f'{head}<unavailable: image bytes not provided>{tail}'}]

    try:
        part, info = build_image_part(raw)
    except ImageRejectedError as exc:
        return [{'type': 'text', 'text': f'{head}<unavailable: {exc}>{tail}'}]

    return [
        {'type': 'text', 'text': head},
        part,
        {'type': 'text', 'text': f'\n[user_image_info] {describe_image_info(info)}{tail}'},
    ]
