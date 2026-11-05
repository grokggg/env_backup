"""Tools for yb_one_agent: bash / read / edit / write / web 工具组 / ask_user_question / handoff / load_skill.

web 工具组（web_fetch / web_search / picture_search / video_search）通过 tool_proxy
(work_agent) 调用搜索服务。
"""

import asyncio
import logging
import os
import pathlib
import random
import uuid

from typing import Any, Literal

import requests

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, Field, model_validator

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command


logger = logging.getLogger(__name__)


# ============== Constants ==============

MAX_RESPONSE_LEN = int(os.environ.get('MAX_RESPONSE_LEN', 32000))
TRUNCATED_MESSAGE = (
    '<response clipped><NOTE>To save on context only part of this file has been shown to you. '
    'You should retry this tool after you have searched inside the file with `grep -n` in order to find '
    'the line numbers of what you are looking for.</NOTE>'
)

# ============== v3: 通用工具增强常量 ==============

# bash 输出过大 → 完整落盘 + head/tail 换行感知预览
# 输出超过 OUTPUT_THRESHOLD 字符才落盘（沿用原 16K 阈值）；≤ 阈值直接给全量。
OUTPUT_THRESHOLD = int(os.environ.get('OUTPUT_THRESHOLD', 32000))
# 落盘目录（相对 bash 执行目录 cwd，沙箱/本地都恒有文件系统，无需 workspace_path）。
OUTPUT_DIR_NAME = '.tool_output'
# 预览头/尾字符预算：总预览 ≤ PREVIEW_HEAD + PREVIEW_TAIL，杜绝按行截断时的总量爆炸。
PREVIEW_HEAD = 2000
PREVIEW_TAIL = 1000

# read / edit 统一 unsupported-file-type 门禁（扩展名分类）
OFFICE_EXTS = {'.docx', '.xlsx', '.pptx', '.doc', '.xls', '.ppt'}
PDF_EXTS = {'.pdf'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
NOTEBOOK_EXTS = {'.ipynb'}
_UNSUPPORTED_EXTS = OFFICE_EXTS | PDF_EXTS | IMAGE_EXTS | NOTEBOOK_EXTS
# read_v3 放行图片：图片扩展名从门禁里摘出来，交给 _read_image 解码成 image part。
_READ_V3_UNSUPPORTED_EXTS = _UNSUPPORTED_EXTS - IMAGE_EXTS
BINARY_CHECK_SIZE = 8192

# 读图前用 stat 卡大小，避免 download_file 把超大文件整个拉进内存。
READ_IMAGE_MAX_BYTES = int(os.environ.get('ONE_AGENT_IMAGE_MAX_BYTES', 20 * 1024 * 1024))


# ============== Utility Functions ==============


def maybe_truncate(content: str, max_length: int = MAX_RESPONSE_LEN) -> str:
    if len(content) <= max_length:
        return content
    return content[:max_length] + TRUNCATED_MESSAGE


def _make_output(text: str) -> list[ChatCompletionContentPartParam | dict]:
    """Helper to create standard output format."""
    return [{'type': 'text', 'text': text}]


def _first_upload_error(upload_result) -> str | None:
    """Return the first upload error message, or None when all uploads succeeded."""
    results = getattr(upload_result, 'results', None) or []
    if not results:
        return 'No upload result returned'

    for result in results:
        if not getattr(result, 'success', False):
            path = getattr(result, 'path', 'file')
            message = getattr(result, 'message', None) or 'Unknown upload error'
            return f'{path}: {message}'

    return None


# ============== v3: unsupported-file-type 门禁 ==============
#
# office/pdf/image/.ipynb（扩展名）+ 二进制（内容）归一到 UNSUPPORTED_FILE_TYPE，
# for_model 逐字对齐 reference/tool_proxy/toolcalls/local/for_model.go：
#   "Unsupported file type: {extension}"


def _unsupported_extension(file_path: str, unsupported: set[str] | None = None) -> str | None:
    """Return the extension when the path points to an unsupported file type, else None."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (_UNSUPPORTED_EXTS if unsupported is None else unsupported):
        return ext
    return None


def _is_binary_content(data: bytes) -> bool:
    """Heuristic: a null byte in the leading bytes indicates binary content."""
    return b'\x00' in data


def _unsupported_response(extension: str, *, guidance: bool = False) -> ToolResponse:
    """Build the unified UNSUPPORTED_FILE_TYPE response (for_model aligned with for_model.go).

    *guidance* appends what to reach for instead. Only ``read_v3`` asks for it:
    the bare line leaves the model with nowhere to go and it drops the step
    rather than switching to a skill or to bash. The tools that mirror the
    online proxy stay word-for-word identical to it.
    """
    text = f'Unsupported file type: {extension}'
    if guidance:
        text += (
            '\nread cannot open this format. Load the matching skill with load_skill '
            '(xlsx / docx / pptx / pdf), or parse the file with code via bash.'
        )
    return ToolResponse(
        output=_make_output(text),
        meta={'ok': False, 'error': {'code': 'UNSUPPORTED_FILE_TYPE', 'extension': extension}},
    )


# Paths a model substitutes when it wants to read an image it has no path for:
# /dev/null, /dev/stdin, or the URL itself. On a VLM smoke run that was 5 of 13
# read calls, in 4 of 8 sessions. Wanting the closer look is not the mistake,
# and these replies deliberately do not tell it to stop — hy3_vlm_sft's own
# agent data trains that move, with an assistant re-rendering an image through
# matplotlib to count shapes off the enlarged copy. What kept the guessing
# going is that the old errors ("Path is not a file", "File does not exist")
# name the symptom and end there, leaving the model to try the next
# placeholder. Naming the route that works ends it on the first try.
_PSEUDO_FILE_PREFIXES = ('/dev/', '/proc/')


def _not_a_file_response(file_path: str) -> ToolResponse | None:
    """A corrective reply when *file_path* cannot name a readable local file."""
    if '://' in file_path:
        reason = (
            f'read takes a path to a file on disk, not a URL: {file_path}\n'
            'Download it first with bash — curl -sSL -o /data/workspace/<name> "<url>" — '
            'then read the file you saved.'
        )
    elif file_path.startswith(_PSEUDO_FILE_PREFIXES):
        reason = (
            f'read takes a path to a file in the sandbox, and {file_path} is not one.\n'
            'Pass the path the image is saved at. If you only have a URL, download it first '
            'with bash — curl -sSL -o /data/workspace/<name> "<url>" — and read that path.'
        )
    else:
        return None
    return ToolResponse(
        output=_make_output(f'Error: {reason}'),
        meta={'ok': False, 'error': {'code': 'NOT_A_FILE', 'path': file_path}},
    )


# ============== v3: bash 输出过大处理 ==============
#
# 设计要点（对齐 reference/tool_proxy/toolcalls/local/for_model.go renderBashTruncated）：
#   - 输出 ≤ OUTPUT_THRESHOLD：模型看全量。
#   - 输出 > OUTPUT_THRESHOLD：完整落盘到 <cwd>/.tool_output/，模型看「文件路径 + head/tail 预览」。
#   - 预览按"总字符预算"截断（head + tail），而非按行，避免 grep 大量单行文件时总量爆炸。
#   - 头尾都对齐行边界，绝不留半行；唯一例外是窗口内无换行（base64/混淆代码等单行超长）才硬截。


def _smart_cut(s: str, limit: int, from_end: bool = False) -> str:
    """Cut ``s`` to at most ``limit`` chars, aligned to a line boundary.

    Only when the window contains no newline at all (a single over-long line such
    as base64 / minified code) do we hard-cut mid-line — there is no boundary to
    align to in that case.
    """
    if len(s) <= limit:
        return s
    if from_end:
        window = s[-limit:]
        nl = window.find('\n')
        return window if nl == -1 else window[nl + 1 :]
    window = s[:limit]
    nl = window.rfind('\n')
    return window if nl == -1 else s[:nl]


def _generate_preview(output: str) -> str:
    """Bounded head+tail preview (≤ PREVIEW_HEAD + PREVIEW_TAIL chars)."""
    if len(output) <= PREVIEW_HEAD + PREVIEW_TAIL:
        return output
    head = _smart_cut(output, PREVIEW_HEAD)
    tail = _smart_cut(output, PREVIEW_TAIL, from_end=True)
    return f'{head}\n...\n{tail}'


def _save_large_output(output: str, cwd: str) -> str:
    """Persist the full output under ``<cwd>/.tool_output/<id>.txt`` and return its path."""
    output_dir = os.path.join(cwd, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'{uuid.uuid4().hex[:8]}.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(output)
    return filepath


def _format_large_output(output: str, saved_path: str) -> str:
    """Canonical truncated for_model text (aligned with for_model.go renderBashTruncated)."""
    return f'Output too large. Full output saved to: {saved_path}\n\nPreview:\n{_generate_preview(output)}'


# ============== Shared input base ==============


class _DescribedInput(BaseModel):
    """Base for tool inputs carrying an annotational ``description`` field.

    ``description`` stays REQUIRED in the schema the model sees (so the model is
    nudged to explain itself), but it is purely annotational and unused by the
    tools. If the model omits it, coerce to '' rather than hard-failing and
    burning a turn. Subclasses declare their own ``description`` field with a
    tool-specific prompt; this base only supplies the tolerance validator.
    """

    @model_validator(mode='before')
    @classmethod
    def _tolerate_missing_description(cls, data):
        if isinstance(data, dict) and not data.get('description'):
            data = {**data, 'description': ''}
        return data


# ============== Bash ==============

# Maximum allowed timeout in milliseconds (600000ms = 600s)
MAX_TIMEOUT_MS = 600000


class BashInputV2(_DescribedInput):
    """Input for bash command execution."""

    description: str = Field(
        description='Why execute this command.',
    )
    command: str = Field(description='The bash command to execute.')
    timeout: int = Field(
        default=60000,
        description='Optional timeout in milliseconds (max 600000).',
    )


@tool_registry.register(
    'yb_one_agent/bash_v2',
    meta={
        'name': 'bash',
        'description': (
            'Executes a given bash command in a persistent shell session with optional timeout.\n\n'
            'IMPORTANT: This tool is for terminal operations like git, npm, docker, running scripts, etc. '
            'DO NOT use it for file operations (reading, writing, editing, searching, finding files) '
            '— use the specialized read, edit, write tools instead.\n\n'
            'Usage notes:\n'
            '- Always quote file paths that contain spaces with double quotes.\n'
            '- For independent commands, prefer chaining with && or calling multiple times.\n'
            '- Avoid using find, grep, cat, head, tail, sed, awk — use the dedicated tools instead.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def bash_tool_v2(params: BashInputV2, context: ToolContext) -> ToolResponse:
    """Execute a bash command using the stateless run_command executor.

    Each invocation runs in an independent subprocess — no state is shared
    between calls (environment variables, working directory, etc.).
    """
    # Resolve timeout: convert from milliseconds to seconds, apply max cap
    timeout_ms = min(params.timeout, MAX_TIMEOUT_MS)
    timeout_seconds = timeout_ms / 1000.0

    # Determine working directory from context
    cwd = context.tool_state.get('working_dir', '/data/workspace')

    # Environment variables from tool_state (merged with the base environment)
    extra_env = context.tool_state.get('env') or {}

    logger.debug('bash: executing command=%r, cwd=%s, timeout=%.1fs', params.command, cwd, timeout_seconds)

    # Execute via stateless run_command
    response = await run_command(
        RunCommandRequest(
            command=params.command,
            cwd=cwd,
            timeout=timeout_seconds,
            extra_env=extra_env,
        )
    )

    # Build output text from stdout/stderr
    if response.status == CommandRunStatus.TimeLimitExceeded:
        logger.warning(
            'bash: command timed out after %.1fs, command=%r',
            response.execution_time or timeout_seconds,
            params.command,
        )
        output_parts = []
        if response.stdout:
            output_parts.append(response.stdout)
        if response.stderr:
            output_parts.append(f'[stderr]\n{response.stderr}')
        output_parts.append(f'[Command timed out after {timeout_seconds}s]')
        output_text = '\n'.join(output_parts) if output_parts else f'Command timed out after {timeout_seconds}s'

        return ToolResponse(
            output=_make_output(maybe_truncate(output_text)),
            meta={
                'command': params.command,
                'exit_code': -1,
                'timeout': True,
                'execution_time': response.execution_time,
            },
        )

    if response.status == CommandRunStatus.Error:
        error_text = response.stderr or 'Command execution failed'
        logger.error('bash: command failed, command=%r, error=%s', params.command, error_text)
        return ToolResponse(
            output=_make_output(maybe_truncate(f'Error: {error_text}')),
            meta={
                'command': params.command,
                'exit_code': -1,
                'error': True,
            },
        )

    # CommandRunStatus.Finished
    exit_code = response.return_code or 0
    logger.debug(
        'bash: command finished, command=%r, exit_code=%d, time=%.2fs',
        params.command,
        exit_code,
        response.execution_time or 0,
    )

    output_text = response.stdout or ''
    if response.stderr:
        output_text += f'\n[stderr]\n{response.stderr}'

    if not output_text.strip():
        output_text = '(no output)'

    # v3: 输出过大 → 完整落盘到 <cwd>/.tool_output/ + head/tail 换行感知预览。
    # 落盘用 bash 执行目录 cwd（沙箱/本地都恒有文件系统），不再依赖 workspace_path。
    if len(output_text) > OUTPUT_THRESHOLD:
        saved_path: str | None = None
        try:
            saved_path = _save_large_output(output_text, cwd)
        except OSError:
            logger.warning('bash: failed to persist large output under %s, falling back to inline truncation', cwd)
        if saved_path is not None:
            return ToolResponse(
                output=_make_output(_format_large_output(output_text, saved_path)),
                meta={
                    'command': params.command,
                    'exit_code': exit_code,
                    'output_file': saved_path,
                    'output_size': len(output_text),
                    'execution_time': response.execution_time,
                },
            )
        # 落盘失败 → 退化为字符截断，避免撑爆上下文
        output_text = maybe_truncate(output_text)

    # ≤ 阈值（或落盘失败已截断）：给全量
    output_text += f'\n[Command finished with exit code {exit_code}]'

    return ToolResponse(
        output=_make_output(output_text),
        meta={
            'command': params.command,
            'exit_code': exit_code,
            'stdout': response.stdout,
            'stderr': response.stderr,
            'execution_time': response.execution_time,
        },
    )


# ============== Read ==============


class ReadInputV2(_DescribedInput):
    description: str = Field(
        description='Why need to view this file.',
    )
    file_path: str = Field(description='The absolute path to the file to read.')
    offset: int = Field(default=1, description='The line number to start reading from (1-indexed).')
    limit: int = Field(default=2000, description='The number of lines to read.')


@tool_registry.register(
    'yb_one_agent/read_v2',
    meta={
        'name': 'read',
        'description': (
            'Reads a file from the filesystem.\n\n'
            'Usage:\n'
            '- file_path must be an absolute path.\n'
            '- By default reads up to 2000 lines from the beginning.\n'
            '- Use offset and limit for large files.\n'
            '- Returns content in cat -n format with line numbers.\n'
            '- Can read text files, source code, configs, etc.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def read_tool_v2(params: ReadInputV2, context: ToolContext) -> ToolResponse:
    from nexus.runtime.executors.file import DownloadFileRequest, FileItem, download_file

    logger.debug('read: file_path=%s, offset=%s, limit=%s', params.file_path, params.offset, params.limit)

    # v3: 统一 unsupported-file-type 门禁（扩展名命中 office/pdf/image/.ipynb）
    unsupported_ext = _unsupported_extension(params.file_path)
    if unsupported_ext is not None:
        logger.debug('read: unsupported file type (ext), path=%s, ext=%s', params.file_path, unsupported_ext)
        return _unsupported_response(unsupported_ext)

    try:
        result = await download_file(DownloadFileRequest(files=[FileItem(path=params.file_path, encoding='utf8')]))
        if not result.files:
            error = result.errors[0].message if result.errors else 'File does not exist'
            logger.warning('read: file not found, path=%s, error=%s', params.file_path, error)
            return ToolResponse(output=_make_output(f'Error: {error}'))

        content = result.files[0].content or ''

        # v3: 二进制内容检测（读到的前若干字节含 null byte）
        if _is_binary_content(content.encode('utf-8', errors='replace')[:BINARY_CHECK_SIZE]):
            logger.debug('read: unsupported file type (binary), path=%s', params.file_path)
            return _unsupported_response('binary')

        lines = content.splitlines()

        # Apply offset and limit (1-indexed offset)
        offset = max(params.offset - 1, 0)
        limit = params.limit
        selected = lines[offset : offset + limit]

        # Format as cat -n
        formatted = '\n'.join(f'{offset + i + 1:6}\t{line}' for i, line in enumerate(selected))
        if len(lines) > offset + limit:
            formatted += f'\n... ({len(lines) - offset - limit} more lines)'

        logger.debug('read: success, path=%s, total_lines=%d, returned=%d', params.file_path, len(lines), len(selected))
        return ToolResponse(output=_make_output(maybe_truncate(formatted)))

    except Exception as e:
        logger.error('read: exception, path=%s, error=%s', params.file_path, e)
        return ToolResponse(output=_make_output(f'Error: {e}'))


# ============== Read v3: 图片作为原生 image part 返回 ==============
#
# read_v2 与线上 tool_proxy 逐字对齐，不动；多模态基座需要的能力全部落在这里。
# 相对 v2 多三件事：
#   1. 图片扩展名不再被门禁拦下，改为解码成 image part 交给模型；
#   2. 拿 URL 或 /dev/* 当路径时给出可执行的纠正，而不是"不是文件"；
#   3. 门禁挡下 office/pdf/.ipynb 时补一句该换什么工具。
# 只有 VLM 基座能消费 image part，纯文本基座拿到会直接报错，所以两者不能混用：本分支
# 的工具表统一挂 v3（prompt_builder.DEFAULT_TOOLS / _BUILTIN_TOOL_ORDER），v2 只注册不加载。
#
# 文本读取那段与 v2 重复，是有意的：v2 是线上那份的镜像，等线上对齐后就会删掉，
# v3 不能跟着一起没。重复的代价也小 —— 镜像按定义不会再改，两边不存在各自演进。


async def _read_image(file_path: str, context: ToolContext) -> ToolResponse:
    """Read an image file and hand it to the model as a native image part.

    The executor bounds the actual read before transferring bytes, including
    when a symlink points at an oversized file.  Every failure degrades to a text error rather than an image part:
    a payload the API rejects would be replayed on every subsequent turn and
    permanently break the session.
    """
    import base64 as _base64

    from nexus.extensions.agents.yb_one_agent.multimodal import (
        ImageRejectedError,
        build_image_part,
        describe_image_info,
    )
    from nexus.runtime.executors.file import DownloadFileRequest, FileItem, download_file

    try:
        result = await download_file(
            DownloadFileRequest(files=[FileItem(path=file_path, encoding='base64')], max_bytes=READ_IMAGE_MAX_BYTES)
        )
    except Exception as exc:  # noqa: BLE001
        logger.error('read: image download failed, path=%s, error=%s', file_path, exc)
        return ToolResponse(output=_make_output(f'Error: {exc}'), meta={'ok': False})

    if not result.files:
        error = result.errors[0].message if result.errors else 'File does not exist'
        return ToolResponse(output=_make_output(f'Error: {error}'), meta={'ok': False})

    try:
        raw = _base64.b64decode(result.files[0].content or '')
    except Exception as exc:  # noqa: BLE001
        return ToolResponse(output=_make_output(f'Error: image base64 decode failed: {exc}'), meta={'ok': False})

    try:
        part, info = build_image_part(raw)
    except ImageRejectedError as exc:
        logger.info('read: image rejected, path=%s, reason=%s', file_path, exc)
        return ToolResponse(
            output=_make_output(f'Error: cannot display image {file_path}: {exc}'),
            meta={'ok': False, 'error': {'code': 'IMAGE_REJECTED', 'reason': str(exc)}},
        )

    logger.debug('read: image ok, path=%s, %s', file_path, describe_image_info(info))
    return ToolResponse(
        output=[
            part,
            {'type': 'text', 'text': f'Read image {file_path} ({describe_image_info(info)})'},
        ],
        meta={'ok': True, 'file_type': 'image', **info},
    )


@tool_registry.register(
    'yb_one_agent/read_v3',
    meta={
        # Same public name as v2: the model sees one ``read`` either way, and a
        # trajectory taken with v3 stays comparable with the text-only ones.
        'name': 'read',
        'description': (
            'Reads a file from the filesystem, including images.\n\n'
            'Usage:\n'
            '- file_path must be an absolute path.\n'
            '- By default reads up to 2000 lines from the beginning.\n'
            '- Use offset and limit for large files.\n'
            '- Returns content in cat -n format with line numbers.\n'
            '- Can read text files, source code, configs, etc.\n'
            '- Reading an image file (PNG, JPG, JPEG, GIF, WEBP, BMP) returns the picture '
            'itself. Re-reading the same path returns the same picture, not a sharper one.\n'
            '- Only a file in the sandbox can be read. When you have a URL and no path, '
            'download it first with bash and read the path you saved it to.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def read_tool_v3(params: ReadInputV2, context: ToolContext) -> ToolResponse:
    # Input schema is identical to v2's, so it reuses ReadInputV2 rather than
    # declaring a copy — the two tools must stay indistinguishable to the model.
    from nexus.runtime.executors.file import DownloadFileRequest, FileItem, download_file

    logger.debug('read: file_path=%s, offset=%s, limit=%s', params.file_path, params.offset, params.limit)

    # Before the extension gate: a URL or /dev/* path is the model looking for a
    # file that was never on disk, and a generic error just sends it guessing at
    # the next placeholder.
    not_a_file = _not_a_file_response(params.file_path)
    if not_a_file is not None:
        logger.debug('read: not a local file, path=%s', params.file_path)
        return not_a_file

    # Same gate as v2 minus the image extensions, which the branch below handles.
    unsupported_ext = _unsupported_extension(params.file_path, _READ_V3_UNSUPPORTED_EXTS)
    if unsupported_ext is not None:
        logger.debug('read: unsupported file type (ext), path=%s, ext=%s', params.file_path, unsupported_ext)
        return _unsupported_response(unsupported_ext, guidance=True)

    if os.path.splitext(params.file_path)[1].lower() in IMAGE_EXTS:
        return await _read_image(params.file_path, context)

    try:
        result = await download_file(DownloadFileRequest(files=[FileItem(path=params.file_path, encoding='utf8')]))
        if not result.files:
            error = result.errors[0].message if result.errors else 'File does not exist'
            logger.warning('read: file not found, path=%s, error=%s', params.file_path, error)
            return ToolResponse(output=_make_output(f'Error: {error}'))

        content = result.files[0].content or ''

        # 二进制内容检测（读到的前若干字节含 null byte）
        if _is_binary_content(content.encode('utf-8', errors='replace')[:BINARY_CHECK_SIZE]):
            logger.debug('read: unsupported file type (binary), path=%s', params.file_path)
            return _unsupported_response('binary')

        lines = content.splitlines()

        # Apply offset and limit (1-indexed offset)
        offset = max(params.offset - 1, 0)
        limit = params.limit
        selected = lines[offset : offset + limit]

        # Format as cat -n
        formatted = '\n'.join(f'{offset + i + 1:6}\t{line}' for i, line in enumerate(selected))
        if len(lines) > offset + limit:
            formatted += f'\n... ({len(lines) - offset - limit} more lines)'

        logger.debug('read: success, path=%s, total_lines=%d, returned=%d', params.file_path, len(lines), len(selected))
        return ToolResponse(output=_make_output(maybe_truncate(formatted)))

    except Exception as e:
        logger.error('read: exception, path=%s, error=%s', params.file_path, e)
        return ToolResponse(output=_make_output(f'Error: {e}'))


# ============== Edit ==============


class EditInputV2(_DescribedInput):
    description: str = Field(
        description='Why make these edits.',
    )
    file_path: str = Field(description='The absolute path to the file to modify.')
    old_string: str = Field(description='The exact text to replace. Must be unique in the file.')
    new_string: str = Field(description='The text to replace it with.')
    replace_all: bool = Field(default=False, description='Replace all occurrences instead of just the first.')


@tool_registry.register(
    'yb_one_agent/edit_v2',
    meta={
        'name': 'edit',
        'description': (
            'Performs exact string replacement in a file.\n\n'
            'Usage:\n'
            '- old_string must match the file content exactly (including whitespace and indentation).\n'
            '- old_string must be unique in the file, or use replace_all=true.\n'
            '- The edit will fail if old_string is not found or is ambiguous.\n'
            '- Always read the file with read before editing to get the exact content.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def edit_tool_v2(params: EditInputV2, context: ToolContext) -> ToolResponse:
    from nexus.runtime.executors.file import (
        DownloadFileRequest,
        FileItem,
        UploadFileRequest,
        download_file,
        upload_file,
    )

    logger.debug(
        'edit: file_path=%s, old_string_len=%d, new_string_len=%d, replace_all=%s',
        params.file_path,
        len(params.old_string),
        len(params.new_string),
        params.replace_all,
    )

    # v3: 统一 unsupported-file-type 门禁（如 .ipynb，字符串替换会破坏 JSON 结构）
    unsupported_ext = _unsupported_extension(params.file_path)
    if unsupported_ext is not None:
        logger.debug('edit: unsupported file type (ext), path=%s, ext=%s', params.file_path, unsupported_ext)
        return _unsupported_response(unsupported_ext)

    # v3: old_string == new_string → NO_CHANGES（for_model 对齐 for_model.go）
    if params.old_string == params.new_string:
        logger.debug('edit: no-op edit (old_string == new_string), path=%s', params.file_path)
        return ToolResponse(
            output=_make_output('No changes made: old_string and new_string are identical.'),
            meta={'ok': False, 'error': {'code': 'NO_CHANGES'}},
        )

    try:
        # Read current content
        result = await download_file(DownloadFileRequest(files=[FileItem(path=params.file_path, encoding='utf8')]))
        if not result.files:
            error = result.errors[0].message if result.errors else 'File does not exist'
            logger.warning('edit: file not found, path=%s, error=%s', params.file_path, error)
            return ToolResponse(output=_make_output(f'Error: {error}'))

        content = result.files[0].content or ''

        if params.old_string not in content:
            logger.warning('edit: old_string not found in %s', params.file_path)
            return ToolResponse(
                output=_make_output(
                    f'Error: old_string not found in {params.file_path}. '
                    'Make sure it matches exactly including whitespace.'
                )
            )

        if not params.replace_all:
            count = content.count(params.old_string)
            if count > 1:
                logger.warning('edit: old_string is ambiguous, found %d times in %s', count, params.file_path)
                return ToolResponse(
                    output=_make_output(
                        f'Error: old_string appears {count} times in the file. '
                        'Provide more context to make it unique, or use replace_all=true.'
                    )
                )
            new_content = content.replace(params.old_string, params.new_string, 1)
        else:
            new_content = content.replace(params.old_string, params.new_string)

        # Write back
        upload_result = await upload_file(
            UploadFileRequest(
                files=[FileItem(path=params.file_path, content=new_content, encoding='utf8')],
                overwrite=True,
            )
        )
        upload_error = _first_upload_error(upload_result)
        if upload_error:
            logger.warning('edit: upload failed, path=%s, error=%s', params.file_path, upload_error)
            return ToolResponse(output=_make_output(f'Error: {upload_error}'))

        replacements = 1 if not params.replace_all else content.count(params.old_string)
        logger.debug('edit: success, path=%s, replacements=%d', params.file_path, replacements)
        return ToolResponse(
            output=_make_output(f'Successfully edited {params.file_path} ({replacements} replacement(s)).')
        )

    except Exception as e:
        logger.error('edit: exception, path=%s, error=%s', params.file_path, e)
        return ToolResponse(output=_make_output(f'Error: {e}'))


# ============== Write ==============


class WriteInputV2(_DescribedInput):
    description: str = Field(
        description='Why write the content into this file.',
    )
    file_path: str = Field(description='The absolute path to the file to write (will be created or overwritten).')
    content: str = Field(description='The full content to write to the file.')


@tool_registry.register(
    'yb_one_agent/write_v2',
    meta={
        'name': 'write',
        'description': (
            'Writes content to a file, creating it or overwriting if it already exists.\n\n'
            'Usage:\n'
            '- file_path must be an absolute path.\n'
            '- Prefer edit for modifying existing files — only use write for new files or full rewrites.\n'
            '- If modifying an existing file, use read first to review current content.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def write_tool_v2(params: WriteInputV2, context: ToolContext) -> ToolResponse:
    from nexus.runtime.executors.file import FileItem, UploadFileRequest, upload_file

    logger.debug('write: file_path=%s, content_len=%d', params.file_path, len(params.content))

    try:
        upload_result = await upload_file(
            UploadFileRequest(
                files=[FileItem(path=params.file_path, content=params.content, encoding='utf8')],
                overwrite=True,
            )
        )
        upload_error = _first_upload_error(upload_result)
        if upload_error:
            logger.warning('write: upload failed, path=%s, error=%s', params.file_path, upload_error)
            return ToolResponse(output=_make_output(f'Error: {upload_error}'))

        lines = params.content.count('\n') + 1
        logger.debug('write: success, path=%s, lines=%d', params.file_path, lines)
        return ToolResponse(output=_make_output(f'Successfully wrote {lines} lines to {params.file_path}.'))

    except Exception as e:
        logger.error('write: exception, path=%s, error=%s', params.file_path, e)
        return ToolResponse(output=_make_output(f'Error: {e}'))


# ============== Search Tools (web_fetch / web_search / picture_search / video_search) ==============
#
# 通过 tool_proxy (work_agent) 调用搜索服务。
# user_id 取 ctx.tool_state['search_tools_user_id']，缺省 'default_user'。


# 默认指向 ci-545 泳道的 tool_proxy HTTP 实例（北极星 Test/trpc.yb_agent.tool_proxy.http，
# 标签 env:ci-545）。这是 StatefulSetPlus 的 pod IP，pod 重建后可能变化；一旦变了用环境变量
# SEARCH_TOOLS_API_URL 覆盖即可，不必改代码。web_fetch 会在此泳道上走 v3.0-generate 版本。
_WORK_AGENT_TOOL_PROXY_URL = os.environ.get(
    'SEARCH_TOOLS_API_URL',
    'http://21.90.150.183:8003/batch_tool_call',
)
_WORK_AGENT_TOOL_PROXY_TIMEOUT = float(os.environ.get('SEARCH_TOOLS_API_TIMEOUT', '30'))
_WORK_AGENT_TOOL_VERSION = 'work_agent'
# web_fetch 单独引导到 ci-545 上的 v3.0-generate 部署：prompt 空时由服务端灰度开关决定
# 直出解析原文 / 公众号转 Markdown。其余搜索工具仍走 work_agent 版本，互不影响。
# 用环境变量兜底，方便切回旧版本或指向别的部署而不改代码。
_WEB_FETCH_TOOL_VERSION = os.environ.get('WEB_FETCH_TOOL_VERSION', 'v3.0-generate')
_WORK_AGENT_DEFAULT_USER = 'default_user'


def _get_work_agent_user_id(context: ToolContext) -> str:
    return context.tool_state.get('search_tools_user_id') or _WORK_AGENT_DEFAULT_USER


async def _call_work_agent_tool(
    tool_name: str,
    arguments: dict,
    user_id: str,
    version: str = _WORK_AGENT_TOOL_VERSION,
) -> tuple[bool, str]:
    payload = {
        'calls': [
            {
                'call_id': f'nexus-{tool_name.replace("/", "-")}',
                'name': tool_name,
                'version': version,
                'arguments': arguments,
            }
        ],
        'options': {},
        'context_in': {
            'user': {'user_id': user_id},
            'session': {'round': 1, 'history': [], 'raw_query': ''},
            'caller': {'call_source': 'training'},
        },
    }
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'nexus-yb-one-agent',
    }

    try:
        response = await asyncio.to_thread(
            requests.post,
            _WORK_AGENT_TOOL_PROXY_URL,
            json=payload,
            headers=headers,
            timeout=_WORK_AGENT_TOOL_PROXY_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else '?'
        logger.error('[search_tools_work_agent] HTTP error tool_name=%s status=%s', tool_name, status_code)
        return False, f'tool_proxy HTTP {status_code}: {exc}'
    except requests.exceptions.RequestException as exc:
        logger.error('[search_tools_work_agent] request failed tool_name=%s exc=%s', tool_name, exc)
        return False, f'tool_proxy request failed: {exc}'

    try:
        body = response.json()
    except ValueError:
        logger.error('[search_tools_work_agent] non-JSON response tool_name=%s', tool_name)
        return False, f'tool_proxy non-JSON response: {response.text[:200]}'

    global_status = body.get('status') or {}
    global_code = global_status.get('code', 0)
    if global_code != 0:
        msg = global_status.get('msg', '')
        return False, f'batch error code={global_code} msg={msg}'

    results = body.get('results') or []
    if not results:
        return False, 'tool_proxy returned empty results'

    result = results[0]
    call_status = result.get('status') or {}
    call_code = call_status.get('code', 0)

    chunks = result.get('for_model') or []
    texts = [c.get('content', '') for c in chunks if c.get('type') == 'text' and c.get('content')]
    text = '\n'.join(texts) if texts else '(无内容)'

    if call_code != 0:
        return False, text

    return True, text


# Shared description + input model for the ``web_fetch`` tool
# (RUNTIME ``yb_one_agent/web_fetch``).
WEB_FETCH_DESCRIPTION = (
    'Fetches content from a URL and returns it as markdown.\n\n'
    'Usage:\n'
    '- Use for retrieving documentation, web pages, or any HTTP-accessible content.\n'
    '- The prompt describes what information you want to extract.\n'
    '- prompt is optional: leave it empty to return the full parsed content of the page, '
    'or provide it to run the extraction model and return only the extracted information.\n'
    '- HTTP URLs are automatically upgraded to HTTPS.'
)


class WebFetchInput(BaseModel):
    url: str = Field(description='The URL to fetch content from.')
    prompt: str = Field(
        default='',
        description='What information to extract from the page. Leave empty to return the full parsed content.',
    )


class WebSearchInput(_DescribedInput):
    description: str = Field(description='为什么执行此次网页搜索')
    query: list[str] = Field(description='一组用于文本检索的子任务查询query')


class PictureSearchInput(_DescribedInput):
    description: str = Field(description='为什么执行此次图片搜索')
    query: list[str] = Field(description='一组用于图片检索的子任务查询query')


class VideoSearchInput(_DescribedInput):
    description: str = Field(description='为什么执行此次视频搜索')
    query: list[str] = Field(description='一组用于视频检索的子任务查询query')


@tool_registry.register(
    'yb_one_agent/web_fetch',
    meta={
        'name': 'web_fetch',
        'description': WEB_FETCH_DESCRIPTION,
        'tool_type': ToolType.RUNTIME,
    },
)
async def web_fetch_tool(params: WebFetchInput, context: ToolContext) -> ToolResponse:
    _, text = await _call_work_agent_tool(
        'web_fetch',
        {'url': params.url, 'prompt': params.prompt},
        _get_work_agent_user_id(context),
        version=_WEB_FETCH_TOOL_VERSION,
    )
    return ToolResponse(output=_make_output(text))


@tool_registry.register(
    'yb_one_agent/web_search',
    meta={
        'name': 'web_search',
        'description': '文本搜索工具，基于query执行全文检索，返回相关文本结果。',
        'tool_type': ToolType.LOCAL,
    },
)
async def web_search_tool(params: WebSearchInput, context: ToolContext) -> ToolResponse:
    _, text = await _call_work_agent_tool(
        'web_search',
        {'query': params.query},
        _get_work_agent_user_id(context),
    )
    return ToolResponse(output=_make_output(text))


@tool_registry.register(
    'yb_one_agent/picture_search',
    meta={
        'name': 'picture_search',
        'description': '图片检索工具，基于query搜索相关图片，返回图片URL、标题等元信息。',
        'tool_type': ToolType.LOCAL,
    },
)
async def picture_search_tool(params: PictureSearchInput, context: ToolContext) -> ToolResponse:
    _, text = await _call_work_agent_tool(
        'picture_search',
        {'query': params.query},
        _get_work_agent_user_id(context),
    )
    return ToolResponse(output=_make_output(text))


@tool_registry.register(
    'yb_one_agent/video_search',
    meta={
        'name': 'video_search',
        'description': '视频检索工具，基于query在视频库中执行语义检索，返回相关视频结果。',
        'tool_type': ToolType.LOCAL,
    },
)
async def video_search_tool(params: VideoSearchInput, context: ToolContext) -> ToolResponse:
    _, text = await _call_work_agent_tool(
        'video_search',
        {'query': params.query},
        _get_work_agent_user_id(context),
    )
    return ToolResponse(output=_make_output(text))


# ============== AskUserQuestion ==============


class _QuestionOption(BaseModel):
    label: str = Field(description='The display text for this option. Should be concise (1-5 words).')
    description: str = Field(description='Explanation of what this option means or what will happen if chosen.')


class _Question(BaseModel):
    question: str = Field(
        description='The complete question to ask the user. Should be clear, specific, and end with a question mark.'
    )
    header: str = Field(
        description='Very short label displayed as a chip/tag (max 12 chars). Examples: "Auth method", "Library", "Approach".'
    )
    options: list[_QuestionOption] = Field(
        description="The available choices for this question. Must have 2-4 options. There should be no 'Other' option, that will be provided automatically."
    )
    multi_select: bool = Field(
        default=False, description='Set to true to allow the user to select multiple options instead of just one.'
    )


class AskUserQuestionInput(BaseModel):
    questions: list[_Question] = Field(description='Questions to ask the user (1-4 questions)')


_INTERACTIVE_SIGNAL = '__ask_user_question_interactive__'
"""Sentinel stored in context_updates to signal the agent loop to pause for user input."""


def _format_answers(questions: list[_Question], chosen: list[list[str]]) -> str:
    """Format answered questions into the canonical tool response string.

    Args:
        questions: The list of questions asked.
        chosen: For each question, the list of chosen option labels (multi_select may have >1).

    Returns:
        Formatted string like:
        'User has answered your questions: "Q1"="A1"; "Q2"="A2". You can now continue with the user\'s answers in mind.'
    """
    parts = []
    for q, labels in zip(questions, chosen, strict=False):
        answer = ', '.join(labels)
        parts.append(f'"{q.question}"="{answer}"')
    answers_str = '; '.join(parts)
    return f"User has answered your questions: {answers_str}. You can now continue with the user's answers in mind."


@tool_registry.register(
    'yb_one_agent/ask_user_question',
    meta={
        'name': 'ask_user_question',
        'description': (
            'Asks the user multiple choice questions to gather information, clarify ambiguity, '
            'understand preferences, make decisions or offer them choices.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def ask_user_question_tool(params: AskUserQuestionInput, context: ToolContext) -> ToolResponse:
    strategy: str = context.tool_state.get('ask_user_question_strategy', 'interactive')

    if strategy == 'random':
        # Randomly select option(s) for each question and return immediately.
        chosen: list[list[str]] = []
        for q in params.questions:
            labels = [o.label for o in q.options]
            if q.multi_select:
                # Pick a random non-empty subset
                k = random.randint(1, len(labels))
                chosen.append(random.sample(labels, k))
            else:
                chosen.append([random.choice(labels)])
        content = _format_answers(params.questions, chosen)
        return ToolResponse(output=_make_output(content))

    # interactive strategy — signal the agent loop to pause and ask the user.
    questions_payload = params.model_dump()
    return ToolResponse(
        output=_make_output(_INTERACTIVE_SIGNAL),
        context_updates={
            'ask_user_question_pending': True,
            'ask_user_question_payload': questions_payload,
        },
    )


# ============== Handoff ==============


class HandoffInput(BaseModel):
    subagent_type: Literal['expert_agent'] = Field(
        description='The specialist to route to. Use "expert_agent" for the stronger model.',
        json_schema_extra={'enum': ['expert_agent']},
    )


@tool_registry.register(
    'yb_one_agent/handoff',
    meta={
        'name': 'handoff',
        'description': (
            'Hand off the current request to a more capable specialist model. '
            'Call this when the request is complex enough that a stronger model is required to answer it reliably.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def handoff_tool(params: HandoffInput, context: ToolContext) -> ToolResponse:
    """Defensive fallback; the agent loop must intercept handoff before execution."""
    return ToolResponse(
        output=_make_output('handoff must be intercepted by the agent loop'),
        meta={'error': True},
    )


# ============== LoadSkill ==============

_SKILLS_DIR = pathlib.Path('/data/skills')


class LoadSkillInput(BaseModel):
    skill_name: str = Field(description='要加载的技能名称')


@tool_registry.register(
    'yb_one_agent/load_skill',
    meta={
        'name': 'load_skill',
        'description': '按名称加载技能（skill），返回技能说明、绑定工具及参考资料',
        'tool_type': ToolType.RUNTIME,
    },
)
async def load_skill_tool(params: LoadSkillInput, context: ToolContext) -> ToolResponse:
    skill_md_path = _SKILLS_DIR / params.skill_name / 'SKILL.md'

    if not skill_md_path.exists():
        available = ', '.join(sorted(p.name for p in _SKILLS_DIR.iterdir() if p.is_dir()))
        return ToolResponse(
            output=_make_output(
                f'<skill_error name="{params.skill_name}">Skill not found. Available skills: {available}</skill_error>'
            )
        )

    raw = skill_md_path.read_text(encoding='utf-8')

    # Strip YAML frontmatter (split limit=2 to avoid cutting body on bare ---)
    body = raw
    if raw.startswith('---'):
        parts = raw.split('---', 2)
        if len(parts) >= 3:
            body = parts[2].strip()

    result = f'<skill_instructions>\n{body}\n</skill_instructions>'

    tools_path = _SKILLS_DIR / params.skill_name / 'additional_tools.json'
    if tools_path.exists():
        result += (
            '\n\n<additional_tools>\n'
            'This skill unlocks the following tools. They are available for tool calling exactly like your built-in tools — call them by name with the specified parameters.\n\n'
            f'{tools_path.read_text(encoding="utf-8").strip()}\n'
            '</additional_tools>'
        )

    return ToolResponse(output=_make_output(result))


# The Delegate tool launches a sub-agent (currently only ``deep_research``) to
# handle multi-source, evidence-heavy work the main agent should not do
# inline. Reply is a ``<delegate_result>`` XML block with a ref to the
# sub-agent's result.md — see
# the Delegate I/O contract defined below (DelegateInput /
# contract.


# ── Delegate tool I/O contract (V0.3.0 simplified schema) ───────────────────
# Single source of truth for what the main agent sends to Delegate(deep_research)
# and the ``<delegate_result>`` envelope the dispatcher returns. Lives here,
# alongside the ``delegate_tool`` dispatcher, so the contract and its only
# consumer stay together.


DELEGATE_TOOL_SCHEMA: dict = {
    'name': 'delegate',
    'description': (
        'Launch a specialized sub-agent to handle complex, multi-step tasks '
        'autonomously. Available SubAgents: deep_research - use for '
        'multi-source, evidence-heavy, comparison or report tasks; do not use '
        'for simple lookup, chit-chat or pure reasoning. '
    ),
    'input_schema': {
        'type': 'object',
        'properties': {
            'description': {
                'type': 'string',
                'description': 'A short (3-5 word) description of the task',
            },
            'prompt': {
                'type': 'string',
                'description': 'The task for the agent to perform',
            },
            'subagent_type': {
                'type': 'string',
                'description': 'The type of specialized agent to use for this task',
                'enum': ['deep_research'],
            },
        },
        'required': ['description', 'prompt', 'subagent_type'],
    },
}


class DelegateInput(BaseModel):
    """Pydantic model for the Delegate tool input."""

    description: str = Field(
        ...,
        description='A short (3-5 word) description of the task',
    )
    prompt: str = Field(
        ...,
        description='The task for the agent to perform',
    )
    subagent_type: Literal['deep_research'] = Field(
        ...,
        description='The type of specialized agent to use for this task',
    )


DelegateStatus = Literal['success', 'failure']


class DelegateResult(BaseModel):
    """Typed fields of the ``<delegate_result>`` envelope the main agent reads.

    Assembled by the dispatcher from the runner's derived status/result — NOT
    parsed from model output.
    """

    subagent_type: Literal['deep_research']
    status: DelegateStatus
    turns_used: int
    result: str  # path on success, error description on failure


# In-process registry mapping ExecutionContext.run_id → asyncio.Queue used by
# host code (e.g. the demo SSE loop) to receive live sub-agent turn events
# from the deep_research sub-agent. Kept here (instead of ctx.tool_state) so
# the queue object never enters ToolContext serialization for RUNTIME tools.
DELEGATE_SUBTURN_QUEUES: 'dict[str, Any]' = {}


def _build_delegate_result_xml(
    *,
    status: str,
    turns_used: int,
    result: str,
) -> str:
    """Compose the <delegate_result> XML returned to the main agent.

    Simplified schema:
      - status: success | failure
      - result: absolute path on success, error description on failure

    See ``DelegateResult`` above for field semantics. ``<`` and
    ``>`` in ``result`` are escaped so the canonical parser's unescape step is
    symmetrical.
    """
    safe_result = result.replace('<', '&lt;').replace('>', '&gt;')
    return '\n'.join([
        '<delegate_result>',
        '<subagent_type>deep_research</subagent_type>',
        f'<status>{status}</status>',
        f'<turns_used>{turns_used}</turns_used>',
        f'<result>{safe_result}</result>',
        '</delegate_result>',
    ])


@tool_registry.register(
    'yb_one_agent/delegate',
    meta={
        'name': 'delegate',
        'description': DELEGATE_TOOL_SCHEMA['description'],
        'tool_type': ToolType.LOCAL,
    },
)
async def delegate_tool(params: DelegateInput, context: ToolContext) -> ToolResponse:
    """Dispatch a sub-agent run and return its ``<delegate_result>`` XML block.

    Currently only ``subagent_type='deep_research'`` is supported. The
    sub-agent itself uses the ``write`` tool inside the sandbox to land
    ``result.md``; this dispatcher trusts that path and only assembles the
    ``<delegate_result>`` envelope the main agent reads.

    On any host-side failure (sub-agent registration missing, runner
    exception, empty report) we return a structured ``<delegate_result>``
    with ``<status>failure</status>`` so the main agent has something
    deterministic to react to.
    """

    def _fail(msg: str, *, turns: int = 0) -> ToolResponse:
        return ToolResponse(
            output=_make_output(_build_delegate_result_xml(status='failure', turns_used=turns, result=msg))
        )

    if params.subagent_type != 'deep_research':
        return _fail(f'unsupported subagent_type {params.subagent_type!r}')

    try:
        from nexus.extensions.agents.yb_one_agent.sub_agents.deep_research.agent import (
            DeepResearchInput,
            run_deep_research,
        )

        logger.info('Successfully imported deep_research sub-agent')
    except Exception as e:  # noqa: BLE001
        logger.error(f'Failed to import deep_research sub-agent: {e}')
        return _fail(f'sub-agent unavailable: {type(e).__name__}: {e}')

    try:
        logger.info(f'Calling deep_research with prompt: {params.prompt[:100]}...')
        max_iter = int(os.environ.get('DR_MAX_TURNS', '20'))
        sub_out = await run_deep_research(
            DeepResearchInput(
                user_prompt=params.prompt,
                max_iterations=max_iter,
            )
        )
        logger.info(f'Deep research returned: status={sub_out.status}, iterations={sub_out.iterations}')
    except Exception as e:  # noqa: BLE001
        logger.error(f'Deep research execution failed: {e}')
        return _fail(f'sub-agent crashed: {type(e).__name__}: {e}')

    iterations = int(sub_out.iterations or 0)
    sub_result = (sub_out.result or '').strip()
    if not sub_result:
        return _fail('sub-agent produced empty result', turns=iterations)

    # Stash sub-agent history on the main agent's tool_state so the calling
    # SSE stream / demo UI can render it without leaking through the
    # <delegate_result> XML the main LLM sees. Append-only list keyed by
    # call index. The demo reads + clears it after emitting to the client.
    try:
        from nexus.core.context import require_context as _require_context

        _ctx = _require_context()
        history_log = _ctx.tool_state.setdefault('delegate_subhistory', [])
        history_log.append({
            'subagent_type': params.subagent_type,
            'description': params.description,
            'prompt': params.prompt,
            'status': sub_out.status,
            'iterations': iterations,
            'result': sub_result,
            'history': sub_out.history,
        })
    except Exception:  # noqa: BLE001
        # Best-effort — never break the main loop on instrumentation failure.
        pass

    # ``sub_out.result`` is the canonical <result> field already parsed by
    # the runner: success → absolute sandbox path; failure → error string.
    # The sub-agent harness (_finalize / _salvage_via_write /
    # _find_written_result) already guarantees a success path points at a file
    # the model actually wrote (or salvaged via its own write call), so we pass
    # the result through verbatim.
    #
    # We deliberately do NOT re-probe the path here: the sub-agent's ``write``
    # is a RUNTIME tool writing inside the runtime CONTAINER, and a dispatcher-
    # side probe via the local file executor reads the HOST fs — under
    # docker/gongfeng runtimes that always reports "File does not exist" and
    # would wrongly downgrade a perfectly good result to failure.
    status = 'success' if sub_out.status == 'success' else 'failure'
    return ToolResponse(
        output=_make_output(
            _build_delegate_result_xml(
                status=status,
                turns_used=iterations,
                result=sub_result,
            )
        )
    )
