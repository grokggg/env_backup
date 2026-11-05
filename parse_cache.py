"""parse_file 结果的本地【内容级】磁盘缓存。

推理链路里真正昂贵的操作是「COS 上传 + MCP parse_file 解析」。同一份输入文件
（字节完全相同）在多模型对比、多次实验、断点续跑等场景中会被反复解析，纯属浪费。

本模块以【文件内容的 sha256】为 key，把 parse_file 返回的纯文本结果
(``parse_result``) 落到本地磁盘。推理时命中即复用；配套的
``scripts/prebuild_parse_cache.py`` 可在推理前批量预热整份 benchmark。

为什么用内容 hash 而非文件路径做 key：
    1. 内容相同的文件天然去重（跨 task / 跨 benchmark 版本共享一份解析结果）；
    2. benchmark 更新了某个文件后，其 hash 变化 → 旧缓存自动失效，不会脏读。

缓存目录解析顺序：
    1. 环境变量 ``ONE_AGENT_PARSE_CACHE_DIR``（最高优先，绝对/相对路径均可）；
    2. 基于 benchmark_root 向上寻找名为 ``benchmarks`` 的目录，放其下 ``parse_cache``
       （这样同一份 files 内容在不同版本目录间共享同一缓存，且适合提交到 git 供团队复用）；
    3. 兜底：当前工作目录下 ``parse_cache``。

磁盘布局（按 sha 前两位分桶，避免单目录文件过多）：
    <cache_dir>/<sha[:2]>/<sha>.json
        {sha256, filename, suffix, parser_version, parse_result}

Git 友好设计：
    - 目录名不带点前缀（``parse_cache`` 而非 ``.parse_cache``），作为正式目录入 git。
    - 缓存 JSON 不含 ``parsed_at`` 等随运行变化的字段，同一输入文件无论谁解析、
      何时解析，产出的 JSON 完全一致（确定性），不会产生无意义 diff。
    - 排序输出 JSON key，保证字节级可复现。

一致性约定：
    - 仅缓存【成功且非空】的解析结果；parse_file 调用失败不写缓存，下次可重试。
    - ``PARSER_VERSION`` 是解析参数/逻辑的指纹，与 get_file_preview 里
      ``_build_parse_file_arguments`` 的参数保持对应；改动解析参数时应 bump 版本，
      使旧缓存自动失效（load 时校验版本一致才命中）。
    - 写入采用「临时文件 + os.replace」原子替换，多协程并发写同一 key 不会损坏。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile

from pathlib import Path


# 解析参数/逻辑指纹：对应 get_file_preview._build_parse_file_arguments
# （no_xml=True, gen_summary=True, file_parse_param{disable_youtu=False, return_mode=0}）。
# 若上述解析参数变化，请同步 bump 此版本号使旧缓存失效。
PARSER_VERSION = 'v1-noxml-gensummary'

_ENV_CACHE_DIR = 'ONE_AGENT_PARSE_CACHE_DIR'
_ENV_ENABLED = 'ONE_AGENT_PARSE_CACHE'  # '0' 显式关闭；其余（含未设置）视为开启


def cache_enabled() -> bool:
    """缓存开关。默认开启；设 ``ONE_AGENT_PARSE_CACHE=0`` 可关闭。"""
    return os.environ.get(_ENV_ENABLED, '1') != '0'


def resolve_cache_dir(benchmark_root: str | os.PathLike | None = None) -> Path:
    """解析缓存根目录，见模块 docstring 的三级优先级。"""
    env = os.environ.get(_ENV_CACHE_DIR)
    if env:
        return Path(env).expanduser()

    if benchmark_root is not None:
        try:
            root = Path(benchmark_root).resolve()
        except Exception:
            root = Path(benchmark_root)
        for p in (root, *root.parents):
            if p.name == 'benchmarks':
                return p / 'parse_cache'
        return root / 'parse_cache'

    return Path.cwd() / 'parse_cache'


def content_sha256(raw: bytes) -> str:
    """文件字节的 sha256 十六进制字符串。"""
    return hashlib.sha256(raw).hexdigest()


def _entry_path(cache_dir: Path, sha: str) -> Path:
    return cache_dir / sha[:2] / f'{sha}.json'


def load(cache_dir: str | os.PathLike, sha: str) -> str | None:
    """读取缓存的 parse_result；未命中 / 版本不符 / 损坏时返回 None。"""
    p = _entry_path(Path(cache_dir), sha)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text('utf-8'))
    except (OSError, ValueError):
        return None
    if data.get('parser_version') != PARSER_VERSION:
        return None
    return data.get('parse_result') or None


def save(
    cache_dir: str | os.PathLike,
    sha: str,
    filename: str,
    suffix: str,
    parse_result: str,
) -> bool:
    """原子写入一条缓存。仅当 parse_result 非空时落盘，返回是否写入。"""
    if not parse_result:
        return False

    p = _entry_path(Path(cache_dir), sha)
    p.parent.mkdir(parents=True, exist_ok=True)

    record = {
        'filename': filename,
        'parse_result': parse_result,
        'parser_version': PARSER_VERSION,
        'sha256': sha,
        'suffix': suffix,
    }

    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, sort_keys=True)
            f.write('\n')  # 结尾换行，git-friendly
        os.replace(tmp, p)  # 原子替换
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return True


def has(cache_dir: str | os.PathLike, sha: str) -> bool:
    """判断某 sha 是否已有有效缓存（供离线脚本快速跳过）。"""
    return load(cache_dir, sha) is not None
