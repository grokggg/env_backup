"""Image input resolver for ``image_generation/generate``.

Agent contract: ``input_images`` items are typed ``{type, value}`` structs.
The ``value`` string is **one of**

  - public image URL       ``http(s)://``   → fetched + base64-encoded
  - sandbox absolute path  ``/data/...``    → downloaded from runtime
                                              container + base64-encoded
Both paths produce a ``data:image/...;base64,...`` URL that the backend
can consume directly.

Sandbox path restriction: only ``/data/`` prefix is allowed.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from nexus.runtime.client.base import Runtime


log = logging.getLogger('image_generation.resolver')


# ── Limits / config ──────────────────────────────────────────────────────────

_DEFAULT_MAX_BYTES = 20 * 1024 * 1024
_DEFAULT_FETCH_TIMEOUT_S = 30.0
_SANDBOX_PREFIX = '/data/'


def _max_bytes() -> int:
    raw = os.environ.get('IMAGE_GEN_MAX_BYTES', '').strip()
    if raw.isdigit():
        return max(1, int(raw))
    return _DEFAULT_MAX_BYTES


def _fetch_timeout() -> float:
    raw = os.environ.get('IMAGE_GEN_FETCH_TIMEOUT', '').strip()
    try:
        return float(raw) if raw else _DEFAULT_FETCH_TIMEOUT_S
    except ValueError:
        return _DEFAULT_FETCH_TIMEOUT_S


# ── Helpers ──────────────────────────────────────────────────────────────────


def _encode_data_url(content: bytes, mime: str) -> str:
    return f'data:{mime};base64,{base64.b64encode(content).decode("ascii")}'


def _guess_mime_from_path(path: str, fallback: str = 'image/png') -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith('image/'):
        return mime
    return fallback


# ── URL fetcher ──────────────────────────────────────────────────────────────


async def _fetch_and_encode(url: str) -> str:
    """Fetch *url* and return a ``data:image/...;base64,...`` string."""
    import httpx

    max_bytes = _max_bytes()
    timeout_s = _fetch_timeout()

    try:
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            async with client.stream('GET', url) as resp:
                if resp.status_code != 200:
                    raise ValueError(f'HTTP {resp.status_code} fetching {url}')

                ct = (resp.headers.get('content-type') or '').split(';', 1)[0].strip().lower()
                mime = ct if ct.startswith('image/') else _guess_mime_from_path(url)

                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        raise ValueError(f'image at {url} exceeds {max_bytes // 1024 // 1024} MB limit')
                return _encode_data_url(bytes(buf), mime)
    except httpx.HTTPError as exc:
        raise ValueError(f'network error fetching {url}: {exc}') from exc


# ── Sandbox file reader ─────────────────────────────────────────────────────


async def _read_sandbox_file(path: str, runtime: Runtime) -> tuple[bytes, str]:
    """Download a file from the runtime sandbox via base64 transport."""
    if not path.startswith(_SANDBOX_PREFIX):
        raise ValueError(f'sandbox path must start with {_SANDBOX_PREFIX!r}, got {path!r}')

    from nexus.runtime.executors.file import FileItem

    resp = await runtime.download_file(files=[FileItem(path=path, encoding='base64')])
    if not resp.files:
        err_msg = '(no error message)'
        if resp.errors:
            err_msg = resp.errors[0].message or err_msg
        raise ValueError(f'sandbox file not readable {path!r}: {err_msg}')

    item = resp.files[0]
    if not item.content:
        raise ValueError(f'sandbox file {path!r} returned empty content')

    try:
        content = base64.b64decode(item.content)
    except Exception as exc:
        raise ValueError(f'sandbox file {path!r} base64 decode failed: {exc}') from exc

    max_bytes = _max_bytes()
    if len(content) > max_bytes:
        raise ValueError(f'sandbox file {path!r} exceeds {max_bytes // 1024 // 1024} MB limit')

    mime = _guess_mime_from_path(path)
    return content, mime


# ── Public entry point ──────────────────────────────────────────────────────


async def resolve_image_payload(
    item: str,
    runtime: Runtime | None = None,
) -> str:
    """Turn one ``input_images[].value`` into a ``data:image/...;base64,...`` URL.

    Two input types, all producing base64:

      1. ``http(s)://...`` URL → fetch + base64
      2. ``/data/...`` path   → download from sandbox → base64
    """
    if not isinstance(item, str) or not item:
        raise ValueError('empty image input')

    if item.startswith(_SANDBOX_PREFIX):
        if runtime is None:
            raise ValueError(f'sandbox path {item!r} requires a runtime (not available in this context)')
        content, mime = await _read_sandbox_file(item, runtime)
        return _encode_data_url(content, mime)

    if item.startswith(('http://', 'https://')):
        return await _fetch_and_encode(item)

    raise ValueError(
        f'unsupported image input {item[:80]!r}: must be a '
        f'sandbox path ({_SANDBOX_PREFIX}..., type=sandbox) '
        f'or http(s):// URL (type=url)'
    )


# Backwards-compat alias.
resolve_to_data_url = resolve_image_payload
