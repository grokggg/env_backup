"""Image helpers shared by the ``read`` tool and the user-upload path.

Turns raw image bytes into an OpenAI ``image_url`` content part that the VLM
backend will accept.  The hard requirement driving this module is that a
malformed image must **never** reach the messages array: because the chat API
is stateless, every turn resends the whole history, so one rejected content
block bricks the session for good.  Every failure therefore raises
:class:`ImageRejectedError` and callers degrade to plain text instead.

Format detection reads magic bytes rather than trusting the file extension —
a PNG saved as ``.jpg`` is a routine way to trigger an API 400.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os

from typing import Any


logger = logging.getLogger(__name__)


# ============== Limits ==============

#: Hard cap on the input bytes we are willing to decode at all.
MAX_INPUT_BYTES = int(os.environ.get('ONE_AGENT_IMAGE_MAX_BYTES', 20 * 1024 * 1024))
#: Reject absurd dimensions before handing anything to the decoder. Sized off
#: the real upload data (13.4k online images: p99 9.5k, max 45.5k px on the long
#: edge), so nothing a user actually sends is turned away for being large — the
#: earlier 8000 dropped 175 of them, and the 17 with ordinary proportions were
#: high-resolution scans that downscale to a perfectly readable ~860px short
#: edge. Note this guards the *decoder*, not usefulness: a long edge says
#: nothing about whether the image survives the downscale — a 1272x45531 phone
#: screenshot passes here and comes out 44px wide.
MAX_INPUT_DIMENSION = int(os.environ.get('ONE_AGENT_IMAGE_MAX_DIMENSION', 46000))
#: Maximum decoded pixel count; checked before loading pixel data.
MAX_INPUT_PIXELS = int(os.environ.get('ONE_AGENT_IMAGE_MAX_PIXELS', 40_000_000))
#: Long edge we downscale to. Capped at 2000, the hard per-image dimension limit
#: Anthropic enforces on *many-image* requests ("At least one of the image
#: dimensions exceed max allowed size for many-image requests: 2000 pixels"): a
#: multi-image turn (some MMU tasks ship 20+ inputs) 400s outright above it, and
#: because the API is stateless the rejected block bricks every later turn too.
#: HY/taiji has no such limit — this is set for the strictest backend the same
#: bytes may reach. The value stays essentially at HY3.0 VLM's ``vit-max-image-size``
#: (2048): larger inputs are tiled by the encoder's AnyRes anyway, and the student
#: model (HY3.0_A20B_VLM_SFT_v0.3_chris) gains no tokens past ~2048, so the 48px
#: haircut costs nothing. Override via env to push HY-only runs back to 2048.
TARGET_MAX_EDGE = int(os.environ.get('ONE_AGENT_IMAGE_TARGET_EDGE', 2000))
#: Visual-token budget for a single image. Aligned with HY3.0 VLM's
#: ``vit-img-max-token-num`` (4096) — a train-time constant baked into the
#: model's positional embeddings, i.e. a hard ceiling. Empirically the backend
#: saturates at ~4111 tokens (4096 image + a handful of system tokens) no
#: matter how large the input, so any resolution above the resulting
#: ~4.2 megapixels is encoded and then discarded.
MAX_IMAGE_TOKENS = int(os.environ.get('ONE_AGENT_IMAGE_MAX_TOKENS', 4096))
#: Pixels per visual token, i.e. the vision encoder's patch accounting.
#: ``vit-patch-size=16`` combined with ``vit-adaptor-patch-size=2`` (a 2x2
#: patch merge in the adapter) yields 32x32 = 1024 px per token. Matches the
#: measured ratio (~977 px/token) once system tokens are backed out.
PIXELS_PER_TOKEN = 1024
#: Cap on the encoded payload — the chat APIs reject oversized image parts.
#: Independent of the token budget: bytes are a transport limit, tokens are a
#: context limit, and compressing harder never changes the latter.
MAX_ENCODED_BYTES = int(os.environ.get('ONE_AGENT_IMAGE_MAX_ENCODED_BYTES', 5 * 1024 * 1024))

#: MIME types the chat APIs accept for inline image parts.
API_SUPPORTED_MIMES = frozenset({'image/png', 'image/jpeg', 'image/gif', 'image/webp'})

_JPEG_QUALITY_LADDER = (85, 70, 55, 40, 25)
#: When quality alone cannot meet the byte cap, resolution comes down in these
#: steps. 0.8 drops ~36% of the pixels, small enough that the first or second
#: step normally lands.
_SHRINK_FACTOR = 0.8
#: Long edge below which shrinking stops being worth it — a thumbnail of an
#: unreadable image tells the model less than the caller's text fallback does.
_MIN_LONG_EDGE = int(os.environ.get('ONE_AGENT_IMAGE_MIN_LONG_EDGE', 256))


class ImageRejectedError(ValueError):
    """Raised when bytes cannot be turned into a payload the API will accept."""


# ============== Magic-byte detection ==============

_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b'\x89PNG\r\n\x1a\n', 'image/png'),
    (b'\xff\xd8\xff', 'image/jpeg'),
    (b'GIF87a', 'image/gif'),
    (b'GIF89a', 'image/gif'),
    (b'BM', 'image/bmp'),
)


def detect_image_mime(data: bytes) -> str | None:
    """Return the MIME type implied by *data*'s leading bytes, or ``None``.

    Deliberately ignores any filename — the extension is exactly what cannot be
    trusted here.
    """
    if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    for signature, mime in _MAGIC_SIGNATURES:
        if data.startswith(signature):
            return mime
    return None


def is_image_bytes(data: bytes) -> bool:
    """True when *data* starts with a recognizable image signature."""
    return detect_image_mime(data) is not None


# ============== Encoding ==============


def _estimate_tokens(width: int, height: int) -> int:
    """Visual tokens consumed by a *width* x *height* image.

    A native image part is never run through the text tokenizer: the backend
    decodes it and the vision encoder charges by patch area.  Cost therefore
    depends on the pixel dimensions alone — base64 length, source format and
    file size do not enter into it.
    """
    return math.ceil(width * height / PIXELS_PER_TOKEN)


def _target_size(width: int, height: int) -> tuple[int, int]:
    """Largest size satisfying both the long-edge cap and the token budget."""
    scale = min(1.0, TARGET_MAX_EDGE / max(width, height))
    if _estimate_tokens(round(width * scale), round(height * scale)) > MAX_IMAGE_TOKENS:
        scale = math.sqrt(MAX_IMAGE_TOKENS * PIXELS_PER_TOKEN / (width * height))
    if scale >= 1.0:
        return width, height
    return max(1, round(width * scale)), max(1, round(height * scale))


def _encode(image: Any, mime: str, *, quality: int | None = None) -> bytes:
    fmt = {'image/png': 'PNG', 'image/jpeg': 'JPEG', 'image/gif': 'GIF', 'image/webp': 'WEBP'}[mime]
    buf = io.BytesIO()
    if fmt == 'JPEG':
        # JPEG has no alpha channel; flatten onto white so transparent PNGs
        # don't come out with black backgrounds.
        if image.mode in ('RGBA', 'LA', 'P'):
            from PIL import Image

            rgba = image.convert('RGBA')
            flat = Image.new('RGB', rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.split()[-1])
            image = flat
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buf, format='JPEG', quality=quality or 85, optimize=True)
    else:
        image.save(buf, format=fmt)
    return buf.getvalue()


def _fit_to_budget(image: Any, mime: str) -> tuple[bytes, str, tuple[int, int]]:
    """Downscale *image* to the token budget, then compress to the byte cap.

    Every returned payload is at or under :data:`MAX_ENCODED_BYTES`; when that
    is unreachable the image is rejected rather than handed over oversized.
    """
    from PIL import Image

    target = _target_size(*image.size)
    work = image.resize(target, Image.Resampling.LANCZOS) if target != image.size else image

    out_mime = mime if mime in API_SUPPORTED_MIMES else 'image/png'
    encoded = _encode(work, out_mime)
    if len(encoded) <= MAX_ENCODED_BYTES:
        return encoded, out_mime, target

    # Resolution is already at the budget, so only the transport cap is left.
    # Stepping down JPEG quality costs far less recognition accuracy than
    # shrinking further would, and leaves the token count untouched.
    out_mime = 'image/jpeg'
    for quality in _JPEG_QUALITY_LADDER:
        encoded = _encode(work, out_mime, quality=quality)
        if len(encoded) <= MAX_ENCODED_BYTES:
            return encoded, out_mime, target

    # Quality alone does not always get there: content close to noise stays
    # incompressible at every rung, so the ladder can run out while still over
    # the cap. Resolution is the only lever left. Returning the oversized part
    # anyway would break this module's one promise — an API that rejects it
    # bricks the session, and every later turn resends the same block.
    # A smaller image at moderate quality also keeps more structure than a
    # larger one crushed at q25, so each step retries quality from the middle.
    while round(max(target) * _SHRINK_FACTOR) >= _MIN_LONG_EDGE:
        target = (
            max(1, round(target[0] * _SHRINK_FACTOR)),
            max(1, round(target[1] * _SHRINK_FACTOR)),
        )
        work = image.resize(target, Image.Resampling.LANCZOS)
        for quality in (55, _JPEG_QUALITY_LADDER[-1]):
            encoded = _encode(work, out_mime, quality=quality)
            if len(encoded) <= MAX_ENCODED_BYTES:
                logger.warning(
                    'image needed a resolution cut to meet the %dKB cap: now %dx%d at q%d (%dKB)',
                    MAX_ENCODED_BYTES // 1024,
                    target[0],
                    target[1],
                    quality,
                    len(encoded) // 1024,
                )
                return encoded, out_mime, target

    raise ImageRejectedError(
        f'image cannot be compressed below {MAX_ENCODED_BYTES // 1024}KB without '
        f'shrinking past {_MIN_LONG_EDGE}px on the long edge'
    )


def build_image_part(raw: bytes) -> tuple[dict, dict]:
    """Validate *raw* image bytes and build an ``image_url`` content part.

    Returns ``(part, info)``; *info* carries mime / dimensions / byte sizes so
    callers can render a companion text part.  Raises :class:`ImageRejectedError`
    when the bytes are unusable — callers must fall back to text.
    """
    if not raw:
        raise ImageRejectedError('image file is empty')
    if len(raw) > MAX_INPUT_BYTES:
        raise ImageRejectedError(
            f'image is {len(raw) // 1024 // 1024} MB, exceeds the {MAX_INPUT_BYTES // 1024 // 1024} MB limit'
        )

    mime = detect_image_mime(raw)
    if mime is None:
        raise ImageRejectedError('unrecognized image format (magic-byte detection failed)')

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - Pillow is a declared dependency
        raise ImageRejectedError(f'Pillow unavailable: {exc}') from exc

    # verify() catches truncated/corrupt files but leaves the instance unusable,
    # so the image has to be opened a second time for the real work.
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            width, height = probe.size
            if max(width, height) > MAX_INPUT_DIMENSION:
                raise ImageRejectedError(f'image is {width}x{height}, exceeds the {MAX_INPUT_DIMENSION}px limit')
            if width * height > MAX_INPUT_PIXELS:
                raise ImageRejectedError(
                    f'image has {width * height} pixels, exceeds the {MAX_INPUT_PIXELS} pixel limit'
                )
            probe.verify()
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:  # noqa: BLE001 - any decoder failure is a rejection
        raise ImageRejectedError(f'image could not be decoded: {exc}') from exc

    original_width, original_height = image.size
    image = ImageOps.exif_transpose(image) or image

    fits_as_is = (
        mime in API_SUPPORTED_MIMES and _target_size(*image.size) == image.size and len(raw) <= MAX_ENCODED_BYTES
    )
    if fits_as_is:
        encoded, out_mime, out_size = raw, mime, image.size
    else:
        encoded, out_mime, out_size = _fit_to_budget(image, mime)

    b64 = base64.b64encode(encoded).decode('ascii')
    part = {'type': 'image_url', 'image_url': {'url': f'data:{out_mime};base64,{b64}'}}
    info = {
        'mime': out_mime,
        'original_mime': mime,
        'original_size': len(raw),
        'encoded_size': len(encoded),
        'original_width': original_width,
        'original_height': original_height,
        'width': out_size[0],
        'height': out_size[1],
        'estimated_tokens': _estimate_tokens(*out_size),
    }
    return part, info


def describe_image_info(info: dict) -> str:
    """One-line human/model readable summary of a processed image."""
    size_kb = max(1, info['original_size'] // 1024)
    text = f'{info["original_width"]}x{info["original_height"]}, {size_kb}KB, {info["original_mime"]}'
    width, height = info.get('width'), info.get('height')
    if (width, height) != (info['original_width'], info['original_height']):
        text += f' (downscaled to {width}x{height})'
    elif info['encoded_size'] != info['original_size']:
        text += f' (recompressed to {max(1, info["encoded_size"] // 1024)}KB {info["mime"]})'
    return text


# ============== Content assembly ==============


def normalize_content(parts: list[Any]) -> str | list[dict]:
    """Collapse *parts* into message content.

    Adjacent text parts are merged, and a run of nothing but text degrades back
    to a plain ``str`` so text-only conversations serialize exactly as they did
    before multimodal support existed.
    """
    merged: list[dict] = []
    for part in parts:
        if part is None:
            continue
        item = {'type': 'text', 'text': part} if isinstance(part, str) else part
        if not item:
            continue
        if item.get('type') == 'text':
            if not item.get('text'):
                continue
            if merged and merged[-1].get('type') == 'text':
                merged[-1] = {'type': 'text', 'text': merged[-1]['text'] + item['text']}
                continue
        merged.append(item)

    if not merged:
        return ''
    if all(item.get('type') == 'text' for item in merged):
        return ''.join(item['text'] for item in merged)
    return merged


def join_blocks(
    blocks: list[Any],
    *,
    separator: str = '\n\n',
    tail: str = '',
) -> str | list[dict]:
    """Join attachment blocks into message content.

    Each block is either a plain string (``[user_doc N]``) or a list of content
    parts (``[user_image N]``, which embeds a real image part).  When every
    block is text the result collapses back to a single string, so text-only
    prompts keep byte-for-byte the same shape as before.
    """
    parts: list[Any] = []
    for i, block in enumerate(blocks):
        if i:
            parts.append({'type': 'text', 'text': separator})
        if isinstance(block, str):
            parts.append({'type': 'text', 'text': block})
        else:
            parts.extend(block)
    if tail:
        parts.append({'type': 'text', 'text': tail})
    return normalize_content(parts)


def content_to_text(content: Any) -> str:
    """Flatten message content to text, replacing image parts with a placeholder."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or '')
    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            chunks.append(str(item))
        elif item.get('type') == 'text':
            chunks.append(item.get('text', ''))
        elif item.get('type') == 'image_url':
            chunks.append('[image]')
    return ''.join(chunks)
