"""Image-generation Pydantic models and shared helpers.

This module contains ONLY data definitions and helper functions — no
``@tool_registry.register`` decorators.  Importing it never triggers
tool registration side effects, so it can be safely used by:

- ``mock_tools.py`` (registers mock tool implementation)
- ``real_tools.py`` (registers real backend tool implementation)
- Skill plugins that need ``GenerateInput`` for validation
"""

from __future__ import annotations

import json
import re

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from nexus.core import ToolResponse


# ── Response helpers ──────────────────────────────────────────────────────────


def extract_real_backend_error(rec: dict) -> str | None:
    """Best-effort: dig the ``topic`` (e.g. "图片生成失败") from raw SSE chunks."""
    raw = rec.get('raw_chunks_debug')
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        chunks = json.loads(raw)
    except Exception:
        return None
    if not isinstance(chunks, list):
        return None
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for rep in chunk.get('replaces') or []:
            if isinstance(rep, dict):
                topic = rep.get('topic')
                if isinstance(topic, str) and topic.strip():
                    return topic.strip()
    return None


def ok(text: str) -> ToolResponse:
    return ToolResponse(output=[{'type': 'text', 'text': f'[ok] true\n{text}'}])


def err(msg: str) -> ToolResponse:
    return ToolResponse(
        output=[{'type': 'text', 'text': f'[ok] false\n[error] {msg}'}],
        meta={'error': True},
    )


def format_output_images(urls: list[str]) -> str:
    if not urls:
        return '[output_images] (none)'
    lines = [f'[output_images] {len(urls)}']
    for i, url in enumerate(urls, 1):
        lines.append(f'[image {i}]')
        lines.append(f'[url] {url}')
    return '\n'.join(lines)


# ── Input models ──────────────────────────────────────────────────────────────


class ImageInput(BaseModel):
    """A single typed image input for image_generation/generate."""

    type: Literal['url', 'sandbox'] = Field(description='url | sandbox')
    value: str = Field(description='对应类型的值')


def coerce_input_images(raw: Any) -> list[ImageInput]:
    """Normalize various LLM tool-call shapes to ``list[ImageInput]``.

    Each element must carry an explicit ``type`` (``url`` / ``sandbox``).
    Elements without a valid ``type`` are silently dropped.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        if s.startswith('['):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return coerce_input_images(parsed)
            except Exception:
                pass
        return []
    if isinstance(raw, list):
        out: list[ImageInput] = []
        for x in raw:
            if isinstance(x, dict):
                t = str(x.get('type', '')).strip()
                v = str(x.get('value', '')).strip()
                if t and v:
                    out.append(ImageInput(type=t, value=v))  # type: ignore[arg-type]
            elif isinstance(x, str):
                xs = x.strip()
                if xs.startswith('[') and xs.endswith(']'):
                    try:
                        parsed = json.loads(xs)
                        if isinstance(parsed, list):
                            out.extend(coerce_input_images(parsed))
                            continue
                    except Exception:
                        pass
            elif isinstance(x, ImageInput):
                out.append(x)
        return out
    return []


class DstIntent(str, Enum):
    TEXT2IMAGE = 'Text2Image'
    IMAGE2IMAGE = 'Image2Image'


VALID_DST_INTENTS = frozenset({DstIntent.TEXT2IMAGE, DstIntent.IMAGE2IMAGE})


#: 单次调用允许的最大输入图张数（多图融合/风格迁移/元素合成等）。
MAX_INPUT_IMAGES = 3


_RATIO_RE = re.compile(r'^\d{1,3}:\d{1,3}$')
_RATIO_REF_RE = re.compile(r'^ref:\d+$')


def coerce_image_ratio(v: Any) -> str:
    """Normalize image_ratio to '' (auto), a 'W:H' integer ratio, or 'ref:N'.

    The reference form is ``ref:N`` ONLY — no other spellings are accepted.
    'W:H' tolerates full-width colon and ``16x9`` / ``16*9`` separators.
    Anything else falls back to '' (auto).
    """
    if v is None:
        return ''
    s = str(v).strip()
    if not s:
        return ''
    low = s.lower()
    if _RATIO_REF_RE.match(low):
        return low
    norm = s.replace('：', ':').replace('x', ':').replace('X', ':').replace('*', ':')
    if _RATIO_RE.match(norm):
        return norm
    return ''


class GenerateInput(BaseModel):
    dst_intent: str = Field(description='Text2Image 或 Image2Image')
    prompt: str = Field(description='图像描述')
    input_images: list[ImageInput] = Field(default_factory=list, description='图生图时的输入图片列表')
    seed: int | None = Field(default=None, ge=1, le=10, description='随机种子 1-10，用于复现同一构图结果（可选）')
    image_ratio: str = Field(
        default='',
        description='输出图比例。可填 "W:H" 整数比（如 16:9 / 3:4 / 1:1），'
        '或 "ref:N" 跟随第 N 张输入图的比例（1-based，N 同 input_images 顺序、也是 prompt 里的「第 N 张图」）。'
        '留空为 auto：图生图默认跟随第一张输入图，文生图用后端默认。',
    )

    @field_validator('input_images', mode='before')
    @classmethod
    def _normalize_input_images(cls, v: Any) -> list[ImageInput]:
        return coerce_input_images(v)

    @field_validator('image_ratio', mode='before')
    @classmethod
    def _coerce_image_ratio(cls, v: Any) -> str:
        return coerce_image_ratio(v)

    @field_validator('seed', mode='before')
    @classmethod
    def _coerce_seed(cls, v: Any) -> int | None:
        """Tolerant coercion: agents sometimes send seed as string or float."""
        if v is None:
            return None
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return None
        if iv < 1 or iv > 10:
            return None
        return iv

    def image_values(self) -> list[str]:
        """Extract raw value strings for resolver / backend consumption."""
        return [img.value for img in self.input_images]

    def combination_error(self) -> str | None:
        """Validate the whole parameter combination before calling the downstream tool.

        Returns the first violated rule as a human-readable string (surfaced as a
        tool error), or None when every cross-field rule holds. Centralizing the
        rules here keeps the mock / real tool entrypoints in sync and lets the
        model self-correct from a single clear message instead of silently
        producing a wrong image.
        """
        if self.dst_intent not in VALID_DST_INTENTS:
            return f'dst_intent 必须是 Text2Image 或 Image2Image，got "{self.dst_intent}"'
        if not self.prompt.strip():
            return 'prompt 不能为空'

        n = len(self.input_images)
        is_ref_ratio = self.image_ratio.startswith('ref:')

        if self.dst_intent == DstIntent.TEXT2IMAGE:
            if n:
                return f'文生图（Text2Image）不应传入 input_images，请传 []（当前传入 {n} 张）'
            if is_ref_ratio:
                return '文生图无输入图可参考，image_ratio 不能用 ref:N，请用 "W:H" 整数比或留空'
            return None

        # Image2Image
        if n == 0:
            return '图生图（Image2Image）必须提供至少 1 条 input_images'
        if self.seed is not None:
            return 'seed 仅文生图（Text2Image）可用，图生图请勿传 seed'
        if n > MAX_INPUT_IMAGES:
            return f'图生图最多传入 {MAX_INPUT_IMAGES} 张输入图，当前传入 {n} 张'
        if is_ref_ratio:
            ref = self.image_ratio.split(':', 1)[1]
            try:
                ref_n = int(ref)
            except ValueError:
                ref_n = -1
            if not (1 <= ref_n <= n):
                return f'image_ratio=ref:{ref} 指向不存在的输入图（当前共 {n} 张）'
        return None


# ── State store ───────────────────────────────────────────────────────────────

generations: dict[str, dict] = {}

# Session-indexed store for eval snapshot support.
# mock_tools.generate() writes here (via record_generation) so both Mock and
# Real executor paths populate the same state that the plugin reads.
_session_generations: dict[str, list[dict]] = {}


def record_generation(session_id: str, rec: dict) -> None:
    """Append a generation record to the session-indexed store."""
    _session_generations.setdefault(session_id, []).append(rec)


def get_session_generations(session_id: str) -> list[dict]:
    """Return all generation records for a session (snapshot-safe copy)."""
    return [dict(r) for r in _session_generations.get(session_id, [])]


def seed_session_generations(session_id: str, records: list[dict]) -> int:
    """Pre-populate generation records for a session."""
    lst = _session_generations.setdefault(session_id, [])
    for r in records:
        lst.append(dict(r))
    return len(records)


def clear_session(session_id: str) -> None:
    """Clear all generation records for a session."""
    _session_generations.pop(session_id, None)
