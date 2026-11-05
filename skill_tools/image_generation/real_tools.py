"""Real backend implementation of image_generation/generate.

Calls the SSE image generation API via ``_real_backend.real_generate``.
Registers the same tool ID as ``mock_tools.py`` — only ONE of the two
modules may be loaded per process (via ``Registry.import_module``).

This file is self-contained and does not import from ``mock_tools.py``,
so it can be independently deployed to production.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


# ── State ─────────────────────────────────────────────────────────────────────

_generations: dict[str, dict] = {}


# ── Input models ──────────────────────────────────────────────────────────────


class ImageInput(BaseModel):
    type: Literal['url', 'sandbox'] = Field(description='url | sandbox')
    value: str = Field(description='对应类型的值')


def _coerce_input_images(raw: Any) -> list[ImageInput]:
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
                    return _coerce_input_images(parsed)
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
                            out.extend(_coerce_input_images(parsed))
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


_VALID_DST_INTENTS = frozenset({DstIntent.TEXT2IMAGE, DstIntent.IMAGE2IMAGE})


#: 单次调用允许的最大输入图张数（多图融合/风格迁移/元素合成等）。
_MAX_INPUT_IMAGES = 3


_RATIO_RE = re.compile(r'^\d{1,3}:\d{1,3}$')
_RATIO_REF_RE = re.compile(r'^ref:\d+$')


def _coerce_image_ratio(v: Any) -> str:
    """Normalize image_ratio to '' (auto), a 'W:H' integer ratio, or 'ref:N'.

    The reference form is ``ref:N`` ONLY — no other spellings are accepted.
    Anything unrecognized falls back to '' (auto).
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
        return _coerce_input_images(v)

    @field_validator('image_ratio', mode='before')
    @classmethod
    def _coerce_image_ratio_field(cls, v: Any) -> str:
        return _coerce_image_ratio(v)

    @field_validator('seed', mode='before')
    @classmethod
    def _coerce_seed(cls, v: Any) -> int | None:
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
        return [img.value for img in self.input_images]

    def combination_error(self) -> str | None:
        """Validate the whole parameter combination before calling the downstream tool.

        Returns the first violated rule as a human-readable string (surfaced as a
        tool error), or None when every cross-field rule holds.
        """
        if self.dst_intent not in _VALID_DST_INTENTS:
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
        if n > _MAX_INPUT_IMAGES:
            return f'图生图最多传入 {_MAX_INPUT_IMAGES} 张输入图，当前传入 {n} 张'
        if is_ref_ratio:
            ref = self.image_ratio.split(':', 1)[1]
            try:
                ref_n = int(ref)
            except ValueError:
                ref_n = -1
            if not (1 <= ref_n <= n):
                return f'image_ratio=ref:{ref} 指向不存在的输入图（当前共 {n} 张）'
        return None


# ── Response helpers ──────────────────────────────────────────────────────────


def _ok(text: str) -> ToolResponse:
    return ToolResponse(output=[{'type': 'text', 'text': f'[ok] true\n{text}'}])


def _err(msg: str) -> ToolResponse:
    return ToolResponse(
        output=[{'type': 'text', 'text': f'[ok] false\n[error] {msg}'}],
        meta={'error': True},
    )


def _format_output_images(urls: list[str]) -> str:
    if not urls:
        return '[output_images] (none)'
    lines = [f'[output_images] {len(urls)}']
    for i, url in enumerate(urls, 1):
        lines.append(f'[image {i}]')
        lines.append(f'[url] {url}')
    return '\n'.join(lines)


log = logging.getLogger('image_generation.real_tools')


def _shorten_url(long_url: str) -> str:
    """尝试将长链转短链，失败时返回原链。"""
    from nexus.extensions.agents.yb_one_agent.short_url_client import ShortLinkClient

    try:
        client = ShortLinkClient()
        resp = client.shorten(long_url)
        short_url = resp.get('short_url')
        if short_url:
            return short_url
        code = resp.get('short_code')
        if code:
            domain = os.environ.get('SHORT_URL_DOMAIN', 'http://yb-test.woa.com')
            return f'{domain.rstrip("/")}/{code}'
    except Exception as exc:
        log.warning('shorten failed for %s: %s', long_url[:80], exc)
    return long_url


def _shorten_urls(urls: list[str]) -> list[str]:
    return [_shorten_url(u) for u in urls]


# ── Tool implementation ──────────────────────────────────────────────────────


@tool_registry.register(
    'yb_one_agent/image_generation/generate',
    meta={'name': 'image_generation/generate', 'description': '图像生成统一入口', 'tool_type': ToolType.LOCAL},
)
async def generate(req: GenerateInput, ctx: ToolContext) -> ToolResponse:
    from nexus.extensions.agents.yb_one_agent.skill_tools.image_generation._real_backend import real_generate
    from nexus.extensions.agents.yb_one_agent.skill_tools.image_generation._resolver import resolve_image_payload

    combo_err = req.combination_error()
    if combo_err:
        return _err(combo_err)

    image_values = req.image_values()

    tool_state = getattr(ctx, 'tool_state', None) or {}
    session_id = tool_state.get('terminal_session_id') or uuid.uuid4().hex[:8]
    trace_id: str | None = tool_state.get('trace_id')

    runtime = None
    if image_values:
        try:
            from nexus.core.context import require_context

            exec_ctx = require_context()
            runtime = await exec_ctx.get_runtime('main')
        except Exception:
            runtime = None

        try:
            resolved_urls = await asyncio.gather(*[
                resolve_image_payload(item, runtime=runtime)
                for item in image_values  # type: ignore[arg-type]
            ])
        except ValueError as exc:
            return _err(f'图片输入解析失败：{exc}')
    else:
        resolved_urls = []

    rec = await asyncio.to_thread(
        real_generate,
        session_id,
        req.dst_intent,
        req.prompt.strip(),
        list(resolved_urls),
        trace_id=trace_id,
        seed=req.seed,
        image_ratio=req.image_ratio,
    )
    rec['input_images'] = [img.model_dump() for img in req.input_images]

    output_urls = rec.get('output_image_urls') or []

    _generations[rec['gen_id']] = rec

    status = rec.get('status')
    if status != 'ok':
        reason = rec.get('error') or f'生成未返回结果（status={status}）'
        return _err(f'图像生成失败：{reason}')

    truncated_urls = output_urls[:1]
    if os.environ.get('IMAGE_GEN_SHORT_URL', '1') != '0':
        short_urls = await asyncio.to_thread(_shorten_urls, truncated_urls)
    else:
        short_urls = truncated_urls

    from nexus.extensions.agents.yb_one_agent.skill_tools.image_generation._state import record_generation

    rec['output_image_urls_truncated'] = truncated_urls
    rec['output_image_short_urls'] = short_urls
    record_generation(session_id, rec)

    lines = [
        f'[gen_id] {rec["gen_id"]}',
        f'[dst_intent] {rec["dst_intent"]}',
        f'[prompt] {req.prompt}',
        _format_output_images(short_urls),
    ]
    return _ok('\n'.join(lines))
