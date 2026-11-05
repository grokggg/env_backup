"""Mock in-memory implementation of image_generation tools (nexus-style stub).

The pydantic ``GenerateInput`` model is defined in ``_state.py`` and serves as
the **single source of truth** for tool argument shape & coercion: the eval
framework (``skill_plugins/image_generation/state.py``) imports and reuses it
for validation, while production runtime invokes the registered ``generate``
callable through ``nexus.tool_registry``.
"""

import uuid

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry

from ._state import (
    DstIntent,
    GenerateInput,
    coerce_input_images,
    err,
    extract_real_backend_error,
    format_output_images,
    generations,
    ok,
    record_generation,
)


# Re-export for backward compatibility
_extract_real_backend_error = extract_real_backend_error
_ok = ok
_err = err
_format_output_images = format_output_images
_coerce_input_images = coerce_input_images
_generations = generations


# ── Mock generate ─────────────────────────────────────────────────────────────


def _mock_generate(req: GenerateInput) -> dict:
    """Deterministic mock — fabricates 1 stable URL (matches real backend agent_draw_times=1)."""
    import hashlib

    from nexus.extensions.agents.yb_one_agent.skill_tools.image_generation._real_backend import (
        _DST_INTENT_API_ALIASES,
        fake_cos_signed_url,
    )

    if req.seed is not None:
        gen_id = hashlib.sha256(f'seed={req.seed}'.encode()).hexdigest()[:12]
    else:
        gen_id = uuid.uuid4().hex[:12]
    url_path_prefix = 'img2img' if req.dst_intent == DstIntent.IMAGE2IMAGE else 'text2img'
    output_urls = [fake_cos_signed_url(gen_id, 0, dst_intent=url_path_prefix)]
    return {
        'gen_id': gen_id,
        'dst_intent': _DST_INTENT_API_ALIASES.get(req.dst_intent, req.dst_intent),
        'prompt': req.prompt,
        'input_images': [img.model_dump() for img in req.input_images],
        'output_image_urls': output_urls,
        'seed': req.seed,
        'status': 'ok',
        'backend': 'mock',
    }


# ── Tool implementation ──────────────────────────────────────────────────────


@tool_registry.register(
    'yb_one_agent/image_generation/generate',
    meta={'name': 'image_generation/generate', 'description': '图像生成统一入口', 'tool_type': ToolType.LOCAL},
)
async def generate(req: GenerateInput, ctx: ToolContext) -> ToolResponse:
    combo_err = req.combination_error()
    if combo_err:
        return err(combo_err)

    rec = _mock_generate(req)

    output_urls = rec.get('output_image_urls') or []

    generations[rec['gen_id']] = rec

    tool_state = getattr(ctx, 'tool_state', None) or {}
    session_id = tool_state.get('terminal_session_id')
    if session_id:
        rec['output_image_urls_truncated'] = output_urls[:1]
        record_generation(session_id, rec)

    status = rec.get('status')
    if status != 'ok':
        reason = rec.get('error') or extract_real_backend_error(rec) or f'生成未返回结果（status={status}）'
        return err(f'图像生成失败：{reason}')

    truncated_urls = output_urls[:1]

    lines = [
        f'[gen_id] {rec["gen_id"]}',
        f'[dst_intent] {rec["dst_intent"]}',
        f'[prompt] {req.prompt}',
        format_output_images(truncated_urls),
    ]
    return ok('\n'.join(lines))
