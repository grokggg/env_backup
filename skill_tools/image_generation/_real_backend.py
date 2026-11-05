"""Shared real-SSE backend for ``image_generation/generate``.

Used by **both** the production tool (``mock_tools.py``) and the evaluation
mock store (``nexus/extensions/tasks/one_agent_skill_eval/skill_plugins/image_generation/state.py``) so
the streaming-SSE / URL-extraction logic lives in exactly one place.

Returns a dict shaped identically to ``_mock_generate`` (one
``output_image_urls`` list, ``status='ok'|'no_urls'|'error'``, plus a
``backend='real'`` marker) so callers can format the result uniformly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import string
import time
import uuid

from typing import Any


log = logging.getLogger('image_generation.real_backend')


# ════════════════════════════════════════════════════════════════════════════
# Mock URL fabrication (shared by chat-side and eval-side mock paths)
# ════════════════════════════════════════════════════════════════════════════

#: Mock COS bucket hosts — kept identical to the real ones so any downstream
#: URL parsing (judge / SFT pipeline / ref store) treats mock and real URLs
#: through the same code path.  Real-vs-mock is *not* signalled in the URL
#: itself (would leak into shuffled training data); use the per-record
#: ``backend='mock'`` field if you need to filter.
#:
#: Split into internal (``cos-internal.*``, only reachable from within the
#: Tencent internal network) vs external (public-facing) hosts so
#: ``IMAGE_GEN_MOCK_URL_INTERNAL`` can pick which flavor the fabricated URL
#: uses — mirrors the ``IMAGE_GEN_USE_REAL`` mock/real toggle.
_MOCK_COS_HOSTS_INTERNAL = [
    'hunyuan-multimodal-1258344703.cos-internal.ap-guangzhou.tencentcos.cn',
    'hunyuan-prod-1258344703.cos-internal.ap-guangzhou.tencentcos.cn',
    'hunyuan-test-1258344703.cos-internal.ap-guangzhou.tencentcos.cn',
]
_MOCK_COS_HOSTS_EXTERNAL = [
    'ep-prod-1258344703.cos.ap-guangzhou.tencentcos.cn',
    'hunyuan-multimodal-1258344703.cos.ap-guangzhou.myqcloud.com',
    'hunyuan-prod-1258344703.cos.ap-guangzhou.myqcloud.com',
    'hunyuan-test-1258344703.cos.ap-guangzhou.myqcloud.com',
]

#: Whether ``fake_cos_signed_url`` should fabricate an internal-network host
#: (``cos-internal.*``) instead of a public-facing one. Off by default,
#: matching the pre-existing (external-leaning, mixed) host pool behavior.
_MOCK_URL_INTERNAL = os.environ.get('IMAGE_GEN_MOCK_URL_INTERNAL', '').lower() in ('1', 'true', 'yes')

#: COS signed-URLs typically have a 1-year validity window.
_MOCK_SIGN_VALIDITY_S = 365 * 24 * 3600

_BASE62 = string.ascii_letters + string.digits  # 62 chars


def _base62(data: bytes, length: int) -> str:
    """Map ``data`` bytes onto base62 alphabet, truncated to ``length``."""
    return ''.join(_BASE62[b % 62] for b in data[:length])


def fake_cos_signed_url(gen_id: str, idx: int, dst_intent: str = 'text2img') -> str:
    """Fabricate a mock URL that mirrors real Hunyuan T2I/I2I COS-signed URLs.

    Args:
        gen_id: Generation ID (used as deterministic seed).
        idx: Image index within the generation batch.
        dst_intent: ``'text2img'`` or ``'img2img'`` — determines the URL path prefix.

    Real example::

        https://hunyuan-test-1258344703.cos.ap-guangzhou.myqcloud.com/text2img/
            <32hex>/<YYYYMMDDHHMMSS>_h2_<4>_<32hex>
            ?q-sign-algorithm=sha1&q-ak=AKID<32 base62>&q-sign-time=<now>;<now+1y>
            &q-key-time=<now>;<now+1y>&q-header-list=host&q-url-param-list=
            &q-signature=<40 hex>

    The output is indistinguishable from a real URL by inspection — host,
    path layout, and every signed-query field uses the real shape and
    plausible-looking values.  Use the per-record ``backend`` field (``mock``
    vs ``real``) if you need to programmatically tell them apart.

    Determinism: every random-looking field is derived from
    ``sha*(gen_id||idx||<salt>)`` so the same ``(gen_id, idx)`` pair always
    produces the same URL within a process.  The wall-clock timestamp segment
    is the one exception (changes across runs, stable within a session).

    Host selection: pulled from ``_MOCK_COS_HOSTS_INTERNAL`` when
    ``IMAGE_GEN_MOCK_URL_INTERNAL=1/true/yes``, else from
    ``_MOCK_COS_HOSTS_EXTERNAL`` (default).
    """
    seed = f'{gen_id}|{idx}'.encode()
    sha1_hex = hashlib.sha1(seed).hexdigest()  # 40 hex
    sha256_bytes = hashlib.sha256(seed).digest()  # 32 bytes
    sig_hex = hashlib.sha1(seed + b'|sig').hexdigest()  # 40 hex

    user_hash = sha1_hex[:32]
    file_hash = (sha1_hex + sha1_hex)[8:40]
    tag = _base62(sha256_bytes[16:20], 4)  # mixed case
    ts_path = time.strftime('%Y%m%d%H%M%S')
    path = f'{dst_intent}/{user_hash}/{ts_path}_h2_{tag}_{file_hash}'

    ak_body = _base62(sha256_bytes, 32)
    now = int(time.time())
    expire = now + _MOCK_SIGN_VALIDITY_S
    sign_time = f'{now};{expire}'

    query = (
        'q-sign-algorithm=sha1'
        f'&q-ak=AKID{ak_body}'
        f'&q-sign-time={sign_time}'
        f'&q-key-time={sign_time}'
        '&q-header-list=host'
        '&q-url-param-list='
        f'&q-signature={sig_hex}'
    )
    hosts = _MOCK_COS_HOSTS_INTERNAL if _MOCK_URL_INTERNAL else _MOCK_COS_HOSTS_EXTERNAL
    host = hosts[sha256_bytes[0] % len(hosts)]
    return f'https://{host}/{path}?{query}'


REAL_API_URL = os.environ.get(
    'IMAGE_GEN_API_URL',
    'http://21.90.144.102:8008/v1/chat/completions',
)
REAL_TIMEOUT_S = float(os.environ.get('IMAGE_GEN_API_TIMEOUT', '200'))

# ``extra.from`` — 调用来源标识，按接口文档 / 工程联调示例固定为 ``oneAgent``
# （小写 o），可经 env 覆盖。
_API_FROM = os.environ.get('IMAGE_GEN_API_FROM', 'oneAgent')

# ``extra.from`` — 调用来源标识，按接口文档 / 工程联调示例固定为 ``oneAgent``
# （小写 o），可经 env 覆盖。
_API_FROM = os.environ.get('IMAGE_GEN_API_FROM', 'oneAgent')

# ``extra.from`` — 调用来源标识，按接口文档 / 工程联调示例固定为 ``oneAgent``
# （小写 o），可经 env 覆盖。
_API_FROM = os.environ.get('IMAGE_GEN_API_FROM', 'oneAgent')


# Internal-only translation between the public ``dst_intent`` exposed to the
# agent / SKILL.md and the value the real backend actually accepts. Keep this
# table local to the real backend so the agent stays unaware of backend quirks
# (mock + eval still see the original ``Text2Image`` / ``Image2Image`` values).
_DST_INTENT_API_ALIASES = {
    'Text2Image': 'Draw',
}


_IMAGE_URL_KEYS = frozenset({'url', 'previewUrl', 'downloadUrl'})


def _check_sensitive(chunk: Any) -> bool:
    """Detect content-safety rejection in an SSE chunk.

    The backend signals this via ``finish_reason: "sensitive"`` inside
    ``choices[]``, or a top-level ``sensitive_sent: "1"`` flag.
    """
    if not isinstance(chunk, dict):
        return False
    if chunk.get('sensitive_sent') in ('1', 1, True):
        return True
    for choice in chunk.get('choices', []):
        if isinstance(choice, dict) and choice.get('finish_reason') == 'sensitive':
            return True
    return False


def _extract_error_topic(chunk: Any) -> str | None:
    """Extract error topic from a replace chunk (e.g. "图片生成失败")."""
    if not isinstance(chunk, dict):
        return None
    for rep in chunk.get('replaces') or []:
        if isinstance(rep, dict):
            topic = rep.get('topic')
            if isinstance(topic, str) and topic.strip():
                return topic.strip()
    return None


def _unwrap_data_value(chunk: dict) -> dict:
    """Unwrap the nested ``dataValue`` field used by the real SSE backend.

    The real endpoint wraps each chunk as:
        {"dataValue": "data: {\"type\":\"replace\",...}", ...}
    We parse the inner JSON so ``_extract_urls`` can see the actual payload.
    """
    dv = chunk.get('dataValue')
    if not isinstance(dv, str):
        return chunk
    s = dv.strip()
    if s.startswith('data:'):
        s = s[5:].strip()
    if not s:
        return chunk
    try:
        return json.loads(s)
    except Exception:
        return chunk


def _extract_urls(obj: Any, urls_out: list[str], text_out: list[str]) -> None:
    """Recursively walk a chunk dict and collect any image-like URLs / text.

    The real backend nests image URLs inside:
        replaces[].multimedias[].{url, previewUrl, downloadUrl}
    We also handle:
    - {"image_url": {"url": ...}} / {"image_url": "..."}
    - {"resourceId": "..."} → assemble standard download URL
    - text deltas under {"delta": {"content": "..."}} / {"text": "..."}
    """
    if isinstance(obj, dict):
        # Multimedias entry: prefer downloadUrl > url > previewUrl
        if any(k in obj for k in _IMAGE_URL_KEYS):
            for k in ('downloadUrl', 'url', 'previewUrl'):
                v = obj.get(k)
                if isinstance(v, str) and v.startswith('http'):
                    if v not in urls_out:
                        urls_out.append(v)
                    break

        for k, v in obj.items():
            if k == 'image_url':
                if isinstance(v, dict) and isinstance(v.get('url'), str):
                    urls_out.append(v['url'])
                elif isinstance(v, str):
                    urls_out.append(v)
            elif k == 'resourceId' and isinstance(v, str) and v:
                urls_out.append(f'https://yuanbao.test.hunyuan.woa.com/api/resource/download?resourceId={v}')
            elif k in ('text', 'content') and isinstance(v, str) and v.strip():
                text_out.append(v)
            elif k in ('replaces', 'multimedias', 'choices', 'delta') or isinstance(v, (dict, list)):
                _extract_urls(v, urls_out, text_out)
    elif isinstance(obj, list):
        for item in obj:
            _extract_urls(item, urls_out, text_out)


def real_generate(
    session_id: str,
    dst_intent: str,
    prompt: str,
    image_urls: list[str],
    trace_id: str | None = None,
    seed: int | None = None,
    image_ratio: str = '',
    **extra: Any,
) -> dict[str, Any]:
    """Call the real SSE endpoint and aggregate the streamed response.

    Returns a record with output_image_urls extracted from the stream. On any
    failure, returns a record with status="error" plus an error message.

    *trace_id*: unique per-request identifier propagated to the backend via
    ``X-TraceID`` header for end-to-end tracing / debugging.  If ``None``,
    one is generated locally.
    """
    import httpx  # local import to keep mock path zero-dep

    gen_id = uuid.uuid4().hex[:12]
    if not trace_id:
        trace_id = uuid.uuid4().hex

    # ``image_ratio="ref:N"`` → 让下游跟随第 N 张（1-based）输入图的比例：在该图的
    # image_url 上打 ``use_ratio=true``（见接口文档 §2.2）。``image_ratio="W:H"`` 则
    # 走 extra 顶层的 ``ratio`` 字段（见下方 extra_payload）。
    ref_idx: int | None = None
    if image_ratio.startswith('ref:'):
        try:
            ref_idx = int(image_ratio.split(':', 1)[1])
        except ValueError:
            ref_idx = None

    msg_content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt}]
    for i, url in enumerate(image_urls, start=1):
        image_url_obj: dict[str, Any] = {'url': url}
        if ref_idx is not None and i == ref_idx:
            image_url_obj['use_ratio'] = True
        msg_content.append({'type': 'image_url', 'image_url': image_url_obj})

    inner = {
        'user_id': f'yb_{session_id}',
        'conversation_id': session_id,
    }
    api_dst_intent = _DST_INTENT_API_ALIASES.get(dst_intent, dst_intent)
    # Backend contract (verified against 21.215.143.152:8008):
    #   - ``inner_option``: JSON-encoded **string**.
    #   - ``messages``:     JSON-encoded **string** (NOT a bare array).
    #     Array format silently fails for Image2Image+base64 — backend
    #     returns "生成失败" instead of producing an image.  String format
    #     works for both Text2Image and Image2Image.
    messages = [{'role': 'user', 'content': msg_content}]
    # 注：当前链路已移除 LLM 改写，``enable_rewrite`` 传了也不消费，故不下发。
    extra_payload: dict[str, Any] = {
        'dst_intent': api_dst_intent,
        'from': _API_FROM,
        'inner_option': json.dumps(inner, ensure_ascii=False),
        'messages': json.dumps(messages, ensure_ascii=False),
        'agent_draw_times': 1,
    }
    # 顶层 ``ratio`` 只接受 "W:H" 整数比（见接口文档 §2.1）；``ref:N`` 已在上面落到
    # 对应输入图的 ``image_url.use_ratio``，不走这里。
    if image_ratio and not image_ratio.startswith('ref:'):
        extra_payload['ratio'] = image_ratio
    if seed is not None and seed != 0:
        extra_payload['agent_draw_seed'] = seed
    body = {'extra': extra_payload}

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
        'X-TraceID': trace_id,
    }

    log.info(
        'real_generate [%s] dst_intent=%s prompt=%.60s images=%d trace_id=%s',
        gen_id,
        api_dst_intent,
        prompt,
        len(image_urls),
        trace_id,
    )

    output_urls: list[str] = []
    aggregated_text: list[str] = []
    raw_chunks: list[Any] = []
    err: str | None = None
    is_sensitive = False
    error_topic: str | None = None

    try:
        with httpx.Client(timeout=REAL_TIMEOUT_S) as client:
            with client.stream(
                'POST',
                REAL_API_URL,
                headers=headers,
                json=body,
            ) as resp:
                if resp.status_code != 200:
                    # ``client.stream(...)`` returns a streaming response —
                    # accessing ``.text`` / ``.content`` directly raises
                    # ``ResponseNotRead`` because the body hasn't been
                    # consumed yet. Must call ``.read()`` first to materialize
                    # the body before reading text. Without this fix all
                    # non-200 responses got swallowed into a meaningless
                    # ``ResponseNotRead`` error string and the actual
                    # downstream HTTP error was hidden.
                    try:
                        body_bytes = resp.read()
                        body_text = body_bytes.decode('utf-8', errors='replace')
                    except Exception as read_exc:
                        body_text = f'<unable to read body: {read_exc}>'
                    err = f'HTTP {resp.status_code}: {body_text[:200]}'
                else:
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        line = raw.strip()
                        if line.startswith('data:'):
                            line = line[5:].strip()
                        if not line or line in ('[DONE]', 'DONE'):
                            continue
                        try:
                            chunk = json.loads(line)
                        except Exception:
                            aggregated_text.append(line)
                            continue
                        unwrapped = _unwrap_data_value(chunk)
                        raw_chunks.append(unwrapped)
                        _extract_urls(unwrapped, output_urls, aggregated_text)
                        if not is_sensitive:
                            is_sensitive = _check_sensitive(unwrapped)
                        if not error_topic:
                            error_topic = _extract_error_topic(unwrapped)
                    if not output_urls:
                        log.warning(
                            'SSE stream returned %d chunks but no image URLs extracted. '
                            'error_topic=%s raw_chunks (first 3): %s',
                            len(raw_chunks),
                            error_topic,
                            json.dumps(raw_chunks[:3], ensure_ascii=False, default=str)[:2000],
                        )
    except Exception as exc:
        err = f'{type(exc).__name__}: {exc}'

    raw_text = ''.join(aggregated_text)[:500] if aggregated_text else ''

    if is_sensitive and not output_urls:
        log.warning('real image_generation blocked by content safety, trace_id=%s', trace_id)
        return {
            'gen_id': gen_id,
            'trace_id': trace_id,
            'dst_intent': dst_intent,
            'prompt': prompt,
            'input_image_urls': list(image_urls),
            'output_image_urls': [],
            'status': 'sensitive',
            'backend': 'real',
            'error': f'内容安全审核未通过，请修改 prompt 或输入图片后重试。（{raw_text}）'
            if raw_text
            else '内容安全审核未通过，请修改 prompt 或输入图片后重试。',
            **{k: v for k, v in extra.items() if v is not None},
        }

    if err and not output_urls:
        log.warning('real image_generation failed (%s) trace_id=%s', err, trace_id)
        return {
            'gen_id': gen_id,
            'trace_id': trace_id,
            'dst_intent': dst_intent,
            'prompt': prompt,
            'input_image_urls': list(image_urls),
            'output_image_urls': [],
            'status': 'error',
            'backend': 'real',
            'error': err,
            **{k: v for k, v in extra.items() if v is not None},
        }

    raw_debug = ''
    if not output_urls and raw_chunks:
        raw_debug = json.dumps(raw_chunks[:3], ensure_ascii=False, default=str)[:2000]

    if not output_urls and error_topic:
        return {
            'gen_id': gen_id,
            'trace_id': trace_id,
            'dst_intent': dst_intent,
            'prompt': prompt,
            'input_image_urls': list(image_urls),
            'output_image_urls': [],
            'status': 'error',
            'backend': 'real',
            'error': error_topic,
            **{k: v for k, v in extra.items() if v is not None},
        }

    return {
        'gen_id': gen_id,
        'trace_id': trace_id,
        'dst_intent': dst_intent,
        'prompt': prompt,
        'input_image_urls': list(image_urls),
        'output_image_urls': output_urls,
        'raw_text': raw_text,
        'raw_chunks_debug': raw_debug,
        'status': 'ok' if output_urls else 'no_urls',
        'backend': 'real',
        **{k: v for k, v in extra.items() if v is not None},
    }
