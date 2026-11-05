"""Wiki ingest tool — the single structured entry-creation tool for /data/library.

Registered as ``wiki/ingest_entry``. This is a RUNTIME tool: it runs inside the
sandbox where ``/data/library`` and ``/data/skills`` live.

Why this exists (design note)
-----------------------------
The old flow made the model hand-write a full JSON manifest into a staging file
(the ``write`` tool) and then shell out to a content-bundle CLI. That put
two "produce exact text" steps into the model's free-generation channel:

1. Assembling the ``body`` markdown — the model re-summarised ``keyInsights``
   while typing the body, silently dropping numbers / proper nouns.
2. Serialising the whole manifest to a JSON string — one un-escaped Chinese
   quote broke JSON parsing and crashed the ingest command.

This tool removes both. The model fills structured Pydantic fields (native
tool-calling → schema-constrained, no manual JSON escaping). The framework
serialises the arguments; the tool renders the body deterministically (Jinja,
for the ``content_digest`` render type) and calls ``ingest_one`` in-process (a
plain function call on a dict — no subprocess, no heredoc, no JSON string).

Rendering happens in code, so ``keyInsights[].text`` is copied verbatim into the
body — it can no longer be rewritten on the way out.
"""

from __future__ import annotations

import json
import logging
import pathlib

from typing import Any, Literal
from urllib.parse import urlparse

import jinja2

from pydantic import BaseModel, Field, field_validator

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.agents.yb_one_agent.skill_tools.wiki.ingest import ingest_one


logger = logging.getLogger(__name__)


# ── sandbox paths (hard-coded, same convention as tools.py::load_skill_tool) ──

_SKILLS_DIR = pathlib.Path('/data/skills')
_TEMPLATE_DIR = _SKILLS_DIR / 'wiki' / 'templates'

# Render type → template filename. Empty / 'plain' means "no render, use body as-is".
_RENDER_TEMPLATES: dict[str, str] = {
    'content_digest': 'content_digest.md.j2',
}


# ── lazy singletons ───────────────────────────────────────────────────────────

_jinja_env: jinja2.Environment | None = None


def _get_jinja_env() -> jinja2.Environment:
    global _jinja_env
    if _jinja_env is None:
        _jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
            undefined=jinja2.StrictUndefined,
        )
    return _jinja_env


# ── input models ──────────────────────────────────────────────────────────────


class KeyInsight(BaseModel):
    text: str = Field(description='一句自包含要点；点睛处在 text 内自行 **加粗**')
    diff: str = Field(default='', description='added / repeated / contradicted / 空串')
    ref: str = Field(default='', description='repeated/contradicted 必填的历史条目 id；来自召回命中，不编造')
    group: str = Field(default='', description='可选分组小标题（长文/多主题分组）；单组/短文留空')


class ReadGuide(BaseModel):
    referenceRecordId: str = Field(default='', description='锚定历史 ent_id（单条、不是标题）；弱相关留空走新知态')
    diff: str = Field(default='', description='要点级聚合后的整篇 diff：contradicted>added>repeated / 空')
    readSuggestion: str = Field(default='', description='full / skim / skip / 空')
    note: str = Field(default='', description='开篇条整句（模型直接产），前端只加显性标不拼句')


class Recommendation(BaseModel):
    verdict: str = Field(default='not_useful', description='useful / not_useful')
    reason: str = Field(default='', description='useful 时一句具体用处 ≤约30字；not_useful 填空串、不展示')


class IngestEntryInput(BaseModel):
    """入库参数。一个模型覆盖两种场景，由 ``render_type`` 显式区分（必填，无默认值，避免漏传导致
    静默走错分支）：

    - ``render_type=''``：wiki 静默保存，``body`` 由调用方直接给。
    - ``render_type='content_digest'``：单篇解读，``body`` 由 Jinja 模板从判断字段渲染，
      ``advisoryNotes`` / ``readGuide`` / ``recommendation`` 仅用于渲染、不落库。

    ``render_type`` 反映的是"这次调用要不要做完整解读渲染"，跟 ``source_kind``（内容从哪来）是
    两个独立维度：静默保存和单篇解读都可能是 ``source_kind='url'``，靠 ``render_type`` 区分，不能
    从 ``source_kind`` 推断。
    """

    # ── JSON-string fallback (minimal) ────────────────────────────────────────
    # Some models serialize array/object params as JSON strings instead of native
    # values. We do a simple json.loads attempt — if the model produces broken JSON
    # (unescaped quotes etc), the validation error surfaces so the model learns from
    # the error message. The tool description already instructs correct format.

    @field_validator('topicTags', 'advisoryNotes', 'keyInsights', mode='before')
    @classmethod
    def _parse_string_list(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return v

    @field_validator('readGuide', 'recommendation', mode='before')
    @classmethod
    def _parse_string_object(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return v

    # ── 落库通用字段 ──
    title: str = Field(description='条目标题')
    category: str = Field(description='wiki §3.1 的 7 类单词枚举，选一个')
    topicTags: list[str] = Field(description='主题标签 2–4 个，优先复用已有标签')
    summary: str = Field(description='一句话摘要，只讲本篇')
    keyInsights: list[KeyInsight] = Field(default_factory=list, description='结构化要点')
    content: str = Field(
        default='',
        description='解析后的原始正文（写入 raw/<id>/content.md）。仅上传文件类需要留正文时传入',
    )
    type: str = Field(default='article', description='article / file / image / video / other')
    status: str = Field(default='parsed', description='parsed / source_only')
    source_kind: str = Field(default='url', description='upload / url / artifact')
    source_raw: str = Field(
        default='',
        description='原始来源定位，含义随 source_kind 变：url 类直接填 URL；链接类必填',
    )
    source_file: str = Field(
        default='',
        description='上传文件类的二进制原件沙箱绝对路径（如 /data/inputs/x.pdf），拷贝进 raw/<id>/source.<ext> 留底后'
        '工具自动改写 source_raw 为该相对路径；链接类留空',
    )

    # ── 渲染控制 ──
    render_type: Literal['', 'content_digest'] = Field(
        description="body 渲染方式：'' 用 body 原样；'content_digest' 用模板渲染；必须显式传入"
    )
    body: str = Field(default='', description='render_type 为空时必填、由调用方给；content_digest 时忽略、由模板渲染')

    # ── content_digest 专属：仅渲染用，不落库 ──
    advisoryNotes: list[str] = Field(default_factory=list, description='风险提示，默认 []、最多 1 条')
    readGuide: ReadGuide | None = Field(default=None, description='开篇条（content_digest 渲染用）')
    recommendation: Recommendation | None = Field(default=None, description='用处判断（content_digest 渲染用）')


# ── rendering ─────────────────────────────────────────────────────────────────


def _render_body(params: IngestEntryInput) -> str:
    """Return the entry body. Deterministic — no LLM involved.

    ``render_type`` empty → the caller-supplied ``body`` verbatim.
    ``render_type`` known → the Jinja template rendered from structured fields;
    ``keyInsights[].text`` appears verbatim because the template only interpolates
    the structured values.
    """
    if not params.render_type:
        return params.body

    template_name = _RENDER_TEMPLATES.get(params.render_type)
    if template_name is None:
        raise ValueError(f'unknown render_type={params.render_type!r}, known: {sorted(_RENDER_TEMPLATES)}')

    template = _get_jinja_env().get_template(template_name)
    ctx = {
        'title': params.title,
        'category': params.category,
        'summary': params.summary,
        'advisoryNotes': params.advisoryNotes,
        'readGuide': (params.readGuide or ReadGuide()).model_dump(),
        'keyInsights': [ki.model_dump() for ki in params.keyInsights],
        'recommendation': (params.recommendation or Recommendation()).model_dump(),
    }
    return template.render(**ctx).strip()


def _make_output(text: str) -> list[Any]:
    return [{'type': 'text', 'text': text}]


def _ok(text: str, *, meta: dict[str, Any] | None = None) -> ToolResponse:
    return ToolResponse(output=_make_output(f'[ok] true\n{text}'), meta=meta or {})


def _err(msg: str, *, body: str | None = None) -> ToolResponse:
    lines = [f'[error] {msg}']
    if body is not None:
        lines.append('[body]')
        lines.append(body)
    return ToolResponse(output=_make_output('[ok] false\n' + '\n'.join(lines)), meta={'error': True})


# ── tool ──────────────────────────────────────────────────────────────────────


@tool_registry.register(
    'yb_one_agent/wiki/ingest_entry',
    meta={
        'name': 'wiki/ingest_entry',
        'description': (
            '将一条内容结构化入库到个人知识库 /data/library，并（可选）用模板渲染展示 body。'
            '这是 wiki 唯一的入库入口，模型只需填结构化字段，无需手写 JSON 或 markdown。'
            'render_type 必填，不可省略：留空字符串时 body 由你直接给；content_digest 时 body 由单篇解读模板从'
            'summary/advisoryNotes/readGuide/keyInsights/recommendation 渲染（keyInsights 逐字进 body）。'
            '链接类内容传 source_raw=URL；用户上传的文件类内容传 source_file（沙箱绝对路径）留原件底，'
            'source_raw 会被工具自动改写为落库后的相对路径。'
            '返回纯文本：首行 [ok] true/false，随后是入库回执字段，最后 [body] 后面是渲染好的展示正文；'
            '把 [body] 后面的内容原样发给用户，不要重写或概括。'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def ingest_entry_tool(params: IngestEntryInput, context: ToolContext) -> ToolResponse:
    # The runtime's invoke path may pass a raw dict instead of a validated model.
    # Coerce here to guarantee we always work with an IngestEntryInput instance.
    if isinstance(params, dict):
        params = IngestEntryInput.model_validate(params)

    # 1. Render body deterministically (never fails on ingest errors — done first).
    try:
        body = _render_body(params)
    except Exception as e:
        logger.error('wiki/ingest_entry: body render failed: %s', e)
        return _err(f'渲染失败: {e}')

    # 2. Build the ingest manifest (a dict — no JSON string, no escaping).
    content_digest = {
        'summary': params.summary,
        'advisoryNotes': params.advisoryNotes,
        'readGuide': (params.readGuide or ReadGuide()).model_dump(),
        'keyInsights': [ki.model_dump() for ki in params.keyInsights],
        'recommendation': (params.recommendation or Recommendation()).model_dump(),
    }
    manifest: dict[str, Any] = {
        'title': params.title,
        'type': params.type,
        'status': params.status,
        'source_kind': params.source_kind,
        'source_raw': params.source_raw,
        'category': params.category,
        'topicTags': params.topicTags,
        'summary': params.summary,
        'keyInsights': [ki.model_dump() for ki in params.keyInsights],
        'body': body,
        'content': params.content,
        'content_digest': content_digest,
    }
    if params.source_file:
        manifest['source_file'] = params.source_file

    # 3. Ingest in-process. Ingest failure must NOT block display — the body is
    #    already rendered, so surface it either way.
    receipt: dict[str, Any]
    try:
        receipt = ingest_one(manifest)
    except Exception as e:
        logger.error('wiki/ingest_entry: ingest raised: %s', e)
        receipt = {'ok': False, 'error': str(e)}

    if not receipt.get('ok'):
        return _err(receipt.get('error', '未知错误'), body=body)

    lines = [
        f'[id] {receipt["id"]}',
        f'[title] {receipt["title"]}',
        f'[category] {receipt["category"]}',
        f'[topicTags] {", ".join(receipt["topicTags"])}',
        f'[keyInsights] {receipt["keyInsights"]}',
        '[body]',
        body,
    ]
    meta: dict[str, Any] = {}
    snapshot = receipt.get('card_snapshot')
    parsed_source = urlparse(snapshot.get('source_url', '')) if isinstance(snapshot, dict) else None
    # A missing source URL must not hide a committed Wiki entry from the library.
    if (
        isinstance(snapshot, dict)
        and params.source_kind == 'url'
        and (
            not snapshot.get('source_url', '').strip()
            or (parsed_source is not None and parsed_source.scheme in ('http', 'https') and parsed_source.netloc)
        )
    ):
        meta['llm_wiki_changed'] = {'card_snapshot': snapshot}
    return _ok('\n'.join(lines), meta=meta)
