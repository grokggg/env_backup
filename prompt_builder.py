"""System prompt builder for one-agent.

Assembles the system prompt modularly and per-request meta information.
Designed for maximum KV-cache reuse:

    Layer 1  人设（身份 + 回复风格）          — 永远存在，内容固定
    Layer 2  工具与技能                       — 按启用工具集条件组装
    ─────────────────────────────────────── KV Cache 边界
    Layer 3  用户信息                         — 每次请求动态注入，置于最末
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib

from datetime import datetime
from typing import Any

import jinja2


# ── paths ────────────────────────────────────────────────────────────────────

# Static schema fallback file
_SCHEMA_FILE = pathlib.Path(__file__).parents[4] / 'one_agent_demo' / 'tool_schemas.json'

# Hard-coded container path for skills
_SKILLS_CONTAINER_DIR = '/data/skills'

# ── Jinja2 template ──────────────────────────────────────────────────────────

_TEMPLATE_DIR = pathlib.Path(__file__).parent
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=jinja2.StrictUndefined,
)

# ── skills metadata ──────────────────────────────────────────────────────────
# Hardcoded from SKILL.md frontmatter — no submodule required for prompt building.
# Keep in sync with skills/skills/*/SKILL.md when skill descriptions change.

_SKILLS_META: list[dict[str, str]] = [
    {
        'name': 'docx',
        'description': (
            '当用户需要创建、读取、编辑或处理 Word 文档（.docx 文件）时使用本 Skill。'
            '触发场景包括：提到「Word 文档」「word doc」「word document」「.docx」；'
            '生成带排版的专业文档（目录、标题、页码、信笺抬头、页眉页脚）；'
            '提取或重组 .docx 中的内容；在文档中插入或替换图片；在 Word 文件里做查找替换；'
            '处理 tracked changes（修订）或 comments（批注）；将内容整理成一份成品 Word 文件。'
            '用户要求产出「公文」「报告」「备忘录」「信函」「模板」等 Word / .docx 文件时同样适用。'
            '请勿用于 PDF、Excel、Google Docs 或与文档生成无关的通用代码任务。'
            '纯格式转换、不涉及编辑的情况，请优先使用 file-format-converter。'
        ),
    },
    {
        'name': 'file-format-converter',
        'description': (
            '将用户提供的文件从当前格式转换为指定目标格式。'
            '纯格式转换、不涉及编辑的情况，请优先使用本技能。'
            '仅适用于文件及图片格式转换任务，不要用于总结、改写、翻译、压缩、OCR、内容编辑或重新设计。'
        ),
    },
    {
        'name': 'image_generation',
        'description': (
            '文本生成/编辑图片（文生图/图生图）：产出绘画、配图、海报、设计、示意图等视觉作品，'
            '或对已有图片做编辑、调整、再创作等各类处理。'
            '只要可能需要产出带画面感/设计版式的图像，或需基于已有图片做这类处理，'
            '就先 load 本技能、按正文判定与执行——'
            '不要想当然用 matplotlib/PIL/reportlab 代码自绘，或用 picture_search 搜图代替。'
            '纯格式转换、不涉及编辑的情况，请优先使用 file-format-converter。'
            '具体触发/不触发边界、参数与写法以技能正文为准。'
        ),
    },
    {
        'name': 'pdf',
        'description': (
            '当用户需要对 PDF 文件进行任何操作时使用此技能。包括但不限于：'
            '读取或提取 PDF 中的文字/表格、合并多个 PDF、拆分 PDF 页面、旋转页面、'
            '添加水印、从零创建 PDF、填写 PDF 表单、加密/解密 PDF、提取图片、'
            '对扫描件进行 OCR 识别等。只要用户提及 .pdf 文件或要求生成 PDF，都应使用此技能。'
            '纯格式转换、不涉及编辑的情况，请优先使用 file-format-converter。'
        ),
    },
    {
        'name': 'personal-plan',
        'description': (
            '将一个个人目标澄清、校验并拆成一次性可执行计划；'
            '完整计划默认生成聊天计划卡和XLSX，轻量安排可只用聊天，用户也可指定DOCX、PPTX或HTML等格式；'
            '可在用户确认后创建少量高价值定时任务。'
            '用于制定或重排学习备考、健身减脂、职业定位、求职冲刺、面试备战、Offer决策与入职准备及其他个人生活目标计划，'
            '尤其适合目标模糊、期限明确、需要阶段拆解、材料分析或日程安排的请求。'
            '不用于单纯知识问答、解题、动作咨询、普通提醒、工作项目排期、简历或文章单次改写，'
            '以及由专门Skill负责的旅行、理财和写作创作。'
        ),
    },
    {
        'name': 'pptx',
        'description': (
            '处理所有与 .pptx 相关的任务：创建演示文稿、编辑现有幻灯片、读取或提取内容、'
            '按模板生成、合并或拆分文件，以及处理版式、备注和评论。'
            '只要用户提到 deck、slides、presentation，或出现任何 .pptx 文件名'
            '（无论后续用于编辑、分析、总结或转写），都应使用此技能。'
            '纯格式转换、不涉及编辑的情况，请优先使用 file-format-converter。'
        ),
    },
    {
        'name': 'task_schedule',
        'description': (
            '用于管理用户的定时/例行委托。以当前日期为基准，当用户指定未来时间并要求做某事'
            '（提醒、写作、生成图片、推送报道等），即属于定时委托，无需关注任务执行方式等细节，'
            '而应通过本技能操作任务而非当场执行，含查看、创建、更新、暂停/恢复、删除；'
            '每条有标题、到点执行说明、时间或重复规则等。'
        ),
    },
    {
        'name': 'tdocs-cli',
        'description': (
            '腾讯文档（在线）CLI。**仅在以下情况使用**：'
            '用户明确要求将主要交付物创建或保存为「腾讯文档 / 在线文档 / 在线编辑文档」，'
            '或明确提到「docs.qq.com / tdocs-cli / 腾讯文档 CLI」；'
            '主要交付物是独立思维导图（优先使用本 skill）；'
            '需要导入 / 导出腾讯文档文件；或需操作用户已给出的腾讯文档链接。'
            '能力覆盖：文档编辑（doc，Word 类文档的创建与编辑）、思维导图（mind）、流程图（flowchart）、'
            '智能文档（smartcanvas，仅在用户明确要求「智能文档 / smartcanvas」时新建，或读取 / 增量编辑已有文档）、'
            '文件管理与导入导出（manage / workflow）。'
            '未明确要求在线或腾讯文档时，Word / 表格 / PPT 分别使用 docx / xlsx / pptx skill；'
            '思维导图仅作为本地 Word / PPT 的内嵌内容时，由对应主交付物 skill 处理。'
        ),
    },
    {
        'name': 'westock-data',
        'description': (
            '金融市场结构化数据查询的权威入口。'
            '支持股票（A股/港股/美股/日韩股）、ETF、指数、板块、期货、外汇、可转债的'
            '行情、财报、研报、新闻、公告、事件、股东、分红、ETF持仓、热搜榜、新股/投资日历、龙虎榜等数据查询；'
            '同时支持行业经营数据、申万行业估值/盈利预测/财务、全球宏观经济等数据查询。'
            '命中能力域时禁止 web_search、HTTP 直连或其它金融 Skill 替代；筛股用 westock-screener。'
        ),
    },
    {
        'name': 'westock-screener',
        'description': (
            '选股/选基工具——按条件、策略、标签、事件、排行批量筛选股票或 ETF。'
            '用户问「哪些股票/帮我选/排行榜/TOP/筛选/MACD金叉/央企/ST/高股息ETF/技术面选股」时使用。'
            '只做批量筛选；单股明细用 westock-data。'
        ),
    },
    {
        'name': 'xlsx',
        'description': (
            '当电子表格文件是主要输入或输出时使用此技能。适用于以下任务：'
            '打开、读取、编辑或修复现有的 .xlsx、.xlsm、.csv 或 .tsv 文件'
            '（如添加列、计算公式、格式设置、图表制作、数据清洗）；'
            '从零开始或从其他数据源创建新的电子表格。'
            '当用户通过名称或路径提及电子表格文件时（即使是随意提及，如"我下载目录里的 xlsx"），'
            '并且希望对其进行操作或从中生成内容时，应特别触发此技能。'
            '同样适用于将混乱的表格数据文件（格式错误的行、错位的表头、垃圾数据）'
            '清洗或重组为规范的电子表格。最终交付物必须是电子表格文件。'
            '当主要交付物是 Word 文档、HTML 报告、独立 Python 脚本、'
            '数据库管道或 Google Sheets API 集成时，即使涉及表格数据，也不要触发此技能。'
            '纯格式转换、不涉及编辑的情况，请优先使用 file-format-converter。'
        ),
    },
]

# Skills disabled by default when skills=None (must be explicitly enabled to use).
_DEFAULT_OFF_SKILLS: frozenset[str] = frozenset()

# Fast lookup: name → meta dict
_SKILLS_BY_NAME: dict[str, dict[str, str]] = {s['name']: s for s in _SKILLS_META}

# ── tool sets ────────────────────────────────────────────────────────────────

#: All tools enabled by default.
#: ``read_v3`` rather than ``read_v2``: this branch feeds a multimodal base, so
#: images have to come back as native image parts. ``read_v2`` stays registered
#: (mirror of the online tool_proxy) but is not loaded here.
DEFAULT_TOOLS: frozenset[str] = frozenset({
    'delegate',
    'bash_v2',
    'read_v3',
    'edit_v2',
    'write_v2',
    'web_fetch',
    'web_search',
    'picture_search',
    'video_search',
    'ask_user_question',
    'load_skill',
})

#: Tools for scheduled-task execution (trigger context).
#: No ``ask_user_question`` — the user is not online when a task triggers.
TASK_EXECUTION_TOOLS: frozenset[str] = DEFAULT_TOOLS - {'ask_user_question', 'delegate'}

_VALID_TERMINAL_TYPES = ('app', 'web', 'pc')

# ── constants exported for external use ──────────────────────────────────────

TASK_EXECUTION_CONSTRAINT = (
    '\n\n# 执行约束\n'
    '你当前正在执行一个定时任务。 在此上下文中，你 **不能** 创建、修改或删除任何定时任务。 '
    '请专注于完成当前任务的执行，不要尝试调用任何定时任务管理相关的工具。'
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _utc_offset_label(tz_name: str) -> str:
    """Return e.g. ``'UTC+8'`` for an IANA timezone name like ``'Asia/Shanghai'``."""
    try:
        import zoneinfo

        tz = zoneinfo.ZoneInfo(tz_name)
        offset = datetime.now(tz).utcoffset()
        if offset is None:
            return 'UTC'
        total_seconds = int(offset.total_seconds())
        hours, remainder = divmod(abs(total_seconds), 3600)
        sign = '+' if total_seconds >= 0 else '-'
        if remainder:
            minutes = remainder // 60
            return f'UTC{sign}{hours}:{minutes:02d}'
        return f'UTC{sign}{hours}'
    except Exception:
        return 'UTC'


def build_meta_lines(meta: dict[str, Any] | None) -> list[str]:
    """Build the Layer-3 user-info lines (date / location) from ``meta``.

    Shared by the main agent's ``build_system_prompt`` and the deep_research
    sub-agent's user-info section so both render identical date/location lines.
    """
    lines: list[str] = []
    if not meta:
        return lines
    cur_date = meta.get('cur_date')
    if cur_date:
        tz = meta.get('timezone') or os.environ.get('DEFAULT_TZ', 'Asia/Shanghai')
        lines.append(f'当前日期：{cur_date} ({tz}, {_utc_offset_label(tz)})')
    location = meta.get('user_location')
    if location:
        lines.append(f'用户位置：{location}')
    return lines


def get_task_execution_skills() -> list[str]:
    """Return all skill names **except** ``task_schedule`` and default-off skills."""
    return sorted(
        s['name'] for s in _SKILLS_META if s['name'] != 'task_schedule' and s['name'] not in _DEFAULT_OFF_SKILLS
    )


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════


def build_system_prompt(
    skills: list[str] | None = None,
    tools: frozenset[str] | None = None,
    meta: dict[str, Any] | None = None,
    terminal_type: str = 'web',
) -> str:
    """Build the system prompt for one-agent.

    Args:
        skills:
            List of skill names to enable.
            ``None`` → all skills on (default).
            ``[]`` → no skills.
            Any list → only the named skills are shown.
        tools:
            Set of active tool names.  Controls which tool instructions are
            included in the prompt (e.g. ``ask_user_question`` guidance is
            omitted when the tool is not in this set).
            ``None`` → treat all tools as enabled (same as DEFAULT_TOOLS).
        meta:
            Per-request dynamic fields placed at the very end (Layer 3) to
            maximise KV-cache reuse for the static layers above.
            Supported keys (all optional):
              - ``cur_date``      e.g. ``'2026年04月07日'``
              - ``user_location`` e.g. ``'广东省深圳市'``
              - ``timezone``      IANA tz name, defaults to env DEFAULT_TZ
        terminal_type:
            Controls response style and feature availability.
            Valid values: ``'app'`` (mobile), ``'web'`` (browser), ``'pc'`` (desktop).
            ``'app'`` → mobile-optimized style; ``'web'``/``'pc'`` → PC-optimized style.
            ``'web'`` additionally disables task_schedule skill.

    Returns:
        Assembled system prompt string.
    """
    if terminal_type not in _VALID_TERMINAL_TYPES:
        raise ValueError(f'Invalid terminal_type={terminal_type!r}, must be one of {_VALID_TERMINAL_TYPES}')

    # Resolve enabled skill list
    enabled_names = {s['name'] for s in _SKILLS_META} - _DEFAULT_OFF_SKILLS if skills is None else set(skills)

    # Web terminal: task_schedule not supported
    if terminal_type == 'web':
        assert skills is None or 'task_schedule' not in skills, 'task_schedule skill is not supported on web terminal'
        enabled_names.discard('task_schedule')

    # Build ordered skill dicts for template (preserve _SKILLS_META order)
    skill_list = [s for s in _SKILLS_META if s['name'] in enabled_names]

    # Tool flags
    ask_user_enabled = tools is None or 'ask_user_question' in tools
    delegate_enabled = tools is None or 'delegate' in tools
    # Unlike delegate, handoff is opt-in and intentionally absent from
    # DEFAULT_TOOLS so the standard harness behavior remains unchanged.
    handoff_enabled = tools is not None and 'handoff' in tools

    # Build meta lines for Layer 3
    meta_lines = build_meta_lines(meta)

    template = _jinja_env.get_template('system_prompt.j2')
    return template.render(
        terminal_type=terminal_type,
        skills=skill_list,
        ask_user_enabled=ask_user_enabled,
        delegate_enabled=delegate_enabled,
        handoff_enabled=handoff_enabled,
        skills_dir=_SKILLS_CONTAINER_DIR,
        meta_lines=meta_lines,
    ).strip()


_DEFAULT_TASK_EXEC_LOCATION = '中国广东省深圳市'


def build_task_execution_system_prompt(
    *,
    skills: list[str] | None = None,
    location: str | None = _DEFAULT_TASK_EXEC_LOCATION,
    terminal_type: str = 'web',
    extra_meta: dict[str, Any] | None = None,
) -> str:
    """Canonical system-prompt builder for scheduled-task execution.

    Both production (the demo's ``_task_trigger_stream``) and the offline
    evaluator (``task_execution_runner``) call this — guaranteeing byte-for-byte
    identical SPs across the two so eval results actually reflect runtime
    behaviour.

    Behaviour:
      - ``tools=TASK_EXECUTION_TOOLS`` (a trigger may need any capability).
      - ``task_schedule`` skill stripped (auto-discovery via
        :func:`get_task_execution_skills` when *skills* is ``None``).
      - Auto-injects ``cur_date`` (now ± ``DEFAULT_TZ``) and ``user_location``
        into the meta layer.
      - Appends :data:`TASK_EXECUTION_CONSTRAINT` so the agent cannot
        recursively create/edit/delete schedules.

    Args:
        skills: Override skill list (``task_schedule`` is always stripped).
            ``None`` → auto-discover all skills except ``task_schedule``.
        location: User location string for ``user_location`` meta. Pass
            ``None`` or ``''`` to omit. Defaults to the demo's default.
        terminal_type: ``'app'`` | ``'web'`` | ``'pc'`` (controls response style).
        extra_meta: Additional meta keys to inject (e.g. for forward-compat).
    """
    from datetime import date as _date

    effective_skills = get_task_execution_skills() if skills is None else [s for s in skills if s != 'task_schedule']

    tz_name = os.environ.get('DEFAULT_TZ', 'Asia/Shanghai')
    today = _date.today().strftime('%Y-%m-%d %A')
    meta: dict[str, Any] = {'cur_date': today, 'timezone': tz_name}
    if location:
        meta['user_location'] = location
    if extra_meta:
        meta.update(extra_meta)

    return (
        build_system_prompt(
            skills=effective_skills,
            tools=TASK_EXECUTION_TOOLS,
            meta=meta,
            terminal_type=terminal_type,
        )
        + TASK_EXECUTION_CONSTRAINT
    )


# ════════════════════════════════════════════════════════════════════════════
# Tool schema builder
# ════════════════════════════════════════════════════════════════════════════

_BUILTIN_TOOL_REGISTRY_PREFIX = 'yb_one_agent/'

# Ordered list of built-in tool names (defines schema order in the tools list)
_BUILTIN_TOOL_ORDER: list[str] = [
    'delegate',
    'bash_v2',
    'read_v3',
    'edit_v2',
    'write_v2',
    'web_fetch',
    'web_search',
    'picture_search',
    'video_search',
    'ask_user_question',
    'load_skill',
    'handoff',
]

# ``bash/edit/write`` are registered under ``*_v2`` keys and ``read`` under
# ``read_v3`` (multimodal) in tools.py, so the ``_BUILTIN_TOOL_ORDER`` entry IS
# the registry-key suffix — no override mapping needed (same convention as
# harness-v3).


def _registry_key_for(public_name: str) -> str:
    """Map a public tool name to its tool_registry key."""
    return f'{_BUILTIN_TOOL_REGISTRY_PREFIX}{public_name}'


def build_tool_funcs(tools: set[str] | frozenset[str] | None = None) -> list[Any]:
    """Return the registered tool callables for the given public tool names.

    The counterpart of :func:`build_tool_schemas`: same name→registry-key
    resolution, so the executor built from these callables resolves tool calls
    to exactly the functions whose schemas :func:`build_tool_schemas` advertised.
    ``None`` → DEFAULT_TOOLS. Unknown names are skipped (no error) so callers can
    pass a superset (e.g. the main agent passing ``delegate`` here is handled
    elsewhere — this only returns built-in sandbox/search tools).
    """
    if tools is None:
        tools = DEFAULT_TOOLS
    from nexus.core import tool_registry

    funcs: list[Any] = []
    for name in _BUILTIN_TOOL_ORDER:
        if name not in tools:
            continue
        with contextlib.suppress(Exception):
            funcs.append(tool_registry.get(_registry_key_for(name)))
    return [f for f in funcs if f is not None]


def _load_builtin_schemas_from_registry(tool_names: frozenset[str]) -> list[dict[str, Any]]:
    """Load built-in tool schemas from the tool registry (dynamic, always up-to-date).

    Falls back to the static file if the registry is unavailable.
    """
    try:
        from nexus.core import tool_registry
        from nexus.core.tool import get_tool_meta

        schemas: list[dict[str, Any]] = []
        for name in _BUILTIN_TOOL_ORDER:
            if name not in tool_names:
                continue
            try:
                tool = tool_registry.get(_registry_key_for(name))
                meta = get_tool_meta(tool)
                schemas.append(meta.schema)
            except Exception:
                pass
        return schemas
    except Exception:
        return _load_builtin_schemas_from_file(tool_names)


def _load_builtin_schemas_from_file(tool_names: frozenset[str]) -> list[dict[str, Any]]:
    """Fallback: load built-in schemas from the static JSON file."""
    if not _SCHEMA_FILE.exists():
        return []
    try:
        all_schemas: list[dict[str, Any]] = json.loads(_SCHEMA_FILE.read_text(encoding='utf-8'))
        index = {entry['function']['name']: entry for entry in all_schemas}
        return [index[n] for n in _BUILTIN_TOOL_ORDER if n in tool_names and n in index]
    except Exception:
        return []


def build_tool_schemas(
    tools: set[str] | frozenset[str] | None = None,
    skills: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the OpenAI function-calling schemas for the given tools.

    Only built-in tool schemas are returned. Skill-specific tools are intentionally
    excluded — they are declared to the LLM via the ``load_skill`` tool response at
    runtime, not pre-loaded into the tools parameter.

    The returned list can be passed directly as the ``tools`` parameter of an
    OpenAI chat completion request.

    Args:
        tools:
            Set of built-in tool names to include.  ``None`` → use DEFAULT_TOOLS.
            ``set()`` → no built-in tools.
        skills:
            Ignored. Kept for backwards-compatible call sites.

    Returns:
        List of ``{"type": "function", "function": {...}}`` dicts.
    """
    if tools is None:
        tools = DEFAULT_TOOLS

    return _load_builtin_schemas_from_registry(frozenset(tools))
