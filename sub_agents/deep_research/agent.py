"""Deep Research sub-agent — single SP-driven ReAct loop.

Design:
- One unified ReAct loop. The system prompt instructs the model to walk
  plan → research → write itself; the harness does NOT inject ``[PHASE: X]``
  user messages.
- Exit condition mirrors the main agent: a turn with NO ``tool_calls`` ends
  the loop (the model has said its closing sentence). There is no
  model-authored sentinel string — the ``<delegate_result>`` envelope the main
  agent reads is assembled entirely by the dispatcher (tools.py) from the
  runner's derived ``status``/``result``, so the sub-model never hand-writes it.
- ``status``/``result`` are derived from history, not the model's word: a
  successfully-written report ``.md`` (see ``_find_written_result``) → success
  with that real path; otherwise → failure.
- All tools (including ``write``) are exposed every turn.
- The sub-agent reuses the main agent's ``_agent_loop``, so it inherits the
  same fallback/exit semantics (FINISHED / MAX_ITERATIONS / TOOL_ERROR /
  TIMEOUT / TRUNCATION / ERROR). On any non-FINISHED exit, ``_finalize``
  synthesizes a fallback report from collected evidence via one LLM call.
"""

import logging
import pathlib
import re

from typing import Any

import jinja2

from pydantic import BaseModel, Field

from nexus.core import ToolExecutor, require_context
from nexus.extensions.agents.yb_one_agent.agent import _agent_loop
from nexus.extensions.agents.yb_one_agent.prompt_builder import (
    _SKILLS_BY_NAME,
    _SKILLS_CONTAINER_DIR,
    build_meta_lines,
    build_tool_funcs,
    build_tool_schemas,
)


logger = logging.getLogger(__name__)

# ── System prompt (Jinja2 template, same Layer-3 pattern as the main agent) ──
# The SP body lives in ``system_prompt.j2`` next to this module and is rendered
# with the same ``meta_lines`` injection the main agent uses (``build_meta_lines``),
# so the sub-agent gets identical date/location lines without a bespoke append.
_TEMPLATE_DIR = pathlib.Path(__file__).parent
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=jinja2.StrictUndefined,
)


# Skills 暴露给子 agent 的白名单。子 agent 只做「查数据 → 写报告」,交付格式转换
# (docx/pptx/pdf)是主 agent 的活(见 system_prompt.j2 Step 3),故不暴露那些 skill。
# 描述与路径不在这里写死,一律从主 agent 的 ``_SKILLS_META`` 取 —— 单一真源,
# skill 改名/改描述(如 westock-tool → westock-screener)自动跟随,不会再抄旧。
# 名字写错会在 import 时 KeyError,fail-fast 优于静默少一个 skill。
DEEP_RESEARCH_SKILLS: tuple[str, ...] = ('westock-data', 'westock-screener')


def build_system_prompt(meta: dict[str, Any] | None = None) -> str:
    """Render the deep_research sub-agent system prompt with Layer-3 user info.

    Mirrors the main agent's ``build_system_prompt``: ``meta`` →
    :func:`build_meta_lines` → appended ``# 用户信息`` section. Empty meta → no
    section (matches the main agent).

    Skills 段与主 agent 同一数据源(``_SKILLS_META``)、同一模板循环,只是名单收窄到
    :data:`DEEP_RESEARCH_SKILLS`。
    """
    template = _jinja_env.get_template('system_prompt.j2')
    return template.render(
        meta_lines=build_meta_lines(meta),
        skills=[_SKILLS_BY_NAME[n] for n in DEEP_RESEARCH_SKILLS],
        skills_dir=_SKILLS_CONTAINER_DIR,
    ).strip()


# Deep-research sub-agent tool set — same convention the main agent uses in
# prompt_builder._BUILTIN_TOOL_ORDER: the ``_v2`` registry-key suffix for the
# four sandbox tools. Reusing ``build_tool_schemas``/``build_tool_funcs`` keeps
# the schema the model sees identical to the main agent's, so a single
# tool-naming contract is enforced for both.
DEEP_RESEARCH_TOOLS: frozenset[str] = frozenset({
    'bash_v2',
    'read_v3',
    'edit_v2',
    'write_v2',
    'web_fetch',
    'web_search',
    'load_skill',
})


class DeepResearchInput(BaseModel):
    """Input for deep_research sub-agent."""

    user_prompt: str = Field(..., description='The research query')
    max_iterations: int = Field(default=30)


class DeepResearchOutput(BaseModel):
    """Output from deep_research sub-agent."""

    status: str = Field(default='failure')
    result: str = Field(default='')
    iterations: int = Field(default=0)
    citations: list = Field(default_factory=list)
    history: list = Field(default_factory=list, description='Full sub-agent message history (for inspection / SFT).')


def _find_written_result(history: list[dict[str, Any]]) -> str | None:
    """Return the path of the LAST successfully-written ``result.md``, or None.

    We do NOT pair ``write`` tool_calls with their responses at all. The write
    tool echoes the full path in its own success message —
    ``"Successfully wrote N lines to <path>."`` (yb_one_agent/tools.py) —
    so the deliverable path is recoverable from the tool RESPONSE alone.

    This deliberately replaces call↔response pairing (by id OR by position).
    Both were fragile: the HyTuner client emits empty tool_call ids so id-keying
    collapses every response onto ``''``; and a global flatten+zip desyncs the
    moment a turn breaks mid-execution (``early_stop`` / ``observation_time_timeout``
    in ``_agent_loop`` leave that turn's trailing tool_calls unanswered, so
    ``calls`` outnumber ``responses`` and every later pairing shifts). That shift
    is what masked a successful salvage ``write`` on the post-salvage re-check
    (dr:edfec1f9c1a3: write succeeded yet was judged missing the same second).
    Reading the path straight out of the response sidesteps the entire class.

    Scans forward; the LAST successful ``.md`` write under
    ``/data/workspace/dr:`` wins. The filename is no longer pinned to
    ``result.md``: when the delegating prompt names the deliverable, the sub
    writes THAT name (so the main agent can deliver without a rename ``cp``);
    with no name given it still writes ``result.md``, which is why every
    pre-existing trajectory stays valid. "Last wins" is what makes the loose
    match safe — the SP puts the report write last (research-phase artifacts are
    csv/json), and salvage writes last of all. csv/png intermediates are
    filtered out by the ``.md`` suffix check.
    """
    # Greedy ``.+`` anchored on the trailing period the tool appends after the
    # path, so ``.../result.md.`` yields ``.../result.md``.
    pattern = re.compile(r'Successfully wrote \d+ lines to (.+)\.$')
    found = None
    for msg in history:
        if msg.get('role') != 'tool':
            continue
        m = pattern.match((msg.get('content') or '').strip())
        if not m:
            continue
        path = m.group(1)
        if path.endswith('.md') and '/data/workspace/dr:' in path:
            found = path
    return found


# Appended as a final user turn when the sub-agent ends WITHOUT writing
# ``result.md`` — either it exhausted ``max_iterations`` mid-research, or it
# FINISHED with a closing sentence that names a path it never actually wrote.
# We do NOT switch system prompts: the model keeps its original SP + full
# research history and is told to stop researching and call ``write`` now.
#
# The model writes ``result.md`` itself (rather than the harness writing it) for
# two reasons:
#   * the write's success response lands in ``history`` (it's the model's own
#     tool call), so ``_find_written_result`` — which scans tool responses for
#     the write tool's ``"Successfully wrote ... to <path>."`` line — can see it.
#     A harness-written file produces no such response, so the runner would
#     discard it as "wrote no result.md".
#   * the model follows its normal write-phase protocol, so frontmatter /
#     citations come along; we don't synthesize a body.
# The harness no longer writes the file itself.
def _salvage_instruction(trace_id: str) -> str:
    # Filename left to the model (SP Step 3 rules) — the salvaged file is
    # delivered to the user as-is, nobody renames it downstream anymore.
    return (
        '你已用完本轮研究预算，但尚未写出研究报告。停止一切检索与沙盒操作，'
        '立刻基于上述已收集的证据，调用一次 `write` 工具，把最终研究报告'
        '（不写 frontmatter 头，写足正文）写到 '
        f'/data/workspace/dr:{trace_id}/<报告文件名>.md，文件名按 Step 3 的命名规则取。'
        '`write` 返回成功后，下一回合按 Step 4 用一段话收尾（结论要点 + 该报告绝对路径），'
        '不要再调用任何工具，不要输出 XML 块或结构化标签。'
    )


async def _salvage_via_write(
    client: Any,
    history: list[dict[str, Any]],
    trace_id: str,
    *,
    executor: ToolExecutor,
    tools: list[dict[str, Any]],
) -> None:
    """Last-resort: append a "write now" instruction and run 2 more turns.

    Used when the sub-agent ends without writing ``result.md``. We append one
    user instruction (:func:`_salvage_instruction`) telling the model to stop
    researching and call ``write`` once, then run ``_agent_loop`` for 2 more
    turns. ``_agent_loop`` mutates ``history`` in place, so the model's
    ``write`` tool call + its success response land in history for
    ``_find_written_result`` to find on the re-check.

    The tool set is kept IDENTICAL to the main loop (not narrowed to
    ``write``): under the HyTuner client, changing ``tools`` mid-trajectory
    changes the rendered prefix and trips its cache guard (``rendering inputs
    changed mid-trajectory``), crashing salvage before the model writes
    anything. The instruction + 2-turn budget are what steer the model to
    write; we no longer rely on tool restriction to stop it re-researching.
    """
    canonical = f'/data/workspace/dr:{trace_id}/result.md'
    history.append({'role': 'user', 'content': _salvage_instruction(trace_id)})
    logger.warning(
        f'[deep_research] no result.md write in history; salvaging via "write now" instruction (2 turns) → {canonical}'
    )
    async for _ in _agent_loop(
        history=history,
        executor=executor,
        client=client,
        tools=tools,
        max_iterations=2,
    ):
        pass


async def _finalize(
    iterations: int,
    history: list,
    *,
    client: Any,
    trace_id: str,
    executor: ToolExecutor,
    tools: list[dict[str, Any]],
) -> DeepResearchOutput:
    """Derive status/result from history; salvage via ``write`` if needed.

    No model-authored ``<delegate_result>`` is parsed: the sub-model closes out
    by simply ending a turn with no tool_calls. Success is defined by history —
    a successfully-written ``result.md`` (see ``_find_written_result``) — not by
    anything the model claims.

    If no ``result.md`` write is found — whether the sub ran out of budget
    mid-research OR FINISHED with a closing sentence that names a path it never
    wrote — give it one more shot via ``_salvage_via_write`` (a "write now"
    instruction + 2 more turns) so the write lands in history, then re-check. This covers both the
    max-iterations case and the FINISHED-without-write case uniformly; the old
    ``force_salvage`` (non-FINISHED only) gate is gone.

    Contract: on success ``result`` is the sub-agent's closing assistant
    sentence, which MUST contain the report's absolute path. We do NOT trust a
    path the model typed into that sentence (it often names a ``dr:<id>`` dir it
    never actually wrote — see ``_find_written_result``); instead the harness
    resolves the REAL written path from history and guarantees it appears in the
    returned sentence (appending it if the model's closing line omitted it or
    named a different path). The dispatcher (tools.py) wraps this verbatim into
    the ``<delegate_result>`` the main agent reads, and the main agent parses the
    path out of this sentence.
    """
    written = _find_written_result(history)
    if written is None:
        # Sub never wrote result.md (max-iterations mid-research, OR finished
        # with a hallucinated "报告写入 result.md" closing line). Give it a
        # "write now" instruction + 2 more turns so the write lands in history.
        await _salvage_via_write(client, history, trace_id, executor=executor, tools=tools)
        written = _find_written_result(history)
    status = 'success' if written is not None else 'failure'

    if status == 'success':
        assert written is not None
        real_path = written
        # Re-read the last assistant turn AFTER any salvage, so the closing
        # sentence is the post-write one (which per SP names the path). Return
        # it when it already names the real path; otherwise rewrite it so the
        # ONLY path present is the real one — never leave a wrong path in
        # ``<result>`` for the main agent to parse and read a ghost file.
        closing = next(
            (m['content'] for m in reversed(history) if m.get('role') == 'assistant'),
            '',
        )
        closing = (closing or '').strip()
        if closing and real_path and real_path in closing:
            result = closing
        elif closing and '/data/workspace/dr:' in closing:
            # Model named a different (wrong) dr path. Rewrite just the path
            # tokens, not the whole sentence — the closing turn now also carries
            # the conclusion bullets the main agent delivers from, and dropping
            # those would force it to re-read the full report.
            result = re.sub(r'/data/workspace/dr:[^\s，。、）)"\']+', real_path, closing)
        elif closing:
            result = f'{closing}\n报告已写入：{real_path}'
        else:
            result = f'研究已完成，报告写入 {real_path}。'
    else:
        # Failure: real reason, not the model's own (often false) closing claim.
        result = 'sub-agent 未写出 result.md（兜底 write 也未成功）'

    logger.info(f'Deep research completed: status={status}, iterations={iterations}')
    return DeepResearchOutput(
        status=status,
        result=result,
        iterations=iterations,
        citations=[],
        history=history or [],
    )


async def run_deep_research(agent_input: DeepResearchInput) -> DeepResearchOutput:
    """Run deep_research sub-agent as a single SP-driven ReAct loop."""
    ctx = require_context()
    logger.info(f'Deep research sub-agent started with prompt: {agent_input.user_prompt[:100]}...')

    # Prefer dedicated 'deep_research' OpenAI client when configured.
    try:
        client = ctx.get_openai_client('deep_research')
    except KeyError:
        client = ctx.openai_client

    # Tool funcs + schemas both resolve from the single DEEP_RESEARCH_TOOLS
    # name set via prompt_builder (same name→registry-key mapping the main
    # agent uses), so the executor and the model see exactly the same tools.
    tools = build_tool_funcs(DEEP_RESEARCH_TOOLS)
    if not tools:
        logger.error('No tools available for deep_research sub-agent')
        return DeepResearchOutput(status='failure', result='No tools available', iterations=0)

    executor = ToolExecutor(tools)
    tool_schemas = build_tool_schemas(DEEP_RESEARCH_TOOLS)

    system_prompt = build_system_prompt(ctx.task_config.get('meta'))
    _run_id = getattr(ctx, 'run_id', None)
    trace_id = _run_id[-12:] if _run_id else 'unknown'
    history: list[dict[str, Any]] = [
        {'role': 'system', 'content': system_prompt},
        {
            'role': 'user',
            'content': f'{agent_input.user_prompt}\n\n(trace_id: {trace_id})',
        },
    ]

    max_iterations = agent_input.max_iterations
    iterations = 0

    # Reuse the main agent's non-streaming loop so the sub-agent inherits the
    # same fallback/exit semantics (FINISHED / MAX_ITERATIONS / TOOL_ERROR /
    # TIMEOUT / TRUNCATION / ERROR) as the main agent, instead of a bespoke
    # loop. The loop mutates ``history`` in place, capturing the full trajectory
    # for ``_finalize`` to inspect (did it write result.md?).
    exit_status: str | None = None
    error_message: str | None = None
    async for event in _agent_loop(
        history=history,
        executor=executor,
        client=client,
        tools=tool_schemas,
        max_iterations=max_iterations,
    ):
        etype = event.get('type')
        if etype == 'error':
            error_message = event.get('message')
        elif etype == 'done':
            exit_status = event.get('exit_status')
            iterations = event.get('iterations', iterations)

    if exit_status == 'error' or error_message is not None:
        logger.error(f'Deep research failed: {error_message}', exc_info=True)
        return DeepResearchOutput(
            status='failure',
            result=f'Error: {error_message}',
            iterations=iterations,
            citations=[],
        )

    # Derive status/result from history. _finalize salvages via a "write now"
    # instruction + 2 more turns whenever no result.md write is found — covers both the
    # max-iterations case and a FINISHED-without-write closing line.
    return await _finalize(
        iterations,
        history,
        client=client,
        trace_id=trace_id,
        executor=executor,
        tools=tool_schemas,
    )


__all__ = ['run_deep_research', 'DeepResearchInput', 'DeepResearchOutput']
