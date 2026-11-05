"""One-agent implementation for nexus framework.

General-purpose agent that can run tasks interactively (streaming) or non-streaming
(inference / RL training).  Both modes share a single ``_agent_loop`` async generator;
the only runtime difference is ``stream=True/False`` on the LLM call.
"""

import asyncio
import json
import logging
import time

from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    ContextLengthExceededError,
    Registry,
    ToolExecutor,
    WithMainTrajectory,
    WithUserPrompt,
    agent_registry,
    require_context,
    tool_registry,
)
from nexus.extensions.agents.yb_one_agent.llm_errors import is_llm_http_error
from nexus.extensions.agents.yb_one_agent.prompt_builder import (
    DEFAULT_TOOLS,
    build_system_prompt,
    build_tool_schemas,
)
from nexus.extensions.agents.yb_one_agent.tools import _INTERACTIVE_SIGNAL
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.runtime.models import RemoteInvokeError
from nexus.trace import get_current_span_id, update_span


def _tool_history_content(output: list, text: str) -> str | list[dict]:
    """Keep native images in model history while text follows the existing limit."""
    images = [part for part in output if part.get('type') == 'image_url']
    return [{'type': 'text', 'text': text}, *images] if images else text


logger = logging.getLogger(__name__)


# Import tools to register them
Registry.import_module('nexus.extensions.agents.yb_one_agent.tools')
Registry.import_module('nexus.extensions.agents.yb_one_agent.tool')


# ============== Observation Truncation ==============

DEFAULT_TEXT_CONTENT_LIMIT = 50_000

DEFAULT_TRUNCATE_NOTICE = (
    '<response clipped><NOTE>Due to the max output limit, only part of the full response has been shown to you.</NOTE>'
)


def maybe_truncate(
    content: str,
    truncate_after: int | None = None,
    truncate_notice: str = DEFAULT_TRUNCATE_NOTICE,
) -> str:
    """Truncate the middle of content if it exceeds the specified length."""
    if not truncate_after or len(content) <= truncate_after or truncate_after < 0:
        return content
    if len(truncate_notice) >= truncate_after:
        return truncate_notice[:truncate_after]
    available_chars = truncate_after - len(truncate_notice)
    head_chars = available_chars // 2 + (available_chars % 2)
    tail_chars = available_chars - head_chars
    return content[:head_chars] + truncate_notice + (content[-tail_chars:] if tail_chars > 0 else '')


# ============== Extended Thinking ==============


def _serialize_thinking_blocks(message: Any) -> list[dict[str, Any]] | None:
    """Dump Claude ``thinking_blocks`` (with signature) for history / trajectory.

    Anthropic multi-turn tool use requires signed thinking blocks to be echoed
    back; keeping only ``reasoning_content`` text is not enough.
    """
    blocks = getattr(message, 'thinking_blocks', None)
    if not blocks:
        return None
    out: list[dict[str, Any]] = []
    for tb in blocks:
        get = tb.get if isinstance(tb, dict) else (lambda k, default=None, _tb=tb: getattr(_tb, k, default))
        if (get('type') or 'thinking') == 'redacted_thinking':
            data = get('data') or ''
            if data:
                out.append({'type': 'redacted_thinking', 'data': data})
            continue
        entry: dict[str, Any] = {'type': 'thinking', 'thinking': get('thinking') or ''}
        sig = get('signature')
        if sig is not None:
            entry['signature'] = sig
        if entry['thinking'] or entry.get('signature'):
            out.append(entry)
    return out or None


def _merge_thinking_block_deltas(acc: dict[int, dict[str, Any]], deltas: Any) -> None:
    """Accumulate streamed ``thinking_blocks`` deltas into ``acc`` by index."""
    if not deltas:
        return
    for tb_i, tb in enumerate(deltas):
        get = tb.get if isinstance(tb, dict) else (lambda k, default=None, _tb=tb: getattr(_tb, k, default))
        slot = acc.setdefault(
            tb_i,
            {'type': 'thinking', 'thinking': '', 'signature': None, 'data': ''},
        )
        tb_type = get('type')
        if tb_type:
            slot['type'] = tb_type
        thinking = get('thinking')
        if thinking:
            slot['thinking'] += thinking
        signature = get('signature')
        if signature:
            slot['signature'] = signature
        data = get('data')
        if data:
            slot['data'] += data


def _finalize_thinking_blocks_acc(acc: dict[int, dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Turn the streaming accumulator into serializable thinking blocks."""
    if not acc:
        return None
    out: list[dict[str, Any]] = []
    for i in sorted(acc):
        tb = acc[i]
        if tb.get('type') == 'redacted_thinking' and tb.get('data'):
            out.append({'type': 'redacted_thinking', 'data': tb['data']})
        elif tb.get('thinking') or tb.get('signature'):
            entry: dict[str, Any] = {'type': 'thinking', 'thinking': tb.get('thinking') or ''}
            if tb.get('signature') is not None:
                entry['signature'] = tb['signature']
            out.append(entry)
    return out or None


# ============== Default Templates ==============

NEXT_STEP_NO_OUTPUT_TEMPLATE = 'Your last command ran successfully and did not produce any output.'


# ============== Prompt Caching ==============


def _inject_cache_control(history: list[dict]) -> list[dict]:
    """Return a copy of history with cache_control injected on the last content block
    of the *last* message only.

    Anthropic allows at most 4 cache_control blocks per request.  Marking only the
    tail of the conversation tells the API "cache everything up to this point",
    which is exactly what we want for multi-turn / multi-step tool-calling sessions.
    """
    import copy

    if not history:
        return history

    result = [copy.copy(msg) for msg in history]

    # Only touch the very last message.
    last = result[-1]
    content = last.get('content')
    if isinstance(content, str):
        result[-1] = {**last, 'content': [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]}
    elif isinstance(content, list) and content:
        blocks = [copy.copy(b) for b in content]
        blocks[-1] = {**blocks[-1], 'cache_control': {'type': 'ephemeral'}}
        result[-1] = {**last, 'content': blocks}

    return result


class ExitStatus(str, Enum):
    FINISHED = 'finished'
    """Normal exit — model returned a response with no tool calls."""

    HANDOFF = 'handoff'
    """Normal routing exit — model requested the stronger expert agent."""

    MAX_ITERATIONS = 'max_iterations'
    """Agent reached the maximum iterations limit."""

    TOOL_ERROR = 'tool_error'
    """Abnormal exit — tool arguments could not be parsed after all retries."""

    ERROR = 'error'
    """Abnormal exit — unexpected exception, or a dead sandbox / runtime."""

    TRUNCATION = 'truncation'
    """Abnormal exit — context window exceeded."""

    TIMEOUT = 'timeout'
    """Abnormal exit — observation or inference time limit exceeded."""

    AWAITING_USER = 'awaiting_user'
    """Paused — ask_user_question interactive mode waiting for user input."""


EXIT_STATUS_TO_TERMINATION_REASON = {
    ExitStatus.FINISHED: 'ENV_DONE',
    ExitStatus.HANDOFF: 'ENV_DONE',
    ExitStatus.MAX_ITERATIONS: 'MAX_STEPS',
    ExitStatus.TOOL_ERROR: 'FORMAT_ERROR',
    ExitStatus.ERROR: 'UNHANDLED_ERROR',
    ExitStatus.TRUNCATION: 'TRUNCATION',
    ExitStatus.TIMEOUT: 'TASK_TIMEOUT',
    ExitStatus.AWAITING_USER: 'AWAITING_USER',
}


class SandboxInfrastructureError(RuntimeError):
    """Fatal sandbox / runtime failure (Gongfeng death, image pull, health watchdog).

    Must abort the agent loop. Callers should fail the run rather than treat this
    as a recoverable tool observation the model can talk past.
    """


# Watchdog / remote-invoke text that means the sandbox is dead — not a
# model-recoverable tool error. Markers stay specific to avoid false positives
# (e.g. the word "unhealthy" in user documents).
_INFRA_ERROR_MARKERS = (
    'runtimeunhealthyerror',
    'sandboxinfrastructureerror',
    'runtime marked unhealthy',
    'marking unhealthy',
    'sandboxexit',
    ' died:',  # "Sandbox 'main' (...) DIED:"
    '/health unreachable',
    'is no longer in a usable state',
    'oomkilled',
)


def is_sandbox_infrastructure_error(err: BaseException | str) -> bool:
    """True when a tool failure is a dead sandbox / runtime, not an in-domain error."""
    if isinstance(err, SandboxInfrastructureError | RuntimeUnhealthyError):
        return True
    if isinstance(err, RemoteInvokeError):
        if err.cause is not None and is_sandbox_infrastructure_error(err.cause):
            return True
        return _text_looks_like_infra_error(str(err))
    if isinstance(err, BaseException):
        cause = err.__cause__ or err.__context__
        if cause is not None and cause is not err and is_sandbox_infrastructure_error(cause):
            return True
        return _text_looks_like_infra_error(f'{type(err).__name__}: {err}')
    return _text_looks_like_infra_error(err)


def _text_looks_like_infra_error(text: str) -> bool:
    if not text:
        return False
    return any(marker in text.lower() for marker in _INFRA_ERROR_MARKERS)


def _as_infra_error(err: BaseException | str) -> SandboxInfrastructureError:
    if isinstance(err, SandboxInfrastructureError):
        return err
    if isinstance(err, BaseException):
        wrapped = SandboxInfrastructureError(str(err))
        wrapped.__cause__ = err
        return wrapped
    return SandboxInfrastructureError(err)


def _abort_status_for_tool_failure(
    *,
    exc: BaseException | None,
    output_text: str,
    early_stop: bool,
    total_observation_time: float,
    observation_time_timeout: float | None,
) -> tuple[ExitStatus | None, SandboxInfrastructureError | None]:
    """Decide whether a tool failure should abort the agent loop.

    Returns ``(exit_status, infra_error)``. ``exit_status is None`` means continue.
    ``infra_error`` is set only for sandbox / runtime death — the loop must
    re-raise it after emitting ``done`` so the run is marked failed.
    """
    infra_source: BaseException | str | None = None
    if exc is not None and is_sandbox_infrastructure_error(exc):
        infra_source = exc
    elif is_sandbox_infrastructure_error(output_text):
        infra_source = output_text
    if infra_source is not None:
        logger.error('Sandbox/runtime infrastructure failure; aborting agent: %s', infra_source)
        return ExitStatus.ERROR, _as_infra_error(infra_source)
    if early_stop and exc is not None:
        logger.warning('early_stop=True: stopping on tool error')
        return ExitStatus.TOOL_ERROR, None
    if observation_time_timeout and total_observation_time >= observation_time_timeout:
        return ExitStatus.TIMEOUT, None
    return None, None


# ============== Input/Output Models ==============


class OneAgentInput(AgentRunInput, WithUserPrompt):
    """Input configuration for one-agent."""

    max_iterations: int = Field(default=50, description='Maximum number of agent steps.')
    completion_retry_on_tool_error: int = Field(
        default=3,
        description='Retries when tool arguments JSON is invalid or LLM returns empty response. 0 = fail immediately.',
    )
    llm_retry_on_error: int = Field(
        default=3,
        description='Retries on LLM API errors (network, rate-limit, 5xx). Uses exponential backoff.',
    )
    early_stop: bool = Field(
        default=False,
        description=(
            'Stop immediately when a tool raises an in-domain exception. '
            'Sandbox / runtime death (RuntimeUnhealthyError) always aborts, '
            'regardless of this flag.'
        ),
    )
    tools: list[str] | None = Field(
        default=None,
        description=(
            'Set of active tool names. None → use DEFAULT_TOOLS. '
            'Controls which tool instructions appear in the prompt and which schemas are sent to the LLM.'
        ),
    )
    skills: list[str] | None = Field(
        default=None,
        description=(
            'List of skill folder names to include in the Skills section. '
            'None → include all skills. Empty list → omit skills section.'
        ),
    )
    terminal_type: str = Field(
        default='web',
        description="Terminal/client type. 'app' → mobile style; 'web' → PC/web style.",
    )
    history: list[dict] | None = Field(
        default=None,
        description=(
            'Prior conversation messages to insert between the system prompt '
            'and the current user_prompt. Used for multi-turn sessions '
            '(query+answer pairs from previous turns) or resume-from-trajectory '
            '(a partial trajectory ending in a tool result). Each item is an '
            'OpenAI-style message dict (role/content[/tool_calls/tool_call_id]).'
        ),
    )
    resume_from_history: bool = Field(
        default=False,
        description=(
            'When True, `history` already ends with the message the agent '
            'should continue from (e.g. a tool result), so NO new user message '
            'is appended — the loop resumes directly from `history`. '
            '`user_prompt` is ignored in this mode.'
        ),
    )


class OneAgentOutput(AgentRunOutput, WithMainTrajectory):
    """Output from one-agent run."""

    exit_status: ExitStatus = Field(description='How the agent exited.')
    iterations: int = Field(default=0, description='Number of iterations completed.')


# ============== Core Agent Loop ==============


async def _agent_loop(
    history: list[dict],
    executor: ToolExecutor,
    client: Any,
    tools: list[dict],
    max_iterations: int = 50,
    completion_retry_on_tool_error: int = 3,
    llm_retry_on_error: int = 3,
    early_stop: bool = False,
    observation_time_timeout: float | None = None,
    inference_time_timeout: float | None = None,
) -> AsyncGenerator[dict, None]:
    """Non-streaming agent loop for inference / RL training.

    Mutates ``history`` in-place by appending every assistant and tool message,
    ensuring full trajectory capture for RL training.

    Yielded event types:

    - ``turn_start`` — new LLM call is starting
    - ``tool_start`` — tool about to execute: ``{name, args}``
    - ``tool_output`` — tool result (success): ``{name, output}``
    - ``tool_error``  — tool failure, either:
                       (a) JSON parse failure on tool args (pre-flight,
                           tool was never invoked), or
                       (b) tool ran cleanly but returned ``ToolResponse(error=True)``
                           — i.e. an in-domain failure the LLM should see.
                       Payload: ``{name, error}``
    - ``done`` — loop finished: ``{exit_status, iterations, observation_time, inference_time}``
    - ``error`` — unexpected exception: ``{message}`` (replaces ``done``)
    """
    exit_status: ExitStatus | None = None
    iteration = 0
    total_observation_time = 0.0
    total_inference_time = 0.0
    infra_error: SandboxInfrastructureError | None = None
    llm_http_error: BaseException | None = None

    try:
        while iteration < max_iterations:
            logger.info(f'=== one-agent step {iteration + 1} ===')
            yield {'type': 'turn_start'}

            # ── LLM call (with JSON-parse retry and API-error backoff) ──────────

            message = None
            parsed_tool_args: list = []

            for retry_attempt in range(completion_retry_on_tool_error + 1):
                # API-level backoff retry
                for llm_attempt in range(llm_retry_on_error + 1):
                    try:
                        t0 = time.time()
                        response = await client.create_completion(
                            messages=history,
                            tools=tools or None,
                        )
                        message = response.choices[0].message
                        total_inference_time += time.time() - t0
                        break
                    except ContextLengthExceededError:
                        raise
                    except Exception as llm_err:
                        if llm_attempt < llm_retry_on_error:
                            backoff = 2**llm_attempt
                            logger.warning(
                                f'LLM API error ({type(llm_err).__name__}: {llm_err}), '
                                f'retry {llm_attempt + 1}/{llm_retry_on_error} in {backoff}s'
                            )
                            await asyncio.sleep(backoff)
                        else:
                            logger.error(f'LLM API error after {llm_retry_on_error} retries: {llm_err}')
                            raise

                if inference_time_timeout and total_inference_time >= inference_time_timeout:
                    logger.info(f'Inference timeout: {total_inference_time:.1f}s >= {inference_time_timeout}s')
                    exit_status = ExitStatus.TIMEOUT
                    break

                # No tool calls → model is done answering
                if not message.tool_calls:  # type: ignore[union-attr]
                    # Empty response (no content AND no tool_calls) is likely an
                    # LLM service issue (empty packet); retry.
                    if not (message.content or '').strip():  # type: ignore[possibly-missing-attribute]
                        if retry_attempt < completion_retry_on_tool_error:
                            logger.warning(
                                f'LLM returned empty response (no content, no tool_calls), '
                                f'retry {retry_attempt + 1}/{completion_retry_on_tool_error}'
                            )
                            continue  # retry LLM call without touching history
                        logger.error('LLM returned empty response after retries, marking as failed')
                        exit_status = ExitStatus.TOOL_ERROR
                        break
                    exit_status = ExitStatus.FINISHED
                    break

                # Parse tool arguments — retry only on JSON error
                parsed_tool_args = []
                parse_error: Exception | None = None
                for tool_call in message.tool_calls:  # type: ignore[union-attr]
                    try:
                        parsed_tool_args.append(json.loads(tool_call.function.arguments or '{}'))
                    except json.JSONDecodeError as e:
                        parse_error = e
                        break

                if parse_error is not None:
                    if retry_attempt < completion_retry_on_tool_error:
                        logger.warning(
                            f'JSON parse error: {parse_error}, '
                            f'retry {retry_attempt + 1}/{completion_retry_on_tool_error}'
                        )
                        continue  # retry LLM call without touching history
                    logger.error(f'JSON parse error after {completion_retry_on_tool_error} retries: {parse_error}')
                    exit_status = ExitStatus.TOOL_ERROR
                    break

                if any(tc.function.name == 'handoff' for tc in message.tool_calls):  # type: ignore[union-attr]
                    if iteration != 0:
                        logger.warning(f'handoff requested after the first step: step={iteration + 1}')
                    if len(message.tool_calls) != 1:  # type: ignore[union-attr]
                        logger.warning('handoff returned with additional tool calls; none of them will be executed')
                    exit_status = ExitStatus.HANDOFF

                break  # valid response

            # Always append the final assistant message to history (needed for trajectory)
            if message is not None:
                is_handoff = exit_status == ExitStatus.HANDOFF
                assistant_msg: dict = {
                    'role': 'assistant',
                    'content': (message.content or '') if is_handoff else (message.content or None),
                }
                if getattr(message, 'reasoning_content', None):
                    assistant_msg['reasoning_content'] = message.reasoning_content
                thinking_blocks = _serialize_thinking_blocks(message)
                if thinking_blocks:
                    assistant_msg['thinking_blocks'] = thinking_blocks
                if message.tool_calls:
                    if is_handoff:
                        assistant_msg['tool_calls'] = [
                            {
                                'type': 'function',
                                'function': {
                                    'name': tc.function.name,
                                    'arguments': parsed_tool_args[idx],
                                },
                            }
                            for idx, tc in enumerate(message.tool_calls)
                            if tc.function.name == 'handoff'
                        ]
                    else:
                        assistant_msg['tool_calls'] = [
                            {
                                'id': tc.id,
                                'type': 'function',
                                'function': {'name': tc.function.name, 'arguments': tc.function.arguments},
                            }
                            for tc in message.tool_calls
                        ]
                history.append(assistant_msg)

            if exit_status is not None:
                break

            # ── Tool execution ────────────────────────────────────────────────

            for idx, tool_call in enumerate(message.tool_calls):  # type: ignore[union-attr]
                tool_name = tool_call.function.name
                tool_args = parsed_tool_args[idx]

                logger.info(f'Tool call: {tool_name}')
                logger.debug(f'Tool args: {json.dumps(tool_args, ensure_ascii=False)[:500]}')

                yield {'type': 'tool_start', 'name': tool_name, 'args': tool_args}

                t0 = time.time()
                try:
                    tool_response = await executor.call(tool_name, tool_args)
                    raw_output = '\n'.join(
                        part.get('text', '') for part in tool_response.output if part.get('type') == 'text'
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(
                        f'Tool execution error — {tool_name}\n  args: {tool_args}\n  error: {type(e).__name__}: {e}'
                    )
                    raw_output = f'[Tool error: {e}]'
                    yield {'type': 'tool_error', 'name': tool_name, 'error': raw_output}
                    history.append({'role': 'tool', 'tool_call_id': tool_call.id, 'content': raw_output})
                    total_observation_time += time.time() - t0
                    abort, infra_error = _abort_status_for_tool_failure(
                        exc=e,
                        output_text=raw_output,
                        early_stop=early_stop,
                        total_observation_time=total_observation_time,
                        observation_time_timeout=observation_time_timeout,
                    )
                    if abort is not None:
                        exit_status = abort
                        break
                    continue

                total_observation_time += time.time() - t0

                # Truncate and wrap observation
                if raw_output:
                    truncated = maybe_truncate(raw_output, truncate_after=DEFAULT_TEXT_CONTENT_LIMIT)
                    if len(truncated) < len(raw_output):
                        logger.info(f'Observation truncated: {len(raw_output)} → {len(truncated)} chars')
                    output_text = truncated
                else:
                    output_text = NEXT_STEP_NO_OUTPUT_TEMPLATE

                # Honor ``meta['error'] == True`` — tool ran cleanly but signaled
                # an in-domain failure (e.g. backend returned 500, args failed
                # business validation).  Surface as ``tool_error`` so the UI
                # / trace can distinguish from successful results, while still
                # appending the same observation text to history so the LLM
                # can read the error reason and recover — unless the payload
                # is a dead sandbox / runtime, which is fatal.
                if tool_response.meta.get('error'):
                    yield {'type': 'tool_error', 'name': tool_name, 'error': raw_output}
                    abort, infra_error = _abort_status_for_tool_failure(
                        exc=None,
                        output_text=raw_output,
                        early_stop=False,
                        total_observation_time=total_observation_time,
                        observation_time_timeout=observation_time_timeout,
                    )
                    history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': _tool_history_content(tool_response.output, output_text),
                    })
                    if abort is not None:
                        exit_status = abort
                        break
                    if observation_time_timeout and total_observation_time >= observation_time_timeout:
                        logger.info(
                            f'Observation timeout: {total_observation_time:.1f}s >= {observation_time_timeout}s'
                        )
                        exit_status = ExitStatus.TIMEOUT
                        break
                    continue
                yield {'type': 'tool_output', 'name': tool_name, 'output': raw_output}
                history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': _tool_history_content(tool_response.output, output_text),
                })

                logger.debug(f'Tool output: {output_text[:200]}')

                if observation_time_timeout and total_observation_time >= observation_time_timeout:
                    logger.info(f'Observation timeout: {total_observation_time:.1f}s >= {observation_time_timeout}s')
                    exit_status = ExitStatus.TIMEOUT
                    break

            if exit_status is not None:
                break

            iteration += 1

    except ContextLengthExceededError as e:
        logger.error(f'Context length exceeded: {e}')
        exit_status = ExitStatus.TRUNCATION
    except Exception as e:
        logger.exception(f'Agent error: {e}')
        yield {'type': 'error', 'message': str(e)}
        exit_status = ExitStatus.ERROR
        if is_sandbox_infrastructure_error(e):
            infra_error = _as_infra_error(e)
        elif is_llm_http_error(e):
            llm_http_error = e

    if exit_status is None:
        exit_status = ExitStatus.MAX_ITERATIONS
        logger.warning(f'Max iterations reached ({max_iterations})')

    logger.info(f'one-agent done: status={exit_status.value} iterations={iteration + 1}')

    yield {
        'type': 'done',
        'exit_status': exit_status.value,
        'iterations': iteration + 1,
        'observation_time': total_observation_time,
        'inference_time': total_inference_time,
    }
    if infra_error is not None:
        raise infra_error
    if llm_http_error is not None:
        raise llm_http_error


async def _agent_loop_stream(
    history: list[dict],
    executor: ToolExecutor,
    client: Any,
    tools: list[dict],
    max_iterations: int = 50,
    completion_retry_on_tool_error: int = 3,
    llm_retry_on_error: int = 3,
    early_stop: bool = False,
    observation_time_timeout: float | None = None,
    inference_time_timeout: float | None = None,
    extra_headers: dict[str, str] | None = None,
) -> AsyncGenerator[dict, None]:
    """Streaming variant of _agent_loop: identical logic but uses stream=True on LLM calls,
    yielding ``token`` events for each text chunk as it arrives."""

    exit_status: ExitStatus | None = None
    iteration = 0
    total_observation_time = 0.0
    total_inference_time = 0.0
    infra_error: SandboxInfrastructureError | None = None
    llm_http_error: BaseException | None = None

    try:
        while iteration < max_iterations:
            logger.info(f'=== one-agent step {iteration + 1} (stream) ===')
            yield {'type': 'turn_start'}

            # ── Streaming LLM call ────────────────────────────────────────────

            collected_content = ''
            collected_reasoning = ''
            collected_thinking_blocks: dict[int, dict[str, Any]] = {}
            collected_tool_calls: dict[int, dict] = {}
            parsed_tool_args: list = []

            for retry_attempt in range(completion_retry_on_tool_error + 1):
                collected_content = ''
                collected_reasoning = ''
                collected_thinking_blocks = {}
                collected_tool_calls = {}

                for llm_attempt in range(llm_retry_on_error + 1):
                    try:
                        t0 = time.time()

                        # Prompt caching is a Claude-only feature; skip for other models.
                        model_name = str(client.get_effective_request_args().get('model') or '')
                        is_claude = 'claude' in model_name.lower()
                        messages_to_send = _inject_cache_control(history) if is_claude else history
                        req_headers = {'anthropic-beta': 'prompt-caching-2024-07-31'} if is_claude else {}
                        if extra_headers:
                            req_headers.update(extra_headers)

                        request_args = client._apply_enforced_request_args({
                            'model': model_name,
                            'messages': messages_to_send,
                            'tools': tools or None,
                            'stream': True,
                            'stream_options': {'include_usage': True},
                            'extra_headers': req_headers,
                            **{k: v for k, v in client.config.request_args.items() if k != 'model'},
                        })
                        stream_resp = await client.client.chat.completions.create(**request_args)
                        async for chunk in stream_resp:
                            # Check for usage on any chunk (LiteLLM sends it on the last chunk)
                            if chunk.usage:
                                u = chunk.usage
                                cache_info = {
                                    'input_tokens': u.prompt_tokens,
                                    'output_tokens': u.completion_tokens,
                                }
                                raw = u.model_extra or {}
                                if raw.get('cache_read_input_tokens'):
                                    cache_info['cache_read_input_tokens'] = raw['cache_read_input_tokens']
                                if raw.get('cache_creation_input_tokens'):
                                    cache_info['cache_creation_input_tokens'] = raw['cache_creation_input_tokens']
                                logger.info(f'usage: {cache_info}')
                                yield {'type': 'usage', **cache_info}
                            if not chunk.choices:
                                continue
                            delta = chunk.choices[0].delta if chunk.choices else None
                            if delta is None:
                                continue
                            if delta.content:
                                collected_content += delta.content
                                yield {'type': 'token', 'content': delta.content}
                            delta_reasoning = getattr(delta, 'reasoning_content', None)
                            if delta_reasoning:
                                collected_reasoning += delta_reasoning
                                yield {'type': 'reasoning_token', 'content': delta_reasoning}
                            _merge_thinking_block_deltas(
                                collected_thinking_blocks,
                                getattr(delta, 'thinking_blocks', None),
                            )
                            if delta.tool_calls:
                                for tc_delta in delta.tool_calls:
                                    idx = tc_delta.index
                                    if idx not in collected_tool_calls:
                                        collected_tool_calls[idx] = {
                                            'id': '',
                                            'type': 'function',
                                            'function': {'name': '', 'arguments': ''},
                                        }
                                    tc = collected_tool_calls[idx]
                                    if tc_delta.id:
                                        tc['id'] += tc_delta.id
                                    if tc_delta.function:
                                        if tc_delta.function.name:
                                            tc['function']['name'] += tc_delta.function.name
                                        if tc_delta.function.arguments:
                                            tc['function']['arguments'] += tc_delta.function.arguments
                                    # Emit streaming chunk so the UI can show tool call progress
                                    yield {
                                        'type': 'tool_call_chunk',
                                        'index': idx,
                                        'name': tc['function']['name'],
                                        'arguments': tc['function']['arguments'],
                                    }
                        total_inference_time += time.time() - t0
                        break
                    except ContextLengthExceededError:
                        raise
                    except Exception as llm_err:
                        if llm_attempt < llm_retry_on_error:
                            backoff = 2**llm_attempt
                            logger.warning(
                                f'LLM API error ({type(llm_err).__name__}: {llm_err}), '
                                f'retry {llm_attempt + 1}/{llm_retry_on_error} in {backoff}s'
                            )
                            await asyncio.sleep(backoff)
                        else:
                            logger.error(f'LLM API error after {llm_retry_on_error} retries: {llm_err}')
                            raise

                if inference_time_timeout and total_inference_time >= inference_time_timeout:
                    logger.info(f'Inference timeout: {total_inference_time:.1f}s >= {inference_time_timeout}s')
                    exit_status = ExitStatus.TIMEOUT
                    break

                # No tool calls → done
                if not collected_tool_calls:
                    # Empty response (no content AND no tool_calls) is likely an
                    # LLM service issue (empty packet); retry.
                    if not (collected_content or '').strip():
                        if retry_attempt < completion_retry_on_tool_error:
                            logger.warning(
                                f'LLM returned empty response (no content, no tool_calls), '
                                f'retry {retry_attempt + 1}/{completion_retry_on_tool_error}'
                            )
                            yield {'type': 'turn_start'}
                            continue  # retry LLM call without touching history
                        logger.error('LLM returned empty response after retries, marking as failed')
                        exit_status = ExitStatus.TOOL_ERROR
                        break
                    exit_status = ExitStatus.FINISHED
                    break

                # Parse tool arguments
                sorted_tcs = [collected_tool_calls[i] for i in sorted(collected_tool_calls)]
                parsed_tool_args = []
                parse_error: Exception | None = None
                for tc in sorted_tcs:
                    try:
                        parsed_tool_args.append(json.loads(tc['function']['arguments'] or '{}'))
                    except json.JSONDecodeError as e:
                        parse_error = e
                        tool_name = tc['function']['name']
                        err_msg = f'[JSON parse error: {e}]'
                        yield {'type': 'tool_error', 'name': tool_name, 'error': err_msg}
                        break

                if parse_error is not None:
                    if retry_attempt < completion_retry_on_tool_error:
                        logger.warning(
                            f'JSON parse error: {parse_error}, '
                            f'retry {retry_attempt + 1}/{completion_retry_on_tool_error}'
                        )
                        yield {'type': 'turn_start'}  # reset frontend bubble before retry tokens
                        continue
                    logger.error(f'JSON parse error after {completion_retry_on_tool_error} retries: {parse_error}')
                    exit_status = ExitStatus.TOOL_ERROR
                    break

                if any(tc['function']['name'] == 'handoff' for tc in sorted_tcs):
                    if iteration != 0:
                        logger.warning(f'handoff requested after the first step: step={iteration + 1}')
                    if len(sorted_tcs) != 1:
                        logger.warning('handoff returned with additional tool calls; none of them will be executed')
                    exit_status = ExitStatus.HANDOFF

                break  # valid response

            # Build and append assistant message to history
            is_handoff = exit_status == ExitStatus.HANDOFF
            assistant_msg: dict = {
                'role': 'assistant',
                'content': (collected_content or '') if is_handoff else (collected_content or None),
            }
            if collected_reasoning:
                assistant_msg['reasoning_content'] = collected_reasoning
            finalized_tbs = _finalize_thinking_blocks_acc(collected_thinking_blocks)
            if finalized_tbs:
                assistant_msg['thinking_blocks'] = finalized_tbs
            if collected_tool_calls:
                sorted_tcs = [collected_tool_calls[i] for i in sorted(collected_tool_calls)]
                if is_handoff:
                    assistant_msg['tool_calls'] = [
                        {
                            'type': 'function',
                            'function': {
                                'name': tc['function']['name'],
                                'arguments': parsed_tool_args[idx],
                            },
                        }
                        for idx, tc in enumerate(sorted_tcs)
                        if tc['function']['name'] == 'handoff'
                    ]
                else:
                    assistant_msg['tool_calls'] = [
                        {
                            'id': tc['id'],
                            'type': 'function',
                            'function': {'name': tc['function']['name'], 'arguments': tc['function']['arguments']},
                        }
                        for tc in sorted_tcs
                    ]
            history.append(assistant_msg)

            if exit_status is not None:
                break

            # ── Tool execution ────────────────────────────────────────────────

            sorted_tcs = [collected_tool_calls[i] for i in sorted(collected_tool_calls)]
            for idx, tc_data in enumerate(sorted_tcs):
                tool_name = tc_data['function']['name']
                tool_call_id = tc_data['id']
                tool_args = parsed_tool_args[idx]

                logger.info(f'Tool call: {tool_name}')
                logger.debug(f'Tool args: {json.dumps(tool_args, ensure_ascii=False)[:500]}')

                yield {'type': 'tool_start', 'name': tool_name, 'args': tool_args}

                t0 = time.time()
                try:
                    tool_response = await executor.call(tool_name, tool_args)
                    raw_output = '\n'.join(
                        part.get('text', '') for part in tool_response.output if part.get('type') == 'text'
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f'Tool execution error — {tool_name}: {type(e).__name__}: {e}')
                    raw_output = f'[Tool error: {e}]'
                    yield {'type': 'tool_error', 'name': tool_name, 'error': raw_output}
                    history.append({'role': 'tool', 'tool_call_id': tool_call_id, 'content': raw_output})
                    total_observation_time += time.time() - t0
                    abort, infra_error = _abort_status_for_tool_failure(
                        exc=e,
                        output_text=raw_output,
                        early_stop=early_stop,
                        total_observation_time=total_observation_time,
                        observation_time_timeout=observation_time_timeout,
                    )
                    if abort is not None:
                        exit_status = abort
                        break
                    continue

                total_observation_time += time.time() - t0

                # Interactive ask_user_question: pause and let the frontend handle the answer.
                if raw_output == _INTERACTIVE_SIGNAL:
                    payload = tool_response.context_updates.get('ask_user_question_payload', {})
                    logger.info(f'ask_user_question: interactive pause, tool_call_id={tool_call_id}')
                    yield {
                        'type': 'ask_user_question',
                        'tool_call_id': tool_call_id,
                        'payload': payload,
                    }
                    exit_status = ExitStatus.AWAITING_USER
                    break

                if raw_output:
                    truncated = maybe_truncate(raw_output, truncate_after=DEFAULT_TEXT_CONTENT_LIMIT)
                    if len(truncated) < len(raw_output):
                        logger.info(f'Observation truncated: {len(raw_output)} → {len(truncated)} chars')
                    output_text = truncated
                else:
                    output_text = NEXT_STEP_NO_OUTPUT_TEMPLATE

                # See note in non-streaming variant above — ``meta['error']``
                # signals in-domain failure; surface as tool_error. Dead sandbox
                # / runtime text in the payload is fatal.
                if tool_response.meta.get('error'):
                    yield {'type': 'tool_error', 'name': tool_name, 'error': raw_output}
                    abort, infra_error = _abort_status_for_tool_failure(
                        exc=None,
                        output_text=raw_output,
                        early_stop=False,
                        total_observation_time=total_observation_time,
                        observation_time_timeout=observation_time_timeout,
                    )
                    history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call_id,
                        'content': _tool_history_content(tool_response.output, output_text),
                    })
                    if abort is not None:
                        exit_status = abort
                        break
                    if observation_time_timeout and total_observation_time >= observation_time_timeout:
                        logger.info(
                            f'Observation timeout: {total_observation_time:.1f}s >= {observation_time_timeout}s'
                        )
                        exit_status = ExitStatus.TIMEOUT
                        break
                    continue
                yield {'type': 'tool_output', 'name': tool_name, 'output': raw_output}
                history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call_id,
                    'content': _tool_history_content(tool_response.output, output_text),
                })

                logger.debug(f'Tool output: {output_text[:200]}')

                if observation_time_timeout and total_observation_time >= observation_time_timeout:
                    logger.info(f'Observation timeout: {total_observation_time:.1f}s >= {observation_time_timeout}s')
                    exit_status = ExitStatus.TIMEOUT
                    break

            if exit_status is not None:
                break

            iteration += 1

    except ContextLengthExceededError as e:
        logger.error(f'Context length exceeded: {e}')
        exit_status = ExitStatus.TRUNCATION
    except Exception as e:
        logger.exception(f'Agent error: {e}')
        yield {'type': 'error', 'message': str(e)}
        exit_status = ExitStatus.ERROR
        if is_sandbox_infrastructure_error(e):
            infra_error = _as_infra_error(e)
        elif is_llm_http_error(e):
            llm_http_error = e

    if exit_status is None:
        exit_status = ExitStatus.MAX_ITERATIONS
        logger.warning(f'Max iterations reached ({max_iterations})')

    logger.info(f'one-agent done: status={exit_status.value} iterations={iteration + 1}')

    yield {
        'type': 'done',
        'exit_status': exit_status.value,
        'iterations': iteration + 1,
        'observation_time': total_observation_time,
        'inference_time': total_inference_time,
    }
    if infra_error is not None:
        raise infra_error
    if llm_http_error is not None:
        raise llm_http_error


# ============== Public Interfaces ==============


def _make_executor() -> ToolExecutor:
    """Build the standard tool executor for one-agent (built-in + skill tools)."""
    return ToolExecutor([
        # Built-in tools (default: v2 — input schema includes a ``description`` field).
        # The newer ``yb_one_agent/{bash,read,edit,write}`` (without ``description``)
        # are registered alongside but not loaded here by default.
        tool_registry.get('yb_one_agent/delegate'),
        tool_registry.get('yb_one_agent/bash_v2'),
        tool_registry.get('yb_one_agent/read_v3'),
        tool_registry.get('yb_one_agent/edit_v2'),
        tool_registry.get('yb_one_agent/write_v2'),
        tool_registry.get('yb_one_agent/web_fetch'),
        tool_registry.get('yb_one_agent/web_search'),
        tool_registry.get('yb_one_agent/picture_search'),
        tool_registry.get('yb_one_agent/video_search'),
        tool_registry.get('yb_one_agent/ask_user_question'),
        tool_registry.get('yb_one_agent/load_skill'),
        # Skill tools (mock or real, selected via __init__.py env vars)
        tool_registry.get('yb_one_agent/image_generation/generate'),
        tool_registry.get('yb_one_agent/task_schedule/create_task'),
        tool_registry.get('yb_one_agent/task_schedule/list_tasks'),
        tool_registry.get('yb_one_agent/task_schedule/update_task'),
        tool_registry.get('yb_one_agent/task_schedule/delete_task'),
        tool_registry.get('yb_one_agent/wiki/ingest_entry'),
    ])


async def run_agent_stream(
    history: list[dict],
    executor: ToolExecutor,
    client: Any,
    tools: list[dict],
    extra_headers: dict[str, str] | None = None,
) -> AsyncGenerator[dict, None]:
    """Streaming agentic loop for interactive use (e.g. web UI).

    Uses ``stream=True`` on LLM calls so ``token`` events are emitted as text arrives.
    Mutates ``history`` in-place.  The caller is responsible for ContextVar setup.
    """
    async for event in _agent_loop_stream(
        history=history,
        executor=executor,
        client=client,
        tools=tools,
        extra_headers=extra_headers,
    ):
        yield event


@agent_registry.register('yb_one_agent')
async def run_yb_one_agent(agent_input: OneAgentInput) -> OneAgentOutput:
    """Run one-agent non-streaming (inference / RL training).

    Uses ``stream=False`` on LLM calls — full response returned at once.
    Full trajectory is captured in history for training signal.
    """
    ctx = require_context()
    client = ctx.openai_client

    executor = _make_executor()

    # Resolve active tool set
    enabled_tools: frozenset[str] = frozenset(agent_input.tools) if agent_input.tools is not None else DEFAULT_TOOLS

    # Build system prompt driven by tool set / skill / terminal configuration
    system_content = build_system_prompt(
        tools=enabled_tools,
        skills=agent_input.skills,
        meta=ctx.task_config.get('meta'),
        terminal_type=agent_input.terminal_type,
    )
    history: list[dict] = [{'role': 'system', 'content': system_content}]

    # Insert prior-turn / resume history (multi-turn sessions, CI trajectories)
    # between the system prompt and the current user turn. The orchestrating
    # task (e.g. one_agent_infer) passes accumulated (user, assistant) pairs
    # for multi-turn sessions, or a partial trajectory for resume mode.
    if agent_input.history:
        history.extend(agent_input.history)

    if agent_input.resume_from_history:
        # `history` already ends with the message to continue from (e.g. a
        # tool result); do NOT append a new user turn. user_prompt is ignored.
        if not agent_input.history:
            logger.warning(
                'resume_from_history=True but no history supplied; '
                'falling back to user_prompt as the first user message.'
            )
            history.append({'role': 'user', 'content': agent_input.user_prompt})
    else:
        history.append({'role': 'user', 'content': agent_input.user_prompt})

    # Build tool schemas aligned to enabled tools + skills
    tools = build_tool_schemas(tools=enabled_tools, skills=agent_input.skills)

    done_event: dict = {}
    async for event in _agent_loop(
        history=history,
        executor=executor,
        client=client,
        tools=tools,
        max_iterations=agent_input.max_iterations,
        completion_retry_on_tool_error=agent_input.completion_retry_on_tool_error,
        llm_retry_on_error=agent_input.llm_retry_on_error,
        early_stop=agent_input.early_stop,
        observation_time_timeout=ctx.task_config.get('observation_time_timeout'),
        inference_time_timeout=ctx.task_config.get('inference_time_timeout'),
    ):
        if event['type'] == 'done':
            done_event = event

    exit_status = ExitStatus(done_event.get('exit_status', ExitStatus.ERROR))
    iterations = done_event.get('iterations', 0)
    total_observation_time = done_event.get('observation_time', 0.0)

    termination_reason = EXIT_STATUS_TO_TERMINATION_REASON.get(exit_status, 'UNKNOWN')
    span_id = get_current_span_id()
    if span_id:
        update_span(
            span_id,
            attributes={
                '_hytuner_termination_reason': termination_reason,
                '_hytuner_observation_time': total_observation_time,
            },
        )

    return OneAgentOutput(exit_status=exit_status, iterations=iterations, main_trajectory=history)
