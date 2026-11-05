"""Shared helpers for LLM HTTP failures (Taiji 503, CSIG 5xx, etc.).

``backend_taiji`` already raises after its own retries. Agent loops must
re-raise 5xx so infer tasks / the runner mark the run failed instead of success.
"""

from __future__ import annotations


# yb_one_agent uses ExitStatus.ERROR; local_one_agent / main_agent use
# MainExitStatus.INFERENCE_ERROR. Either means the LLM call did not complete.
FATAL_LLM_EXIT_STATUSES = frozenset({'error', 'inference_error'})


def _http_status_code(err: BaseException) -> int | None:
    """Best-effort status code from httpx / OpenAI-style HTTP errors."""
    response = getattr(err, 'response', None)
    status = getattr(response, 'status_code', None) if response is not None else None
    if status is None:
        status = getattr(err, 'status_code', None)
    return status if isinstance(status, int) else None


def is_llm_http_error(err: BaseException | str) -> bool:
    """True when the LLM backend returned 5xx (e.g. Taiji 503 Service Unavailable)."""
    if isinstance(err, str):
        return False
    status = _http_status_code(err)
    if status is not None and status >= 500:
        return True
    cause = err.__cause__ or err.__context__
    if cause is not None and cause is not err:
        return is_llm_http_error(cause)
    return False


def is_fatal_llm_exit(exit_status: object) -> bool:
    """True when the agent finished without raising but reported an LLM failure."""
    value = getattr(exit_status, 'value', exit_status)
    return value in FATAL_LLM_EXIT_STATUSES


def fatal_llm_exit_error(turn_task_id: str, exit_status: object) -> RuntimeError:
    value = getattr(exit_status, 'value', exit_status)
    return RuntimeError(f'[{turn_task_id}] Agent aborted: exit_status={value}')
