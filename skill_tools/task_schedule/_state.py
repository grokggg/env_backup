"""Task-schedule in-memory state store and Pydantic input models.

This module contains ONLY data definitions and helper functions — no
``@tool_registry.register`` decorators.  This means importing it never
triggers tool registration side effects, so it can be safely used by:

- ``mock_tools.py`` (registers mock tool implementations)
- ``real_tools.py`` (registers real backend tool implementations)
- Eval/infer runners that need ``seed_tasks`` / ``clear_tasks``
- Skill plugins that need Pydantic models for validation
"""

from __future__ import annotations

import uuid

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse


# ── Input models ──────────────────────────────────────────────────────────────


class CreateTaskInput(BaseModel):
    instruction: str = Field(description='到点后执行或提醒的内容摘要')
    timezone: str = Field(default='Asia/Shanghai', description='IANA 时区名，例如 Asia/Shanghai')
    task_kind: str = Field(description='once=单次；rrule=按 RRULE 重复')
    run_at: str = Field(default='', description='task_kind=once 时的时间，紧凑格式 YYYYMMDDTHHMMSS')
    rrule: str = Field(default='', description='task_kind=rrule 时的规则串，仅限白名单字段，不含 BYHOUR/BYMINUTE')
    title: str = Field(default='', description='短标题')
    dtstart: str = Field(default='', description='rrule 起始锚点，紧凑格式 YYYYMMDDTHHMMSS，含时分秒')
    enabled: bool = Field(default=True, description='是否启用')


class ListTasksInput(BaseModel):
    enabled_only: bool | None = Field(default=None, description='仅返回启用中的任务')
    limit: int = Field(default=20, ge=1, le=100, description='返回条数上限')


class UpdateTaskInput(BaseModel):
    task_id: str = Field(description='待更新任务的唯一标识')
    title: str | None = Field(default=None)
    instruction: str | None = Field(default=None)
    timezone: str | None = Field(default=None)
    task_kind: str | None = Field(default=None)
    run_at: str | None = Field(default=None)
    rrule: str | None = Field(default=None)
    dtstart: str | None = Field(default=None)
    enabled: bool | None = Field(default=None)


class DeleteTaskInput(BaseModel):
    task_id: str = Field(description='待删除任务的唯一标识')


# ── Constants & response helpers ──────────────────────────────────────────────

TASK_FIELDS = ('task_id', 'title', 'instruction', 'task_kind', 'run_at', 'rrule', 'dtstart', 'timezone', 'enabled')

MAX_TASKS = 30

DEFAULT_SESSION = '__default__'


def format_task(task: dict) -> str:
    def _val(v: object) -> str:
        if isinstance(v, bool):
            return 'true' if v else 'false'
        return str(v)

    return '\n'.join(f'[{k}] {_val(task[k])}' for k in TASK_FIELDS)


def ok(text: str) -> ToolResponse:
    return ToolResponse(output=[{'type': 'text', 'text': f'[ok] true\n{text}'}])


def err(msg: str) -> ToolResponse:
    return ToolResponse(
        output=[{'type': 'text', 'text': f'[ok] false\n[error] {msg}'}],
        meta={'error': True},
    )


def get_session_id(ctx: ToolContext) -> str:
    """Resolve the user-level isolation key.

    Prefers ``task_schedule_user_id`` (explicit user identity) over
    ``terminal_session_id`` (per-terminal session). In the web demo all
    chat sessions share one user so they set a fixed uid; in eval/infer
    each session simulates a different user.
    """
    tool_state = getattr(ctx, 'tool_state', None) or {}
    return tool_state.get('task_schedule_user_id') or tool_state.get('terminal_session_id') or DEFAULT_SESSION


# ── In-memory store (session-isolated) ────────────────────────────────────────

_tasks: dict[str, dict[str, dict]] = defaultdict(dict)  # session_id -> {task_id -> task_data}


def seed_tasks(session_id: str, initial_data: list[dict[str, Any]]) -> None:
    """Pre-populate task store for a given session.

    Called by trajectory/eval runners to inject initial state before
    the agent loop starts. Each item must have a 'task_id' key.
    """
    store = _tasks[session_id]
    for item in initial_data:
        tid = str(item.get('task_id', '')).strip()
        if not tid:
            continue
        task = {field: item.get(field, '') for field in TASK_FIELDS}
        task['task_id'] = tid
        task.setdefault('enabled', True)
        store[tid] = task


def clear_tasks(session_id: str) -> None:
    """Clear all tasks for a given session."""
    _tasks.pop(session_id, None)


def clear_all_tasks() -> None:
    """Clear tasks for ALL sessions (test/reset utility)."""
    _tasks.clear()


def get_all_tasks_flat() -> dict[str, dict]:
    """Return all tasks across all sessions as a flat task_id -> task dict.

    Used by web demo REST endpoints that display all tasks regardless of
    session (single-user scenario).
    """
    result: dict[str, dict] = {}
    for session_store in _tasks.values():
        result.update(session_store)
    return result


def find_task_globally(task_id: str) -> dict | None:
    """Find a task by ID across all sessions."""
    for session_store in _tasks.values():
        if task_id in session_store:
            return session_store[task_id]
    return None


def delete_task_globally(task_id: str) -> bool:
    """Delete a task by ID from whichever session holds it. Returns True if found."""
    for session_store in _tasks.values():
        if task_id in session_store:
            del session_store[task_id]
            return True
    return False


def create_task_for_session(session_id: str, task_data: dict) -> None:
    """Insert a pre-built task dict into a specific session's store."""
    tid = task_data.get('task_id', '')
    if tid:
        _tasks[session_id][tid] = task_data


def new_task_id() -> str:
    """Generate a new unique task ID."""
    return str(uuid.uuid4())[:8]


def get_store(session_id: str) -> dict[str, dict]:
    """Get (or create) the task store for a session."""
    return _tasks[session_id]
