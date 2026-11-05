"""Real backend implementation of task_schedule tools.

Calls the batch_tool_call API via ``_real_backend.call_tool_api``.
Registers the same tool IDs as ``mock_tools.py`` — only ONE of the two
modules may be loaded per process (via ``Registry.import_module``).

User isolation: ``ctx.tool_state['task_schedule_user_id']`` is used as
``context_in.user.user_id`` in the API payload.

The remote API returns fully-formatted text (including ``[ok]`` status),
so we pass it through to ToolResponse as-is — no local formatting.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


# ── Input models (lightweight, only for framework argument parsing) ───────────


class CreateTaskInput(BaseModel):
    instruction: str = Field(description='到点后执行或提醒的内容摘要')
    timezone: str = Field(default='Asia/Shanghai')
    task_kind: str = Field(description='once / rrule')
    run_at: str = Field(default='')
    rrule: str = Field(default='')
    title: str = Field(default='')
    dtstart: str = Field(default='')
    enabled: bool = Field(default=True)


class ListTasksInput(BaseModel):
    enabled_only: bool | None = Field(default=None)
    limit: int = Field(default=20, ge=1, le=100)


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


# ── Helpers ───────────────────────────────────────────────────────────────────

_DEFAULT_USER = 'default_user'


def _get_user_id(ctx: ToolContext) -> str:
    tool_state = getattr(ctx, 'tool_state', None) or {}
    return tool_state.get('task_schedule_user_id') or _DEFAULT_USER


def _respond(text: str) -> ToolResponse:
    return ToolResponse(output=[{'type': 'text', 'text': text}])


# ── Tool implementations ─────────────────────────────────────────────────────


@tool_registry.register(
    'yb_one_agent/task_schedule/create_task',
    meta={'name': 'task_schedule/create_task', 'description': '新建一条定时任务', 'tool_type': ToolType.LOCAL},
)
async def create_task(req: CreateTaskInput, ctx: ToolContext) -> ToolResponse:
    from ._real_backend import call_tool_api

    _, text = await call_tool_api('task_schedule/create_task', req.model_dump(), _get_user_id(ctx))
    return _respond(text)


@tool_registry.register(
    'yb_one_agent/task_schedule/list_tasks',
    meta={'name': 'task_schedule/list_tasks', 'description': '列出当前用户的任务', 'tool_type': ToolType.LOCAL},
)
async def list_tasks(req: ListTasksInput, ctx: ToolContext) -> ToolResponse:
    from ._real_backend import call_tool_api

    _, text = await call_tool_api('task_schedule/list_tasks', req.model_dump(), _get_user_id(ctx))
    return _respond(text)


@tool_registry.register(
    'yb_one_agent/task_schedule/update_task',
    meta={'name': 'task_schedule/update_task', 'description': '按 task_id 部分更新任务', 'tool_type': ToolType.LOCAL},
)
async def update_task(req: UpdateTaskInput, ctx: ToolContext) -> ToolResponse:
    from ._real_backend import call_tool_api

    _, text = await call_tool_api('task_schedule/update_task', req.model_dump(exclude_none=True), _get_user_id(ctx))
    return _respond(text)


@tool_registry.register(
    'yb_one_agent/task_schedule/delete_task',
    meta={
        'name': 'task_schedule/delete_task',
        'description': '按 task_id 永久删除一条任务',
        'tool_type': ToolType.LOCAL,
    },
)
async def delete_task(req: DeleteTaskInput, ctx: ToolContext) -> ToolResponse:
    from ._real_backend import call_tool_api

    _, text = await call_tool_api('task_schedule/delete_task', req.model_dump(), _get_user_id(ctx))
    return _respond(text)
