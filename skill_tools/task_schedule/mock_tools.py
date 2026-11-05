"""Mock in-memory implementation of task_schedule tools.

Stores tasks in a session-keyed dict so concurrent sessions are isolated.
All four tools are registered under yb_one_agent/* so the executor can route them.

Session isolation is keyed by ``ctx.tool_state['terminal_session_id']``,
which is set by both web demo and eval/infer harnesses.
"""

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry

from ._state import (
    DEFAULT_SESSION,
    MAX_TASKS,
    TASK_FIELDS,
    CreateTaskInput,
    DeleteTaskInput,
    ListTasksInput,
    UpdateTaskInput,
    err,
    format_task,
    get_session_id,
    get_store,
    new_task_id,
    ok,
)


# Re-export for backward compatibility
_TASK_FIELDS = TASK_FIELDS
_MAX_TASKS = MAX_TASKS
_DEFAULT_SESSION = DEFAULT_SESSION
_format_task = format_task
_ok = ok
_err = err
_get_session_id = get_session_id


# ── Tool implementations ─────────────────────────────────────────────────────


@tool_registry.register(
    'yb_one_agent/task_schedule/create_task',
    meta={'name': 'task_schedule/create_task', 'description': '新建一条定时任务', 'tool_type': ToolType.LOCAL},
)
async def create_task(req: CreateTaskInput, ctx: ToolContext) -> ToolResponse:
    session_id = get_session_id(ctx)
    store = get_store(session_id)

    if len(store) >= MAX_TASKS:
        return err(f'任务数量已达上限（{MAX_TASKS}条），请删除不需要的任务后重试')
    if req.task_kind not in ('once', 'rrule'):
        return err(f'无效的 task_kind: {req.task_kind}，必须为 once 或 rrule')
    if req.task_kind == 'once' and not req.run_at:
        return err('task_kind=once 时 run_at 不能为空')
    if req.task_kind == 'rrule' and not req.rrule:
        return err('task_kind=rrule 时 rrule 不能为空')

    task_id = new_task_id()
    task = {
        'task_id': task_id,
        'title': req.title or req.instruction[:20],
        'instruction': req.instruction,
        'task_kind': req.task_kind,
        'run_at': req.run_at,
        'rrule': req.rrule,
        'dtstart': req.dtstart,
        'timezone': req.timezone,
        'enabled': req.enabled,
    }
    store[task_id] = task
    return ok(format_task(task))


@tool_registry.register(
    'yb_one_agent/task_schedule/list_tasks',
    meta={'name': 'task_schedule/list_tasks', 'description': '列出当前会话下已创建的任务', 'tool_type': ToolType.LOCAL},
)
async def list_tasks(req: ListTasksInput, ctx: ToolContext) -> ToolResponse:
    session_id = get_session_id(ctx)
    store = get_store(session_id)

    tasks = list(store.values())
    if req.enabled_only is True:
        tasks = [t for t in tasks if t.get('enabled')]
    tasks = tasks[: req.limit]

    parts = [f'[total] {len(tasks)}']
    for i, task in enumerate(tasks, 1):
        parts.append(f'\n[task {i}]\n{format_task(task)}')
    return ok('\n'.join(parts))


@tool_registry.register(
    'yb_one_agent/task_schedule/update_task',
    meta={'name': 'task_schedule/update_task', 'description': '按 task_id 部分更新任务', 'tool_type': ToolType.LOCAL},
)
async def update_task(req: UpdateTaskInput, ctx: ToolContext) -> ToolResponse:
    session_id = get_session_id(ctx)
    store = get_store(session_id)

    task = store.get(req.task_id)
    if task is None:
        return err(f'task_id {req.task_id!r} 不存在')

    updatable = ('title', 'instruction', 'timezone', 'task_kind', 'run_at', 'rrule', 'dtstart', 'enabled')
    for field in updatable:
        val = getattr(req, field)
        if val is not None:
            task[field] = val

    return ok(format_task(task))


@tool_registry.register(
    'yb_one_agent/task_schedule/delete_task',
    meta={
        'name': 'task_schedule/delete_task',
        'description': '按 task_id 永久删除一条任务',
        'tool_type': ToolType.LOCAL,
    },
)
async def delete_task(req: DeleteTaskInput, ctx: ToolContext) -> ToolResponse:
    session_id = get_session_id(ctx)
    store = get_store(session_id)

    if req.task_id not in store:
        return err(f'task_id {req.task_id!r} 不存在')
    task = store.pop(req.task_id)
    return ok(format_task(task))
