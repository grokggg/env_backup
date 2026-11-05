"""Real backend for task_schedule — calls the batch_tool_call HTTP API.

Same protocol as tools_modify.py's _call_task_schedule_tool but with user_id
as a direct parameter (for seed/clear operations and mock_tools real-mode
dispatch where ToolContext.tool_configs is not the source of user_id).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

from typing import Any

import requests


_TASK_SCHEDULE_TOOL_PROXY_URL = os.environ.get(
    'TASK_SCHEDULE_API_URL',
    'http://11.141.126.149:8003/batch_tool_call',
)
_TASK_SCHEDULE_TOOL_PROXY_TIMEOUT = float(os.environ.get('TASK_SCHEDULE_API_TIMEOUT', '30'))
_TASK_SCHEDULE_TOOL_VERSION = 'v1.0'

logger = logging.getLogger(__name__)


async def call_tool_api(tool_name: str, arguments: dict[str, Any], user_id: str) -> tuple[bool, str]:
    """Call the real batch_tool_call API.

    Returns (ok, text) where text is the for_model content from tool_proxy.
    """
    payload = {
        'calls': [
            {
                'call_id': f'nexus-{tool_name.replace("/", "-")}',
                'name': tool_name,
                'version': _TASK_SCHEDULE_TOOL_VERSION,
                'arguments': arguments,
            }
        ],
        'options': {},
        'context_in': {
            'user': {'user_id': user_id},
            'session': {'round': 1, 'history': [], 'raw_query': ''},
            'caller': {'call_source': 'training'},
        },
    }
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'nexus-yb-one-agent',
    }

    try:
        response = await asyncio.to_thread(
            requests.post,
            _TASK_SCHEDULE_TOOL_PROXY_URL,
            json=payload,
            headers=headers,
            timeout=_TASK_SCHEDULE_TOOL_PROXY_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else '?'
        logger.error('[task_schedule_real] HTTP error tool_name=%s status=%s', tool_name, status_code)
        return False, f'tool_proxy HTTP {status_code}: {exc}'
    except requests.exceptions.RequestException as exc:
        logger.error('[task_schedule_real] request failed tool_name=%s exc=%s', tool_name, exc)
        return False, f'tool_proxy request failed: {exc}'

    try:
        body = response.json()
    except ValueError:
        logger.error('[task_schedule_real] non-JSON response tool_name=%s', tool_name)
        return False, f'tool_proxy non-JSON response: {response.text[:200]}'

    global_status = body.get('status') or {}
    global_code = global_status.get('code', 0)
    if global_code != 0:
        msg = global_status.get('msg', '')
        return False, f'batch error code={global_code} msg={msg}'

    results = body.get('results') or []
    if not results:
        return False, 'tool_proxy returned empty results'

    result = results[0]
    call_status = result.get('status') or {}
    call_code = call_status.get('code', 0)

    chunks = result.get('for_model') or []
    texts = [c.get('content', '') for c in chunks if c.get('type') == 'text' and c.get('content')]
    text = '\n'.join(texts) if texts else '(无内容)'

    if call_code != 0:
        return False, text

    return True, text


def _extract_task_ids(list_text: str) -> list[str]:
    """Extract task_id values from list_tasks response text."""
    return re.findall(r'\[task_id\]\s+(\S+)', list_text)


async def clear_tasks_real(user_id: str) -> None:
    """List all tasks for the user, then delete them one by one."""
    ok, text = await call_tool_api('task_schedule/list_tasks', {'enabled_only': False, 'limit': 100}, user_id)
    if not ok:
        logger.warning('[task_schedule_real] clear: list_tasks failed for uid=%s: %s', user_id, text)
        return
    task_ids = _extract_task_ids(text)
    for tid in task_ids:
        await call_tool_api('task_schedule/delete_task', {'task_id': tid}, user_id)
    if task_ids:
        logger.info('[task_schedule_real] cleared %d tasks for uid=%s', len(task_ids), user_id)


async def seed_tasks_real(user_id: str, tasks: list[dict[str, Any]]) -> None:
    """Clear existing tasks, then create each task from the provided list."""
    await clear_tasks_real(user_id)
    for task in tasks:
        args = {
            'instruction': task.get('instruction', ''),
            'timezone': task.get('timezone', 'Asia/Shanghai'),
            'task_kind': task.get('task_kind', 'once'),
            'title': task.get('title', ''),
            'run_at': task.get('run_at', ''),
            'rrule': task.get('rrule', ''),
            'dtstart': task.get('dtstart', ''),
            'enabled': task.get('enabled', True),
        }
        ok, text = await call_tool_api('task_schedule/create_task', args, user_id)
        if not ok:
            logger.warning('[task_schedule_real] seed: create_task failed uid=%s: %s', user_id, text)
    logger.info('[task_schedule_real] seeded %d tasks for uid=%s', len(tasks), user_id)
