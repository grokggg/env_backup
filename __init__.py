"""yb_one_agent implementation for nexus framework.

Usage:
    from nexus.core import Registry
    Registry.import_module('nexus.extensions.agents.yb_one_agent')

    from nexus.core import agent_registry
    agent_fn = agent_registry.get('yb_one_agent')

Example:
    from nexus.extensions.agents.yb_one_agent import OneAgentInput, OneAgentOutput

    input = OneAgentInput(
        user_prompt='Fix the bug in function foo...',
        max_iterations=30,
    )

    # Run with ExecutionContext
    output = await run_yb_one_agent(input)
"""

import os

from nexus.core import Registry
from nexus.extensions.agents.yb_one_agent.agent import (
    ExitStatus,
    OneAgentInput,
    OneAgentOutput,
    SandboxInfrastructureError,
    is_llm_http_error,
    is_sandbox_infrastructure_error,
    run_agent_stream,
    run_yb_one_agent,
)
from nexus.extensions.agents.yb_one_agent.tool import (
    bash_tool,
    edit_tool,
    read_tool,
    write_tool,
)
from nexus.extensions.agents.yb_one_agent.tools import (
    bash_tool_v2,
    edit_tool_v2,
    read_tool_v2,
    read_tool_v3,
    web_fetch_tool,
    write_tool_v2,
)


# ── Skill tool backend selection ─────────────────────────────────────────────

_IMAGE_GEN_USE_REAL = os.environ.get('IMAGE_GEN_USE_REAL', '').lower() in ('1', 'true', 'yes')
_TASK_SCHEDULE_USE_REAL = os.environ.get('TASK_SCHEDULE_USE_REAL', '').lower() in ('1', 'true', 'yes')
_TASK_SCHEDULE_DEMO_USER_ID = os.environ.get('TASK_SCHEDULE_DEMO_USER_ID', 'yb_demo_user')

# ── Module registration ───────────────────────────────────────────────────────

Registry.import_module('nexus.extensions.agents.yb_one_agent.tools')
Registry.import_module('nexus.extensions.agents.yb_one_agent.tool')
Registry.import_module('nexus.extensions.agents.yb_one_agent.skill_tools.wiki.tools')

if _IMAGE_GEN_USE_REAL:
    Registry.import_module('nexus.extensions.agents.yb_one_agent.skill_tools.image_generation.real_tools')
else:
    Registry.import_module('nexus.extensions.agents.yb_one_agent.skill_tools.image_generation.mock_tools')

if _TASK_SCHEDULE_USE_REAL:
    Registry.import_module('nexus.extensions.agents.yb_one_agent.skill_tools.task_schedule.real_tools')
else:
    Registry.import_module('nexus.extensions.agents.yb_one_agent.skill_tools.task_schedule.mock_tools')


__all__ = [
    # Agent
    'run_yb_one_agent',
    'run_agent_stream',
    'OneAgentInput',
    'OneAgentOutput',
    'ExitStatus',
    'SandboxInfrastructureError',
    'is_sandbox_infrastructure_error',
    'is_llm_http_error',
    # Tools (non-v2: input schema without ``description`` field)
    'bash_tool',
    'read_tool',
    'edit_tool',
    'write_tool',
    'web_fetch_tool',
    # Tools (v2: input schema includes ``description`` field — used downstream)
    'bash_tool_v2',
    'read_tool_v2',
    'read_tool_v3',
    'edit_tool_v2',
    'write_tool_v2',
]
