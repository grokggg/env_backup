"""Cross-agent contract: main agent ↔ deep_research sub-agent.

Single source of truth for the contract between yb_one_agent (main agent)
and the deep_research sub-agent. Anything the main agent needs to know about
the sub-agent — its system prompt, the Delegate tool input shape, the
``<delegate_result>`` envelope shape — lives here so SFT data synthesis /
training scripts can import a single module.

Implementation files (loop, runtime) are NOT contract — they stay private to
the sub-agent.

Sources:
  - system_prompt: ``build_system_prompt`` (renders ``system_prompt.j2`` with
    the same ``meta_lines`` Layer-3 the main agent uses) — in ``agent.py``.
  - tools: Delegate tool input + ``DelegateResult`` envelope shape (see yb_one_agent/tools.py).
"""

from nexus.extensions.agents.yb_one_agent.sub_agents.deep_research.agent import (
    build_system_prompt,
)
from nexus.extensions.agents.yb_one_agent.tools import (
    DELEGATE_TOOL_SCHEMA,
    DelegateInput,
    DelegateResult,
)


__all__ = [
    'DELEGATE_TOOL_SCHEMA',
    'DelegateInput',
    'DelegateResult',
    'build_system_prompt',
]
