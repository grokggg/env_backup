"""训练用适配层：把官方 deep_research 子 agent 包成可被 runner 独立驱动的 agent。

官方 ``run_deep_research``（同目录 ``agent.py``）是给主 agent 通过 Delegate 工具
内部 ``await`` 调用设计的：它没注册进 agent_registry，其 ``DeepResearchInput``/
``DeepResearchOutput`` 只继承 ``BaseModel``，过不了 runner 侧 ``validate_agent`` 对
``AgentRunInput``/``AgentRunOutput`` 的 issubclass 校验。因此 runner 无法按
``agent_name`` 找到并直接驱动它——沿用官方链路（runner→主 agent→Delegate→子 agent）
时被训模型是**主 agent**，子 agent 的 ReAct 循环藏在工具内部、不在被训 token 流里。

本文件加一层薄适配（合规 Input/Output + ``@agent_registry.register``），内部直接
``await`` 官方 ``run_deep_research``，让 runner 能把子 agent 当独立 agent 驱动——
rollout 轨迹 = 子 agent 的 plan→research→write history，训的就是子 agent。

**不改官方 ``agent.py``/ SP / 主 agent**，主 agent 训练与线上编排不受影响。

report_path 提取（与主 agent 完全一致）：
- ``report_path`` 直接调官方 ``_find_written_result(history)``——与主 agent 判定 result.md
  的**同一个函数**，两条链路的"报告在哪"口径永不漂移（官方若改名，import 会立即报错，
  fail-fast 优于内联副本静默漂移）。
- harness-v3(commit 1881a1624/846f9a9a0)下 ``_find_written_result`` 按**位置**配对
  write tool_call 与其响应（HyTuner client 给每个 tool_call 发空 id，id 配对会塌缩误判）；
  且 salvage 改为 ``_salvage_via_write``——模型没写出 result.md 时,以 write-only 2 轮
  mini-ReAct **强制模型自己**调一次 write，那次 write + 成功响应**会进 history**。因此
  salvage 写出的 result.md 与模型正常写出的一样能被 ``_find_written_result`` 找到、照常认
  （它本质仍是模型自己 write 的动作，落在被训轨迹里）。
- 只有连 salvage 都没成功写出 result.md（无 runtime / write 失败）→ ``None`` → reward task
  （``task_subagent._fetch_report_text``：``not path`` → 空 solution）→ **score=0**。
``report_path`` 供 reward task 从沙盒 download 报告打分（见
``nexus.extensions.tasks.quest_reward.task_subagent``）。
"""

import logging

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    WithMainTrajectory,
    WithUserPrompt,
    agent_registry,
)
from nexus.extensions.agents.yb_one_agent.sub_agents.deep_research.agent import (
    DeepResearchInput,
    _find_written_result,
    _salvage_instruction,
    run_deep_research,
)


logger = logging.getLogger(__name__)


# salvage 触发时,``_finalize`` 会往 history 追加一条 ``_salvage_instruction`` 的 user 消息
# (agent.py:_salvage_via_write)。官方不在 Output 里透出"是否走过 salvage",这里从 history
# 反推——与 report_path 靠 ``_find_written_result`` 扫 history 同一哲学(不改官方 agent.py)。
# marker 取官方指令的稳定前缀(而非硬编码整句):官方若改文案,import 的函数一起变,marker 不漂移。
_SALVAGE_MARKER = _salvage_instruction('x').split('，')[0]  # "你已用完本轮研究预算"


def _used_salvage(history: list) -> bool:
    """history 里出现 salvage 指令 → 说明模型正常预算内没写出 result.md,靠兜底 write 补救。"""
    return any(msg.get('role') == 'user' and _SALVAGE_MARKER in (msg.get('content') or '') for msg in history)


class DeepResearchTrainInput(AgentRunInput, WithUserPrompt):
    """子 agent 训练入口输入。user_prompt 由 WithUserPrompt 提供。"""

    # 与官方 DeepResearchInput 默认一致；对齐 Delegate 里的 env DR_MAX_TURNS（默认 20）。
    max_iterations: int = Field(default=30)


class DeepResearchTrainOutput(AgentRunOutput, WithMainTrajectory):
    """子 agent 训练出口输出。

    - status: 'success' / 'failure'（官方二值口径）。
    - result: 官方原样透传——v3 成功时是含路径的中文收尾语，失败时是错误串。
    - report_path: history 里真实成功写出的 result.md 路径（``_find_written_result``，正常写
      或 salvage 的模型 write 都算）；连 salvage 都没写出 → None（→ reward score=0）。reward 据此 download。
    - used_salvage: 是否走过兜底(预算内没写 result.md,靠 salvage "write now" 指令补救)。
      正常写出 / 连 salvage 都没触发 → False。区分"正常写出"与"靠兜底写出",供 metric 统计兜底占比。
    - main_trajectory: 子 agent 的完整 history，RL 取轨迹用。
    """

    status: str = 'failure'
    result: str = ''
    report_path: str | None = None
    used_salvage: bool = False
    iterations: int = 0


@agent_registry.register('deep_research_subagent')
async def run_deep_research_train(agent_input: DeepResearchTrainInput) -> DeepResearchTrainOutput:
    """跑官方 deep_research 子 agent（复用实现），把沙盒报告路径透出到 report_path。"""
    sub_out = await run_deep_research(
        DeepResearchInput(
            user_prompt=agent_input.user_prompt
            if isinstance(agent_input.user_prompt, str)
            else str(agent_input.user_prompt),
            max_iterations=agent_input.max_iterations,
        )
    )

    # 与主 agent 同一口径:调官方 _find_written_result,只认 history 里真实成功写出的 result.md。
    # 新版按位置配对(空 id 安全)、salvage 也是模型自己 write 进 history,故正常写/salvage 写都认;
    # 连 salvage 都没写出(无 runtime/write 失败)才 None → reward score=0。
    history = sub_out.history or []
    report_path = _find_written_result(history)
    used_salvage = _used_salvage(history)

    logger.info(
        '[deep_research_subagent] status=%s iterations=%s report_path=%r used_salvage=%s',
        sub_out.status,
        sub_out.iterations,
        report_path,
        used_salvage,
    )

    return DeepResearchTrainOutput(
        status=sub_out.status,
        result=sub_out.result,
        report_path=report_path,
        used_salvage=used_salvage,
        iterations=int(sub_out.iterations or 0),
        main_trajectory=history,
    )
