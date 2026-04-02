"""模块三 Pipeline A：Mentor 场外指导 + 认知伪装法。

实现 inspire.md §3.2：
  1. Mentor Agent 进行考点拆解，输出 Hint
  2. 将 [System, User, Hint] 封装给 Teacher Model
  3. 强制 Teacher 将 Hint 内化为角色本能，禁止暴露指令痕迹
  4. 最终销毁 Hint，仅保留纯净三元组
"""

from __future__ import annotations

import logging

from rp_factory.config import FactoryConfig
from rp_factory.llm_client import LLMClient
from rp_factory.models import Message, Role

logger = logging.getLogger(__name__)

MENTOR_SYSTEM = """\
你是一名 RP 数据质量导师（Mentor Agent）。你的任务是分析当前对话场景，\
为 RP Agent 提供一份冷酷、精准的"考点拆解"。

分析以下信息：
1. 角色 System Prompt（角色定义）
2. 用户的最新发言
3. 对话历史

请输出一份简洁的行动指南（Hint），包含：
- 用户表面在说什么 vs 实际需要什么
- RP Agent 应该触发的行为（纠偏/共情/执行任务等）
- 必须避免的雷区（AI味措辞、破坏人设等）
- 如果涉及工具性任务：任务必须准确完成，人设只包裹在任务外部文本中

直接输出 Hint 文本，不要废话。
"""

TEACHER_WITH_HINT_SYSTEM = """\
{system_prompt}

【最高优先级隔离警告】：
以下信息已被注入你的潜意识。在你输出的 reasoning_content 中，\
绝对禁止以任何形式提及"Mentor"、"考点"、"任务提示"、"规则"、"Hint"。\
你必须将下述指导完全内化为你角色的本能潜意识。\
你的推理必须看起来像是你自己在读完用户发言后，自发产生的神经递质反应。\
不要暴露你在"执行指令"。

【已内化的直觉】：
{hint}
"""


class PipelineA:
    """Pipeline A: Mentor 指导 + 认知伪装法生成管线。"""

    def __init__(
        self,
        config: FactoryConfig,
        mentor_llm: LLMClient,
        teacher_llm: LLMClient,
    ) -> None:
        self.config = config
        self.mentor = mentor_llm
        self.teacher = teacher_llm

    async def generate_hint(
        self,
        system_prompt: str,
        conversation: list[Message],
    ) -> str:
        """调用 Mentor Agent 生成考点拆解 Hint。"""
        history_text = "\n".join(
            f"[{m.role.value}] {m.content}" for m in conversation
        )
        messages = [
            {"role": "system", "content": MENTOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"角色设定：\n{system_prompt}\n\n"
                    f"对话历史：\n{history_text}"
                ),
            },
        ]
        resp = await self.mentor.chat_single(messages)
        return resp["content"]

    async def generate_response(
        self,
        system_prompt: str,
        conversation: list[Message],
    ) -> Message:
        """完整 Pipeline A 流程：Mentor → Hint 注入 → Teacher 生成。"""
        hint = await self.generate_hint(system_prompt, conversation)
        logger.debug("Pipeline A Hint: %s", hint[:120])

        camouflaged_system = TEACHER_WITH_HINT_SYSTEM.format(
            system_prompt=system_prompt,
            hint=hint,
        )

        openai_messages: list[dict[str, str]] = [
            {"role": "system", "content": camouflaged_system},
        ]
        for m in conversation:
            if m.role != Role.SYSTEM:
                openai_messages.append({"role": m.role.value, "content": m.content})

        resp = await self.teacher.chat_single(openai_messages)

        return Message(
            role=Role.ASSISTANT,
            content=resp["content"],
            reasoning_content=resp.get("reasoning_content"),
        )
