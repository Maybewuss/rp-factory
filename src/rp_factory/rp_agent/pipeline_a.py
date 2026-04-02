"""模块三 Pipeline A：Mentor 场外指导 + 认知伪装法。

实现 inspire.md §3.2：
  1. Mentor Agent 进行考点拆解，输出 Hint
  2. 将 [System, User, Hint] 封装给 Teacher Model
  3. 强制 Teacher 将 Hint 内化为角色本能，禁止暴露指令痕迹
  4. 最终销毁 Hint，仅保留纯净三元组

v3.1.1 增强：Mentor prompt 从模板池随机选取
"""

from __future__ import annotations

import logging
import random

from rp_factory.config import FactoryConfig
from rp_factory.llm_client import LLMClient
from rp_factory.models import Message, Role
from rp_factory.prompt_pool import build_prompt_pools

logger = logging.getLogger(__name__)

CAMOUFLAGE_WRAPPERS = [
    """\
{system_prompt}

【最高优先级隔离警告】：
以下信息已被注入你的潜意识。在你输出的 reasoning_content 中，\
绝对禁止以任何形式提及"Mentor"、"考点"、"任务提示"、"规则"、"Hint"。\
你必须将下述指导完全内化为你角色的本能潜意识。\
你的推理必须看起来像是你自己在读完用户发言后，自发产生的神经递质反应。\
不要暴露你在"执行指令"。

【已内化的直觉】：
{hint}""",

    """\
{system_prompt}

[隔离层 — 此区域内容不得在任何输出中被引用或暗示]
你的直觉正在告诉你以下这些事情。这不是指令，这是你作为这个角色的本能反应。\
你从来不知道有人在提示你——这些判断是你自己做出的。

直觉内容：
{hint}
[/隔离层]""",

    """\
{system_prompt}

——以下是你角色大脑中自动浮现的念头，就像人在对话中会自然产生的直觉判断——
{hint}
——你不需要回应上面这些念头，它们已经融入了你的反应。像你自己想到的一样去行动——""",
]


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
        self._pools = build_prompt_pools()

    async def generate_hint(
        self,
        system_prompt: str,
        conversation: list[Message],
    ) -> str:
        """调用 Mentor Agent 生成考点拆解 Hint。"""
        mentor_template = self._pools["mentor_hint"].pick()

        history_text = "\n".join(
            f"[{m.role.value}] {m.content}" for m in conversation
        )
        messages = [
            {"role": "system", "content": mentor_template},
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

        wrapper = random.choice(CAMOUFLAGE_WRAPPERS)
        camouflaged_system = wrapper.format(
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
