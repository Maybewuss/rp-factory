"""模块三 Pipeline A：Mentor 场外指导 + 认知伪装法。"""

from __future__ import annotations

import logging

from rp_factory import prompts
from rp_factory.config import FactoryConfig
from rp_factory.llm_client import LLMClient
from rp_factory.models import Message, Role

logger = logging.getLogger(__name__)


class PipelineA:
    """Pipeline A: Mentor 指导 + 认知伪装法生成管线。"""

    def __init__(
        self, config: FactoryConfig, mentor_llm: LLMClient, teacher_llm: LLMClient,
    ) -> None:
        self.config = config
        self.mentor = mentor_llm
        self.teacher = teacher_llm

    async def generate_hint(self, system_prompt: str, conversation: list[Message]) -> str:
        history_text = "\n".join(f"[{m.role.value}] {m.content}" for m in conversation)
        messages = [
            {"role": "system", "content": prompts.MENTOR_HINT},
            {"role": "user", "content": f"角色设定：\n{system_prompt}\n\n对话历史：\n{history_text}"},
        ]
        resp = await self.mentor.chat_single(messages)
        return resp["content"]

    async def generate_response(self, system_prompt: str, conversation: list[Message]) -> Message:
        hint = await self.generate_hint(system_prompt, conversation)
        logger.debug("Pipeline A Hint: %s", hint[:120])

        camouflaged_system = prompts.CAMOUFLAGE_WRAPPER.format(
            system_prompt=system_prompt, hint=hint,
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
