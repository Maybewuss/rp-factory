"""模块三 Pipeline B：纯净 Prefix + Best-of-N 采样法。"""

from __future__ import annotations

import logging
from typing import Any

from rp_factory import prompts
from rp_factory.config import FactoryConfig
from rp_factory.llm_client import LLMClient, parse_json_response
from rp_factory.models import Message, Role

logger = logging.getLogger(__name__)


class PipelineB:
    """Pipeline B: Best-of-N 纯净采样 + Verifier 收网。"""

    def __init__(
        self, config: FactoryConfig, teacher_llm: LLMClient, verifier_llm: LLMClient,
    ) -> None:
        self.config = config
        self.teacher = teacher_llm
        self.verifier = verifier_llm
        self._n = config.rp_agent.pipeline_b.n_samples
        self._temp = config.rp_agent.pipeline_b.temperature
        self._max_conc = config.rp_agent.pipeline_b.max_concurrent

    async def sample_n(self, system_prompt: str, conversation: list[Message]) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        for m in conversation:
            if m.role != Role.SYSTEM:
                openai_messages.append({"role": m.role.value, "content": m.content})

        candidates = await self.teacher.chat_parallel(
            openai_messages, n=self._n, max_concurrent=self._max_conc, temperature=self._temp,
        )
        logger.info("Pipeline B: 采集到 %d 个候选回复", len(candidates))
        return candidates

    async def verify_candidates(
        self,
        system_prompt: str,
        conversation: list[Message],
        candidates: list[dict[str, Any]],
    ) -> tuple[Message | None, list[dict[str, Any]]]:
        history_text = "\n".join(f"[{m.role.value}] {m.content}" for m in conversation)
        candidates_text = ""
        for i, c in enumerate(candidates):
            candidates_text += f"\n--- 候选 {i} ---\n"
            if c.get("reasoning_content"):
                candidates_text += f"[Reasoning]: {c['reasoning_content']}\n"
            candidates_text += f"[Content]: {c['content']}\n"

        messages = [
            {"role": "system", "content": prompts.VERIFIER},
            {"role": "user", "content": (
                f"角色设定：\n{system_prompt}\n\n"
                f"对话历史：\n{history_text}\n\n"
                f"候选回复：\n{candidates_text}"
            )},
        ]
        resp = await self.verifier.chat_single(messages)

        try:
            evals = parse_json_response(resp["content"])
            if isinstance(evals, dict):
                evals = [evals]
        except Exception:
            logger.warning("Verifier 返回解析失败")
            return None, []

        diamonds = [e for e in evals if e.get("is_diamond")]
        if not diamonds:
            logger.info("Pipeline B: 无钻石级候选，本轮废弃")
            return None, evals

        best = max(diamonds, key=lambda e: e.get("total", 0))
        idx = min(best.get("candidate_index", 0), len(candidates) - 1)
        chosen = candidates[idx]
        logger.info("Pipeline B: 选中候选 %d (score=%.1f)", idx, best.get("total", 0))
        return Message(
            role=Role.ASSISTANT,
            content=chosen["content"],
            reasoning_content=chosen.get("reasoning_content"),
        ), evals

    async def generate_response(self, system_prompt: str, conversation: list[Message]) -> Message | None:
        candidates = await self.sample_n(system_prompt, conversation)
        best, _ = await self.verify_candidates(system_prompt, conversation, candidates)
        return best
