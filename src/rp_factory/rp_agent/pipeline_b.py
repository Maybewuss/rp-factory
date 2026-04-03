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
    """Pipeline B: RP 元指令 + Best-of-N 纯净采样 + Verifier 收网。"""

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
        wrapped_system = prompts.TEACHER_PURE_WRAPPER.format(
            system_prompt=system_prompt,
            meta_instruction=prompts.RP_META_INSTRUCTION,
        )

        openai_messages: list[dict[str, str]] = [
            {"role": "system", "content": wrapped_system},
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
    ) -> tuple[Message, list[dict[str, Any]]]:
        """从候选中选出最佳。如果有钻石级取最佳钻石，否则取得分最高的。"""
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
            logger.warning("Verifier 解析失败，取第一个候选")
            return self._pick_candidate(candidates, 0), []

        diamonds = [e for e in evals if e.get("is_diamond")]
        if diamonds:
            best = max(diamonds, key=lambda e: e.get("total", 0))
        elif evals:
            best = max(evals, key=lambda e: e.get("total", 0))
        else:
            return self._pick_candidate(candidates, 0), evals

        idx = min(best.get("candidate_index", 0), len(candidates) - 1)
        quality = "钻石" if best.get("is_diamond") else "最佳"
        logger.info("Pipeline B: 选中%s候选 %d (score=%.1f)", quality, idx, best.get("total", 0))
        return self._pick_candidate(candidates, idx), evals

    @staticmethod
    def _pick_candidate(candidates: list[dict[str, Any]], idx: int) -> Message:
        chosen = candidates[idx]
        return Message(
            role=Role.ASSISTANT,
            content=chosen["content"],
            reasoning_content=chosen.get("reasoning_content"),
        )

    async def generate_response(self, system_prompt: str, conversation: list[Message]) -> Message:
        """Pipeline B 不再返回 None — 总是选出一个最佳候选。"""
        candidates = await self.sample_n(system_prompt, conversation)
        best, _ = await self.verify_candidates(system_prompt, conversation, candidates)
        return best
