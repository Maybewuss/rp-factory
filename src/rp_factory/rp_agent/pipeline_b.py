"""模块三 Pipeline B：纯净 Prefix + Best-of-N 采样法。

实现 inspire.md §3.3：
  1. 隔离任何外部指导，直接向顶配 Teacher Model 发送纯粹 [System, User]
  2. 高温采样 N 次并发请求
  3. Mentor 作为阅卷法官 (Verifier) 对 N 个样本进行遍历审核
  4. 筛选出"钻石级数据"或整体废弃
"""

from __future__ import annotations

import logging
from typing import Any

from rp_factory.config import FactoryConfig
from rp_factory.llm_client import LLMClient, parse_json_response
from rp_factory.models import Message, Role

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM = """\
你是一名严苛的数据质量阅卷法官（Verifier）。你的任务是从 N 个候选回复中，\
筛选出在毫无提示的情况下最自然、最准确地完成了所有隐藏要求的回答。

评估维度（每项 0-10 分）：
1. persona_adherence: 角色一致性（口癖、性格、说话方式是否贴合设定）
2. deep_understanding: 深层需求理解（是否穿透表面事件识别到用户真实需求）
3. task_completion: 任务完成度（如果有工具性任务，是否准确完成且人设不崩）
4. reasoning_quality: 推理链质量（reasoning 是否自然、有分支探索、非后验辩护）
5. anti_ai_taste: 去 AI 味（是否避免了模板化客服语言）

请对每个候选回复输出 JSON：
{{"candidate_index": int, "scores": {{"persona_adherence": float, "deep_understanding": float, "task_completion": float, "reasoning_quality": float, "anti_ai_taste": float}}, "total": float, "is_diamond": bool, "reason": "..."}}

只有 total >= 40 且所有单项 >= 6 的候选才能标记为 is_diamond=true。
将所有候选评估放在一个 JSON 数组中返回。
"""


class PipelineB:
    """Pipeline B: Best-of-N 纯净采样 + Verifier 收网。"""

    def __init__(
        self,
        config: FactoryConfig,
        teacher_llm: LLMClient,
        verifier_llm: LLMClient,
    ) -> None:
        self.config = config
        self.teacher = teacher_llm
        self.verifier = verifier_llm
        self._n = config.rp_agent.pipeline_b.n_samples
        self._temp = config.rp_agent.pipeline_b.temperature
        self._max_conc = config.rp_agent.pipeline_b.max_concurrent

    async def sample_n(
        self,
        system_prompt: str,
        conversation: list[Message],
    ) -> list[dict[str, Any]]:
        """向 Teacher Model 发送 N 次高温并发请求，收集候选回复。"""
        openai_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        for m in conversation:
            if m.role != Role.SYSTEM:
                openai_messages.append({"role": m.role.value, "content": m.content})

        candidates = await self.teacher.chat_parallel(
            openai_messages,
            n=self._n,
            max_concurrent=self._max_conc,
            temperature=self._temp,
        )
        logger.info("Pipeline B: 采集到 %d 个候选回复", len(candidates))
        return candidates

    async def verify_candidates(
        self,
        system_prompt: str,
        conversation: list[Message],
        candidates: list[dict[str, Any]],
    ) -> tuple[Message | None, list[dict[str, Any]]]:
        """调用 Verifier 从候选中筛选钻石级数据。

        Returns:
            (最佳回复 Message 或 None, 所有评估结果列表)
        """
        history_text = "\n".join(
            f"[{m.role.value}] {m.content}" for m in conversation
        )
        candidates_text = ""
        for i, c in enumerate(candidates):
            candidates_text += f"\n--- 候选 {i} ---\n"
            if c.get("reasoning_content"):
                candidates_text += f"[Reasoning]: {c['reasoning_content']}\n"
            candidates_text += f"[Content]: {c['content']}\n"

        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"角色设定：\n{system_prompt}\n\n"
                    f"对话历史：\n{history_text}\n\n"
                    f"候选回复：\n{candidates_text}"
                ),
            },
        ]
        resp = await self.verifier.chat_single(messages)

        try:
            evals = parse_json_response(resp["content"])
            if isinstance(evals, dict):
                evals = [evals]
        except Exception:
            logger.warning("Verifier 返回解析失败，放弃本轮所有候选")
            return None, []

        diamonds = [e for e in evals if e.get("is_diamond")]
        if not diamonds:
            logger.info("Pipeline B: 无钻石级候选，本轮废弃")
            return None, evals

        best = max(diamonds, key=lambda e: e.get("total", 0))
        idx = best.get("candidate_index", 0)
        if idx >= len(candidates):
            idx = 0

        chosen = candidates[idx]
        logger.info(
            "Pipeline B: 选中候选 %d (score=%.1f)", idx, best.get("total", 0),
        )
        return Message(
            role=Role.ASSISTANT,
            content=chosen["content"],
            reasoning_content=chosen.get("reasoning_content"),
        ), evals

    async def generate_response(
        self,
        system_prompt: str,
        conversation: list[Message],
    ) -> Message | None:
        """完整 Pipeline B 流程：采样 → 验证 → 返回最优或 None。"""
        candidates = await self.sample_n(system_prompt, conversation)
        best, _ = await self.verify_candidates(
            system_prompt, conversation, candidates,
        )
        return best
