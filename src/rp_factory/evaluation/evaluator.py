"""模块五：评估闭环 — 量化训练前后的能力变化。"""

from __future__ import annotations

import logging
import random

from rp_factory import prompts
from rp_factory.config import FactoryConfig
from rp_factory.llm_client import LLMClient, parse_json_response
from rp_factory.models import ConversationRecord, EvalScores, Message, Role

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        config: FactoryConfig,
        judge_llm: LLMClient,
        model_a_llm: LLMClient | None = None,
        model_b_llm: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.judge = judge_llm
        self.model_a = model_a_llm
        self.model_b = model_b_llm

    async def score_response(
        self, system_prompt: str, conversation: list[Message], assistant_response: Message,
    ) -> EvalScores:
        history_text = "\n".join(f"[{m.role.value}] {m.content}" for m in conversation)
        messages = [
            {"role": "system", "content": prompts.EVAL_JUDGE},
            {"role": "user", "content": (
                f"角色设定：\n{system_prompt}\n\n"
                f"对话历史：\n{history_text}\n\n"
                f"模型回复：\n{assistant_response.content}\n\n"
                f"模型推理：\n{assistant_response.reasoning_content or '(无)'}"
            )},
        ]
        resp = await self.judge.chat_single(messages)
        try:
            data = parse_json_response(resp["content"])
            return EvalScores(**{k: float(v) for k, v in data.items() if k in EvalScores.model_fields})
        except Exception:
            return EvalScores()

    async def compare_models(self, records: list[ConversationRecord]) -> dict[str, dict[str, float]]:
        if not self.model_a or not self.model_b:
            return {}

        sample_size = max(1, int(len(records) * self.config.evaluation.sample_ratio))
        sampled = random.sample(records, min(sample_size, len(records)))

        a_scores_list: list[EvalScores] = []
        b_scores_list: list[EvalScores] = []

        for rec in sampled:
            sys_msg = next((m for m in rec.messages if m.role == Role.SYSTEM), None)
            if not sys_msg:
                continue

            prefix = [
                {"role": m.role.value, "content": m.content}
                for m in rec.messages if m.role != Role.ASSISTANT
            ]

            resp_a = await self.model_a.chat_single(prefix)
            resp_b = await self.model_b.chat_single(prefix)

            conversation = [m for m in rec.messages if m.role != Role.ASSISTANT]
            a_scores_list.append(await self.score_response(
                sys_msg.content, conversation, Message(role=Role.ASSISTANT, content=resp_a["content"]),
            ))
            b_scores_list.append(await self.score_response(
                sys_msg.content, conversation, Message(role=Role.ASSISTANT, content=resp_b["content"]),
            ))

        if not a_scores_list:
            return {}

        dims = list(EvalScores.model_fields.keys())
        avg_a = {d: sum(getattr(s, d) for s in a_scores_list) / len(a_scores_list) for d in dims}
        avg_b = {d: sum(getattr(s, d) for s in b_scores_list) / len(b_scores_list) for d in dims}
        return {"model_a": avg_a, "model_b": avg_b, "delta": {d: avg_b[d] - avg_a[d] for d in dims}}
