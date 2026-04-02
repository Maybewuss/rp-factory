"""模块五：评估闭环 — 量化训练前后的能力变化。

实现 inspire.md §5：
  从生产数据中抽样，让训练前后的模型各跑一遍相同的 [System, User]，
  用 Judge LLM 按维度打分对比，驱动数据工厂定向增产。
"""

from __future__ import annotations

import logging
import random

from rp_factory.config import FactoryConfig
from rp_factory.llm_client import LLMClient, parse_json_response
from rp_factory.models import ConversationRecord, EvalScores, Message, Role

logger = logging.getLogger(__name__)

EVAL_JUDGE_SYSTEM = """\
你是一名 RP 模型能力评估专家。请从以下五个维度对模型回复进行评分（每项 0-10）：

1. deep_need_recognition: 深层需求识别 — 是否穿透了表面事件，识别到了用户的深层心理需求
2. persona_consistency: 人设一致性 — 口癖、性格、世界观是否与 System Prompt 一致
3. flavored_task_completion: 风味任务完成 — 如果涉及工具性任务，是否准确完成且人设没崩
4. fact_correction: 事实纠偏 — 面对用户的记忆篡改或错误信息，是否恰当地纠正了
5. cognitive_translation: 认知转译 — 面对超出角色时代的知识，是否用角色的语言体系进行了恰当转译

如果某个维度在当前对话中不适用，给 5 分（中立值）。

请输出 JSON：
{{"deep_need_recognition": float, "persona_consistency": float, "flavored_task_completion": float, "fact_correction": float, "cognitive_translation": float}}
"""


class Evaluator:
    """评估闭环引擎。"""

    def __init__(
        self,
        config: FactoryConfig,
        judge_llm: LLMClient,
        model_a_llm: LLMClient | None = None,
        model_b_llm: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.judge = judge_llm
        self.model_a = model_a_llm  # 训练前
        self.model_b = model_b_llm  # 训练后

    async def score_response(
        self,
        system_prompt: str,
        conversation: list[Message],
        assistant_response: Message,
    ) -> EvalScores:
        """对单条回复进行五维度评分。"""
        history_text = "\n".join(
            f"[{m.role.value}] {m.content}" for m in conversation
        )
        messages = [
            {"role": "system", "content": EVAL_JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"角色设定：\n{system_prompt}\n\n"
                    f"对话历史：\n{history_text}\n\n"
                    f"模型回复：\n{assistant_response.content}\n\n"
                    f"模型推理：\n{assistant_response.reasoning_content or '(无)'}"
                ),
            },
        ]
        resp = await self.judge.chat_single(messages)
        try:
            data = parse_json_response(resp["content"])
            return EvalScores(**{k: float(v) for k, v in data.items() if k in EvalScores.model_fields})
        except Exception:
            logger.warning("评估 JSON 解析失败，返回默认分数")
            return EvalScores()

    async def compare_models(
        self,
        records: list[ConversationRecord],
    ) -> dict[str, dict[str, float]]:
        """A/B 模型对比：在同一组 [System, User] 上跑两个模型并打分。

        Returns:
            {"model_a": {dim: avg_score}, "model_b": {dim: avg_score}, "delta": {dim: diff}}
        """
        if not self.model_a or not self.model_b:
            logger.warning("未配置 A/B 模型，跳过对比")
            return {}

        sample_size = max(1, int(len(records) * self.config.evaluation.sample_ratio))
        sampled = random.sample(records, min(sample_size, len(records)))

        a_scores_list: list[EvalScores] = []
        b_scores_list: list[EvalScores] = []

        for rec in sampled:
            sys_msg = next(
                (m for m in rec.messages if m.role == Role.SYSTEM), None,
            )
            if not sys_msg:
                continue

            prefix = [
                {"role": m.role.value, "content": m.content}
                for m in rec.messages
                if m.role != Role.ASSISTANT
            ]

            resp_a = await self.model_a.chat_single(prefix)
            resp_b = await self.model_b.chat_single(prefix)

            msg_a = Message(role=Role.ASSISTANT, content=resp_a["content"])
            msg_b = Message(role=Role.ASSISTANT, content=resp_b["content"])

            conversation = [m for m in rec.messages if m.role != Role.ASSISTANT]
            score_a = await self.score_response(sys_msg.content, conversation, msg_a)
            score_b = await self.score_response(sys_msg.content, conversation, msg_b)

            a_scores_list.append(score_a)
            b_scores_list.append(score_b)

        if not a_scores_list:
            return {}

        dims = list(EvalScores.model_fields.keys())
        avg_a = {d: sum(getattr(s, d) for s in a_scores_list) / len(a_scores_list) for d in dims}
        avg_b = {d: sum(getattr(s, d) for s in b_scores_list) / len(b_scores_list) for d in dims}
        delta = {d: avg_b[d] - avg_a[d] for d in dims}

        return {"model_a": avg_a, "model_b": avg_b, "delta": delta}
