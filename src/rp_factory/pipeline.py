"""管线编排器 — 串联六大模块的完整数据合成流水线。

按照 inspire.md 的全链路架构：
  User Agent (冰山法) → 语境控制 (扰动注入) → RP Agent (A/B 管线)
  → 质检漏斗 → 评估闭环 → JSONL 落盘
"""

from __future__ import annotations

import asyncio
import logging
import random

from rp_factory.config import FactoryConfig
from rp_factory.context_control.perturbation import PerturbationEngine
from rp_factory.llm_client import LLMClient
from rp_factory.models import (
    ConversationMeta,
    ConversationRecord,
    Message,
    Perturbation,
    PipelineTag,
    QualityVerdict,
    Role,
)
from rp_factory.output.serializer import Serializer
from rp_factory.quality.filters import QualityFilter
from rp_factory.rp_agent.pipeline_a import PipelineA
from rp_factory.rp_agent.pipeline_b import PipelineB
from rp_factory.user_agent.iceberg import IcebergEngine

logger = logging.getLogger(__name__)


class DataFactory:
    """RP 数据合成工厂 — 完整流水线的统一入口。"""

    def __init__(self, config: FactoryConfig) -> None:
        self.config = config

        self.generator_llm = LLMClient(config.llm.generator)
        self.teacher_llm = LLMClient(config.llm.teacher)
        self.mentor_llm = LLMClient(config.llm.mentor)
        self.target_llm = LLMClient(config.llm.target)

        self.iceberg = IcebergEngine(config, self.generator_llm)
        self.perturbation = PerturbationEngine(config, self.generator_llm)
        self.pipeline_a = PipelineA(config, self.mentor_llm, self.teacher_llm)
        self.pipeline_b = PipelineB(config, self.teacher_llm, self.mentor_llm)
        self.quality = QualityFilter(config, self.target_llm, self.mentor_llm)
        self.serializer = Serializer(config)

    def _choose_pipeline(self) -> PipelineTag:
        """根据配置的混合比例随机选择管线。"""
        ratio_a = self.config.rp_agent.mix_ratio.get("pipeline_a", 0.5)
        if random.random() < ratio_a:
            return PipelineTag.PIPELINE_A
        return PipelineTag.PIPELINE_B

    async def _generate_rp_response(
        self,
        system_prompt: str,
        conversation: list[Message],
        pipeline: PipelineTag,
    ) -> Message | None:
        """根据选定的管线生成 RP Agent 回复。"""
        if pipeline == PipelineTag.PIPELINE_A:
            return await self.pipeline_a.generate_response(system_prompt, conversation)
        else:
            return await self.pipeline_b.generate_response(system_prompt, conversation)

    async def generate_conversation(
        self,
        system_prompt: str,
        num_turns: int | None = None,
    ) -> ConversationRecord | None:
        """生成一条完整的多轮对话数据。

        Args:
            system_prompt: RP 角色的 System Prompt
            num_turns: 对话轮数（一轮 = 一次 User + 一次 Assistant）

        Returns:
            通过质检的 ConversationRecord，或 None（如果质检全部失败）。
        """
        turns = num_turns or self.config.pipeline.conversation_turns
        pipeline = self._choose_pipeline()
        logger.info("开始生成对话 | 管线=%s | 轮数=%d", pipeline.value, turns)

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
        ]
        perturbations: list[Perturbation] = []

        for turn_idx in range(turns):
            # --- User 发言 ---
            perturbed_msg, perturbation = await self.perturbation.maybe_perturb(
                round_index=turn_idx,
                system_prompt=system_prompt,
                conversation_history=messages,
            )

            if perturbed_msg and perturbation:
                user_content = perturbed_msg
                perturbations.append(perturbation)
            else:
                iceberg = await self.iceberg.generate(
                    system_prompt,
                    conversation_history=messages if turn_idx > 0 else None,
                )
                user_content = iceberg.user_message

            messages.append(Message(role=Role.USER, content=user_content))

            # --- RP Agent 回复 ---
            assistant_msg = await self._generate_rp_response(
                system_prompt, messages, pipeline,
            )

            if assistant_msg is None:
                logger.warning("轮次 %d: 生成失败，跳过", turn_idx)
                messages.pop()
                continue

            # --- 逐轮质检（第一级） ---
            l1_result = self.quality.level1_filter(assistant_msg)
            if l1_result.verdict != QualityVerdict.PASS:
                logger.info(
                    "轮次 %d: 未通过 L1 质检 (%s), 尝试重新生成",
                    turn_idx, l1_result.verdict.value,
                )
                retry_msg = await self._generate_rp_response(
                    system_prompt, messages, pipeline,
                )
                if retry_msg:
                    l1_retry = self.quality.level1_filter(retry_msg)
                    if l1_retry.verdict == QualityVerdict.PASS:
                        assistant_msg = retry_msg
                    else:
                        logger.warning("轮次 %d: 重试仍未通过 L1, 使用原始结果", turn_idx)

            messages.append(assistant_msg)

        # --- 整体第二级质检（增益过滤） ---
        last_assistant = next(
            (m for m in reversed(messages) if m.role == Role.ASSISTANT), None,
        )
        if not last_assistant:
            logger.warning("对话无有效 Assistant 回复，废弃")
            return None

        l2_result = await self.quality.level2_gain_filter(
            system_prompt,
            [m for m in messages if m.role == Role.USER],
            last_assistant,
        )

        if l2_result.verdict != QualityVerdict.PASS:
            logger.info("未通过 L2 增益过滤: %s", l2_result.details)
            return None

        meta = ConversationMeta(
            system_prompt_source=system_prompt[:200],
            pipeline=pipeline,
            perturbations=perturbations,
            quality=l2_result,
        )

        return ConversationRecord(messages=messages, meta=meta)

    async def run_batch(
        self,
        system_prompts: list[str],
        num_turns: int | None = None,
        output_file: str = "output.jsonl",
    ) -> list[ConversationRecord]:
        """批量生成数据并落盘。"""
        sem = asyncio.Semaphore(self.config.pipeline.max_concurrent_conversations)

        async def _one(sp: str) -> ConversationRecord | None:
            async with sem:
                return await self.generate_conversation(sp, num_turns)

        results = await asyncio.gather(*[_one(sp) for sp in system_prompts])
        records = [r for r in results if r is not None]

        logger.info(
            "批次完成: %d/%d 条数据通过质检", len(records), len(system_prompts),
        )

        if records:
            self.serializer.write_jsonl(records, output_file)
            self.serializer.write_training_only(records, output_file.replace(".jsonl", "_train.jsonl"))

        return records
