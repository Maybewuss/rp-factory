"""管线编排器 — 串联六大模块的完整数据合成流水线。

按照 inspire.md 的全链路架构：
  User Agent (冰山法) → 语境控制 (扰动注入) → RP Agent (A/B 管线)
  → 质检漏斗 → 评估闭环 → JSONL 落盘

v3.2 增强：
  - 自动角色生成：无 system prompt 时 LLM 动态生成多样化角色
  - 批次前种子预热：根据批次大小动态扩充种子池
  - 风格动态生成：运行时扩充用户风格标签
"""

from __future__ import annotations

import asyncio
import logging
import random

from rp_factory.config import FactoryConfig
from rp_factory.context_control.perturbation import PerturbationEngine
from rp_factory.generators import PersonaGenerator, SeedExpander
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

        self.persona_gen = PersonaGenerator(self.generator_llm)
        self.seed_expander = SeedExpander(self.generator_llm)

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

    async def warmup_seeds(self, batch_size: int) -> None:
        """批次前种子预热：根据批次大小判断是否需要扩充种子池。

        当 batch_size 超过当前种子池容量的 60% 时，自动调用 LLM 扩充。
        """
        intent_count = len(self.iceberg._intent_texts)
        event_count = len(self.iceberg._event_texts)
        style_count = len(self.iceberg._styles)
        task_count = len(self.perturbation._flat_tasks)

        logger.info(
            "种子池状态: intents=%d, events=%d, styles=%d, tasks=%d",
            intent_count, event_count, style_count, task_count,
        )

        expand_tasks = []

        if batch_size > intent_count * 0.6:
            needed = max(5, batch_size - intent_count)
            logger.info("intent 池不足，将扩充 %d 条", needed)
            expand_tasks.append(self._expand_intents(needed))

        if batch_size > event_count * 0.6:
            needed = max(8, batch_size - event_count)
            logger.info("event 池不足，将扩充 %d 条", needed)
            expand_tasks.append(self._expand_events(needed))

        if style_count < 10:
            logger.info("style 池较小，将扩充")
            expand_tasks.append(self._expand_styles(8))

        if batch_size > task_count * 0.8:
            needed = max(5, batch_size - task_count)
            logger.info("task 池不足，将扩充 %d 条", needed)
            expand_tasks.append(self._expand_tasks(needed))

        if expand_tasks:
            await asyncio.gather(*expand_tasks)
            logger.info(
                "种子预热完成: intents=%d, events=%d, styles=%d, tasks=%d",
                len(self.iceberg._intent_texts),
                len(self.iceberg._event_texts),
                len(self.iceberg._styles),
                len(self.perturbation._flat_tasks),
            )

    async def _expand_intents(self, count: int) -> None:
        new = await self.seed_expander.expand_intents(count, self.iceberg._intent_texts)
        self.iceberg.inject_intents(new)

    async def _expand_events(self, count: int) -> None:
        new = await self.seed_expander.expand_events(count, self.iceberg._event_texts)
        self.iceberg.inject_events(new)

    async def _expand_styles(self, count: int) -> None:
        new = await self.seed_expander.expand_styles(count, self.iceberg._styles)
        self.iceberg.inject_styles(new)

    async def _expand_tasks(self, count: int) -> None:
        existing_prompts = [t.get("prompt", "") for t in self.perturbation._flat_tasks]
        new = await self.seed_expander.expand_tasks(count, existing_prompts)
        for task in new:
            if isinstance(task, dict) and task.get("prompt"):
                self.perturbation._flat_tasks.append(task)

    async def generate_personas(self, count: int) -> list[str]:
        """LLM 动态生成多样化角色 System Prompt。"""
        logger.info("开始 LLM 生成 %d 个角色人设", count)
        return await self.persona_gen.generate_batch(
            count=count,
            existing_personas=self.persona_gen.all_generated,
        )

    async def generate_conversation(
        self,
        system_prompt: str,
        num_turns: int | None = None,
    ) -> ConversationRecord | None:
        """生成一条完整的多轮对话数据。"""
        turns = num_turns or self.config.pipeline.conversation_turns
        pipeline = self._choose_pipeline()
        logger.info("开始生成对话 | 管线=%s | 轮数=%d", pipeline.value, turns)

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
        ]
        perturbations: list[Perturbation] = []

        for turn_idx in range(turns):
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

            assistant_msg = await self._generate_rp_response(
                system_prompt, messages, pipeline,
            )

            if assistant_msg is None:
                logger.warning("轮次 %d: 生成失败，跳过", turn_idx)
                messages.pop()
                continue

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
        system_prompts: list[str] | None = None,
        count: int | None = None,
        num_turns: int | None = None,
        output_file: str = "output.jsonl",
    ) -> list[ConversationRecord]:
        """批量生成数据并落盘。

        Args:
            system_prompts: 提供的角色列表。如果为 None 或数量不足，自动 LLM 生成。
            count: 要生成的对话总数。如果 system_prompts 数量不足则自动补齐。
            num_turns: 每条对话的轮数。
            output_file: 输出文件名。
        """
        prompts = list(system_prompts or [])
        target_count = count or len(prompts) or 10

        if len(prompts) < target_count:
            need = target_count - len(prompts)
            logger.info("需要 %d 条对话但只有 %d 个角色，将自动生成 %d 个", target_count, len(prompts), need)
            new_personas = await self.generate_personas(need)
            prompts.extend(new_personas)

        await self.warmup_seeds(len(prompts))
        self.iceberg.diversity.seed_tracker.reset_batch()

        sem = asyncio.Semaphore(self.config.pipeline.max_concurrent_conversations)

        async def _one(sp: str) -> ConversationRecord | None:
            async with sem:
                return await self.generate_conversation(sp, num_turns)

        results = await asyncio.gather(*[_one(sp) for sp in prompts])
        records = [r for r in results if r is not None]

        logger.info(
            "批次完成: %d/%d 条数据通过质检", len(records), len(prompts),
        )

        if records:
            self.serializer.write_jsonl(records, output_file)
            self.serializer.write_training_only(records, output_file.replace(".jsonl", "_train.jsonl"))

        return records
