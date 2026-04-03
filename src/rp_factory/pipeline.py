"""管线编排器 — 串联六大模块的完整数据合成流水线。"""

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
    """RP 数据合成工厂。"""

    def __init__(self, config: FactoryConfig) -> None:
        self.config = config

        self.generator_llm = LLMClient(config.llm.generator)
        self.teacher_llm = LLMClient(config.llm.teacher)
        self.mentor_llm = LLMClient(config.llm.mentor)
        self.target_llm = (
            LLMClient(config.llm.target) if config.llm.target.api_key else None
        )

        self.iceberg = IcebergEngine(config, self.generator_llm)
        self.perturbation = PerturbationEngine(config, self.generator_llm)
        self.pipeline_a = PipelineA(config, self.mentor_llm, self.teacher_llm)
        self.pipeline_b = PipelineB(config, self.teacher_llm, self.mentor_llm)
        self.quality = QualityFilter(config, self.target_llm, self.mentor_llm)
        self.serializer = Serializer(config)

        self.persona_gen = PersonaGenerator(self.generator_llm)
        self.seed_expander = SeedExpander(self.generator_llm)

    def _choose_pipeline(self) -> PipelineTag:
        ratio_a = self.config.rp_agent.mix_ratio.get("pipeline_a", 0.5)
        return PipelineTag.PIPELINE_A if random.random() < ratio_a else PipelineTag.PIPELINE_B

    async def _generate_rp_response(
        self, system_prompt: str, conversation: list[Message], pipeline: PipelineTag,
    ) -> Message | None:
        if pipeline == PipelineTag.PIPELINE_A:
            return await self.pipeline_a.generate_response(system_prompt, conversation)
        return await self.pipeline_b.generate_response(system_prompt, conversation)

    async def warmup_seeds(self, batch_size: int) -> None:
        """根据批次大小预判种子池是否够用，不够则 LLM 扩充。"""
        pools = [
            ("intents", self.iceberg.intents, 0.6),
            ("events", self.iceberg.events, 0.6),
        ]
        tasks = []
        for category, pool, ratio in pools:
            if batch_size > len(pool) * ratio:
                needed = max(5, batch_size - len(pool))
                logger.info("%s 池不足，将扩充 %d 条", category, needed)
                tasks.append(self._expand_pool(category, pool, needed))

        if len(self.iceberg.styles) < 10:
            tasks.append(self._expand_pool("styles", self.iceberg.styles, 8))

        if tasks:
            await asyncio.gather(*tasks)

    async def _expand_pool(self, category: str, pool: "Pool", count: int) -> None:  # noqa: F821
        new_items = await self.seed_expander.expand(category, count, pool.items)
        added = pool.extend(new_items)
        logger.info("%s 池扩充完成: +%d, 总计 %d", category, added, len(pool))

    async def generate_personas(self, count: int) -> list[str]:
        return await self.persona_gen.generate_batch(
            count=count, existing_personas=self.persona_gen.all_generated,
        )

    async def generate_conversation(
        self, system_prompt: str, num_turns: int | None = None,
    ) -> ConversationRecord | None:
        turns = num_turns or self.config.pipeline.conversation_turns
        pipeline = self._choose_pipeline()

        messages: list[Message] = [Message(role=Role.SYSTEM, content=system_prompt)]
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

            assistant_msg = await self._generate_rp_response(system_prompt, messages, pipeline)
            if assistant_msg is None:
                messages.pop()
                continue

            l1 = self.quality.level1_filter(assistant_msg)
            if l1.verdict != QualityVerdict.PASS:
                retry = await self._generate_rp_response(system_prompt, messages, pipeline)
                if retry and self.quality.level1_filter(retry).verdict == QualityVerdict.PASS:
                    assistant_msg = retry

            messages.append(assistant_msg)

        last_assistant = next((m for m in reversed(messages) if m.role == Role.ASSISTANT), None)
        if not last_assistant:
            return None

        l2 = await self.quality.level2_gain_filter(
            system_prompt, [m for m in messages if m.role == Role.USER], last_assistant,
        )
        if l2.verdict != QualityVerdict.PASS:
            return None

        return ConversationRecord(
            messages=messages,
            meta=ConversationMeta(
                system_prompt_source=system_prompt[:200],
                pipeline=pipeline,
                perturbations=perturbations,
                quality=l2,
            ),
        )

    async def run_batch(
        self,
        system_prompts: list[str] | None = None,
        count: int | None = None,
        num_turns: int | None = None,
        output_file: str = "output.jsonl",
    ) -> list[ConversationRecord]:
        prompts = list(system_prompts or [])
        target_count = count or len(prompts) or 10

        if len(prompts) < target_count:
            need = target_count - len(prompts)
            logger.info("需要 %d 条但只有 %d 个角色，自动生成 %d 个", target_count, len(prompts), need)
            prompts.extend(await self.generate_personas(need))

        await self.warmup_seeds(len(prompts))

        sem = asyncio.Semaphore(self.config.pipeline.max_concurrent_conversations)

        async def _one(sp: str) -> ConversationRecord | None:
            async with sem:
                return await self.generate_conversation(sp, num_turns)

        results = await asyncio.gather(*[_one(sp) for sp in prompts])
        records = [r for r in results if r is not None]

        logger.info("批次完成: %d/%d 通过质检", len(records), len(prompts))

        if records:
            self.serializer.write_jsonl(records, output_file)
            self.serializer.write_training_only(records, output_file.replace(".jsonl", "_train.jsonl"))

        return records
