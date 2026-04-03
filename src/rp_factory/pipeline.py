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

MAX_L1_RETRIES = 3


class DataFactory:
    """RP 数据合成工厂。"""

    def __init__(self, config: FactoryConfig) -> None:
        self.config = config

        self.generator_llm = LLMClient(config.llm.generator)
        self.teacher_llm = LLMClient(config.llm.teacher)
        self.mentor_llm = LLMClient(config.llm.mentor)

        self.iceberg = IcebergEngine(config, self.generator_llm)
        self.perturbation = PerturbationEngine(config, self.generator_llm)
        self.pipeline_a = PipelineA(config, self.mentor_llm, self.teacher_llm)
        self.pipeline_b = PipelineB(config, self.teacher_llm, self.mentor_llm)
        self.quality = QualityFilter(config, self.mentor_llm)
        self.serializer = Serializer(config)

        self.persona_gen = PersonaGenerator(self.generator_llm)
        self.seed_expander = SeedExpander(self.generator_llm)

    def _choose_pipeline(self) -> PipelineTag:
        ratio_a = self.config.rp_agent.mix_ratio.get("pipeline_a", 0.5)
        return PipelineTag.PIPELINE_A if random.random() < ratio_a else PipelineTag.PIPELINE_B

    async def _generate_rp_response(
        self, system_prompt: str, conversation: list[Message], pipeline: PipelineTag,
    ) -> Message:
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
                tasks.append(self._expand_pool(category, pool, needed))

        if len(self.iceberg.styles) < 10:
            tasks.append(self._expand_pool("styles", self.iceberg.styles, 8))

        if tasks:
            await asyncio.gather(*tasks)

    async def _expand_pool(self, category: str, pool: "Pool", count: int) -> None:  # noqa: F821
        new_items = await self.seed_expander.expand(category, count, pool.items)
        added = pool.extend(new_items)
        logger.info("%s 池扩充: +%d → %d", category, added, len(pool))

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
        deep_intent: str | None = None

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
                    deep_intent_override=deep_intent,
                )
                user_content = iceberg.user_message
                if deep_intent is None:
                    deep_intent = iceberg.deep_intent

            messages.append(Message(role=Role.USER, content=user_content))

            # --- RP Agent 回复 + L1 质检（带重试） ---
            assistant_msg = await self._generate_rp_response(system_prompt, messages, pipeline)

            for retry in range(MAX_L1_RETRIES):
                l1 = await self.quality.level1_filter(assistant_msg, system_prompt)
                if l1.verdict == QualityVerdict.PASS:
                    break
                logger.info("轮次 %d L1 未通过 (%s), 重试 %d/%d",
                            turn_idx, l1.details, retry + 1, MAX_L1_RETRIES)
                assistant_msg = await self._generate_rp_response(system_prompt, messages, pipeline)

            messages.append(assistant_msg)

        if not any(m.role == Role.ASSISTANT for m in messages):
            return None

        return ConversationRecord(
            messages=messages,
            meta=ConversationMeta(
                system_prompt_source=system_prompt[:200],
                pipeline=pipeline,
                perturbations=perturbations,
            ),
        )

    async def run_batch(
        self,
        system_prompts: list[str] | None = None,
        count: int | None = None,
        num_turns: int | None = None,
        output_file: str = "output.jsonl",
    ) -> list[ConversationRecord]:
        prompts_list = list(system_prompts or [])
        target_count = count or len(prompts_list) or 10

        if len(prompts_list) < target_count:
            need = target_count - len(prompts_list)
            logger.info("需要 %d 条但只有 %d 个角色，自动生成 %d 个", target_count, len(prompts_list), need)
            prompts_list.extend(await self.generate_personas(need))

        await self.warmup_seeds(len(prompts_list))

        sem = asyncio.Semaphore(self.config.pipeline.max_concurrent_conversations)

        async def _one(sp: str) -> ConversationRecord | None:
            async with sem:
                return await self.generate_conversation(sp, num_turns)

        results = await asyncio.gather(*[_one(sp) for sp in prompts_list])
        records = [r for r in results if r is not None]

        logger.info("批次完成: %d/%d 通过", len(records), len(prompts_list))

        if records:
            self.serializer.write_jsonl(records, output_file)
            self.serializer.write_training_only(records, output_file.replace(".jsonl", "_train.jsonl"))

        return records
