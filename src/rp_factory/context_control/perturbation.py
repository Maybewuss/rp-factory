"""模块二：语境控制与异常处理 — 风味任务注入、记忆投毒、认知转译。

实现 inspire.md §2 的三大扰动机制：
  2.1 风味任务 (Flavored Tasks) — 保持人设的工具性任务
  2.2 记忆与逻辑投毒 (Memory & Logic Poisoning)
  2.3 认知转译 (Cognitive Translation) — 唐代诗人悖论

v3.1.1 增强：
  - Prompt 模板池替代固定模板
  - 种子去重追踪避免批次内重复
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import yaml

from rp_factory.config import FactoryConfig
from rp_factory.diversity import SeedTracker
from rp_factory.llm_client import LLMClient
from rp_factory.models import (
    MemoryPoisonSeverity,
    Message,
    Perturbation,
    PerturbationType,
)
from rp_factory.prompt_pool import build_prompt_pools

logger = logging.getLogger(__name__)

_SEEDS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "seeds"


def _load_flavored_tasks() -> list[dict[str, Any]]:
    path = _SEEDS_DIR / "flavored_tasks.yaml"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


class PerturbationEngine:
    """语境扰动引擎：管理风味任务注入、记忆投毒和认知转译。"""

    def __init__(self, config: FactoryConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm
        self._task_seeds = _load_flavored_tasks()
        self._flat_tasks = self._flatten_tasks()
        self._pools = build_prompt_pools()
        self._seed_tracker = SeedTracker()

    def _flatten_tasks(self) -> list[dict[str, str]]:
        flat: list[dict[str, str]] = []
        for cat in self._task_seeds:
            for task in cat.get("tasks", []):
                flat.append({
                    "prompt": task.get("prompt", "帮我处理一下这个："),
                    "payload": task.get("payload") or "",
                    "type": cat.get("type", "unknown"),
                })
        return flat

    # ----- 2.1 风味任务注入 -----
    def should_inject_task(self, round_index: int) -> bool:
        cfg = self.config.context_control.flavored_task
        if round_index not in cfg.injection_rounds:
            return False
        return random.random() < cfg.injection_probability

    def pick_flavored_task(self) -> dict[str, str]:
        """从种子库随机选取一个风味任务（带去重追踪）。"""
        if not self._flat_tasks:
            return {
                "prompt": "帮我把这段话翻译成英文，急用：",
                "payload": "明天下午三点的会议改到五点了，请通知所有人。",
            }
        _, task = self._seed_tracker.pick_unique("flavored_task", self._flat_tasks)
        return task

    def create_flavored_task_message(self, round_index: int) -> tuple[str, Perturbation]:
        """生成风味任务用户发言及对应的扰动记录。"""
        task = self.pick_flavored_task()
        msg = task["prompt"]
        if task["payload"]:
            msg += "\n\n" + task["payload"]
        perturbation = Perturbation(
            type=PerturbationType.FLAVORED_TASK,
            round_index=round_index,
            detail=task["prompt"][:100],
        )
        return msg, perturbation

    # ----- 2.2 记忆投毒 -----
    def should_inject_poison(self, round_index: int) -> bool:
        if round_index < 3:
            return False
        return random.random() < self.config.context_control.memory_poisoning.injection_probability

    def _pick_severity(self) -> MemoryPoisonSeverity:
        dist = self.config.context_control.memory_poisoning.severity_distribution
        choices = list(dist.keys())
        weights = list(dist.values())
        selected = random.choices(choices, weights=weights, k=1)[0]
        return MemoryPoisonSeverity(selected)

    async def create_memory_poison_message(
        self,
        conversation_history: list[Message],
        round_index: int,
    ) -> tuple[str, Perturbation]:
        """基于对话历史生成记忆投毒用户发言。"""
        severity = self._pick_severity()
        template = self._pools["memory_poison"].pick()

        history_text = "\n".join(
            f"[{m.role.value}] {m.content}" for m in conversation_history[-6:]
        )
        messages = [
            {
                "role": "system",
                "content": template.format(severity=severity.value),
            },
            {"role": "user", "content": f"对话历史：\n{history_text}"},
        ]
        resp = await self.llm.chat_single(messages)
        user_msg = resp["content"].strip().strip('"')

        perturbation = Perturbation(
            type=PerturbationType.MEMORY_POISON,
            round_index=round_index,
            detail=f"severity={severity.value}",
            severity=severity,
        )
        return user_msg, perturbation

    # ----- 2.3 认知转译 -----
    def needs_cognitive_translation(self, system_prompt: str) -> bool:
        """检测 System Prompt 是否涉及历史/古代人物设定。"""
        historical_hints = [
            "古代", "唐代", "宋代", "明代", "清代", "汉代", "三国", "春秋", "战国",
            "诗人", "武将", "皇帝", "将军", "丞相", "书生",
            "仙", "修真", "江湖", "武林", "古风", "历史人物",
            "中世纪", "维多利亚", "文艺复兴", "古罗马", "古希腊",
        ]
        sandbox_tag = self.config.context_control.cognitive_translation.strict_sandbox_tag
        if sandbox_tag in system_prompt:
            return False
        return any(h in system_prompt for h in historical_hints)

    async def create_cognitive_translation_question(
        self, system_prompt: str, round_index: int,
    ) -> tuple[str, Perturbation]:
        """为历史角色生成涉及现代知识的用户问题。"""
        template = self._pools["cognitive_translation"].pick()
        messages = [
            {
                "role": "system",
                "content": template.format(persona_summary=system_prompt[:500]),
            },
            {"role": "user", "content": "请生成一个涉及现代知识的用户问题："},
        ]
        resp = await self.llm.chat_single(messages)
        user_msg = resp["content"].strip().strip('"')

        perturbation = Perturbation(
            type=PerturbationType.COGNITIVE_TRANSLATION,
            round_index=round_index,
            detail="现代知识问题注入",
        )
        return user_msg, perturbation

    # ----- 综合调度 -----
    async def maybe_perturb(
        self,
        round_index: int,
        system_prompt: str,
        conversation_history: list[Message],
    ) -> tuple[str | None, Perturbation | None]:
        """根据当前轮次决定是否注入扰动，返回 (扰动消息, 扰动记录) 或 (None, None)。"""
        if self.should_inject_task(round_index):
            logger.info("轮次 %d: 注入风味任务", round_index)
            return self.create_flavored_task_message(round_index)

        if self.should_inject_poison(round_index):
            logger.info("轮次 %d: 注入记忆投毒", round_index)
            return await self.create_memory_poison_message(
                conversation_history, round_index,
            )

        if (
            self.needs_cognitive_translation(system_prompt)
            and round_index >= 3
            and random.random() < 0.2
        ):
            logger.info("轮次 %d: 注入认知转译测试", round_index)
            return await self.create_cognitive_translation_question(
                system_prompt, round_index,
            )

        return None, None
