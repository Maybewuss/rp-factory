"""模块二：语境控制与异常处理 — 风味任务注入、记忆投毒、认知转译。

实现 inspire.md §2 的三大扰动机制：
  2.1 风味任务 (Flavored Tasks) — 保持人设的工具性任务
  2.2 记忆与逻辑投毒 (Memory & Logic Poisoning)
  2.3 认知转译 (Cognitive Translation) — 唐代诗人悖论
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import yaml

from rp_factory.config import FactoryConfig
from rp_factory.llm_client import LLMClient
from rp_factory.models import (
    MemoryPoisonSeverity,
    Message,
    Perturbation,
    PerturbationType,
)

logger = logging.getLogger(__name__)

_SEEDS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "seeds"


def _load_flavored_tasks() -> list[dict[str, Any]]:
    path = _SEEDS_DIR / "flavored_tasks.yaml"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

MEMORY_POISON_SYSTEM = """\
你是一个对话数据质量工程师。给定一段多轮对话历史，请生成一条"记忆投毒"用户发言。

投毒等级：{severity}

规则：
- critical: 故意篡改角色之前明确表达过的核心事实（名字、核心设定、明确喜好），\
  例如角色说过讨厌甜食，你就让用户说"你不是最喜欢吃马卡龙吗"。
- trivial: 对无关紧要的细节进行模糊记忆偏差，例如把"前天"说成"上周"。
- correct: 生成一条记忆正确的正常回复（用于配比训练，防止模型变成杠精）。

请直接输出用户会说的那句话（纯对白）。
"""

COGNITIVE_TRANSLATION_SYSTEM = """\
你是一个对话数据工程师。当前角色的设定是一个古代/历史人物。\
请生成一个涉及现代知识的用户问题，用于测试角色的"认知转译"能力。

角色设定摘要：{persona_summary}

要求：
- 问题涉及现代科技、概念或事物（如黑洞、互联网、量子力学等）
- 角色应该用自己的世界观词汇包装现代知识来回答，而非装傻

请直接输出用户的问题（纯对白）。
"""


class PerturbationEngine:
    """语境扰动引擎：管理风味任务注入、记忆投毒和认知转译。"""

    def __init__(self, config: FactoryConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm
        self._task_seeds = _load_flavored_tasks()

    # ----- 2.1 风味任务注入 -----
    def should_inject_task(self, round_index: int) -> bool:
        cfg = self.config.context_control.flavored_task
        if round_index not in cfg.injection_rounds:
            return False
        return random.random() < cfg.injection_probability

    def pick_flavored_task(self) -> dict[str, str]:
        """从种子库随机选取一个风味任务。"""
        if not self._task_seeds:
            return {
                "prompt": "帮我把这段话翻译成英文，急用：",
                "payload": "明天下午三点的会议改到五点了，请通知所有人。",
            }
        category = random.choice(self._task_seeds)
        tasks = category.get("tasks", [])
        task = random.choice(tasks) if tasks else {}
        prompt = task.get("prompt", "帮我处理一下这个：")
        payload = task.get("payload", "")
        return {"prompt": prompt, "payload": payload or ""}

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
        history_text = "\n".join(
            f"[{m.role.value}] {m.content}" for m in conversation_history[-6:]
        )
        messages = [
            {
                "role": "system",
                "content": MEMORY_POISON_SYSTEM.format(severity=severity.value),
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
            "古代", "唐代", "宋代", "明代", "清代", "诗人", "武将", "皇帝",
            "仙", "修真", "江湖", "武林", "古风", "历史人物",
        ]
        sandbox_tag = self.config.context_control.cognitive_translation.strict_sandbox_tag
        if sandbox_tag in system_prompt:
            return False
        return any(h in system_prompt for h in historical_hints)

    async def create_cognitive_translation_question(
        self, system_prompt: str, round_index: int,
    ) -> tuple[str, Perturbation]:
        """为历史角色生成涉及现代知识的用户问题。"""
        messages = [
            {
                "role": "system",
                "content": COGNITIVE_TRANSLATION_SYSTEM.format(
                    persona_summary=system_prompt[:500],
                ),
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
