"""模块二：语境控制与异常处理 — 风味任务注入、记忆投毒、认知转译。"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import yaml

from rp_factory import prompts
from rp_factory.config import FactoryConfig
from rp_factory.diversity import Pool
from rp_factory.llm_client import LLMClient
from rp_factory.models import (
    MemoryPoisonSeverity,
    Message,
    Perturbation,
    PerturbationType,
)

logger = logging.getLogger(__name__)

_SEEDS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "seeds"


def _load_flavored_tasks() -> list[dict[str, str]]:
    path = _SEEDS_DIR / "flavored_tasks.yaml"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    flat: list[dict[str, str]] = []
    for cat in raw:
        for task in cat.get("tasks", []):
            flat.append({
                "prompt": task.get("prompt", "帮我处理一下这个："),
                "payload": task.get("payload") or "",
                "type": cat.get("type", "unknown"),
            })
    return flat


class PerturbationEngine:
    """语境扰动引擎。"""

    def __init__(self, config: FactoryConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm
        self.tasks = Pool(_load_flavored_tasks())

    def should_inject_task(self, round_index: int) -> bool:
        cfg = self.config.context_control.flavored_task
        if round_index not in cfg.injection_rounds:
            return False
        return random.random() < cfg.injection_probability

    def create_flavored_task_message(self, round_index: int) -> tuple[str, Perturbation]:
        if not len(self.tasks):
            task = {"prompt": "帮我把这段话翻译成英文：", "payload": "明天三点的会议改到五点。"}
        else:
            task = self.tasks.pick()
        msg = task["prompt"]
        if task.get("payload"):
            msg += "\n\n" + task["payload"]
        return msg, Perturbation(
            type=PerturbationType.FLAVORED_TASK,
            round_index=round_index,
            detail=task["prompt"][:100],
        )

    def should_inject_poison(self, round_index: int) -> bool:
        if round_index < 3:
            return False
        return random.random() < self.config.context_control.memory_poisoning.injection_probability

    def _pick_severity(self) -> MemoryPoisonSeverity:
        dist = self.config.context_control.memory_poisoning.severity_distribution
        selected = random.choices(list(dist.keys()), weights=list(dist.values()), k=1)[0]
        return MemoryPoisonSeverity(selected)

    async def create_memory_poison_message(
        self, conversation_history: list[Message], round_index: int,
    ) -> tuple[str, Perturbation]:
        severity = self._pick_severity()
        history_text = "\n".join(
            f"[{m.role.value}] {m.content}" for m in conversation_history[-6:]
        )
        messages = [
            {"role": "system", "content": prompts.MEMORY_POISON.format(severity=severity.value)},
            {"role": "user", "content": f"对话历史：\n{history_text}"},
        ]
        resp = await self.llm.chat_single(messages)
        return resp["content"].strip().strip('"'), Perturbation(
            type=PerturbationType.MEMORY_POISON,
            round_index=round_index,
            detail=f"severity={severity.value}",
            severity=severity,
        )

    def needs_cognitive_translation(self, system_prompt: str) -> bool:
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
        messages = [
            {"role": "system", "content": prompts.COGNITIVE_TRANSLATION.format(
                persona_summary=system_prompt[:500],
            )},
            {"role": "user", "content": "请生成一个涉及现代知识的用户问题："},
        ]
        resp = await self.llm.chat_single(messages)
        return resp["content"].strip().strip('"'), Perturbation(
            type=PerturbationType.COGNITIVE_TRANSLATION,
            round_index=round_index,
            detail="现代知识问题注入",
        )

    async def maybe_perturb(
        self,
        round_index: int,
        system_prompt: str,
        conversation_history: list[Message],
    ) -> tuple[str | None, Perturbation | None]:
        if self.should_inject_task(round_index):
            return self.create_flavored_task_message(round_index)
        if self.should_inject_poison(round_index):
            return await self.create_memory_poison_message(conversation_history, round_index)
        if self.needs_cognitive_translation(system_prompt) and round_index >= 3 and random.random() < 0.2:
            return await self.create_cognitive_translation_question(system_prompt, round_index)
        return None, None
