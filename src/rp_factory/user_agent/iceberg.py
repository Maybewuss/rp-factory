"""模块一：User Agent 引擎 — 冰山三步法动机混淆流水线。

Step 1: 动机反推 → Step 2: 实体锚定 → Step 3: 加密生成
非首轮对话使用上下文感知续写。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from rp_factory import prompts
from rp_factory.config import FactoryConfig
from rp_factory.diversity import NGramMonitor, Pool
from rp_factory.llm_client import LLMClient, parse_json_response
from rp_factory.models import IcebergLayers, Message, Role

logger = logging.getLogger(__name__)

_SEEDS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "seeds"


def _load_yaml(filename: str) -> list[Any]:
    path = _SEEDS_DIR / filename
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


class IcebergEngine:
    """冰山三步法 User 发言生成器。"""

    def __init__(self, config: FactoryConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

        raw_intents = _load_yaml("deep_intents.yaml")
        raw_events = _load_yaml("proxy_events.yaml")

        self.intents = Pool([s["seed"] for s in raw_intents if s.get("seed")])
        self.events = Pool([
            ev for cat in raw_events for ev in cat.get("events", [])
        ])
        self.styles = Pool(list(config.user_agent.iceberg.diversity.user_styles))

        self.intent_monitor = NGramMonitor(
            threshold=config.user_agent.iceberg.diversity.ngram_collapse_threshold,
        )
        self.event_monitor = NGramMonitor(
            threshold=config.user_agent.iceberg.diversity.ngram_collapse_threshold,
        )

    async def step1_reverse_intent(self, system_prompt: str) -> str:
        seed_hint = ""
        if len(self.intents):
            seed_hint = f"\n\n参考方向（可以偏离）：{self.intents.pick()}"

        messages = [
            {"role": "system", "content": prompts.STEP1_REVERSE_INTENT},
            {"role": "user", "content": f"角色设定：\n{system_prompt}{seed_hint}"},
        ]
        resp = await self.llm.chat_single(messages)
        try:
            result = parse_json_response(resp["content"])["deep_intent"]
        except (KeyError, Exception):
            result = resp["content"].strip()

        ratio = self.intent_monitor.add_text(result)
        if self.intent_monitor.is_collapsing():
            logger.warning("intent n-gram 重复率 %.2f 超阈值，建议扩充", ratio)
        return result

    async def step2_event_anchoring(self, deep_intent: str) -> str:
        seed_hint = ""
        if len(self.events):
            seed_hint = f"\n\n参考事件（请生成一个不同的）：{self.events.pick()}"

        messages = [
            {"role": "system", "content": prompts.STEP2_EVENT_ANCHORING},
            {"role": "user", "content": f"用户深层动机：\n{deep_intent}{seed_hint}"},
        ]
        resp = await self.llm.chat_single(messages)
        try:
            result = parse_json_response(resp["content"])["proxy_event"]
        except (KeyError, Exception):
            result = resp["content"].strip()

        ratio = self.event_monitor.add_text(result)
        if self.event_monitor.is_collapsing():
            logger.warning("event n-gram 重复率 %.2f 超阈值，建议扩充", ratio)
        return result

    async def step3_obfuscated_generation(
        self, deep_intent: str, proxy_event: str, user_style: str,
    ) -> str:
        system = prompts.STEP3_OBFUSCATED_GENERATION.format(
            proxy_event=proxy_event, deep_intent=deep_intent, user_style=user_style,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "直接说话："},
        ]
        resp = await self.llm.chat_single(messages)
        return resp["content"].strip().strip('"').strip("「」")

    async def generate_continuation(
        self, conversation_history: list[Message], deep_intent: str, user_style: str,
    ) -> str:
        history_text = "\n".join(
            f"[{m.role.value}] {m.content}"
            for m in conversation_history if m.role != Role.SYSTEM
        )
        system = prompts.CONTINUATION.format(
            history=history_text[-2000:], deep_intent=deep_intent, user_style=user_style,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "继续："},
        ]
        resp = await self.llm.chat_single(messages)
        return resp["content"].strip().strip('"').strip("「」")

    async def generate(
        self,
        system_prompt: str,
        conversation_history: list[Message] | None = None,
        deep_intent_override: str | None = None,
    ) -> IcebergLayers:
        """生成 User 发言。

        Args:
            deep_intent_override: 复用之前的 deep_intent（同一段对话的心理动机应该贯穿始终）。
                首轮传 None 触发 step1 生成，后续轮次传首轮的 intent。
        """
        style = self.styles.pick() if len(self.styles) else "自然随意型"
        has_history = (
            conversation_history
            and any(m.role == Role.ASSISTANT for m in conversation_history)
        )

        deep_intent = deep_intent_override or await self.step1_reverse_intent(system_prompt)

        if has_history:
            user_message = await self.generate_continuation(
                conversation_history, deep_intent, style,  # type: ignore[arg-type]
            )
            return IcebergLayers(
                deep_intent=deep_intent,
                proxy_event="(续写模式)",
                user_message=user_message,
                user_style=style,
            )

        proxy_event = await self.step2_event_anchoring(deep_intent)
        user_message = await self.step3_obfuscated_generation(deep_intent, proxy_event, style)

        return IcebergLayers(
            deep_intent=deep_intent,
            proxy_event=proxy_event,
            user_message=user_message,
            user_style=style,
        )
