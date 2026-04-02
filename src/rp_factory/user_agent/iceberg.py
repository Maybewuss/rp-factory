"""模块一：User Agent 引擎 — 冰山三步法动机混淆流水线。

实现 inspire.md §1 中的核心流程：
  Step 1: 动机反推 (Reverse-Engineering the Deficit)
  Step 2: 实体锚定 (Event Anchoring)
  Step 3: 加密生成 (Obfuscated Generation)

v3.2 增强：
  - LLM 驱动的种子动态扩充：n-gram 塌缩时自动调用 SeedExpander
  - 可变种子池：运行时可注入新种子和风格
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import yaml

from rp_factory.config import FactoryConfig
from rp_factory.diversity import DiversityGuard
from rp_factory.generators import SeedExpander
from rp_factory.llm_client import LLMClient, parse_json_response
from rp_factory.models import IcebergLayers, Message, Role
from rp_factory.prompt_pool import build_prompt_pools

logger = logging.getLogger(__name__)

_SEEDS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "seeds"


def _load_seeds(filename: str) -> list[dict[str, Any]]:
    path = _SEEDS_DIR / filename
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


class IcebergEngine:
    """冰山三步法 User 发言生成器，支持 LLM 驱动的动态种子扩充。"""

    def __init__(self, config: FactoryConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

        self._intent_seeds = _load_seeds("deep_intents.yaml")
        self._event_seeds = _load_seeds("proxy_events.yaml")
        self._styles = list(config.user_agent.iceberg.diversity.user_styles)
        self._flat_events = self._flatten_events()

        # 可变种子列表（运行时动态追加）
        self._intent_texts: list[str] = [s.get("seed", "") for s in self._intent_seeds if s.get("seed")]
        self._event_texts: list[str] = list(self._flat_events)

        self._pools = build_prompt_pools()
        self.diversity = DiversityGuard(
            collapse_threshold=config.user_agent.iceberg.diversity.ngram_collapse_threshold,
        )

        self._expander = SeedExpander(llm)
        self._register_auto_expand()

    def _flatten_events(self) -> list[str]:
        flat: list[str] = []
        for cat in self._event_seeds:
            for ev in cat.get("events", []):
                flat.append(ev)
        return flat

    def _register_auto_expand(self) -> None:
        """将种子扩充回调注册到 DiversityGuard。"""

        async def _expand_intent(category: str) -> None:
            new = await self._expander.expand_intents(
                count=5, existing=self._intent_texts,
            )
            for s in new:
                if s and s not in self._intent_texts:
                    self._intent_texts.append(s)
            logger.info("intent 池扩充至 %d 条", len(self._intent_texts))

        async def _expand_event(category: str) -> None:
            new = await self._expander.expand_events(
                count=8, existing=self._event_texts,
            )
            for s in new:
                if s and s not in self._event_texts:
                    self._event_texts.append(s)
            logger.info("event 池扩充至 %d 条", len(self._event_texts))

        self.diversity.register_expand_callback("intent", _expand_intent)
        self.diversity.register_expand_callback("event", _expand_event)

    # ----- 种子 & 风格选取 -----

    def _pick_style(self) -> str:
        if self._styles and self.config.user_agent.iceberg.diversity.enable_style_injection:
            return random.choice(self._styles)
        return "自然随意型"

    def _pick_seed_intent(self) -> str | None:
        if not self._intent_texts:
            return None
        idx, text = self.diversity.seed_tracker.pick_unique("intent", self._intent_texts)
        return text

    def _pick_seed_event(self) -> str | None:
        if not self._event_texts:
            return None
        idx, event = self.diversity.seed_tracker.pick_unique("event", self._event_texts)
        return event

    def inject_intents(self, new_intents: list[str]) -> int:
        """运行时注入新的深层动机种子。返回新增数量。"""
        added = 0
        for s in new_intents:
            if s and s not in self._intent_texts:
                self._intent_texts.append(s)
                added += 1
        return added

    def inject_events(self, new_events: list[str]) -> int:
        """运行时注入新的表面事件种子。返回新增数量。"""
        added = 0
        for s in new_events:
            if s and s not in self._event_texts:
                self._event_texts.append(s)
                added += 1
        return added

    def inject_styles(self, new_styles: list[str]) -> int:
        """运行时注入新的用户风格标签。返回新增数量。"""
        added = 0
        for s in new_styles:
            if s and s not in self._styles:
                self._styles.append(s)
                added += 1
        return added

    # ----- Step 1: 动机反推 -----
    async def step1_reverse_intent(self, system_prompt: str) -> str:
        """基于角色 System Prompt 反推用户深层心理动机。"""
        template = self._pools["step1_reverse_intent"].pick()

        seed_hint = ""
        seed = self._pick_seed_intent()
        if seed:
            seed_hint = f"\n\n参考方向（可以偏离）：{seed}"

        messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": f"角色设定：\n{system_prompt}{seed_hint}"},
        ]
        resp = await self.llm.chat_single(messages)
        content = resp["content"]

        try:
            data = parse_json_response(content)
            result = data["deep_intent"]
        except (KeyError, Exception):
            result = content.strip()

        self.diversity.check_and_warn("intent", result)
        return result

    # ----- Step 2: 实体锚定 -----
    async def step2_event_anchoring(self, deep_intent: str) -> str:
        """生成用于掩盖动机的琐碎事件。"""
        template = self._pools["step2_event_anchoring"].pick()

        seed_event = self._pick_seed_event()
        seed_hint = ""
        if seed_event:
            seed_hint = f"\n\n参考事件（请生成一个不同的）：{seed_event}"

        messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": f"用户深层动机：\n{deep_intent}{seed_hint}"},
        ]
        resp = await self.llm.chat_single(messages)
        content = resp["content"]

        try:
            data = parse_json_response(content)
            result = data["proxy_event"]
        except (KeyError, Exception):
            result = content.strip()

        self.diversity.check_and_warn("event", result)
        return result

    # ----- Step 3: 加密生成 -----
    async def step3_obfuscated_generation(
        self,
        deep_intent: str,
        proxy_event: str,
        user_style: str,
    ) -> str:
        """融合深层动机与表面事件，输出最终的用户台词。"""
        template = self._pools["step3_obfuscated_generation"].pick()
        system = template.format(
            proxy_event=proxy_event,
            deep_intent=deep_intent,
            user_style=user_style,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "请直接以用户身份说话："},
        ]
        resp = await self.llm.chat_single(messages)
        result = resp["content"].strip().strip('"').strip("「」")
        self.diversity.check_and_warn("output", result)
        return result

    # ----- 上下文感知续写 -----
    async def generate_continuation(
        self,
        conversation_history: list[Message],
        deep_intent: str,
        user_style: str,
    ) -> str:
        """基于对话历史生成延续上文的 User 发言（非首轮使用）。"""
        template = self._pools["continuation"].pick()
        history_text = "\n".join(
            f"[{m.role.value}] {m.content}"
            for m in conversation_history
            if m.role != Role.SYSTEM
        )
        system = template.format(
            history=history_text[-2000:],
            deep_intent=deep_intent,
            user_style=user_style,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "继续："},
        ]
        resp = await self.llm.chat_single(messages)
        result = resp["content"].strip().strip('"').strip("「」")
        self.diversity.check_and_warn("output", result)
        return result

    # ----- 完整流程 -----
    async def generate(
        self,
        system_prompt: str,
        conversation_history: list[Message] | None = None,
    ) -> IcebergLayers:
        """执行冰山三步法或上下文续写，返回三层结构。"""
        style = self._pick_style()
        has_history = (
            conversation_history
            and any(m.role == Role.ASSISTANT for m in conversation_history)
        )

        if has_history:
            logger.info("上下文续写模式 | 风格=%s", style)
            deep_intent = await self.step1_reverse_intent(system_prompt)
            user_message = await self.generate_continuation(
                conversation_history, deep_intent, style,  # type: ignore[arg-type]
            )
            return IcebergLayers(
                deep_intent=deep_intent,
                proxy_event="(续写模式，无独立事件)",
                user_message=user_message,
                user_style=style,
            )

        logger.info("冰山三步法开始 | 风格=%s", style)
        deep_intent = await self.step1_reverse_intent(system_prompt)
        logger.debug("Step1 动机反推完成: %s", deep_intent[:80])

        proxy_event = await self.step2_event_anchoring(deep_intent)
        logger.debug("Step2 实体锚定完成: %s", proxy_event[:80])

        user_message = await self.step3_obfuscated_generation(
            deep_intent, proxy_event, style,
        )
        logger.debug("Step3 加密生成完成: %s", user_message[:80])

        return IcebergLayers(
            deep_intent=deep_intent,
            proxy_event=proxy_event,
            user_message=user_message,
            user_style=style,
        )
