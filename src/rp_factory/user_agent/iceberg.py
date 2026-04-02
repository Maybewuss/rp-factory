"""模块一：User Agent 引擎 — 冰山三步法动机混淆流水线。

实现 inspire.md §1 中的核心流程：
  Step 1: 动机反推 (Reverse-Engineering the Deficit)
  Step 2: 实体锚定 (Event Anchoring)
  Step 3: 加密生成 (Obfuscated Generation)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import yaml

from rp_factory.config import FactoryConfig
from rp_factory.llm_client import LLMClient, parse_json_response
from rp_factory.models import IcebergLayers

logger = logging.getLogger(__name__)

_SEEDS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "seeds"


def _load_seeds(filename: str) -> list[dict[str, Any]]:
    path = _SEEDS_DIR / filename
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

STEP1_SYSTEM = """\
你是一名精通精神分析的心理咨询师。你的任务是基于精神分析的"对偶性"原理，\
反推一个真实人类在什么极端脆弱的心理状态下，会渴望与某种特定角色交流。

请基于下方给出的 RP 角色设定（System Prompt），输出一段核心的心理动机描述。\
要求：具体、深刻、带有个人创伤色彩，不要泛泛而谈。

请用 JSON 格式输出：{"deep_intent": "..."}
"""

STEP2_SYSTEM = """\
你是一名创意写作专家。你的任务是生成一个日常生活中极度琐碎、倒霉、\
且与用户真实心理困境毫无关联的微小物理事件，作为情绪的伪装载体。

要求：
1. 事件必须极其琐碎（洒咖啡、碎屏、丢钥匙级别）
2. 事件与深层动机之间不能有逻辑关联
3. 事件必须带有轻微的倒霉感

请用 JSON 格式输出：{"proxy_event": "..."}
"""

STEP3_SYSTEM = """\
你是一个正在经历以下琐碎事件的真实用户：
{proxy_event}

但你的内心实际上处于以下状态：
{deep_intent}

现在你要向一个角色发起对话。

【绝对红线】：
- 你的发言中绝对不能出现任何直接暴露内心真实动机的词汇
- 你必须把所有情绪（崩溃、委屈、无名火）全部发泄在这件琐碎事件上
- 模拟真实人类在情绪激动时的口语，允许逻辑跳跃、语病、重复
- 用户画像风格：{user_style}

请直接输出用户会说的话（纯对白，不要任何解释或标注）。
"""


class IcebergEngine:
    """冰山三步法 User 发言生成器。"""

    def __init__(self, config: FactoryConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm
        self._intent_seeds = _load_seeds("deep_intents.yaml")
        self._event_seeds = _load_seeds("proxy_events.yaml")
        self._styles = config.user_agent.iceberg.diversity.user_styles

    def _pick_style(self) -> str:
        if self._styles and self.config.user_agent.iceberg.diversity.enable_style_injection:
            return random.choice(self._styles)
        return "自然随意型"

    def _pick_seed_intent(self) -> str | None:
        if self._intent_seeds:
            return random.choice(self._intent_seeds).get("seed")
        return None

    def _pick_seed_event(self) -> str | None:
        if self._event_seeds:
            cat = random.choice(self._event_seeds)
            events = cat.get("events", [])
            if events:
                return random.choice(events)
        return None

    # ----- Step 1: 动机反推 -----
    async def step1_reverse_intent(self, system_prompt: str) -> str:
        """基于角色 System Prompt 反推用户深层心理动机。"""
        seed_hint = ""
        seed = self._pick_seed_intent()
        if seed:
            seed_hint = f"\n\n参考方向（可以偏离）：{seed}"

        messages = [
            {"role": "system", "content": STEP1_SYSTEM},
            {"role": "user", "content": f"角色设定：\n{system_prompt}{seed_hint}"},
        ]
        resp = await self.llm.chat_single(messages)
        try:
            data = parse_json_response(resp["content"])
            return data["deep_intent"]
        except (KeyError, Exception):
            return resp["content"].strip()

    # ----- Step 2: 实体锚定 -----
    async def step2_event_anchoring(self, deep_intent: str) -> str:
        """生成用于掩盖动机的琐碎事件。"""
        seed_event = self._pick_seed_event()
        seed_hint = ""
        if seed_event:
            seed_hint = f"\n\n参考事件（请生成一个不同的）：{seed_event}"

        messages = [
            {"role": "system", "content": STEP2_SYSTEM},
            {"role": "user", "content": f"用户深层动机：\n{deep_intent}{seed_hint}"},
        ]
        resp = await self.llm.chat_single(messages)
        try:
            data = parse_json_response(resp["content"])
            return data["proxy_event"]
        except (KeyError, Exception):
            return resp["content"].strip()

    # ----- Step 3: 加密生成 -----
    async def step3_obfuscated_generation(
        self,
        deep_intent: str,
        proxy_event: str,
        user_style: str,
    ) -> str:
        """融合深层动机与表面事件，输出最终的用户台词。"""
        system = STEP3_SYSTEM.format(
            proxy_event=proxy_event,
            deep_intent=deep_intent,
            user_style=user_style,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "请直接以用户身份说话："},
        ]
        resp = await self.llm.chat_single(messages)
        return resp["content"].strip().strip('"').strip("「」")

    # ----- 完整流程 -----
    async def generate(self, system_prompt: str) -> IcebergLayers:
        """执行完整的冰山三步法，返回三层结构。"""
        style = self._pick_style()
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
