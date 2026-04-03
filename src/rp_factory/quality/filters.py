"""质检漏斗：正则黑名单 + 代码块穿透检测 + LLM AI 味结构检测。"""

from __future__ import annotations

import logging
import re

from rp_factory import prompts
from rp_factory.config import FactoryConfig
from rp_factory.llm_client import LLMClient
from rp_factory.models import Message, QualityResult, QualityVerdict

logger = logging.getLogger(__name__)


class QualityFilter:

    def __init__(self, config: FactoryConfig, judge_llm: LLMClient | None = None) -> None:
        self.config = config
        self.judge_llm = judge_llm
        self._blacklist_re = self._compile_blacklist()

    def _compile_blacklist(self) -> re.Pattern[str]:
        patterns = self.config.quality.level1.persona_blacklist_patterns
        if not patterns:
            return re.compile(r"(?!)")
        return re.compile("|".join(re.escape(p) for p in patterns), re.IGNORECASE)

    def check_persona_blacklist(self, content: str) -> bool:
        return not self._blacklist_re.search(content)

    def check_payload_integrity(self, content: str) -> bool:
        code_blocks = re.findall(r"```[\s\S]*?```", content)
        bleed_re = re.compile(
            r"[#//]\s*.{0,5}(笨蛋|白痴|废物|呆瓜)"
            r"|[#//]\s*.{0,5}(给你|帮你|本[大少])"
            r"|[（(][^)）]{0,10}(叹气|翻白眼|冷笑)[)）]"
        )
        return not any(bleed_re.search(block) for block in code_blocks)

    async def check_ai_taste(self, system_prompt: str, content: str) -> bool:
        """LLM 快速判断是否有结构性 AI 味。返回 True 表示通过。"""
        if not self.judge_llm:
            return True
        messages = [
            {"role": "system", "content": prompts.AI_TASTE_JUDGE},
            {"role": "user", "content": (
                f"角色设定（前200字）：\n{system_prompt[:200]}\n\n"
                f"待检测回复：\n{content}"
            )},
        ]
        try:
            resp = await self.judge_llm.chat_single(messages, temperature=0.0, max_tokens=64)
            return resp["content"].strip().lower().startswith("pass")
        except Exception:
            return True

    async def filter(
        self, assistant_msg: Message, system_prompt: str = "",
    ) -> QualityResult:
        content = assistant_msg.content

        if not self.check_persona_blacklist(content):
            return QualityResult(verdict=QualityVerdict.REJECT_PERSONA, details="命中 AI 味黑名单")

        if self.config.quality.level1.payload_lint_enabled and not self.check_payload_integrity(content):
            return QualityResult(verdict=QualityVerdict.REJECT_TASK, details="代码块人设穿透")

        if not await self.check_ai_taste(system_prompt, content):
            return QualityResult(verdict=QualityVerdict.REJECT_PERSONA, details="LLM 检测到结构性 AI 味")

        return QualityResult(verdict=QualityVerdict.PASS)
