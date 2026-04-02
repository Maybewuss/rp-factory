"""模块四：Rejection Sampling 与质检漏斗。

第一级：人设校验（黑名单正则）+ 任务校验（代码块无穿透）
第二级：目标模型增益过滤 (Target-Model-in-the-Loop)
"""

from __future__ import annotations

import logging
import re

from rp_factory import prompts
from rp_factory.config import FactoryConfig
from rp_factory.llm_client import LLMClient, parse_json_response
from rp_factory.models import Message, QualityResult, QualityVerdict, Role

logger = logging.getLogger(__name__)


class QualityFilter:
    """两级质检漏斗。"""

    def __init__(
        self,
        config: FactoryConfig,
        target_llm: LLMClient | None = None,
        judge_llm: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.target_llm = target_llm
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

    def level1_filter(self, assistant_msg: Message) -> QualityResult:
        content = assistant_msg.content
        if not self.check_persona_blacklist(content):
            return QualityResult(verdict=QualityVerdict.REJECT_PERSONA, details="命中 AI 味黑名单")
        if self.config.quality.level1.payload_lint_enabled and not self.check_payload_integrity(content):
            return QualityResult(verdict=QualityVerdict.REJECT_TASK, details="代码块人设穿透")
        return QualityResult(verdict=QualityVerdict.PASS)

    async def level2_gain_filter(
        self, system_prompt: str, conversation: list[Message], teacher_msg: Message,
    ) -> QualityResult:
        if not self.config.quality.level2.enabled or not self.target_llm or not self.judge_llm:
            return QualityResult(verdict=QualityVerdict.PASS, details="L2 跳过")

        openai_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for m in conversation:
            if m.role != Role.SYSTEM:
                openai_messages.append({"role": m.role.value, "content": m.content})

        target_resp = await self.target_llm.chat_single(openai_messages)

        judge_messages = [
            {"role": "system", "content": prompts.GAIN_JUDGE},
            {"role": "user", "content": (
                f"角色设定：\n{system_prompt}\n\n"
                f"用户最后发言：\n{conversation[-1].content if conversation else ''}\n\n"
                f"--- Target ---\n{target_resp['content']}\n\n"
                f"--- Teacher ---\n{teacher_msg.content}\n\n"
                f"--- Teacher Reasoning ---\n{teacher_msg.reasoning_content or '(无)'}"
            )},
        ]
        judge_resp = await self.judge_llm.chat_single(judge_messages)

        try:
            data = parse_json_response(judge_resp["content"])
            has_gain = data.get("has_gain", False)
            gain_score = float(data.get("gain_score", 0.0))
        except Exception:
            return QualityResult(verdict=QualityVerdict.PASS, gain_delta=0.5)

        if not has_gain or gain_score < 0.2:
            return QualityResult(
                verdict=QualityVerdict.REJECT_LOW_GAIN,
                gain_delta=gain_score,
                details=data.get("reason", ""),
            )
        return QualityResult(
            verdict=QualityVerdict.PASS,
            gain_delta=gain_score,
            details=data.get("reason", ""),
        )
