"""模块四：Rejection Sampling 与质检漏斗。

实现 inspire.md §4 的两级过滤：
  4.1 第一级：人设校验（黑名单正则 + AI味检测）+ 任务校验（载荷无污染）
  4.2 第二级：目标模型增益过滤 (Target-Model-in-the-Loop)
"""

from __future__ import annotations

import logging
import re

from rp_factory.config import FactoryConfig
from rp_factory.llm_client import LLMClient
from rp_factory.models import (
    Message,
    QualityResult,
    QualityVerdict,
    Role,
)

logger = logging.getLogger(__name__)

GAIN_JUDGE_SYSTEM = """\
你是一名数据增益评估专家。你的任务是对比两个回复的质量差异，判断 Teacher 回复\
相对于 Target 回复是否具有训练增益。

评估维度：
1. 深层需求识别：Teacher 是否比 Target 更准确地识别了用户的隐藏需求？
2. 人设一致性：Teacher 是否比 Target 更好地维持了角色设定？
3. 推理质量：Teacher 的 reasoning 是否展示了 Target 缺失的推理步骤？
4. 去 AI 味：Teacher 是否比 Target 更少使用模板化表达？

请输出 JSON：
{{"has_gain": bool, "gain_score": float, "reason": "..."}}

gain_score 范围 0-1：
- 0.0: Target 已经很好，无增益
- 0.3-0.5: 有轻微增益
- 0.5-0.8: 有显著增益
- 0.8-1.0: Target 严重失败，极高增益
"""


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
            return re.compile(r"(?!)")  # never matches
        escaped = [re.escape(p) for p in patterns]
        return re.compile("|".join(escaped), re.IGNORECASE)

    # ----- 第一级漏斗 -----

    def check_persona_blacklist(self, content: str) -> bool:
        """检查 AI 味黑名单。返回 True 表示通过（未命中黑名单）。"""
        if self._blacklist_re.search(content):
            return False
        return True

    def check_payload_integrity(self, content: str) -> bool:
        """检查代码块内是否存在人设穿透（Persona Bleed）。

        规则：代码块（```...```）内部不应包含角色口癖、颜文字、骂人等内容。
        """
        code_blocks = re.findall(r"```[\s\S]*?```", content)
        persona_bleed_patterns = [
            r"[#//]\s*.{0,5}(笨蛋|白痴|废物|呆瓜)",
            r"[#//]\s*.{0,5}(给你|帮你|本[大少])",
            r"[（(][^)）]{0,10}(叹气|翻白眼|冷笑)[)）]",
        ]
        bleed_re = re.compile("|".join(persona_bleed_patterns))
        for block in code_blocks:
            if bleed_re.search(block):
                return False
        return True

    def level1_filter(self, assistant_msg: Message) -> QualityResult:
        """第一级漏斗：人设 + 任务校验。"""
        content = assistant_msg.content

        if not self.check_persona_blacklist(content):
            return QualityResult(
                verdict=QualityVerdict.REJECT_PERSONA,
                details="命中 AI 味黑名单正则",
            )

        if self.config.quality.level1.payload_lint_enabled:
            if not self.check_payload_integrity(content):
                return QualityResult(
                    verdict=QualityVerdict.REJECT_TASK,
                    details="代码块内检测到人设穿透 (Persona Bleed)",
                )

        return QualityResult(verdict=QualityVerdict.PASS)

    # ----- 第二级漏斗 -----

    async def level2_gain_filter(
        self,
        system_prompt: str,
        conversation: list[Message],
        teacher_msg: Message,
    ) -> QualityResult:
        """第二级漏斗：目标模型增益过滤。

        将前缀喂给 Target Model，对比其回答与 Teacher 回答的差异。
        """
        if not self.config.quality.level2.enabled:
            return QualityResult(verdict=QualityVerdict.PASS, details="L2 disabled")

        if not self.target_llm or not self.judge_llm:
            return QualityResult(verdict=QualityVerdict.PASS, details="no target/judge LLM")

        openai_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        for m in conversation:
            if m.role != Role.SYSTEM:
                openai_messages.append({"role": m.role.value, "content": m.content})

        target_resp = await self.target_llm.chat_single(openai_messages)

        judge_messages = [
            {"role": "system", "content": GAIN_JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"角色设定：\n{system_prompt}\n\n"
                    f"用户最后发言：\n{conversation[-1].content if conversation else ''}\n\n"
                    f"--- Target Model 回复 ---\n{target_resp['content']}\n\n"
                    f"--- Teacher Model 回复 ---\n{teacher_msg.content}\n\n"
                    f"--- Teacher Reasoning ---\n{teacher_msg.reasoning_content or '(无)'}\n\n"
                    f"--- Target Reasoning ---\n{target_resp.get('reasoning_content', '(无)')}"
                ),
            },
        ]
        judge_resp = await self.judge_llm.chat_single(judge_messages)

        try:
            from rp_factory.llm_client import parse_json_response
            data = parse_json_response(judge_resp["content"])
            has_gain = data.get("has_gain", False)
            gain_score = float(data.get("gain_score", 0.0))
        except Exception:
            logger.warning("增益评估 JSON 解析失败，默认保留")
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

    # ----- 完整过滤 -----

    async def filter(
        self,
        system_prompt: str,
        conversation: list[Message],
        assistant_msg: Message,
    ) -> QualityResult:
        """执行两级质检漏斗。"""
        l1 = self.level1_filter(assistant_msg)
        if l1.verdict != QualityVerdict.PASS:
            return l1

        return await self.level2_gain_filter(
            system_prompt, conversation, assistant_msg,
        )
