"""测试质检漏斗。"""

import pytest

from rp_factory.config import load_config
from rp_factory.models import Message, QualityVerdict, Role
from rp_factory.quality.filters import QualityFilter


@pytest.fixture
def quality_filter():
    cfg = load_config()
    return QualityFilter(cfg)


class TestLevel1Filter:
    def test_pass_clean_response(self, quality_filter: QualityFilter):
        msg = Message(
            role=Role.ASSISTANT,
            content="啧，你这破事也好意思来烦我？算了，看在你可怜的份上跟你说两句。",
        )
        result = quality_filter.level1_filter(msg)
        assert result.verdict == QualityVerdict.PASS

    def test_reject_ai_taste_chinese(self, quality_filter: QualityFilter):
        msg = Message(
            role=Role.ASSISTANT,
            content="好的，为您解答这个问题。作为AI，我可以帮助您。",
        )
        result = quality_filter.level1_filter(msg)
        assert result.verdict == QualityVerdict.REJECT_PERSONA

    def test_reject_ai_taste_pattern(self, quality_filter: QualityFilter):
        msg = Message(
            role=Role.ASSISTANT,
            content="如果您还需要帮助，请随时告诉我。",
        )
        result = quality_filter.level1_filter(msg)
        assert result.verdict == QualityVerdict.REJECT_PERSONA

    def test_pass_code_block_clean(self, quality_filter: QualityFilter):
        msg = Message(
            role=Role.ASSISTANT,
            content='切，就这点破代码也好意思说难？\n\n```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\n\n拿去用吧，别再来烦我。',
        )
        result = quality_filter.level1_filter(msg)
        assert result.verdict == QualityVerdict.PASS

    def test_reject_persona_bleed_in_code(self, quality_filter: QualityFilter):
        msg = Message(
            role=Role.ASSISTANT,
            content='```python\n# 给你这笨蛋写的代码\ndef fib(n):\n    pass\n```',
        )
        result = quality_filter.level1_filter(msg)
        assert result.verdict == QualityVerdict.REJECT_TASK


class TestBlacklistPatterns:
    def test_multiple_patterns(self, quality_filter: QualityFilter):
        bad_phrases = [
            "作为AI，我认为",
            "好的，为您处理",
            "如果您还需要任何帮助",
            "很高兴为您服务",
        ]
        for phrase in bad_phrases:
            msg = Message(role=Role.ASSISTANT, content=phrase)
            result = quality_filter.level1_filter(msg)
            assert result.verdict == QualityVerdict.REJECT_PERSONA, f"应拒绝: {phrase}"

    def test_persona_rich_response(self, quality_filter: QualityFilter):
        msg = Message(
            role=Role.ASSISTANT,
            content="哈？你问我黑洞是什么？本仙尊称之为太虚深渊，万物归墟之地。",
        )
        result = quality_filter.level1_filter(msg)
        assert result.verdict == QualityVerdict.PASS
