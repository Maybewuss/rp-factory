"""测试质检漏斗。"""

import pytest

from rp_factory.config import load_config
from rp_factory.models import Message, QualityVerdict, Role
from rp_factory.quality.filters import QualityFilter


@pytest.fixture
def qf():
    return QualityFilter(load_config())


class TestQualityFilter:
    @pytest.mark.asyncio
    async def test_pass_clean_response(self, qf: QualityFilter):
        msg = Message(role=Role.ASSISTANT, content="啧，你这破事也好意思来烦我？")
        r = await qf.filter(msg)
        assert r.verdict == QualityVerdict.PASS

    @pytest.mark.asyncio
    async def test_reject_ai_taste_chinese(self, qf: QualityFilter):
        msg = Message(role=Role.ASSISTANT, content="好的，为您解答这个问题。作为AI，我可以帮助您。")
        r = await qf.filter(msg)
        assert r.verdict == QualityVerdict.REJECT_PERSONA

    @pytest.mark.asyncio
    async def test_reject_ai_taste_pattern(self, qf: QualityFilter):
        msg = Message(role=Role.ASSISTANT, content="如果您还需要帮助，请随时告诉我。")
        r = await qf.filter(msg)
        assert r.verdict == QualityVerdict.REJECT_PERSONA

    @pytest.mark.asyncio
    async def test_pass_code_block_clean(self, qf: QualityFilter):
        msg = Message(
            role=Role.ASSISTANT,
            content='切，就这代码？\n\n```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\n\n拿去。',
        )
        r = await qf.filter(msg)
        assert r.verdict == QualityVerdict.PASS

    @pytest.mark.asyncio
    async def test_reject_persona_bleed_in_code(self, qf: QualityFilter):
        msg = Message(role=Role.ASSISTANT, content='```python\n# 给你这笨蛋写的\ndef f(): pass\n```')
        r = await qf.filter(msg)
        assert r.verdict == QualityVerdict.REJECT_TASK


class TestBlacklistPatterns:
    @pytest.mark.asyncio
    async def test_multiple_patterns(self, qf: QualityFilter):
        for phrase in ["作为AI，我认为", "好的，为您处理", "如果您还需要任何帮助", "很高兴为您服务"]:
            msg = Message(role=Role.ASSISTANT, content=phrase)
            r = await qf.filter(msg)
            assert r.verdict == QualityVerdict.REJECT_PERSONA, f"应拒绝: {phrase}"

    @pytest.mark.asyncio
    async def test_persona_rich_response(self, qf: QualityFilter):
        msg = Message(role=Role.ASSISTANT, content="哈？你问我黑洞是什么？本仙尊称之为太虚深渊。")
        r = await qf.filter(msg)
        assert r.verdict == QualityVerdict.PASS
