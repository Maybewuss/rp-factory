import unittest

from src.rp_factory.models import FlavoredTask, Scenario, TeacherResponse
from src.rp_factory.quality import run_quality_funnel


class QualityFunnelTests(unittest.TestCase):
    def test_rejects_blacklist_language(self) -> None:
        scenario = Scenario(
            name="blacklist",
            system_prompt="你是一个暴躁但可靠的角色。",
            persona_tags=["暴躁"],
            intent_tags=["高压"],
            intensity="high",
            style_pool=["暴躁直接型"],
        )
        response = TeacherResponse(
            reasoning_content="表层线索：有。 深层缺口：有。",
            content="好的，为您处理如下。",
            scores={},
        )
        report = run_quality_funnel(scenario, response)
        self.assertFalse(report.accepted)
        self.assertTrue(any("黑名单" in reason for reason in report.reasons))

    def test_rejects_persona_bleed_in_code(self) -> None:
        scenario = Scenario(
            name="payload",
            system_prompt="你是一个嘴硬但靠谱的程序员角色。",
            persona_tags=["嘴硬"],
            intent_tags=["执行力"],
            intensity="medium",
            style_pool=["理性压抑型"],
            flavored_task=FlavoredTask(
                kind="python",
                payload="写一个重置状态函数",
                expected_keywords=["reset"],
                code_language="python",
            ),
        )
        response = TeacherResponse(
            reasoning_content="表层线索：有。 深层缺口：有。 任务剥离：有。",
            content="```python\n# 给你这笨蛋写的\nprint('x')\n```",
            scores={},
        )
        report = run_quality_funnel(scenario, response)
        self.assertFalse(report.accepted)
        self.assertTrue(any("人设污染" in reason for reason in report.reasons))


if __name__ == "__main__":
    unittest.main()
