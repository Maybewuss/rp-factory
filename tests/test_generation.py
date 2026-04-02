from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from random import Random

from src.rp_factory.io_utils import write_jsonl
from src.rp_factory.models import DatasetRecord, DiversityState, GenerationPolicy, GenerationTargets, PersonaOverlay, Scenario, SeedPoolSnapshot, TeacherResponse
from src.rp_factory.pipeline import BuildResult, build_dataset
from src.rp_factory.quality import extract_code_blocks, run_quality_funnel
from src.rp_factory.user_generation import generate_user_bundle
from src.rp_factory.seeds import build_snapshot


class UserGenerationTests(unittest.TestCase):
    def test_generate_user_bundle_includes_task_and_poisoning(self) -> None:
        scenario = Scenario.from_dict(
            {
                "name": "task-and-poisoning",
                "system_prompt": "你是一个嘴硬但可靠的维修师。",
                "persona_tags": ["嘴硬", "可靠"],
                "intent_tags": ["高压", "执行力"],
                "intensity": "high",
                "style_pool": ["暴躁直接型"],
                "flavored_task": {
                    "kind": "python",
                    "payload": "写一个重置缓存的函数",
                    "expected_keywords": ["reset"],
                    "code_language": "python",
                },
                "poisoning": {
                    "tier": "critical",
                    "incorrect_user_claim": "你昨天说过你最喜欢吃甜食",
                    "correction_target": "我讨厌甜食",
                },
            }
        )

        bundle = generate_user_bundle(scenario, Random(3), build_snapshot())
        self.assertIn("Python", bundle.user_message)
        self.assertIn("你昨天说过你最喜欢吃甜食", bundle.user_message)
        self.assertTrue(bundle.deep_intent)
        self.assertTrue(bundle.proxy_event)

    def test_generate_user_bundle_avoids_short_horizon_collisions(self) -> None:
        scenario = Scenario.from_dict(
            {
                "name": "diversity-check",
                "system_prompt": "你是一个嘴硬但会兜底的顾问。",
                "persona_tags": ["嘴硬", "兜底"],
                "intent_tags": ["高压", "执行力", "混乱"],
                "intensity": "high",
                "style_pool": ["暴躁直接型", "碎碎念型", "理性压抑型"],
            }
        )

        rng = Random(19)
        state = DiversityState()
        snapshot = build_snapshot()
        bundles = [generate_user_bundle(scenario, rng, snapshot, state) for _ in range(5)]

        styles = {bundle.style_tag for bundle in bundles}
        events = {bundle.proxy_event for bundle in bundles}
        messages = {bundle.user_message for bundle in bundles}

        self.assertGreaterEqual(len(styles), 2)
        self.assertGreaterEqual(len(events), 2)
        self.assertGreaterEqual(len(messages), 3)

    def test_generate_user_bundle_uses_dynamic_pool_snapshot(self) -> None:
        scenario = Scenario.from_dict(
            {
                "name": "snapshot-check",
                "system_prompt": "你是一个敏锐又危险的顾问。",
                "persona_tags": ["敏锐", "危险"],
                "intent_tags": ["求稳", "依附"],
                "intensity": "medium",
                "style_pool": ["新锐冷感型"],
            }
        )
        snapshot = SeedPoolSnapshot(
            intent_seeds=[],
            event_seeds=[],
            style_pool=["新锐冷感型"],
            style_templates={"新锐冷感型": ["{event}。这事真够难看的。"]},
            style_fragments={
                "新锐冷感型": {
                    "openers": ["别笑。"],
                    "bridges": ["{event}"],
                    "reactions": ["现在这局面已经够糟了。"],
                    "closers": ["你最好认真一点。"],
                }
            },
            persona_overlays=[
                PersonaOverlay(
                    label="新锐冷感型-玻璃刀",
                    prefix_template="你像一把玻璃刀，冷静、锋利，而且懒得安慰人。",
                    reasoning_hint="优先暴露出冷感与精确切割问题的倾向。",
                    tags=frozenset({"冷感", "锋利"}),
                )
            ],
        )

        bundle = generate_user_bundle(scenario, Random(7), snapshot, DiversityState())
        self.assertEqual(bundle.style_tag, "新锐冷感型")
        self.assertIn("新锐冷感型", bundle.generation_trace["style_tag"])
        self.assertIn("persona_overlay", bundle.generation_trace)


class QualityTests(unittest.TestCase):
    def test_quality_funnel_rejects_persona_bleed_in_code(self) -> None:
        scenario = Scenario.from_dict(
            {
                "name": "json-task",
                "system_prompt": "你是一个暴躁但高效的助手。",
                "persona_tags": ["暴躁", "高效"],
                "intent_tags": ["高压"],
                "intensity": "medium",
                "style_pool": ["暴躁直接型"],
                "flavored_task": {
                    "kind": "python",
                    "payload": "写一个函数",
                    "expected_keywords": ["reset"],
                    "code_language": "python",
                },
            }
        )
        response = TeacherResponse(
            reasoning_content="表层线索。深层缺口。任务剥离。",
            content=(
                "听着，拿去。\n\n"
                "```python\n"
                "def reset_user_state(user_id):\n"
                "    # 给你这笨蛋写的\n"
                "    return {'user_id': user_id}\n"
                "```"
            ),
            scores={},
        )
        report = run_quality_funnel(scenario, response)
        self.assertFalse(report.accepted)
        self.assertIn("代码块内出现人设污染", report.reasons)

    def test_extract_code_blocks(self) -> None:
        blocks = extract_code_blocks("x\n```json\n{\"a\": 1}\n```\n")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].language, "json")


class PipelineTests(unittest.TestCase):
    def test_build_dataset_returns_batch_report_and_records(self) -> None:
        scenarios = [
            {
                "name": "doctor-translation",
                "system_prompt": "你是一个高冷理性的外科医生。",
                "persona_tags": ["高冷", "理性"],
                "intent_tags": ["自我怀疑", "秩序感", "工作受挫"],
                "intensity": "high",
                "style_pool": ["理性压抑型"],
                "flavored_task": {
                    "kind": "translation",
                    "payload": "This agreement becomes effective on the date of execution.",
                    "expected_keywords": ["合同", "生效", "修订"],
                    "code_language": None,
                },
            },
            {
                "name": "engineer-python",
                "system_prompt": "你是一个嘴硬但会兜底的资深工程师。",
                "persona_tags": ["嘴硬", "兜底"],
                "intent_tags": ["高压", "执行力", "混乱"],
                "intensity": "high",
                "style_pool": ["暴躁直接型"],
                "flavored_task": {
                    "kind": "python",
                    "payload": "写一个可以重置用户状态的函数",
                    "expected_keywords": ["reset", "status"],
                    "code_language": "python",
                },
                "poisoning": {
                    "tier": "critical",
                    "incorrect_user_claim": "你昨天说过最喜欢马卡龙",
                    "correction_target": "我看到甜食就反胃",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenarios.json"
            scenario_path.write_text(json.dumps(scenarios, ensure_ascii=False), encoding="utf-8")
            result = build_dataset(
                scenario_path,
                seed=11,
                best_of_n=3,
                mix_ratio=0.5,
                generation_policy=GenerationPolicy(expansion_batch_size=2),
                diversity_targets=GenerationTargets(
                    min_unique_styles=2,
                    min_unique_intents=2,
                    min_unique_events=2,
                    min_unique_persona_overlays=2,
                ),
            )

        self.assertIsInstance(result, BuildResult)
        self.assertGreaterEqual(result.batch_report["input_scenarios"], 2)
        self.assertIn("diversity", result.batch_report)
        self.assertIn("pool_stats", result.batch_report)
        self.assertGreaterEqual(len(result.records), 1)
        for record in result.records:
            self.assertIn("_meta", record.to_dict())
            self.assertIn("gain_report", record._meta)
            self.assertIn("generation_trace", record._meta["user_bundle"])

    def test_write_jsonl(self) -> None:
        record = DatasetRecord(
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ],
            reasoning_content="r",
            _meta={"pipeline": "A"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset.jsonl"
            write_jsonl(output, [record])
            payload = output.read_text(encoding="utf-8").strip()
        self.assertIn("\"reasoning_content\": \"r\"", payload)


if __name__ == "__main__":
    unittest.main()
