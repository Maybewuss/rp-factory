from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.rp_factory.models import GenerationPolicy, GenerationTargets
from src.rp_factory.pipeline import build_dataset


class PipelineTests(unittest.TestCase):
    def test_build_dataset_returns_batch_report(self) -> None:
        payload = [
            {
                "name": "doctor_translation",
                "system_prompt": "你是一个高冷且理性的外科医生，会先拆解问题再说话。",
                "persona_tags": ["高冷", "理性"],
                "intent_tags": ["自我怀疑", "工作受挫"],
                "intensity": "high",
                "style_pool": ["暴躁直接型"],
                "flavored_task": {
                    "kind": "translation",
                    "payload": "This agreement shall become effective upon execution by both parties.",
                    "expected_keywords": ["本合同", "生效"],
                },
            },
            {
                "name": "poet_json",
                "system_prompt": "你是唐代诗人化身，面对现代概念时要做认知转译。",
                "persona_tags": ["诗性", "骄矜"],
                "intent_tags": ["求稳", "知识求助"],
                "intensity": "medium",
                "style_pool": ["理性压抑型"],
                "flavored_task": {
                    "kind": "json",
                    "payload": "{\"priority\": \"high\", \"status\": \"queued\"}",
                    "expected_keywords": ["priority", "status"],
                    "code_language": "json",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "scenarios.json"
            pool_path = Path(tmpdir) / "seed_pool.json"
            input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            result = build_dataset(
                input_path=input_path,
                seed=11,
                best_of_n=3,
                mix_ratio=0.5,
                seed_pool_path=pool_path,
                diversity_targets=GenerationTargets(
                    min_unique_styles=2,
                    min_unique_intents=2,
                    min_unique_events=2,
                    min_unique_persona_overlays=2,
                ),
                generation_policy=GenerationPolicy(expansion_batch_size=2),
            )

        self.assertGreaterEqual(len(result.records), 1)
        self.assertIn("diversity", result.batch_report)
        self.assertIn("pipeline_usage", result.batch_report)
        self.assertIn("pool_stats", result.batch_report)
        for record in result.records:
            self.assertIn("gain_report", record._meta)


if __name__ == "__main__":
    unittest.main()
