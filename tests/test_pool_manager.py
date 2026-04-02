from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from random import Random

from src.rp_factory.models import DiversityState, GenerationTargets, Scenario
from src.rp_factory.pool_manager import PoolManager, RuleBasedExpansionBackend
from src.rp_factory.user_generation import generate_user_bundle


class PoolManagerTests(unittest.TestCase):
    def test_expand_pool_persists_new_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_path = Path(tmpdir) / "seed_pool.json"
            manager = PoolManager(pool_path)
            snapshot = manager.load()
            scenario = Scenario.from_dict(
                {
                    "name": "expand",
                    "system_prompt": "你是一个高冷顾问。",
                    "persona_tags": ["高冷", "克制"],
                    "intent_tags": ["自我怀疑", "高压"],
                    "intensity": "high",
                    "style_pool": ["理性压抑型", "暴躁直接型"],
                }
            )
            state = DiversityState()
            expanded = manager.expand_pool(
                scenario=scenario,
                state=state,
                snapshot=snapshot,
                targets=GenerationTargets(
                    min_unique_intents=len(snapshot.intent_seeds) + 2,
                    min_unique_events=len(snapshot.event_seeds) + 2,
                    min_unique_styles=2,
                    min_unique_persona_overlays=2,
                ),
                rng=Random(7),
            )
            reloaded = manager.load()

        self.assertGreaterEqual(len(expanded.intent_seeds), len(snapshot.intent_seeds) + 2)
        self.assertEqual(len(expanded.intent_seeds), len(reloaded.intent_seeds))
        self.assertIn("理性压抑型", reloaded.style_fragments)

    def test_generated_bundle_can_use_expanded_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_path = Path(tmpdir) / "seed_pool.json"
            pool_path.write_text(
                json.dumps(
                    {
                        "intent_seeds": [
                            {
                                "text": "用户在公开场合被打断后持续反刍，表面不说，心里已经开始怀疑自己是不是永远不值得被认真听完。",
                                "tags": ["自我怀疑", "公开尴尬"],
                                "intensity": "high",
                            }
                        ],
                        "event_seeds": [
                            {
                                "text": "地铁闸机刷了三次都不开，后面一排人盯着看。",
                                "tags": ["公开尴尬", "日常", "高压"],
                                "intensity": "high",
                            }
                        ],
                        "style_pool": ["理性压抑型"],
                        "style_templates": {"理性压抑型": []},
                        "style_fragments": {
                            "理性压抑型": {
                                "openers": ["从结果看，"],
                                "bridges": ["{event}"],
                                "reactions": ["这件事本身不致命，但我能感觉到自己已经在失控边缘。"],
                                "closers": ["我现在没有多少缓冲。"],
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            manager = PoolManager(pool_path, expansion_backend=RuleBasedExpansionBackend())
            snapshot = manager.load()
            scenario = Scenario.from_dict(
                {
                    "name": "use-expanded",
                    "system_prompt": "你是一个高冷顾问。",
                    "persona_tags": ["高冷", "克制"],
                    "intent_tags": ["自我怀疑", "公开尴尬"],
                    "intensity": "high",
                    "style_pool": ["理性压抑型"],
                }
            )
            bundle = generate_user_bundle(
                scenario=scenario,
                rng=Random(3),
                seed_pool=snapshot,
                diversity_state=DiversityState(),
            )

        self.assertIn("地铁闸机", bundle.user_message)


if __name__ == "__main__":
    unittest.main()
