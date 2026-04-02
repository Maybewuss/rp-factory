"""测试 Prompt 模板池。"""

import pytest

from rp_factory.prompt_pool import PromptPool, build_prompt_pools


class TestPromptPool:
    def test_single_variant(self):
        pool = PromptPool(["只有一个"])
        assert pool.pick() == "只有一个"

    def test_multiple_variants_all_returned(self):
        variants = ["A", "B", "C"]
        pool = PromptPool(variants)
        picked = {pool.pick() for _ in range(10)}
        assert picked == {"A", "B", "C"}

    def test_no_repeat_until_exhausted(self):
        variants = ["A", "B", "C"]
        pool = PromptPool(variants)
        first_round = [pool.pick() for _ in range(3)]
        assert len(set(first_round)) == 3

    def test_reset_allows_re_pick(self):
        pool = PromptPool(["X", "Y"])
        pool.pick()
        pool.pick()
        pool.reset()
        result = pool.pick()
        assert result in ("X", "Y")

    def test_add_variant(self):
        pool = PromptPool(["A"])
        pool.add_variant("B")
        assert len(pool) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            PromptPool([])


class TestBuildPromptPools:
    def test_all_keys_present(self):
        pools = build_prompt_pools()
        expected_keys = [
            "step1_reverse_intent", "step2_event_anchoring",
            "step3_obfuscated_generation", "mentor_hint",
            "memory_poison", "cognitive_translation", "continuation",
        ]
        for key in expected_keys:
            assert key in pools, f"缺少 prompt pool: {key}"

    def test_multiple_variants_per_pool(self):
        pools = build_prompt_pools()
        for key, pool in pools.items():
            assert len(pool) >= 3, f"{key} 变体数不足 3: {len(pool)}"

    def test_step1_variants_contain_json_instruction(self):
        pools = build_prompt_pools()
        pool = pools["step1_reverse_intent"]
        for _ in range(len(pool)):
            text = pool.pick()
            assert "deep_intent" in text

    def test_step3_variants_contain_placeholders(self):
        pools = build_prompt_pools()
        pool = pools["step3_obfuscated_generation"]
        pool.reset()
        for _ in range(len(pool)):
            text = pool.pick()
            assert "{proxy_event}" in text
            assert "{deep_intent}" in text
            assert "{user_style}" in text

    def test_continuation_variants_contain_placeholders(self):
        pools = build_prompt_pools()
        pool = pools["continuation"]
        pool.reset()
        for _ in range(len(pool)):
            text = pool.pick()
            assert "{history}" in text
            assert "{deep_intent}" in text
            assert "{user_style}" in text
