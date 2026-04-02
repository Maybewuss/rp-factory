"""测试语境控制与扰动引擎。"""

import pytest

from rp_factory.config import load_config
from rp_factory.context_control.perturbation import PerturbationEngine
from rp_factory.llm_client import LLMClient
from rp_factory.models import PerturbationType


@pytest.fixture
def engine():
    cfg = load_config()
    llm = LLMClient(cfg.llm.generator)
    return PerturbationEngine(cfg, llm)


class TestFlavoredTask:
    def test_pick_task(self, engine: PerturbationEngine):
        task = engine.pick_flavored_task()
        assert "prompt" in task
        assert isinstance(task["prompt"], str)
        assert len(task["prompt"]) > 0

    def test_create_task_message(self, engine: PerturbationEngine):
        msg, pert = engine.create_flavored_task_message(round_index=3)
        assert isinstance(msg, str)
        assert len(msg) > 0
        assert pert.type == PerturbationType.FLAVORED_TASK
        assert pert.round_index == 3

    def test_should_inject_task_wrong_round(self, engine: PerturbationEngine):
        assert not engine.should_inject_task(0)
        assert not engine.should_inject_task(1)


class TestCognitiveTranslation:
    def test_detects_historical_persona(self, engine: PerturbationEngine):
        assert engine.needs_cognitive_translation("你是一个唐代诗人")
        assert engine.needs_cognitive_translation("你是一位古代武将")
        assert not engine.needs_cognitive_translation("你是一个现代程序员")

    def test_strict_sandbox_overrides(self, engine: PerturbationEngine):
        prompt = "你是一个唐代诗人 <strict_historical_sandbox>"
        assert not engine.needs_cognitive_translation(prompt)


class TestSeedLoading:
    def test_flavored_tasks_loaded(self, engine: PerturbationEngine):
        assert len(engine._task_seeds) > 0

    def test_multiple_task_types(self, engine: PerturbationEngine):
        types = {cat.get("type") for cat in engine._task_seeds}
        assert "translation" in types
        assert "code_writing" in types
