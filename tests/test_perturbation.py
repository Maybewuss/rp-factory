"""测试语境控制与扰动引擎。"""

import pytest

from rp_factory.config import load_config
from rp_factory.context_control.perturbation import PerturbationEngine
from rp_factory.llm_client import LLMClient
from rp_factory.models import PerturbationType


@pytest.fixture
def engine():
    cfg = load_config()
    return PerturbationEngine(cfg, LLMClient(cfg.llm.generator))


class TestFlavoredTask:
    def test_create_task_message(self, engine: PerturbationEngine):
        msg, pert = engine.create_flavored_task_message(round_index=3)
        assert isinstance(msg, str) and len(msg) > 0
        assert pert.type == PerturbationType.FLAVORED_TASK

    def test_should_inject_task_wrong_round(self, engine: PerturbationEngine):
        assert not engine.should_inject_task(0)
        assert not engine.should_inject_task(1)


class TestCognitiveTranslation:
    def test_detects_historical_persona(self, engine: PerturbationEngine):
        assert engine.needs_cognitive_translation("你是一个唐代诗人")
        assert engine.needs_cognitive_translation("你是一位古代武将")
        assert not engine.needs_cognitive_translation("你是一个现代程序员")

    def test_strict_sandbox_overrides(self, engine: PerturbationEngine):
        assert not engine.needs_cognitive_translation("你是唐代诗人 <strict_historical_sandbox>")


class TestSeedLoading:
    def test_tasks_loaded(self, engine: PerturbationEngine):
        assert len(engine.tasks) > 0

    def test_multiple_task_types(self, engine: PerturbationEngine):
        types = {t.get("type") for t in engine.tasks.items}
        assert "translation" in types
        assert "code_writing" in types
