"""测试 LLM 动态生成器的结构和维度数据。"""

from rp_factory.generators import (
    _ERAS,
    _MOODS,
    _PERSONALITIES,
    _PROFESSIONS,
    _SPEECH_STYLES,
)


class TestPersonaDimensions:
    """验证角色生成的维度池足够丰富且无重复。"""

    def test_professions_count(self):
        assert len(_PROFESSIONS) >= 20

    def test_professions_unique(self):
        assert len(_PROFESSIONS) == len(set(_PROFESSIONS))

    def test_eras_count(self):
        assert len(_ERAS) >= 10

    def test_eras_unique(self):
        assert len(_ERAS) == len(set(_ERAS))

    def test_personalities_count(self):
        assert len(_PERSONALITIES) >= 10

    def test_personalities_unique(self):
        assert len(_PERSONALITIES) == len(set(_PERSONALITIES))

    def test_speech_styles_count(self):
        assert len(_SPEECH_STYLES) >= 8

    def test_moods_count(self):
        assert len(_MOODS) >= 6


class TestPromptTemplateStructure:
    """验证 Prompt 模板的占位符完整性。"""

    def test_persona_gen_system_has_placeholders(self):
        from rp_factory.generators import PERSONA_GEN_SYSTEM
        assert "{count}" in PERSONA_GEN_SYSTEM
        assert "{existing_hint}" in PERSONA_GEN_SYSTEM

    def test_persona_single_system_has_constraint(self):
        from rp_factory.generators import PERSONA_SINGLE_SYSTEM
        assert "{constraint}" in PERSONA_SINGLE_SYSTEM

    def test_expand_intents_system_has_placeholders(self):
        from rp_factory.generators import EXPAND_INTENTS_SYSTEM
        assert "{count}" in EXPAND_INTENTS_SYSTEM
        assert "{existing_hint}" in EXPAND_INTENTS_SYSTEM

    def test_expand_events_system_has_placeholders(self):
        from rp_factory.generators import EXPAND_EVENTS_SYSTEM
        assert "{count}" in EXPAND_EVENTS_SYSTEM

    def test_expand_styles_system_has_placeholders(self):
        from rp_factory.generators import EXPAND_STYLES_SYSTEM
        assert "{count}" in EXPAND_STYLES_SYSTEM

    def test_expand_tasks_system_has_placeholders(self):
        from rp_factory.generators import EXPAND_TASKS_SYSTEM
        assert "{count}" in EXPAND_TASKS_SYSTEM


class TestIcebergEngineInjection:
    """测试 IcebergEngine 的运行时注入能力。"""

    def test_inject_intents(self):
        from rp_factory.config import load_config
        from rp_factory.llm_client import LLMClient
        from rp_factory.user_agent.iceberg import IcebergEngine

        cfg = load_config()
        llm = LLMClient(cfg.llm.generator)
        engine = IcebergEngine(cfg, llm)

        original_count = len(engine._intent_texts)
        added = engine.inject_intents(["新动机A", "新动机B"])
        assert added == 2
        assert len(engine._intent_texts) == original_count + 2

    def test_inject_dedup(self):
        from rp_factory.config import load_config
        from rp_factory.llm_client import LLMClient
        from rp_factory.user_agent.iceberg import IcebergEngine

        cfg = load_config()
        llm = LLMClient(cfg.llm.generator)
        engine = IcebergEngine(cfg, llm)

        engine.inject_intents(["唯一动机"])
        added = engine.inject_intents(["唯一动机"])
        assert added == 0

    def test_inject_events(self):
        from rp_factory.config import load_config
        from rp_factory.llm_client import LLMClient
        from rp_factory.user_agent.iceberg import IcebergEngine

        cfg = load_config()
        llm = LLMClient(cfg.llm.generator)
        engine = IcebergEngine(cfg, llm)

        original_count = len(engine._event_texts)
        added = engine.inject_events(["新事件X"])
        assert added == 1
        assert len(engine._event_texts) == original_count + 1

    def test_inject_styles(self):
        from rp_factory.config import load_config
        from rp_factory.llm_client import LLMClient
        from rp_factory.user_agent.iceberg import IcebergEngine

        cfg = load_config()
        llm = LLMClient(cfg.llm.generator)
        engine = IcebergEngine(cfg, llm)

        original_count = len(engine._styles)
        added = engine.inject_styles(["阴阳怪气型", "学术论文型"])
        assert added == 2
        assert len(engine._styles) == original_count + 2


class TestDiversityGuardCallback:
    """测试 DiversityGuard 的回调注册。"""

    def test_register_callback(self):
        from rp_factory.diversity import DiversityGuard

        guard = DiversityGuard()
        called = []

        async def fake_cb(cat: str) -> None:
            called.append(cat)

        guard.register_expand_callback("intent", fake_cb)
        assert "intent" in guard._expand_callbacks
