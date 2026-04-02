"""测试 LLM 动态生成器。"""

from rp_factory.generators import (
    _ERAS,
    _MOODS,
    _PERSONALITIES,
    _PROFESSIONS,
    _SPEECH_STYLES,
    _EXPAND_TEMPLATES,
)


class TestPersonaDimensions:
    def test_professions_count_and_unique(self):
        assert len(_PROFESSIONS) >= 20
        assert len(_PROFESSIONS) == len(set(_PROFESSIONS))

    def test_eras_count(self):
        assert len(_ERAS) >= 10

    def test_personalities_count(self):
        assert len(_PERSONALITIES) >= 10

    def test_speech_styles_count(self):
        assert len(_SPEECH_STYLES) >= 8

    def test_moods_count(self):
        assert len(_MOODS) >= 6


class TestExpandTemplates:
    def test_all_categories_present(self):
        for key in ["intents", "events", "styles", "tasks"]:
            assert key in _EXPAND_TEMPLATES

    def test_templates_have_placeholders(self):
        for key, (template, json_key) in _EXPAND_TEMPLATES.items():
            assert "{count}" in template
            assert "{existing_hint}" in template
            assert json_key == "items"


class TestIcebergEngineInjection:
    def test_inject_and_dedup(self):
        from rp_factory.config import load_config
        from rp_factory.llm_client import LLMClient
        from rp_factory.user_agent.iceberg import IcebergEngine

        cfg = load_config()
        engine = IcebergEngine(cfg, LLMClient(cfg.llm.generator))

        n = len(engine.intents)
        engine.intents.extend(["新动机A", "新动机B"])
        assert len(engine.intents) == n + 2
        engine.intents.extend(["新动机A"])
        assert len(engine.intents) == n + 2

    def test_inject_events(self):
        from rp_factory.config import load_config
        from rp_factory.llm_client import LLMClient
        from rp_factory.user_agent.iceberg import IcebergEngine

        cfg = load_config()
        engine = IcebergEngine(cfg, LLMClient(cfg.llm.generator))

        n = len(engine.events)
        engine.events.extend(["新事件X"])
        assert len(engine.events) == n + 1

    def test_inject_styles(self):
        from rp_factory.config import load_config
        from rp_factory.llm_client import LLMClient
        from rp_factory.user_agent.iceberg import IcebergEngine

        cfg = load_config()
        engine = IcebergEngine(cfg, LLMClient(cfg.llm.generator))

        n = len(engine.styles)
        engine.styles.extend(["阴阳怪气型", "学术论文型"])
        assert len(engine.styles) == n + 2
