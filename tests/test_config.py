"""测试配置加载。"""


from rp_factory.config import FactoryConfig, load_config


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, FactoryConfig)
    assert cfg.llm.generator.model == "gpt-4o"
    assert cfg.pipeline.conversation_turns == 6


def test_load_nonexistent_fallback():
    cfg = load_config("/nonexistent/path.yaml")
    assert isinstance(cfg, FactoryConfig)
    assert cfg.llm.generator.model == "gpt-4o"


def test_config_pipeline_settings():
    cfg = load_config()
    assert cfg.rp_agent.pipeline_b.n_samples == 8
    assert cfg.rp_agent.mix_ratio["pipeline_a"] == 0.5


def test_config_quality_blacklist():
    cfg = load_config()
    patterns = cfg.quality.level1.persona_blacklist_patterns
    assert "作为AI" in patterns
    assert len(patterns) >= 4


def test_config_user_styles():
    cfg = load_config()
    styles = cfg.user_agent.iceberg.diversity.user_styles
    assert "暴躁直接型" in styles
    assert len(styles) >= 5


def test_config_context_control():
    cfg = load_config()
    assert cfg.context_control.flavored_task.injection_probability == 0.3
    assert cfg.context_control.memory_poisoning.injection_probability == 0.25
    assert 3 in cfg.context_control.flavored_task.injection_rounds


def test_config_evaluation_dimensions():
    cfg = load_config()
    dims = cfg.evaluation.dimensions
    assert "deep_need_recognition" in dims
    assert "persona_consistency" in dims
    assert len(dims) == 5
