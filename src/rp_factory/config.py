"""全局配置加载与管理。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "default.yaml"


# ---------------------------------------------------------------------------
# LLM 配置
# ---------------------------------------------------------------------------

class LLMEndpoint(BaseModel):
    model: str = "gpt-4o"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 2048

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


class LLMConfig(BaseModel):
    generator: LLMEndpoint = Field(default_factory=LLMEndpoint)
    teacher: LLMEndpoint = Field(default_factory=LLMEndpoint)
    mentor: LLMEndpoint = Field(default_factory=LLMEndpoint)
    target: LLMEndpoint = Field(default_factory=LLMEndpoint)


# ---------------------------------------------------------------------------
# 各模块配置
# ---------------------------------------------------------------------------

class DiversityConfig(BaseModel):
    enable_style_injection: bool = True
    user_styles: list[str] = Field(default_factory=lambda: [
        "话少冷淡型", "碎碎念型", "暴躁直接型", "故作轻松型", "理性压抑型",
    ])
    ngram_collapse_threshold: float = 0.7


class IcebergConfig(BaseModel):
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)


class UserAgentConfig(BaseModel):
    iceberg: IcebergConfig = Field(default_factory=IcebergConfig)


class FlavoredTaskConfig(BaseModel):
    injection_probability: float = 0.3
    injection_rounds: list[int] = Field(default_factory=lambda: [3, 4, 5])
    task_types: list[str] = Field(default_factory=lambda: [
        "translation", "code_writing", "summarization", "knowledge_qa",
    ])


class MemoryPoisoningConfig(BaseModel):
    injection_probability: float = 0.25
    severity_distribution: dict[str, float] = Field(default_factory=lambda: {
        "critical": 0.3, "trivial": 0.4, "correct": 0.3,
    })


class CognitiveTranslationConfig(BaseModel):
    default_route: str = "route_two"
    strict_sandbox_tag: str = "<strict_historical_sandbox>"


class ContextControlConfig(BaseModel):
    flavored_task: FlavoredTaskConfig = Field(default_factory=FlavoredTaskConfig)
    memory_poisoning: MemoryPoisoningConfig = Field(default_factory=MemoryPoisoningConfig)
    cognitive_translation: CognitiveTranslationConfig = Field(default_factory=CognitiveTranslationConfig)


class PipelineAConfig(BaseModel):
    enabled: bool = True
    perplexity_check: bool = True
    perplexity_threshold: float = 1.5


class PipelineBConfig(BaseModel):
    enabled: bool = True
    n_samples: int = 8
    temperature: float = 0.9
    max_concurrent: int = 4


class RPAgentConfig(BaseModel):
    pipeline_a: PipelineAConfig = Field(default_factory=PipelineAConfig)
    pipeline_b: PipelineBConfig = Field(default_factory=PipelineBConfig)
    mix_ratio: dict[str, float] = Field(default_factory=lambda: {
        "pipeline_a": 0.5, "pipeline_b": 0.5,
    })


class Level1QualityConfig(BaseModel):
    persona_blacklist_patterns: list[str] = Field(default_factory=lambda: [
        "作为AI", "好的，为您", "如果您还需要", "很高兴为您",
    ])
    payload_lint_enabled: bool = True


class Level2QualityConfig(BaseModel):
    enabled: bool = True
    reasoning_depth_check: bool = True


class QualityConfig(BaseModel):
    level1: Level1QualityConfig = Field(default_factory=Level1QualityConfig)
    level2: Level2QualityConfig = Field(default_factory=Level2QualityConfig)


class EvaluationConfig(BaseModel):
    sample_ratio: float = 0.1
    dimensions: list[str] = Field(default_factory=lambda: [
        "deep_need_recognition", "persona_consistency",
        "flavored_task_completion", "fact_correction", "cognitive_translation",
    ])


class OutputConfig(BaseModel):
    format: str = "jsonl"
    output_dir: str = "output"
    include_meta: bool = True
    compress: bool = False


class PipelineRunConfig(BaseModel):
    conversation_turns: int = 6
    batch_size: int = 10
    max_concurrent_conversations: int = 5


# ---------------------------------------------------------------------------
# 顶层配置
# ---------------------------------------------------------------------------

class FactoryConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    user_agent: UserAgentConfig = Field(default_factory=UserAgentConfig)
    context_control: ContextControlConfig = Field(default_factory=ContextControlConfig)
    rp_agent: RPAgentConfig = Field(default_factory=RPAgentConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    pipeline: PipelineRunConfig = Field(default_factory=PipelineRunConfig)


def load_config(path: str | Path | None = None) -> FactoryConfig:
    """从 YAML 加载配置，缺省值自动补齐。"""
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return FactoryConfig.model_validate(raw)
    return FactoryConfig()
