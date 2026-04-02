"""RP 数据合成工厂 v3.2 — 基于认知仿真与对偶动机的训练数据生成系统。"""

__version__ = "3.2.0"

from rp_factory.config import FactoryConfig, load_config
from rp_factory.generators import PersonaGenerator, SeedExpander
from rp_factory.models import (
    ConversationMeta,
    ConversationRecord,
    EvalScores,
    IcebergLayers,
    Message,
    Perturbation,
    PipelineTag,
    QualityResult,
    QualityVerdict,
    Role,
)
from rp_factory.pipeline import DataFactory

__all__ = [
    "DataFactory",
    "FactoryConfig",
    "load_config",
    "PersonaGenerator",
    "SeedExpander",
    "ConversationMeta",
    "ConversationRecord",
    "EvalScores",
    "IcebergLayers",
    "Message",
    "Perturbation",
    "PipelineTag",
    "QualityResult",
    "QualityVerdict",
    "Role",
]
