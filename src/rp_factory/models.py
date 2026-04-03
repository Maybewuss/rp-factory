"""数据模型定义 — 贯穿整条数据合成管线的核心类型。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 基础消息类型（对齐 OpenAI messages 格式）
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str
    reasoning_content: str | None = None


# ---------------------------------------------------------------------------
# 冰山三层结构（User Agent 引擎）
# ---------------------------------------------------------------------------

class IcebergLayers(BaseModel):
    """冰山理论三层：深层动机 / 表面事件 / 加密输出。"""
    deep_intent: str = Field(..., description="冰山底部：深层心理动机")
    proxy_event: str = Field(..., description="冰山中间：掩盖动机的琐碎事件")
    user_message: str = Field(..., description="冰山顶部：最终用户台词")
    user_style: str | None = Field(None, description="注入的用户表达风格标签")


# ---------------------------------------------------------------------------
# 上下文扰动事件（语境控制模块）
# ---------------------------------------------------------------------------

class PerturbationType(str, Enum):
    FLAVORED_TASK = "flavored_task"
    MEMORY_POISON = "memory_poison"
    COGNITIVE_TRANSLATION = "cognitive_translation"


class MemoryPoisonSeverity(str, Enum):
    CRITICAL = "critical"
    TRIVIAL = "trivial"
    CORRECT = "correct"


class Perturbation(BaseModel):
    """在对话中注入的扰动事件。"""
    type: PerturbationType
    round_index: int = Field(..., description="注入发生的对话轮次")
    detail: str = Field(..., description="扰动具体描述")
    severity: MemoryPoisonSeverity | None = None


# ---------------------------------------------------------------------------
# 生成管线标签
# ---------------------------------------------------------------------------

class PipelineTag(str, Enum):
    PIPELINE_A = "pipeline_a"
    PIPELINE_B = "pipeline_b"


# ---------------------------------------------------------------------------
# 质检结果
# ---------------------------------------------------------------------------

class QualityVerdict(str, Enum):
    PASS = "pass"
    REJECT_PERSONA = "reject_persona"
    REJECT_TASK = "reject_task"


class QualityResult(BaseModel):
    verdict: QualityVerdict
    details: str = ""


# ---------------------------------------------------------------------------
# 评估维度得分
# ---------------------------------------------------------------------------

class EvalScores(BaseModel):
    deep_need_recognition: float = 0.0
    persona_consistency: float = 0.0
    flavored_task_completion: float = 0.0
    fact_correction: float = 0.0
    cognitive_translation: float = 0.0


# ---------------------------------------------------------------------------
# 单条落盘数据
# ---------------------------------------------------------------------------

class ConversationMeta(BaseModel):
    """溯源元数据（不进入训练）。"""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    system_prompt_source: str = ""
    pipeline: PipelineTag | None = None
    perturbations: list[Perturbation] = Field(default_factory=list)
    quality: QualityResult | None = None
    eval_scores: EvalScores | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ConversationRecord(BaseModel):
    """最终落盘的一条数据：纯净 messages + _meta。"""
    messages: list[Message]
    meta: ConversationMeta = Field(default_factory=ConversationMeta)

    model_config = {"populate_by_name": True}
