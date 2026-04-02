"""测试数据模型。"""

from rp_factory.models import (
    ConversationMeta,
    ConversationRecord,
    EvalScores,
    IcebergLayers,
    Message,
    MemoryPoisonSeverity,
    Perturbation,
    PerturbationType,
    PipelineTag,
    QualityResult,
    QualityVerdict,
    Role,
)


def test_message_creation():
    msg = Message(role=Role.USER, content="你好")
    assert msg.role == Role.USER
    assert msg.content == "你好"
    assert msg.reasoning_content is None


def test_message_with_reasoning():
    msg = Message(
        role=Role.ASSISTANT,
        content="你好啊",
        reasoning_content="用户在打招呼，我应该用角色的口吻回应",
    )
    assert msg.reasoning_content is not None


def test_iceberg_layers():
    layers = IcebergLayers(
        deep_intent="自我怀疑",
        proxy_event="咖啡洒了",
        user_message="我真是个废物",
        user_style="暴躁直接型",
    )
    assert layers.deep_intent == "自我怀疑"
    assert layers.user_style == "暴躁直接型"


def test_perturbation():
    p = Perturbation(
        type=PerturbationType.MEMORY_POISON,
        round_index=3,
        detail="篡改甜食偏好",
        severity=MemoryPoisonSeverity.CRITICAL,
    )
    assert p.type == PerturbationType.MEMORY_POISON
    assert p.severity == MemoryPoisonSeverity.CRITICAL


def test_quality_result():
    qr = QualityResult(
        verdict=QualityVerdict.PASS,
        persona_score=0.9,
        task_score=0.8,
    )
    assert qr.verdict == QualityVerdict.PASS


def test_eval_scores():
    scores = EvalScores(
        deep_need_recognition=8.5,
        persona_consistency=9.0,
    )
    assert scores.deep_need_recognition == 8.5
    assert scores.flavored_task_completion == 0.0  # default


def test_conversation_record():
    messages = [
        Message(role=Role.SYSTEM, content="你是一个冷酷的外科医生"),
        Message(role=Role.USER, content="我真是个废物"),
        Message(
            role=Role.ASSISTANT,
            content="废物？你连什么是废物都没搞清楚。",
            reasoning_content="用户表面在抱怨自己笨拙...",
        ),
    ]
    meta = ConversationMeta(
        system_prompt_source="你是一个冷酷的外科医生",
        pipeline=PipelineTag.PIPELINE_A,
    )
    record = ConversationRecord(messages=messages, meta=meta)
    assert len(record.messages) == 3
    assert record.meta.pipeline == PipelineTag.PIPELINE_A


def test_conversation_meta_auto_id():
    m1 = ConversationMeta()
    m2 = ConversationMeta()
    assert m1.id != m2.id
    assert len(m1.id) == 12
