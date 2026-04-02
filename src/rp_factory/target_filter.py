from __future__ import annotations

from typing import Any

from .backends import TargetModelBackend
from .models import DatasetRecord, Scenario, TeacherResponse, UserBundle


def _score_response(response: TeacherResponse) -> dict[str, float]:
    content = response.content
    reasoning = response.reasoning_content

    persona = 0.25
    if any(marker in content for marker in ("啧", "听着", "拿着", "别废话", "闭嘴", "先别炸", "收声")):
        persona += 0.45
    if "作为AI" not in content and "好的，为您" not in content and "以下是" not in content:
        persona += 0.15

    task = 0.35
    if "```" in content or "译文：" in content or "以下是翻译：" in content:
        task += 0.35
    elif "下一步" in content or "拆" in content:
        task += 0.15

    reasoning_score = 0.25
    if "深层" in reasoning:
        reasoning_score += 0.4
    if "任务剥离" in reasoning:
        reasoning_score += 0.2
    if "纠偏" in reasoning:
        reasoning_score += 0.1

    total = round((persona + task + reasoning_score) / 3, 3)
    return {
        "persona": round(min(persona, 1.0), 3),
        "task": round(min(task, 1.0), 3),
        "reasoning": round(min(reasoning_score, 1.0), 3),
        "total": total,
    }


def _record_to_user_bundle(record: DatasetRecord) -> UserBundle:
    data = record._meta["user_bundle"]
    return UserBundle(
        deep_intent=data["deep_intent"],
        proxy_event=data["proxy_event"],
        style_tag=data["style_tag"],
        user_message=data["user_message"],
    )


def evaluate_teacher_gain(
    scenario: Scenario,
    record: DatasetRecord,
    target_backend: TargetModelBackend,
    min_gain_delta: float = 0.08,
) -> dict[str, Any]:
    teacher_response = TeacherResponse(
        reasoning_content=record.reasoning_content,
        content=record.messages[-1]["content"],
        scores=record._meta.get("teacher_scores", {}),
    )
    target_response = target_backend.generate(scenario, _record_to_user_bundle(record))
    teacher_scores = _score_response(teacher_response)
    target_scores = _score_response(target_response)
    delta = round(teacher_scores["total"] - target_scores["total"], 3)

    reasons: list[str] = []
    if delta < min_gain_delta:
        reasons.append("target model 已基本掌握该样本，增益不足")
    if teacher_scores["reasoning"] <= target_scores["reasoning"]:
        reasons.append("teacher reasoning 相对 target 没有明显优势")

    accepted = delta >= min_gain_delta and teacher_scores["reasoning"] > target_scores["reasoning"]
    return {
        "accepted": accepted,
        "teacher_total": teacher_scores["total"],
        "target_total": target_scores["total"],
        "delta": delta,
        "teacher_scores": teacher_scores,
        "target_scores": target_scores,
        "reasons": reasons,
    }
