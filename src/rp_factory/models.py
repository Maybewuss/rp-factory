from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class FlavoredTask:
    kind: str
    payload: str
    expected_keywords: list[str] = field(default_factory=list)
    code_language: str | None = None


@dataclass(slots=True)
class PoisoningSpec:
    tier: str
    incorrect_user_claim: str
    correction_target: str


@dataclass(slots=True)
class Scenario:
    name: str
    system_prompt: str
    persona_tags: list[str]
    intent_tags: list[str]
    intensity: str
    style_pool: list[str]
    flavored_task: FlavoredTask | None = None
    poisoning: PoisoningSpec | None = None
    strict_historical_sandbox: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        flavored = data.get("flavored_task")
        poisoning = data.get("poisoning")
        return cls(
            name=data["name"],
            system_prompt=data["system_prompt"],
            persona_tags=list(data.get("persona_tags", [])),
            intent_tags=list(data.get("intent_tags", [])),
            intensity=data.get("intensity", "medium"),
            style_pool=list(data.get("style_pool", [])),
            flavored_task=FlavoredTask(**flavored) if flavored else None,
            poisoning=PoisoningSpec(**poisoning) if poisoning else None,
            strict_historical_sandbox=bool(data.get("strict_historical_sandbox", False)),
        )


@dataclass(slots=True)
class UserBundle:
    deep_intent: str
    proxy_event: str
    style_tag: str
    user_message: str


@dataclass(slots=True)
class TeacherResponse:
    reasoning_content: str
    content: str
    scores: dict[str, float]


@dataclass(slots=True)
class QualityReport:
    accepted: bool
    reasons: list[str]
    scores: dict[str, float]


@dataclass(slots=True)
class DatasetRecord:
    messages: list[dict[str, str]]
    reasoning_content: str
    _meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "reasoning_content": self.reasoning_content,
            "_meta": self._meta,
        }


def dataclass_to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
