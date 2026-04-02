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
    generation_trace: dict[str, Any] = field(default_factory=dict)


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


@dataclass(slots=True)
class DiversityState:
    max_recent: int = 6
    _recent_by_namespace: dict[str, list[str]] = field(default_factory=dict)
    _counts_by_namespace: dict[str, dict[str, int]] = field(default_factory=dict)

    def recent_items(self, namespace: str) -> list[str]:
        return list(self._recent_by_namespace.get(namespace, []))

    def usage_count(self, namespace: str, value: str) -> int:
        return self._counts_by_namespace.get(namespace, {}).get(value, 0)

    def remember(self, namespace: str, value: str) -> None:
        history = self._recent_by_namespace.setdefault(namespace, [])
        counts = self._counts_by_namespace.setdefault(namespace, {})
        history.append(value)
        counts[value] = counts.get(value, 0) + 1
        if len(history) > self.max_recent:
            history.pop(0)


@dataclass(slots=True)
class TaggedSeed:
    text: str
    tags: frozenset[str]
    intensity: str = "medium"
    source: str = "default"


@dataclass(slots=True)
class PersonaOverlay:
    label: str
    prefix_template: str
    reasoning_hint: str
    tags: frozenset[str]
    source: str = "default"


@dataclass(slots=True)
class SeedPoolSnapshot:
    intent_seeds: list[TaggedSeed]
    event_seeds: list[TaggedSeed]
    style_pool: list[str]
    style_templates: dict[str, list[str]]
    style_fragments: dict[str, dict[str, list[str]]]
    persona_overlays: list[PersonaOverlay]


@dataclass(slots=True)
class GenerationTargets:
    min_unique_styles: int = 3
    min_unique_intents: int = 3
    min_unique_events: int = 3
    min_unique_persona_overlays: int = 3


@dataclass(slots=True)
class GenerationPolicy:
    expansion_batch_size: int = 3
    max_generation_attempts: int = 4
