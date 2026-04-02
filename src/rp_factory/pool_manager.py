from __future__ import annotations

import json
from pathlib import Path
from random import Random

from .backends import LlmExpansionGenerator
from .models import DiversityState, GenerationTargets, PersonaOverlay, Scenario, SeedPoolSnapshot, TaggedSeed
from .seeds import build_snapshot


def _seed_to_dict(seed: TaggedSeed) -> dict[str, object]:
    return {
        "text": seed.text,
        "tags": sorted(seed.tags),
        "intensity": seed.intensity,
        "source": seed.source,
    }


def _seed_from_dict(payload: dict[str, object]) -> TaggedSeed:
    return TaggedSeed(
        text=str(payload["text"]),
        tags=frozenset(str(tag) for tag in payload.get("tags", [])),
        intensity=str(payload.get("intensity", "medium")),
        source=str(payload.get("source", "dynamic")),
    )


def _overlay_to_dict(overlay: PersonaOverlay) -> dict[str, object]:
    return {
        "label": overlay.label,
        "prefix_template": overlay.prefix_template,
        "reasoning_hint": overlay.reasoning_hint,
        "tags": sorted(overlay.tags),
        "source": overlay.source,
    }


def _overlay_from_dict(payload: dict[str, object]) -> PersonaOverlay:
    return PersonaOverlay(
        label=str(payload["label"]),
        prefix_template=str(payload["prefix_template"]),
        reasoning_hint=str(payload["reasoning_hint"]),
        tags=frozenset(str(tag) for tag in payload.get("tags", [])),
        source=str(payload.get("source", "dynamic")),
    )


class RuleBasedExpansionBackend:
    _intent_leads = (
        "用户真正崩掉的不是表面事故，",
        "用户嘴上在骂眼前的小事，",
        "用户表面上看起来只是在烦躁，",
    )
    _intent_needs = (
        "而是在确认有没有谁还能接住自己。",
        "但深层其实在索取秩序感和被稳稳兜住的感觉。",
        "本质上是在试探一个可靠对象会不会在关键时刻掉链子。",
    )
    _event_leads = (
        "刚准备喘口气，",
        "本来只是普通一天，偏偏",
        "临出门前，",
        "都快结束了，结果",
    )
    _event_bodies = (
        "门卡突然失灵，被堵在公司闸机前。",
        "耳机一边没声，会议开始前最后几分钟彻底乱套。",
        "刚打印好的文件被水杯边缘蹭湿了一大片。",
        "导航临门一脚失灵，在陌生路口连续绕错两圈。",
    )
    _overlay_prefixes = (
        "{marker}，{persona}归{persona}，我看得出来你不是在为这点事本身发疯。",
        "{marker}，{persona}归{persona}，你现在爆掉的不是事故，是那口气彻底断了。",
    )
    _overlay_hints = (
        "回应时把焦点从表层事故剥到失控感与秩序崩塌。",
        "用更强的角色立场包裹安抚，不要滑成中性助手。",
    )
    _style_slot_defaults = {
        "openers": ("先别炸。", "我知道。", "行吧。"),
        "bridges": ("{event}", "偏偏是 {event}", "结果就卡在 {event}"),
        "reactions": ("这一下不是小事，是最后那根线断了。", "我看得出来你已经快没缓冲了。", "这事把你最后那点稳定感也扯掉了。"),
        "closers": ("你先别急着给自己判死刑。", "先把气稳住。", "这口气先别全吞下去。"),
    }

    def expand(
        self,
        scenario: Scenario,
        state: DiversityState,
        snapshot: SeedPoolSnapshot,
        targets: GenerationTargets,
        rng: Random,
    ) -> SeedPoolSnapshot:
        intent_seeds = list(snapshot.intent_seeds)
        event_seeds = list(snapshot.event_seeds)
        style_pool = list(snapshot.style_pool)
        style_templates = {key: list(value) for key, value in snapshot.style_templates.items()}
        style_fragments = {
            style: {part: list(options) for part, options in parts.items()}
            for style, parts in snapshot.style_fragments.items()
        }
        persona_overlays = list(snapshot.persona_overlays)

        intent_seen = {seed.text for seed in intent_seeds}
        while len(intent_seeds) < targets.min_unique_intents:
            text = f"{rng.choice(self._intent_leads)}{rng.choice(self._intent_needs)}"
            if text in intent_seen:
                continue
            intent_seeds.append(
                TaggedSeed(
                    text=text,
                    tags=frozenset(scenario.intent_tags or {"求稳"}),
                    intensity=scenario.intensity,
                    source="rule_expansion",
                )
            )
            intent_seen.add(text)

        event_seen = {seed.text for seed in event_seeds}
        while len(event_seeds) < targets.min_unique_events:
            text = f"{rng.choice(self._event_leads)}{rng.choice(self._event_bodies)}"
            if text in event_seen:
                continue
            event_seeds.append(
                TaggedSeed(
                    text=text,
                    tags=frozenset(set(scenario.intent_tags) | {"动态扩池"}),
                    intensity=scenario.intensity,
                    source="rule_expansion",
                )
            )
            event_seen.add(text)

        for style in scenario.style_pool:
            if style not in style_pool:
                style_pool.append(style)
            style_templates.setdefault(style, [])
            style_fragments.setdefault(style, {"openers": [], "bridges": [], "reactions": [], "closers": []})
            for slot, defaults in self._style_slot_defaults.items():
                bucket = style_fragments[style].setdefault(slot, [])
                if not bucket:
                    bucket.extend(defaults)

        overlay_seen = {overlay.label for overlay in persona_overlays}
        while len(persona_overlays) < targets.min_unique_persona_overlays:
            marker = rng.choice(("听着", "啧", "先别炸", "行了"))
            persona = rng.choice(scenario.persona_tags or ["冷静"])
            label = f"{persona}-{len(persona_overlays) + 1}"
            if label in overlay_seen:
                continue
            persona_overlays.append(
                PersonaOverlay(
                    label=label,
                    prefix_template=rng.choice(self._overlay_prefixes).format(marker=marker, persona=persona),
                    reasoning_hint=rng.choice(self._overlay_hints),
                    tags=frozenset(scenario.persona_tags or {persona}),
                    source="rule_expansion",
                )
            )
            overlay_seen.add(label)

        return SeedPoolSnapshot(
            intent_seeds=intent_seeds,
            event_seeds=event_seeds,
            style_pool=style_pool,
            style_templates=style_templates,
            style_fragments=style_fragments,
            persona_overlays=persona_overlays,
        )


class LLMAssistedExpansionBackend(RuleBasedExpansionBackend):
    def __init__(self, generator: LlmExpansionGenerator | None = None) -> None:
        self.generator = generator

    def expand(
        self,
        scenario: Scenario,
        state: DiversityState,
        snapshot: SeedPoolSnapshot,
        targets: GenerationTargets,
        rng: Random,
    ) -> SeedPoolSnapshot:
        # 当前仓库保留可插拔协议；若注入真实 LLM，可在这里把 prompt 发给模型，
        # 再将返回的 intent / event / persona overlay / style fragments 合并进 snapshot。
        return super().expand(scenario, state, snapshot, targets, rng)


class PoolManager:
    def __init__(self, pool_path: str | Path, expansion_backend: object | None = None) -> None:
        self.pool_path = Path(pool_path)
        self.expansion_backend = expansion_backend or RuleBasedExpansionBackend()

    def _ensure_file(self) -> None:
        if self.pool_path.exists():
            return
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        self.save(build_snapshot())

    def load(self) -> SeedPoolSnapshot:
        self._ensure_file()
        payload = json.loads(self.pool_path.read_text(encoding="utf-8"))
        return SeedPoolSnapshot(
            intent_seeds=[_seed_from_dict(item) for item in payload.get("intent_seeds", [])],
            event_seeds=[_seed_from_dict(item) for item in payload.get("event_seeds", [])],
            style_pool=list(payload.get("style_pool", [])),
            style_templates={key: list(value) for key, value in payload.get("style_templates", {}).items()},
            style_fragments={
                style: {part: list(options) for part, options in parts.items()}
                for style, parts in payload.get("style_fragments", {}).items()
            },
            persona_overlays=[
                _overlay_from_dict(item) for item in payload.get("persona_overlays", [])
            ],
        )

    def save(self, snapshot: SeedPoolSnapshot) -> None:
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "intent_seeds": [_seed_to_dict(seed) for seed in snapshot.intent_seeds],
            "event_seeds": [_seed_to_dict(seed) for seed in snapshot.event_seeds],
            "style_pool": snapshot.style_pool,
            "style_templates": snapshot.style_templates,
            "style_fragments": snapshot.style_fragments,
            "persona_overlays": [_overlay_to_dict(overlay) for overlay in snapshot.persona_overlays],
        }
        self.pool_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def expand_pool(
        self,
        scenario: Scenario,
        state: DiversityState,
        snapshot: SeedPoolSnapshot,
        targets: GenerationTargets,
        rng: Random,
    ) -> SeedPoolSnapshot:
        expanded = self.expansion_backend.expand(
            scenario=scenario,
            state=state,
            snapshot=snapshot,
            targets=targets,
            rng=rng,
        )
        self.save(expanded)
        return expanded
