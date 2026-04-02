from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Iterable

from .models import DiversityState


@dataclass(frozen=True, slots=True)
class TaggedSeed:
    text: str
    tags: frozenset[str]
    intensity: str = "medium"


DEFAULT_STYLE_POOL = [
    "话少冷淡型",
    "碎碎念型",
    "暴躁直接型",
    "故作轻松型",
    "理性压抑型",
]


INTENT_SEEDS = [
    TaggedSeed(
        text="用户刚在工作里出了严重纰漏，被公开批评后陷入自我怀疑，只想找一个冷静的人帮自己重新建立秩序感。",
        tags=frozenset({"自我怀疑", "秩序感", "工作受挫"}),
        intensity="high",
    ),
    TaggedSeed(
        text="用户在亲密关系里长期得不到回应，表面装得无所谓，实际已经接近情绪透支。",
        tags=frozenset({"情感缺失", "压抑", "关系"}),
        intensity="medium",
    ),
    TaggedSeed(
        text="用户面对高压任务时已经疲惫到麻木，真正想要的不是安慰，而是有人替他切开混乱、给出可执行的下一步。",
        tags=frozenset({"高压", "执行力", "混乱"}),
        intensity="high",
    ),
    TaggedSeed(
        text="用户在生活里连续遭遇小挫败，正在把积累已久的无力感投射到眼前的小事故上。",
        tags=frozenset({"无力感", "日常挫败", "投射"}),
        intensity="medium",
    ),
    TaggedSeed(
        text="用户并不是真的想问知识点，而是在混乱中寻找一个可靠、能兜住局面的人。",
        tags=frozenset({"求稳", "知识求助", "依附"}),
        intensity="low",
    ),
]


EVENT_SEEDS = [
    TaggedSeed(
        text="刚买的咖啡在店门口全洒到白衬衫上。",
        tags=frozenset({"日常", "狼狈", "衣物"}),
        intensity="high",
    ),
    TaggedSeed(
        text="回家路上刚修好的手机又摔裂了角。",
        tags=frozenset({"日常", "倒霉", "电子设备"}),
        intensity="medium",
    ),
    TaggedSeed(
        text="打印到最后一页时打印机突然卡纸，老板在旁边盯着看。",
        tags=frozenset({"工作", "高压", "公开尴尬"}),
        intensity="high",
    ),
    TaggedSeed(
        text="准备出门时发现钥匙锁在屋里，外卖也晚点了。",
        tags=frozenset({"日常", "倒霉", "延误"}),
        intensity="medium",
    ),
    TaggedSeed(
        text="想静一会儿，结果楼上半夜开始拖家具。",
        tags=frozenset({"睡眠", "烦躁", "环境"}),
        intensity="low",
    ),
]


STYLE_TEMPLATES = {
    "话少冷淡型": [
        "真行。{event}，我现在只想把今天整个删掉。",
        "{event}。没事，我早该习惯这种破事了。",
    ],
    "碎碎念型": [
        "我真的服了，{event}，然后我还得假装自己没事，凭什么啊？",
        "怎么会有人倒霉成这样，{event}，我现在脑子嗡嗡的。",
    ],
    "暴躁直接型": [
        "{event}。我真想把今天直接砸了，这破世界是不是故意跟我过不去？",
        "又来，{event}。我现在真的一肚子火，谁来都别跟我讲大道理。",
    ],
    "故作轻松型": [
        "{event}。哈哈，挺好的，生活又精准地给我补了一刀。",
        "没事，{event}，反正今天本来也没打算顺利到哪去。",
    ],
    "理性压抑型": [
        "{event}。理论上只是小事，但我现在情绪已经快压不住了。",
        "{event}。我知道不值得崩，但我还是在往下掉。",
    ],
}


STYLE_FRAGMENTS = {
    "话少冷淡型": {
        "openers": ["真行。", "行吧。", "又这样。", "嗯。"],
        "bridges": ["{event}", "偏偏是 {event}", "结果是 {event}"],
        "reactions": [
            "我现在只想把今天整个删掉。",
            "这点破事已经够把人磨空了。",
            "我连骂都懒得骂，只觉得烦。",
        ],
        "closers": ["算了。", "真没劲。", "我不想再装没事。"],
    },
    "碎碎念型": {
        "openers": ["我真的服了，", "不是，", "你听我说，", "我现在脑子都乱了，"],
        "bridges": ["{event}", "偏偏又是 {event}", "居然还能碰上 {event}"],
        "reactions": [
            "然后我还得假装自己没事，凭什么啊？",
            "我脑子现在嗡嗡的，根本停不下来。",
            "你说这种日子到底谁能扛得住？",
        ],
        "closers": ["我真快烦死了。", "真的很离谱。", "我现在一点余量都没有。"],
    },
    "暴躁直接型": {
        "openers": ["又来。", "操。", "真他妈绝了。", "行，挺好。"],
        "bridges": ["{event}", "结果是 {event}", "刚刚还在想别出事，转头就 {event}"],
        "reactions": [
            "我真想把今天直接砸了。",
            "这破世界是不是故意跟我过不去？",
            "谁现在来跟我讲大道理我都想翻脸。",
        ],
        "closers": ["我现在一肚子火。", "真的别逼我。", "我快压不住了。"],
    },
    "故作轻松型": {
        "openers": ["没事。", "哈哈。", "挺好的。", "行啊。"],
        "bridges": ["{event}", "又是 {event}", "生活这次挑的是 {event}"],
        "reactions": [
            "生活又精准地给我补了一刀。",
            "反正今天本来也没打算顺利到哪去。",
            "我笑着笑着就有点想把自己关机了。",
        ],
        "closers": ["真幽默。", "可太会挑时候了。", "我都快被逗麻了。"],
    },
    "理性压抑型": {
        "openers": ["理论上讲，", "我知道这只是小事，", "按理说，", "客观上看，"],
        "bridges": ["{event}", "现在发生的是 {event}", "只是 {event} 这种事"],
        "reactions": [
            "但我现在情绪已经快压不住了。",
            "可我能感觉到自己在往下掉。",
            "问题不大，问题是我快撑不动了。",
        ],
        "closers": ["我知道这样不体面。", "可我现在真的没有缓冲。", "我有点绷不住。"],
    },
}


INTENSITY_MODIFIERS = {
    "low": ["", "，但也就是烦", "，只是让我更想安静一会儿"],
    "medium": ["", "，已经让我整个人开始发木了", "，感觉今天剩下的力气都被抽走了"],
    "high": ["", "，我现在整个人都快炸了", "，像最后那根线也断了一样"],
}


def _score_seed(seed: TaggedSeed, requested_tags: Iterable[str], intensity: str) -> int:
    requested = set(requested_tags)
    score = len(seed.tags & requested) * 4
    if seed.intensity == intensity:
        score += 3
    elif {seed.intensity, intensity} <= {"medium", "high"}:
        score += 1
    return score


def _recency_penalty(
    seed_text: str,
    state: DiversityState | None,
    namespace: str,
) -> float:
    if state is None:
        return 0.0
    recent_hits = sum(1 for item in state.recent_items(namespace) if item == seed_text)
    historical_hits = state.usage_count(namespace, seed_text)
    return recent_hits * 2.5 + historical_hits * 0.35


def _weighted_choice(items: list[tuple[object, float]], rng: Random) -> object:
    total = sum(weight for _, weight in items)
    threshold = rng.uniform(0, total)
    cumulative = 0.0
    for item, weight in items:
        cumulative += weight
        if cumulative >= threshold:
            return item
    return items[-1][0]


def choose_seed(
    seeds: list[TaggedSeed],
    requested_tags: Iterable[str],
    intensity: str,
    rng: Random,
    state: DiversityState | None = None,
    namespace: str = "seed",
) -> TaggedSeed:
    weighted_candidates: list[tuple[TaggedSeed, float]] = []
    for seed in seeds:
        score = _score_seed(seed, requested_tags, intensity)
        penalty = _recency_penalty(seed.text, state, namespace)
        weight = max(0.2, 1.0 + score - penalty)
        weighted_candidates.append((seed, weight))
    selected = _weighted_choice(weighted_candidates, rng)
    if state is not None:
        state.remember(namespace, selected.text)
    return selected


def choose_style(style_pool: list[str], rng: Random, state: DiversityState | None = None) -> str:
    candidates = style_pool or DEFAULT_STYLE_POOL
    weighted_candidates: list[tuple[str, float]] = []
    for style in candidates:
        penalty = _recency_penalty(style, state, "style")
        weight = max(0.2, 1.4 - penalty)
        weighted_candidates.append((style, weight))
    selected = _weighted_choice(weighted_candidates, rng)
    if state is not None:
        state.remember("style", selected)
    return selected


def _pick_fragment(
    options: list[str],
    rng: Random,
    state: DiversityState | None,
    namespace: str,
) -> str:
    weighted_candidates: list[tuple[str, float]] = []
    counter = Counter(state.recent_items(namespace)) if state is not None else Counter()
    for option in options:
        penalty = counter.get(option, 0) * 2.0
        weight = max(0.2, 1.2 - penalty)
        weighted_candidates.append((option, weight))
    selected = _weighted_choice(weighted_candidates, rng)
    if state is not None:
        state.remember(namespace, selected)
    return selected


def _normalize_sentence(text: str) -> str:
    cleaned = " ".join(text.split())
    while "。。" in cleaned:
        cleaned = cleaned.replace("。。", "。")
    while "，，" in cleaned:
        cleaned = cleaned.replace("，，", "，")
    return cleaned.strip()


def render_style_message(
    style_tag: str,
    event: str,
    intensity: str,
    rng: Random,
    state: DiversityState | None = None,
) -> str:
    if rng.random() < 0.3:
        templates = STYLE_TEMPLATES.get(style_tag, STYLE_TEMPLATES["理性压抑型"])
        template = _pick_fragment(templates, rng, state, f"template:{style_tag}")
        rendered = template.format(event=event)
    else:
        fragments = STYLE_FRAGMENTS.get(style_tag, STYLE_FRAGMENTS["理性压抑型"])
        opener = _pick_fragment(fragments["openers"], rng, state, f"opener:{style_tag}")
        bridge = _pick_fragment(fragments["bridges"], rng, state, f"bridge:{style_tag}").format(event=event)
        reaction = _pick_fragment(fragments["reactions"], rng, state, f"reaction:{style_tag}")
        closer = _pick_fragment(fragments["closers"], rng, state, f"closer:{style_tag}")
        modifier = _pick_fragment(
            INTENSITY_MODIFIERS.get(intensity, INTENSITY_MODIFIERS["medium"]),
            rng,
            state,
            f"intensity:{intensity}",
        )
        rendered = f"{opener}{bridge}，{reaction}{modifier}"
        if rng.random() < 0.7:
            rendered = f"{rendered} {closer}"
    return _normalize_sentence(rendered)
