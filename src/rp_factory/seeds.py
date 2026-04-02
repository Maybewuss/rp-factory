from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable


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


def _score_seed(seed: TaggedSeed, requested_tags: Iterable[str], intensity: str) -> int:
    requested = set(requested_tags)
    score = len(seed.tags & requested) * 4
    if seed.intensity == intensity:
        score += 3
    elif {seed.intensity, intensity} <= {"medium", "high"}:
        score += 1
    return score


def choose_seed(
    seeds: list[TaggedSeed],
    requested_tags: Iterable[str],
    intensity: str,
    rng: Random,
) -> TaggedSeed:
    scored = sorted(
        seeds,
        key=lambda seed: (_score_seed(seed, requested_tags, intensity), seed.text),
        reverse=True,
    )
    top_score = _score_seed(scored[0], requested_tags, intensity)
    top_candidates = [seed for seed in scored if _score_seed(seed, requested_tags, intensity) == top_score]
    return rng.choice(top_candidates)


def choose_style(style_pool: list[str], rng: Random) -> str:
    candidates = style_pool or DEFAULT_STYLE_POOL
    return rng.choice(candidates)


def render_style_message(style_tag: str, event: str, rng: Random) -> str:
    templates = STYLE_TEMPLATES.get(style_tag, STYLE_TEMPLATES["理性压抑型"])
    template = rng.choice(templates)
    return template.format(event=event)
