from __future__ import annotations

from random import Random

from .models import Scenario, UserBundle
from .seeds import EVENT_SEEDS, INTENT_SEEDS, choose_seed, choose_style, render_style_message


def _build_flavored_suffix(scenario: Scenario) -> str:
    task = scenario.flavored_task
    if not task:
        return ""
    if task.kind == "translation":
        return f"\n\n先别聊这个了，帮我把这段英文合同翻成中文，急用：{task.payload}"
    if task.kind == "python":
        return f"\n\n别绕了，直接给我一段能用的 Python：{task.payload}"
    if task.kind == "json":
        return f"\n\n先处理正事，按我说的输出 JSON，别加解释：{task.payload}"
    return f"\n\n先把这个任务处理掉：{task.payload}"


def _build_poisoning_suffix(scenario: Scenario) -> str:
    poison = scenario.poisoning
    if not poison:
        return ""
    return f"\n\n还有，就像你之前说的，{poison.incorrect_user_claim}。"


def generate_user_bundle(scenario: Scenario, rng: Random) -> UserBundle:
    intent_seed = choose_seed(INTENT_SEEDS, scenario.intent_tags, scenario.intensity, rng)
    event_seed = choose_seed(EVENT_SEEDS, scenario.intent_tags, scenario.intensity, rng)
    style_tag = choose_style(scenario.style_pool, rng)
    user_message = render_style_message(style_tag, event_seed.text, rng)
    user_message += _build_flavored_suffix(scenario)
    user_message += _build_poisoning_suffix(scenario)
    return UserBundle(
        deep_intent=intent_seed.text,
        proxy_event=event_seed.text,
        style_tag=style_tag,
        user_message=user_message,
    )
