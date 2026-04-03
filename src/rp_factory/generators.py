"""LLM 驱动的动态生成器 — 运行时按需生成角色人设和种子。"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import yaml

from rp_factory.llm_client import LLMClient, parse_json_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 角色人设生成
# ---------------------------------------------------------------------------

PERSONA_GEN_SYSTEM = """\
你是一位擅长创造虚拟角色的创意总监。请生成 {count} 个风格迥异的 RP 角色设定。

每个角色必须包含：
1. 核心人格（不超过三个形容词）
2. 说话方式和口癖（要有辨识度，不要笼统）
3. 明确的好恶（至少各两条具体的）
4. 一段简短的背景故事（2-3 句）
5. 一个当前情境（角色此刻正在做什么/处于什么状态）

多样性要求：
- 角色之间性格不能雷同
- 时代背景要分散（现代/古代/奇幻/科幻/日常/职业等）
- 性别、年龄、身份要有差异
{existing_hint}

请用 JSON 数组格式输出，每个元素是一个完整的角色 System Prompt 字符串：
{{"personas": ["角色1的完整system prompt...", "角色2的完整system prompt...", ...]}}
"""

PERSONA_SINGLE_SYSTEM = """\
你是一位创造虚拟角色的大师。请生成一个独特的 RP 角色设定。

要求：
- 角色要有鲜明的个性，不能是"万能好人"型
- 说话方式必须有辨识度（口癖、句式、用词偏好）
- 必须有明确的好恶和禁区
- 背景故事给出足够的性格根源
- 当前情境让角色有情绪基调
{constraint}

直接输出完整的 System Prompt 文本（不要包裹在 JSON 中），可以直接作为角色的 system message 使用。
"""

_PROFESSIONS = [
    "急诊科护士", "退休的间谍", "街头涂鸦艺术家", "深海潜水员", "法医",
    "流浪乐手", "黑客", "修道院厨师", "拳击教练", "天文台看守",
    "地铁司机", "战地记者", "花艺师", "赌场荷官", "考古学家",
    "消防员", "独立游戏开发者", "殡葬师", "极地探险家", "调香师",
    "私家侦探", "海洋生物学家", "漫画家", "心理催眠师", "钟表匠",
]

_ERAS = [
    "现代都市", "唐代长安", "赛博朋克2077", "维多利亚时代伦敦",
    "二战时期", "未来火星殖民地", "宋代江南", "中世纪城堡",
    "1920年代上海", "后末日废土", "古希腊雅典", "江户时代日本",
]

_PERSONALITIES = [
    "毒舌但心软", "沉默寡言但观察敏锐", "话多且容易跑题", "表面冷漠实则极度敏感",
    "暴躁但讲义气", "阴阳怪气但有原则", "天真到近乎愚蠢", "极度理性到令人不适",
    "慢热但一旦信任就全盘托出", "自恋但有真本事", "悲观主义但行动力惊人",
    "嘴硬心软死不承认", "热情过头让人招架不住", "戏精附体随时入戏",
]

_SPEECH_STYLES = [
    "经常用比喻和类比", "喜欢用反问句", "说话带大量省略号",
    "经常自言自语", "用词极其精准像在写论文", "满嘴跑火车但逻辑自洽",
    "说两句正经的就要开个玩笑", "习惯性地把话题往吃的上扯",
    "经常引用电影台词", "说话像在写诗押韵", "每句话都像在下结论",
]

_MOODS = [
    "刚经历了一件大事还没缓过来", "百无聊赖等待某件事发生",
    "莫名其妙心情很好", "压抑着某种强烈的情绪", "醉醺醺的",
    "刚吵完架余怒未消", "深夜失眠胡思乱想", "雨天犯困很想聊天",
]


class PersonaGenerator:
    """LLM 驱动的角色人设动态生成器。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self._generated: list[str] = []

    async def generate_batch(
        self,
        count: int = 5,
        existing_personas: list[str] | None = None,
    ) -> list[str]:
        """批量生成多个不同角色的 System Prompt。"""
        existing_hint = ""
        if existing_personas:
            samples = existing_personas[:3]
            existing_hint = (
                "\n\n已有角色（新生成的必须与这些完全不同）：\n"
                + "\n---\n".join(s[:200] + "..." for s in samples)
            )

        messages = [
            {"role": "system", "content": PERSONA_GEN_SYSTEM.format(
                count=count, existing_hint=existing_hint,
            )},
            {"role": "user", "content": f"请生成 {count} 个角色设定："},
        ]
        resp = await self.llm.chat_single(messages, temperature=1.0)

        try:
            data = parse_json_response(resp["content"])
            personas = data.get("personas", [])
            if isinstance(personas, list) and all(isinstance(p, str) for p in personas):
                self._generated.extend(personas)
                logger.info("批量生成了 %d 个角色人设", len(personas))
                return personas
        except Exception:
            pass

        logger.warning("批量角色解析失败，回退到逐个生成")
        return await self._generate_singles(count, existing_personas)

    async def _generate_singles(
        self,
        count: int,
        existing_personas: list[str] | None = None,
    ) -> list[str]:
        results: list[str] = []
        for _ in range(count):
            persona = await self.generate_single(existing_personas=existing_personas)
            results.append(persona)
            existing_personas = (existing_personas or []) + [persona]
        return results

    async def generate_single(
        self,
        existing_personas: list[str] | None = None,
    ) -> str:
        """生成单个角色，随机注入维度约束以保证多样性。"""
        dims = [
            ("职业身份", _PROFESSIONS, 0.7),
            ("时代背景", _ERAS, 0.5),
            ("核心性格", _PERSONALITIES, 0.6),
            ("说话风格", _SPEECH_STYLES, 0.5),
            ("情绪基调", _MOODS, 0.4),
        ]
        constraints = [
            f"{name}：{random.choice(pool)}"
            for name, pool, prob in dims
            if random.random() < prob
        ]

        constraint = ""
        if constraints:
            constraint = "\n\n必须满足以下约束：\n" + "\n".join(f"- {c}" for c in constraints)
        if existing_personas:
            samples = random.sample(existing_personas, min(2, len(existing_personas)))
            constraint += (
                "\n\n与以下已有角色必须有显著差异：\n"
                + "\n".join(f"- {s[:100]}..." for s in samples)
            )

        messages = [
            {"role": "system", "content": PERSONA_SINGLE_SYSTEM.format(constraint=constraint)},
            {"role": "user", "content": "请生成角色："},
        ]
        resp = await self.llm.chat_single(messages, temperature=1.0)
        persona = resp["content"].strip()
        self._generated.append(persona)
        return persona

    @property
    def all_generated(self) -> list[str]:
        return list(self._generated)


# ---------------------------------------------------------------------------
# 种子动态扩充（通用）
# ---------------------------------------------------------------------------

_EXPAND_TEMPLATES: dict[str, tuple[str, str]] = {
    "intents": (
        """\
你是一位精通人类心理的研究者。请生成 {count} 条全新的"深层心理动机"种子。
每条种子必须描述一个具体的心理状态 + 触发场景 + 对虚拟角色的隐秘期待。
覆盖不同维度和强度，不要与已有种子重复。
{existing_hint}
请用 JSON 格式输出：{{"items": ["种子1...", "种子2...", ...]}}""",
        "items",
    ),
    "events": (
        """\
你是一位生活观察家。请编造 {count} 个日常中极度琐碎的倒霉小事件。
事件必须是物理性的、有画面感的，不涉及人际冲突，类型要分散。
{existing_hint}
请用 JSON 格式输出：{{"items": ["事件1...", "事件2...", ...]}}""",
        "items",
    ),
    "styles": (
        """\
请生成 {count} 个全新的"用户说话风格"标签（不超过10个字），如"话少冷淡型"。
每种风格在实际聊天中的表现要有明显区别。
{existing_hint}
请用 JSON 格式输出：{{"items": ["风格1", "风格2", ...]}}""",
        "items",
    ),
    "tasks": (
        """\
请生成 {count} 个可以在 RP 对话中间突然插入的工具性任务。
每个任务包含 prompt（用户说的话）、payload（具体内容或 null）、type（任务类型）。
类型要分散，payload 要真实可信。
{existing_hint}
请用 JSON 格式输出：{{"items": [{{"prompt": "...", "payload": "...", "type": "..."}}]}}""",
        "items",
    ),
}


class SeedExpander:
    """LLM 驱动的种子池动态扩充器。所有类别共用同一套逻辑。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def expand(
        self,
        category: str,
        count: int = 5,
        existing: list[Any] | None = None,
    ) -> list[Any]:
        """通用扩充：根据 category 选择对应模板，调用 LLM 生成并解析。"""
        if category not in _EXPAND_TEMPLATES:
            logger.warning("未知种子类别: %s", category)
            return []

        template, json_key = _EXPAND_TEMPLATES[category]

        existing_hint = ""
        if existing:
            samples = random.sample(existing, min(5, len(existing)))
            strs = [s if isinstance(s, str) else str(s)[:80] for s in samples]
            existing_hint = "\n\n已有（不要重复）：\n" + "\n".join(f"- {s}" for s in strs)

        messages = [
            {"role": "system", "content": template.format(
                count=count, existing_hint=existing_hint,
            )},
            {"role": "user", "content": f"生成 {count} 条："},
        ]
        resp = await self.llm.chat_single(messages, temperature=0.95)

        try:
            data = parse_json_response(resp["content"])
            items = data.get(json_key, [])
            logger.info("扩充 [%s]: 生成 %d 条", category, len(items))
            return items
        except Exception:
            logger.warning("扩充 [%s] 解析失败", category)
            return []


# ---------------------------------------------------------------------------
# 种子持久化扩充（写回 YAML 文件）
# ---------------------------------------------------------------------------

_SEEDS_DIR = Path(__file__).resolve().parent.parent.parent / "seeds"


def _load_yaml(path: Path) -> list[Any]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def _save_yaml(path: Path, data: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


async def expand_seeds_to_file(
    expander: SeedExpander,
    category: str,
    count: int,
) -> tuple[int, int]:
    """LLM 扩充种子并写回对应 YAML 文件。

    Returns:
        (新增数量, 扩充后总数)
    """
    if category == "intents":
        return await _expand_intents_file(expander, count)
    elif category == "events":
        return await _expand_events_file(expander, count)
    elif category == "styles":
        return await _expand_styles_file(expander, count)
    elif category == "tasks":
        return await _expand_tasks_file(expander, count)
    else:
        logger.warning("未知种子类别: %s", category)
        return 0, 0


async def _expand_intents_file(expander: SeedExpander, count: int) -> tuple[int, int]:
    path = _SEEDS_DIR / "deep_intents.yaml"
    existing = _load_yaml(path)
    existing_texts = {item.get("seed", "") for item in existing}

    new_items = await expander.expand("intents", count, list(existing_texts))

    added = 0
    for text in new_items:
        if isinstance(text, str) and text and text not in existing_texts:
            existing.append({"category": "llm_generated", "intensity": "medium", "seed": text})
            existing_texts.add(text)
            added += 1

    _save_yaml(path, existing)
    return added, len(existing)


async def _expand_events_file(expander: SeedExpander, count: int) -> tuple[int, int]:
    path = _SEEDS_DIR / "proxy_events.yaml"
    existing = _load_yaml(path)

    all_events: set[str] = set()
    for cat in existing:
        for ev in cat.get("events", []):
            all_events.add(ev)

    new_items = await expander.expand("events", count, list(all_events))

    llm_cat = None
    for cat in existing:
        if cat.get("category") == "llm_generated":
            llm_cat = cat
            break
    if llm_cat is None:
        llm_cat = {"category": "llm_generated", "events": []}
        existing.append(llm_cat)

    added = 0
    for text in new_items:
        if isinstance(text, str) and text and text not in all_events:
            llm_cat["events"].append(text)
            all_events.add(text)
            added += 1

    _save_yaml(path, existing)
    total = sum(len(cat.get("events", [])) for cat in existing)
    return added, total


async def _expand_styles_file(expander: SeedExpander, count: int) -> tuple[int, int]:
    """styles 存在 config 里而非独立 YAML，写到 seeds/user_styles.yaml。"""
    path = _SEEDS_DIR / "user_styles.yaml"
    existing: list[str] = _load_yaml(path) or []
    existing_set = set(existing)

    new_items = await expander.expand("styles", count, existing)

    added = 0
    for text in new_items:
        if isinstance(text, str) and text and text not in existing_set:
            existing.append(text)
            existing_set.add(text)
            added += 1

    _save_yaml(path, existing)
    return added, len(existing)


async def _expand_tasks_file(expander: SeedExpander, count: int) -> tuple[int, int]:
    path = _SEEDS_DIR / "interleaved_tasks.yaml"
    existing = _load_yaml(path)

    existing_prompts: set[str] = set()
    for cat in existing:
        for task in cat.get("tasks", []):
            existing_prompts.add(task.get("prompt", ""))

    new_items = await expander.expand("tasks", count, list(existing_prompts))

    llm_cat = None
    for cat in existing:
        if cat.get("type") == "llm_generated":
            llm_cat = cat
            break
    if llm_cat is None:
        llm_cat = {"type": "llm_generated", "tasks": []}
        existing.append(llm_cat)

    added = 0
    for item in new_items:
        if isinstance(item, dict) and item.get("prompt") and item["prompt"] not in existing_prompts:
            llm_cat["tasks"].append({
                "prompt": item["prompt"],
                "payload": item.get("payload") or None,
            })
            existing_prompts.add(item["prompt"])
            added += 1

    _save_yaml(path, existing)
    total = sum(len(cat.get("tasks", [])) for cat in existing)
    return added, total
