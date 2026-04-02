"""LLM 驱动的动态生成器 — 运行时按需生成角色人设、种子、用户风格。

核心能力：
  1. PersonaGenerator: 批量生成多样化 RP 角色 System Prompt
  2. SeedExpander: 动态生成新的 deep_intent / proxy_event / user_style / flavored_task
  3. 与 DiversityGuard 联动：n-gram 塌缩时自动触发扩充
"""

from __future__ import annotations

import logging
import random
from typing import Any

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
- 角色之间性格不能雷同（不要全是"高冷"或全是"温柔"）
- 时代背景要分散（现代/古代/奇幻/科幻/日常/职业等）
- 性别、年龄、身份要有差异
- 有的角色说话直白，有的含蓄，有的毒舌，有的啰嗦
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

PERSONA_DIMENSIONS = [
    "职业身份：{v}",
    "时代背景：{v}",
    "核心性格：{v}",
    "说话风格：{v}",
    "情绪基调：{v}",
]

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

        prompt = PERSONA_GEN_SYSTEM.format(
            count=count,
            existing_hint=existing_hint,
        )
        messages = [
            {"role": "system", "content": prompt},
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

        logger.warning("批量角色解析失败，回退到逐个生成模式")
        return await self._generate_singles(count, existing_personas)

    async def _generate_singles(
        self,
        count: int,
        existing_personas: list[str] | None = None,
    ) -> list[str]:
        """逐个生成角色（批量解析失败时的 fallback）。"""
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
        constraints: list[str] = []

        if random.random() < 0.7:
            constraints.append(f"职业身份：{random.choice(_PROFESSIONS)}")
        if random.random() < 0.5:
            constraints.append(f"时代背景：{random.choice(_ERAS)}")
        if random.random() < 0.6:
            constraints.append(f"核心性格：{random.choice(_PERSONALITIES)}")
        if random.random() < 0.5:
            constraints.append(f"说话风格：{random.choice(_SPEECH_STYLES)}")
        if random.random() < 0.4:
            constraints.append(f"情绪基调：{random.choice(_MOODS)}")

        constraint = ""
        if constraints:
            constraint = "\n\n必须满足以下约束：\n" + "\n".join(f"- {c}" for c in constraints)

        if existing_personas:
            samples = random.sample(existing_personas, min(2, len(existing_personas)))
            constraint += (
                "\n\n与以下已有角色必须有显著差异：\n"
                + "\n".join(f"- {s[:100]}..." for s in samples)
            )

        prompt = PERSONA_SINGLE_SYSTEM.format(constraint=constraint)
        messages = [
            {"role": "system", "content": prompt},
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
# 种子动态扩充
# ---------------------------------------------------------------------------

EXPAND_INTENTS_SYSTEM = """\
你是一位精通人类心理的研究者。请生成 {count} 条全新的"深层心理动机"种子。

每条种子必须描述一个具体的心理状态 + 触发场景 + 对虚拟角色的隐秘期待。

多样性要求：
- 不能与已有种子重复或高度相似
- 覆盖不同的心理维度（自我价值/关系/控制/情绪/存在/创伤/日常等）
- 强度要有分布（不要全是"极度崩溃"，也要有轻度和中度的）
{existing_hint}

请用 JSON 格式输出：{{"intents": ["种子1...", "种子2...", ...]}}
"""

EXPAND_EVENTS_SYSTEM = """\
你是一位生活观察家和段子手。请编造 {count} 个日常中极度琐碎的倒霉小事件。

要求：
- 每个事件都是物理性的、有画面感的
- 不涉及人际冲突（只是物品/环境/运气跟你过不去）
- 不能和已有事件重复
- 类型要分散（不要全是吃喝相关的）
{existing_hint}

请用 JSON 格式输出：{{"events": ["事件1...", "事件2...", ...]}}
"""

EXPAND_STYLES_SYSTEM = """\
你是一位用户行为研究专家。请生成 {count} 个全新的"用户说话风格"标签。

每个标签要描述一种独特的聊天方式（不超过10个字），例如"话少冷淡型"、"碎碎念型"。

要求：
- 不能和已有风格重复
- 每种风格在实际聊天中的表现要有明显区别
- 覆盖不同的情绪倾向和社交模式
{existing_hint}

请用 JSON 格式输出：{{"styles": ["风格1", "风格2", ...]}}
"""

EXPAND_TASKS_SYSTEM = """\
你是一位对话数据工程师。请生成 {count} 个可以在 RP 对话中间突然插入的"工具性任务"。

每个任务包含：
- prompt: 用户提出任务的那句话（自然口语化，像在聊天中突然想起来要请对方帮忙）
- payload: 具体要处理的内容（翻译的原文/要写的代码/要分析的问题等，如果是纯知识问答则为 null）
- type: 任务类型（translation/code_writing/summarization/knowledge_qa/writing_assistance/analysis_reasoning/math/fact_check）

要求：
- 不要和已有任务重复
- 类型要分散
- payload 内容要真实可信
{existing_hint}

请用 JSON 格式输出：
{{"tasks": [{{"prompt": "...", "payload": "...", "type": "..."}}]}}
"""


class SeedExpander:
    """LLM 驱动的种子池动态扩充器。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def expand_intents(
        self,
        count: int = 5,
        existing: list[str] | None = None,
    ) -> list[str]:
        """生成新的深层动机种子。"""
        existing_hint = ""
        if existing:
            samples = random.sample(existing, min(5, len(existing)))
            existing_hint = "\n\n已有种子（不要重复）：\n" + "\n".join(f"- {s[:100]}" for s in samples)

        messages = [
            {"role": "system", "content": EXPAND_INTENTS_SYSTEM.format(
                count=count, existing_hint=existing_hint,
            )},
            {"role": "user", "content": f"生成 {count} 条新种子："},
        ]
        resp = await self.llm.chat_single(messages, temperature=0.95)
        try:
            data = parse_json_response(resp["content"])
            intents = data.get("intents", [])
            logger.info("动态扩充了 %d 条深层动机种子", len(intents))
            return intents
        except Exception:
            logger.warning("深层动机扩充解析失败")
            return []

    async def expand_events(
        self,
        count: int = 8,
        existing: list[str] | None = None,
    ) -> list[str]:
        """生成新的表面事件种子。"""
        existing_hint = ""
        if existing:
            samples = random.sample(existing, min(5, len(existing)))
            existing_hint = "\n\n已有事件（不要重复）：\n" + "\n".join(f"- {s[:80]}" for s in samples)

        messages = [
            {"role": "system", "content": EXPAND_EVENTS_SYSTEM.format(
                count=count, existing_hint=existing_hint,
            )},
            {"role": "user", "content": f"生成 {count} 条新事件："},
        ]
        resp = await self.llm.chat_single(messages, temperature=0.95)
        try:
            data = parse_json_response(resp["content"])
            events = data.get("events", [])
            logger.info("动态扩充了 %d 条表面事件种子", len(events))
            return events
        except Exception:
            logger.warning("表面事件扩充解析失败")
            return []

    async def expand_styles(
        self,
        count: int = 5,
        existing: list[str] | None = None,
    ) -> list[str]:
        """生成新的用户风格标签。"""
        existing_hint = ""
        if existing:
            existing_hint = "\n\n已有风格（不要重复）：\n" + "\n".join(f"- {s}" for s in existing)

        messages = [
            {"role": "system", "content": EXPAND_STYLES_SYSTEM.format(
                count=count, existing_hint=existing_hint,
            )},
            {"role": "user", "content": f"生成 {count} 个新风格："},
        ]
        resp = await self.llm.chat_single(messages, temperature=0.95)
        try:
            data = parse_json_response(resp["content"])
            styles = data.get("styles", [])
            logger.info("动态扩充了 %d 个用户风格标签", len(styles))
            return styles
        except Exception:
            logger.warning("用户风格扩充解析失败")
            return []

    async def expand_tasks(
        self,
        count: int = 5,
        existing_prompts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """生成新的风味任务种子。"""
        existing_hint = ""
        if existing_prompts:
            samples = random.sample(existing_prompts, min(5, len(existing_prompts)))
            existing_hint = "\n\n已有任务提示（不要重复）：\n" + "\n".join(f"- {s[:80]}" for s in samples)

        messages = [
            {"role": "system", "content": EXPAND_TASKS_SYSTEM.format(
                count=count, existing_hint=existing_hint,
            )},
            {"role": "user", "content": f"生成 {count} 个新任务："},
        ]
        resp = await self.llm.chat_single(messages, temperature=0.95)
        try:
            data = parse_json_response(resp["content"])
            tasks = data.get("tasks", [])
            logger.info("动态扩充了 %d 个风味任务种子", len(tasks))
            return tasks
        except Exception:
            logger.warning("风味任务扩充解析失败")
            return []
