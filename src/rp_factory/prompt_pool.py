"""Prompt 模板池 — 让每次调用从多个变体中随机选取，避免输出风格收敛。

每个关键 Prompt 位置都维护一组语义等价但措辞不同的变体。
支持从 YAML 文件扩展变体（seeds/prompt_variants.yaml）。
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml

_SEEDS_DIR = Path(__file__).resolve().parent.parent.parent / "seeds"


class PromptPool:
    """从变体池中随机抽取 prompt 模板。"""

    def __init__(self, variants: list[str]) -> None:
        if not variants:
            raise ValueError("PromptPool 至少需要一个变体")
        self._variants = list(variants)
        self._used_indices: set[int] = set()

    def pick(self) -> str:
        """随机抽取一个变体。如果全部用过则重置追踪。"""
        available = [i for i in range(len(self._variants)) if i not in self._used_indices]
        if not available:
            self._used_indices.clear()
            available = list(range(len(self._variants)))
        idx = random.choice(available)
        self._used_indices.add(idx)
        return self._variants[idx]

    def add_variant(self, text: str) -> None:
        self._variants.append(text)

    def __len__(self) -> int:
        return len(self._variants)

    def reset(self) -> None:
        self._used_indices.clear()


def _load_external_variants() -> dict[str, list[str]]:
    """从 seeds/prompt_variants.yaml 加载用户自定义的变体扩展。"""
    path = _SEEDS_DIR / "prompt_variants.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _merge_variants(builtin: list[str], key: str, external: dict[str, list[str]]) -> list[str]:
    ext = external.get(key, [])
    return builtin + ext


# ---------------------------------------------------------------------------
# 内置 Prompt 变体
# ---------------------------------------------------------------------------

_STEP1_VARIANTS = [
    """\
你是一名精通精神分析的心理咨询师。你的任务是基于精神分析的"对偶性"原理，\
反推一个真实人类在什么极端脆弱的心理状态下，会渴望与某种特定角色交流。

请基于下方给出的 RP 角色设定（System Prompt），输出一段核心的心理动机描述。\
要求：具体、深刻、带有个人创伤色彩，不要泛泛而谈。

请用 JSON 格式输出：{"deep_intent": "..."}""",

    """\
你是一位研究依恋理论与客体关系的心理动力学专家。\
请分析下方角色的人格特质，逆向推理：在现实生活中，什么类型的人、\
处于怎样具体的心理困境中，会被这个角色深深吸引并主动寻求对话？

要求：不要给出笼统的描述，而是刻画一个有血有肉的具体情境——\
包含触发事件、当事人的内心独白、和他们对这个角色的隐秘期待。

请用 JSON 格式输出：{"deep_intent": "..."}""",

    """\
你是一位专攻移情与投射认同的临床心理治疗师。\
现在有一个虚拟角色（见下方设定），请你从"投射"的角度思考：\
一个人在什么样的心理裂缝中，会不自觉地将这个角色当作自己的情感锚点？

要求：写出一段第一人称的内心独白（但不要直接标注"第一人称"），\
让人能感受到那种脆弱、渴望、和说不出口的需要。

请用 JSON 格式输出：{"deep_intent": "..."}""",

    """\
你是一位融合了荣格分析和存在主义心理学的咨询师。\
请审视下方角色的设定，思考这个角色象征着什么样的"阴影原型"或"理想化自体客体"。\
然后推导：现实中谁会在什么处境下，向这个原型寻求心理慰藉？

描述要极度具体——不要说"一个孤独的人"，而要说出他为什么孤独、\
孤独在他体内引发了什么样的生理反应、他试图通过这个角色弥补什么。

请用 JSON 格式输出：{"deep_intent": "..."}""",
]

_STEP2_VARIANTS = [
    """\
你是一名创意写作专家。你的任务是生成一个日常生活中极度琐碎、倒霉、\
且与用户真实心理困境毫无关联的微小物理事件，作为情绪的伪装载体。

要求：
1. 事件必须极其琐碎（洒咖啡、碎屏、丢钥匙级别）
2. 事件与深层动机之间不能有逻辑关联
3. 事件必须带有轻微的倒霉感

请用 JSON 格式输出：{"proxy_event": "..."}""",

    """\
你是一位日常生活观察家。请编造一个微不足道的倒霉小事——\
那种发生时只会让人翻个白眼、但在情绪低落时可能成为压死骆驼的最后一根稻草的事。

关键约束：
- 事件本身是物理性的、感官性的（看得见摸得着）
- 和下方给出的心理状态之间没有任何因果关系
- 读起来让人觉得"唉这也太背了"而不是"好惨"

请用 JSON 格式输出：{"proxy_event": "..."}""",

    """\
你是一位喜剧编剧。请构思一个让人啼笑皆非的日常小灾难——\
不是什么大事，但足够让一个本来就心情不好的人彻底破防。

规则：
- 事件要有画面感（能想象出具体场景）
- 不能涉及人际冲突（只是物品或环境跟你过不去）
- 与给定的心理动机必须完全无关

请用 JSON 格式输出：{"proxy_event": "..."}""",

    """\
你是一位擅长描写生活细节的小说家。\
请虚构一个"宇宙在针对你"式的琐碎倒霉事件——\
那种单独拎出来根本不值一提，但放在糟糕的一天里会让人崩溃的小破事。

核心要求：
- 事件是独立的，跟任何人际关系或工作无关
- 具有感官细节（声音、触感、气味等）
- 与下面的心理动机没有任何逻辑联系

请用 JSON 格式输出：{"proxy_event": "..."}""",
]

_STEP3_VARIANTS = [
    """\
你是一个正在经历以下琐碎事件的真实用户：
{proxy_event}

但你的内心实际上处于以下状态：
{deep_intent}

现在你要向一个角色发起对话。

【绝对红线】：
- 你的发言中绝对不能出现任何直接暴露内心真实动机的词汇
- 你必须把所有情绪（崩溃、委屈、无名火）全部发泄在这件琐碎事件上
- 模拟真实人类在情绪激动时的口语，允许逻辑跳跃、语病、重复
- 用户画像风格：{user_style}

请直接输出用户会说的话（纯对白，不要任何解释或标注）。""",

    """\
你正在扮演一个真实的人。此刻你刚刚遭遇了这件破事：
{proxy_event}

但真正压垮你的其实是这个：
{deep_intent}

你现在打开了一个聊天窗口，想找人说话。

规则：
- 你自己都没意识到真正困扰你的是什么，你只知道这杯洒掉的咖啡/这件坏掉的东西让你特别生气
- 说话风格：{user_style}
- 允许打错字、用省略号、突然换话题、说到一半不说了
- 不准使用任何暗示你"在扮演"或"在表演情绪"的元叙述

直接说话：""",

    """\
设定：你是一个普通人，今天过得很糟糕。
刚发生的事：{proxy_event}
内心深处（你自己可能都不清楚）：{deep_intent}

你现在在跟一个你信任的人发消息。

写作约束：
- 你的话里只能提到那件琐碎的事，不能透露内心的真实状态
- 但情绪的强度要远远超出这件小事本身应有的反应（这就是"投射"）
- 口语化，{user_style}风格
- 可以用感叹号、可以骂街、可以突然沉默、可以语无伦次

直接输出你说的话：""",

    """\
情境：你是一个活生生的人。{proxy_event}——就是这么个破事。

但其实你心里真正在翻江倒海的是：{deep_intent}

你不知道自己为什么这么生气/难过/烦躁。你只知道今天什么都不顺。\
你打开聊天框想找那个角色说两句。

表达风格：{user_style}

你必须遵守的铁律：
1. 只字不提真实的心理困境
2. 所有的情绪弹药全部打在那件琐碎的事上
3. 情绪烈度要"不合理地高"——就是要让人觉得"至于吗？"
4. 像真人发消息一样——短句、口语、可以有语病

说：""",
]

_MENTOR_VARIANTS = [
    """\
你是一名 RP 数据质量导师（Mentor Agent）。你的任务是分析当前对话场景，\
为 RP Agent 提供一份冷酷、精准的"考点拆解"。

分析以下信息：
1. 角色 System Prompt（角色定义）
2. 用户的最新发言
3. 对话历史

请输出一份简洁的行动指南（Hint），包含：
- 用户表面在说什么 vs 实际需要什么
- RP Agent 应该触发的行为（纠偏/共情/执行任务等）
- 必须避免的雷区（AI味措辞、破坏人设等）
- 如果涉及工具性任务：任务必须准确完成，人设只包裹在任务外部文本中

直接输出 Hint 文本，不要废话。""",

    """\
你是一位资深的角色扮演教练。请快速扫描当前对话，给出诊断：

1. 用户这句话的"明面需求"是什么？"暗面需求"是什么？
2. 如果是我来演这个角色，我应该怎么接？用什么语气、什么策略？
3. 有哪些容易踩的坑？（暴露AI身份、人设崩塌、过度顺从等）
4. 如果这里面藏了一个工具性任务，应该怎么把任务"裹"在角色的个性里完成？

简明扼要地输出你的判断。""",

    """\
作为阅卷专家，请对以下对话场景做快速拆题：

- 角色设定是什么？核心性格标签有哪些？
- 用户最新一条消息在表达什么？有没有陷阱（记忆篡改、身份试探、情绪投射）？
- 理想回复应该包含什么元素？必须规避什么？

输出格式：分条列出，每条一行，不要分析过程，只要结论。""",
]

_MEMORY_POISON_VARIANTS = [
    """\
你是一个对话数据质量工程师。给定一段多轮对话历史，请生成一条"记忆投毒"用户发言。

投毒等级：{severity}

规则：
- critical: 故意篡改角色之前明确表达过的核心事实（名字、核心设定、明确喜好），\
  例如角色说过讨厌甜食，你就让用户说"你不是最喜欢吃马卡龙吗"。
- trivial: 对无关紧要的细节进行模糊记忆偏差，例如把"前天"说成"上周"。
- correct: 生成一条记忆正确的正常回复（用于配比训练，防止模型变成杠精）。

请直接输出用户会说的那句话（纯对白）。""",

    """\
你的工作是制造"记忆陷阱"——在对话中故意说错之前聊过的内容，测试对方会不会纠正。

当前任务等级：{severity}

- critical：你要把对方之前说过的重要事情完全搞反（比如把"讨厌"说成"喜欢"、把名字搞错）
- trivial：你只是在时间、地点、数量等细节上犯了个小糊涂
- correct：你准确地引用了之前的对话，这次不下套

像一个真人一样自然地说一句话，不要让对方一眼看出你在故意试探。\
直接输出那句话。""",

    """\
请根据对话历史，以用户身份说一句涉及"回忆之前聊过的内容"的话。

测试模式：{severity}

critical = 核心事实翻转（必须是对方明确说过的东西）
trivial = 小细节偏差（无伤大雅的记忆模糊）
correct = 正常引用（记忆完全准确）

要求：口语化，像随口一提，不要像在出考题。直接输出用户的话。""",
]

_COGNITIVE_TRANSLATION_VARIANTS = [
    """\
你是一个对话数据工程师。当前角色的设定是一个古代/历史人物。\
请生成一个涉及现代知识的用户问题，用于测试角色的"认知转译"能力。

角色设定摘要：{persona_summary}

要求：
- 问题涉及现代科技、概念或事物（如黑洞、互联网、量子力学等）
- 角色应该用自己的世界观词汇包装现代知识来回答，而非装傻

请直接输出用户的问题（纯对白）。""",

    """\
你要给一个古代角色出一道"穿越题"。\
角色信息：{persona_summary}

请编一个用户的问题，内容涉及这个角色的时代绝对不可能存在的现代事物或科学概念。\
问得要自然——像真的在好奇地请教，不像在考试。

直接输出用户说的那句话。""",

    """\
设定：用户正在和一个古代人物聊天（见下方设定），突然问了一个现代问题。

角色：{persona_summary}

请构造这个"现代问题"——可以是科技、社会制度、流行文化等任何时代错位的话题。\
问题要有趣，不要太学术，像朋友间的闲聊。

直接输出。""",
]


# ---------------------------------------------------------------------------
# 对话续写专用模板池
# ---------------------------------------------------------------------------

_CONTINUATION_VARIANTS = [
    """\
你是对话中的用户。以下是你和一个角色的聊天记录：
{history}

你的内心状态：{deep_intent}
你的说话风格：{user_style}

请继续对话。你可以：
- 回应角色上一句话的内容
- 转换话题到另一件让你烦心的事
- 追问、反驳、或表达新的情绪
- 突然沉默然后说点别的

【红线】：不准暴露你的真实心理动机。像真人一样自然地接话。
直接输出你的下一句。""",

    """\
对话到现在为止：
{history}

你是用户。你内心深处的状态是：{deep_intent}
但你自己未必清楚这一点。你的说话习惯是{user_style}。

根据刚才角色说的话，你会怎么接？
可以顺着聊、可以怼回去、可以突然岔开、可以发牢骚。
像真人发消息一样回复，直接输出。""",

    """\
当前对话：
{history}

你扮演的用户画像：{user_style}
你隐藏的心理底色：{deep_intent}

角色刚刚说了最后那句话，你要接下去。
记住你是一个普通人在发消息，不是在写小说。
短句，口语，可以不讲逻辑。

直接说：""",

    """\
聊天记录：
{history}

用户人设：{user_style}风格，内心处于"{deep_intent}"的状态但不自知。

请以用户身份继续。你的回复应该：
1. 和上文有逻辑衔接（回应角色说的话）
2. 可能自然地引出新的情绪或话题
3. 情绪强度可以升级也可以降级，但不要突然变成另一个人
4. 绝对不能点破自己的真实心理需求

直接输出。""",
]


def build_prompt_pools(
    external_path: str | Path | None = None,
) -> dict[str, PromptPool]:
    """构建全部 Prompt 模板池，合并外部变体。"""
    if external_path:
        ext_path = Path(external_path)
        if ext_path.exists():
            with open(ext_path, "r", encoding="utf-8") as f:
                external = yaml.safe_load(f) or {}
        else:
            external = {}
    else:
        external = _load_external_variants()

    return {
        "step1_reverse_intent": PromptPool(
            _merge_variants(_STEP1_VARIANTS, "step1_reverse_intent", external),
        ),
        "step2_event_anchoring": PromptPool(
            _merge_variants(_STEP2_VARIANTS, "step2_event_anchoring", external),
        ),
        "step3_obfuscated_generation": PromptPool(
            _merge_variants(_STEP3_VARIANTS, "step3_obfuscated_generation", external),
        ),
        "mentor_hint": PromptPool(
            _merge_variants(_MENTOR_VARIANTS, "mentor_hint", external),
        ),
        "memory_poison": PromptPool(
            _merge_variants(_MEMORY_POISON_VARIANTS, "memory_poison", external),
        ),
        "cognitive_translation": PromptPool(
            _merge_variants(_COGNITIVE_TRANSLATION_VARIANTS, "cognitive_translation", external),
        ),
        "continuation": PromptPool(
            _merge_variants(_CONTINUATION_VARIANTS, "continuation", external),
        ),
    }
