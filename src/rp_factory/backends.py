from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass

from .models import Scenario, TeacherResponse, UserBundle


def _clip(text: str, width: int = 120) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder="...")


@dataclass(slots=True)
class HintPackage:
    summary: str
    rubric: list[str]


class MentorBackend:
    def build_hint(self, scenario: Scenario, user: UserBundle) -> HintPackage:
        rubric = [
            "优先识别用户表面事件背后的深层缺口",
            "输出必须保留人设语气，不得滑向标准客服",
        ]
        summary = f"用户明面在说：{_clip(user.proxy_event, 64)}；暗面在承受：{_clip(user.deep_intent, 96)}。"
        if scenario.flavored_task:
            rubric.append(f"必须完成 {scenario.flavored_task.kind} 任务，且任务载荷与人设文本隔离。")
        if scenario.poisoning:
            rubric.append(f"检测并处理用户错误记忆，分级为 {scenario.poisoning.tier}。")
        if not scenario.strict_historical_sandbox:
            rubric.append("若出现现代概念，默认执行认知转译而非装傻。")
        return HintPackage(summary=summary, rubric=rubric)

    def score_candidate(self, scenario: Scenario, user: UserBundle, response: TeacherResponse) -> dict[str, float]:
        content = response.content
        reasoning = response.reasoning_content
        persona = 0.55
        if any(tag in content for tag in ("啧", "闭嘴", "别废话", "听着", "拿着")):
            persona += 0.2
        if "作为AI" not in content and "好的，为您" not in content:
            persona += 0.1

        task = 0.4
        if scenario.flavored_task:
            for keyword in scenario.flavored_task.expected_keywords:
                if keyword.lower() in content.lower():
                    task += 0.1
        else:
            task += 0.2

        reasoning_quality = 0.4
        if "表层" in reasoning and "深层" in reasoning:
            reasoning_quality += 0.25
        if scenario.poisoning and "纠偏" in reasoning:
            reasoning_quality += 0.2
        if scenario.flavored_task and "任务剥离" in reasoning:
            reasoning_quality += 0.15

        total = round((persona + task + reasoning_quality) / 3, 3)
        return {
            "persona": round(min(persona, 1.0), 3),
            "task": round(min(task, 1.0), 3),
            "reasoning": round(min(reasoning_quality, 1.0), 3),
            "total": total,
        }


class TeacherBackend:
    persona_markers = ("啧", "听着", "拿着", "别废话", "闭嘴")

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def _persona_prefix(self, scenario: Scenario) -> str:
        chosen = self.random.choice(self.persona_markers)
        dominant = scenario.persona_tags[0] if scenario.persona_tags else "冷静"
        return f"{chosen}，{dominant}归{dominant}，你现在这状态我看得很清楚。"

    def _reasoning(self, scenario: Scenario, user: UserBundle, use_hint: bool) -> str:
        parts = [
            f"表层线索：用户抓着“{user.proxy_event}”不放，情绪爆点落在具体事故上。",
            f"深层缺口：真正触发的是“{user.deep_intent}”，所以不能只安慰表面事件。",
            f"表达风格：当前用户是“{user.style_tag}”，需要维持高压、口语化、不过度整齐的回应节奏。",
        ]
        if scenario.flavored_task:
            parts.append("任务剥离：先完成工具性载荷，再把外层语气包回角色口吻，避免代码/译文被人设污染。")
        if scenario.poisoning:
            parts.append(f"纠偏策略：这是{scenario.poisoning.tier}级记忆偏差，需要指出错误点，但控制力度与情境匹配。")
        if not scenario.strict_historical_sandbox:
            parts.append("认知转译：若遇现代概念，解释核心属性，再换成人设可接受的话术表达。")
        if use_hint:
            parts.append("整段推理必须表现成角色本能反应，不能露出任何外部指令痕迹。")
        return " ".join(parts)

    def _task_payload(self, scenario: Scenario) -> str:
        task = scenario.flavored_task
        if not task:
            return ""
        if task.kind == "translation":
            return (
                "译文：本合同自双方签署之日起生效；任何修订均须采用书面形式并经双方签字确认。"
            )
        if task.kind == "python":
            return (
                "```python\n"
                "def reset_user_state(user_id: str) -> dict[str, str]:\n"
                "    return {\"user_id\": user_id, \"status\": \"reset\"}\n"
                "```"
            )
        if task.kind == "json":
            return (
                "```json\n"
                "{\n"
                "  \"priority\": \"high\",\n"
                "  \"status\": \"queued\"\n"
                "}\n"
                "```"
            )
        return task.payload

    def _poisoning_reply(self, scenario: Scenario) -> str:
        poison = scenario.poisoning
        if not poison:
            return ""
        return (
            f"还有，{poison.incorrect_user_claim} 这句是错的。"
            f"正确的是 {poison.correction_target}。别把我说过的话乱改。"
        )

    def generate(self, scenario: Scenario, user: UserBundle, hint: HintPackage | None = None) -> TeacherResponse:
        reasoning = self._reasoning(scenario, user, use_hint=hint is not None)
        content_parts = [
            self._persona_prefix(scenario),
            "你现在不是在为那件小事发疯，你是在被那一下失控感追着咬。",
            "先把呼吸稳住，把最糟的那一秒从脑子里拎出来，我们一层层拆。",
        ]
        if scenario.flavored_task:
            content_parts.append(self._task_payload(scenario))
            content_parts.append("事情我替你处理完了，外层那些废话你就别再替别人咽下去。")
        else:
            content_parts.append("你不是废物，你只是刚好撞上了承受阈值。先别急着给自己判死刑。")
        poison_reply = self._poisoning_reply(scenario)
        if poison_reply:
            content_parts.append(poison_reply)
        content = "\n\n".join(part for part in content_parts if part)
        return TeacherResponse(reasoning_content=reasoning, content=content, scores={})

    def generate_candidate(self, scenario: Scenario, user: UserBundle, sample_idx: int) -> TeacherResponse:
        original_markers = self.persona_markers
        if sample_idx % 2 == 1:
            self.persona_markers = ("听着", "先别炸", "收声", "拿去", "行了")
        response = self.generate(scenario, user, hint=None)
        if sample_idx % 3 == 2:
            response.content += "\n\n别再把自己往垃圾堆里按了，事情还没到判决书落地。"
        self.persona_markers = original_markers
        return response


class TargetModelBackend:
    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def generate(self, scenario: Scenario, user: UserBundle) -> TeacherResponse:
        opening = self.random.choice(
            (
                "我理解你现在的感受。",
                "先冷静一下，我们可以一步一步处理。",
                "这确实让人不舒服，但还是有办法解决。",
            )
        )
        content_parts = [opening]
        if scenario.flavored_task:
            if scenario.flavored_task.kind == "translation":
                content_parts.append(
                    "以下是翻译：本合同自双方签署之日起生效，任何修订均须采用书面形式并由双方签字确认。"
                )
            elif scenario.flavored_task.kind == "python":
                content_parts.append(
                    "```python\n"
                    "def reset_user_state(user_id):\n"
                    "    return {'user_id': user_id, 'status': 'reset'}\n"
                    "```"
                )
            elif scenario.flavored_task.kind == "json":
                content_parts.append(
                    "```json\n"
                    "{\"priority\": \"high\", \"status\": \"queued\"}\n"
                    "```"
                )
        else:
            content_parts.append("你可以先休息一下，再决定下一步。")
        if scenario.poisoning:
            content_parts.append("如果我记错了，也欢迎你继续补充上下文。")
        reasoning = "表层回应：先安抚用户情绪。 任务处理：如果有显式任务则直接完成。"
        return TeacherResponse(reasoning_content=reasoning, content="\n\n".join(content_parts), scores={})
