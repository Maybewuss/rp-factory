from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .models import QualityReport, Scenario, TeacherResponse


BLACKLIST_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"作为\s*ai",
        r"好的，为您",
        r"抱歉",
        r"如果您还需要",
        r"很高兴为您服务",
    )
]

AI_FLAVOR_MARKERS = (
    "首先",
    "其次",
    "总结一下",
    "以下是",
    "1.",
    "2.",
    "3.",
    "总的来说",
)


@dataclass(slots=True)
class CodeBlock:
    language: str
    body: str


def extract_code_blocks(content: str) -> list[CodeBlock]:
    pattern = re.compile(r"```([a-zA-Z0-9_-]+)?\n(.*?)```", re.DOTALL)
    blocks: list[CodeBlock] = []
    for match in pattern.finditer(content):
        language = (match.group(1) or "").strip().lower()
        body = match.group(2)
        blocks.append(CodeBlock(language=language, body=body))
    return blocks


def _validate_python_block(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _validate_json_block(code: str) -> bool:
    import json

    try:
        json.loads(code)
        return True
    except json.JSONDecodeError:
        return False


def _persona_bleed_in_code(code: str) -> bool:
    lower = code.lower()
    suspicious = ("笨蛋", "闭嘴", "别废话", "垃圾", "傻", "啧")
    return any(token in code or token in lower for token in suspicious)


def run_quality_funnel(scenario: Scenario, response: TeacherResponse) -> QualityReport:
    reasons: list[str] = []
    content = response.content

    for pattern in BLACKLIST_PATTERNS:
        if pattern.search(content):
            reasons.append(f"命中黑名单话术: {pattern.pattern}")

    ai_flavor_hits = sum(marker in content for marker in AI_FLAVOR_MARKERS)
    if ai_flavor_hits >= 3:
        reasons.append("结构性 AI 味过重")

    task_score = 1.0
    if scenario.flavored_task:
        blocks = extract_code_blocks(content)
        if scenario.flavored_task.kind in {"python", "json"} and not blocks:
            reasons.append("缺少代码载荷")
            task_score = 0.0
        for block in blocks:
            if _persona_bleed_in_code(block.body):
                reasons.append("代码块内出现人设污染")
                task_score = 0.0
            if block.language == "python" and not _validate_python_block(block.body):
                reasons.append("Python 代码块语法错误")
                task_score = 0.0
            if block.language == "json" and not _validate_json_block(block.body):
                reasons.append("JSON 代码块语法错误")
                task_score = 0.0
        if scenario.flavored_task.kind == "translation":
            matched = sum(
                keyword.lower() in content.lower()
                for keyword in scenario.flavored_task.expected_keywords
            )
            if matched == 0:
                reasons.append("翻译载荷缺少目标关键词")
                task_score = 0.2

    persona_score = 1.0
    if not any(marker in content for marker in ("啧", "听着", "拿着", "别废话", "先别炸", "收声")):
        persona_score = 0.45
        reasons.append("角色外壳不足")

    reasoning_score = 1.0
    reasoning = response.reasoning_content
    if "深层" not in reasoning:
        reasoning_score = 0.4
        reasons.append("reasoning 未触达深层需求")
    if scenario.flavored_task and "任务剥离" not in reasoning:
        reasoning_score = min(reasoning_score, 0.5)
        reasons.append("reasoning 缺少任务剥离")
    if scenario.poisoning and "纠偏" not in reasoning:
        reasoning_score = min(reasoning_score, 0.5)
        reasons.append("reasoning 缺少纠偏")

    scores = {
        "persona": round(persona_score, 3),
        "task": round(task_score, 3),
        "reasoning": round(reasoning_score, 3),
    }
    scores["total"] = round(sum(scores.values()) / len(scores), 3)

    accepted = scores["total"] >= 0.72 and not any(reason.startswith("命中黑名单") for reason in reasons)
    return QualityReport(accepted=accepted, reasons=reasons, scores=scores)
