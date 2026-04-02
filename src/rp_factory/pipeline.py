from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any

from .backends import MentorBackend, TargetModelBackend, TeacherBackend
from .models import (
    DatasetRecord,
    DiversityState,
    QualityReport,
    Scenario,
    TeacherResponse,
    dataclass_to_dict,
)
from .io_utils import load_scenarios
from .quality import run_quality_funnel
from .target_filter import evaluate_teacher_gain
from .user_generation import generate_user_bundle
from .diversity import summarize_diversity


@dataclass(slots=True)
class BuildResult:
    records: list[DatasetRecord]
    batch_report: dict[str, Any]


class PipelineA:
    def __init__(self, mentor: MentorBackend, teacher: TeacherBackend) -> None:
        self.mentor = mentor
        self.teacher = teacher

    def run(
        self,
        scenario: Scenario,
        rng: Random,
        diversity_state: DiversityState,
    ) -> tuple[DatasetRecord | None, QualityReport]:
        user = generate_user_bundle(scenario, rng, diversity_state)
        hint = self.mentor.build_hint(scenario, user)
        response = self.teacher.generate(scenario, user, hint=hint)
        mentor_scores = self.mentor.score_candidate(scenario, user, response)
        response.scores = mentor_scores
        report = run_quality_funnel(scenario, response)
        if not report.accepted:
            return None, report
        record = DatasetRecord(
            messages=[
                {"role": "system", "content": scenario.system_prompt},
                {"role": "user", "content": user.user_message},
                {"role": "assistant", "content": response.content},
            ],
            reasoning_content=response.reasoning_content,
            _meta={
                "pipeline": "A",
                "scenario": scenario.name,
                "user_bundle": dataclass_to_dict(user),
                "hint_used": True,
                "teacher_scores": mentor_scores,
                "quality_report": asdict(report),
            },
        )
        return record, report


class PipelineB:
    def __init__(self, mentor: MentorBackend, teacher: TeacherBackend, best_of_n: int = 4) -> None:
        self.mentor = mentor
        self.teacher = teacher
        self.best_of_n = max(best_of_n, 1)

    def _select_best(
        self,
        scenario: Scenario,
        user,
        candidates: list[TeacherResponse],
    ) -> tuple[TeacherResponse | None, QualityReport]:
        scored: list[tuple[float, TeacherResponse, QualityReport]] = []
        best_report = QualityReport(accepted=False, reasons=["无候选样本"], scores={"total": 0.0})
        for candidate in candidates:
            mentor_scores = self.mentor.score_candidate(scenario, user, candidate)
            candidate.scores = mentor_scores
            report = run_quality_funnel(scenario, candidate)
            if report.accepted:
                composite = mentor_scores["total"] * 0.6 + report.scores["total"] * 0.4
                scored.append((composite, candidate, report))
            elif report.scores.get("total", 0.0) > best_report.scores.get("total", 0.0):
                best_report = report
        if not scored:
            return None, best_report
        scored.sort(key=lambda item: item[0], reverse=True)
        _, winner, winner_report = scored[0]
        return winner, winner_report

    def run(
        self,
        scenario: Scenario,
        rng: Random,
        diversity_state: DiversityState,
    ) -> tuple[DatasetRecord | None, QualityReport]:
        user = generate_user_bundle(scenario, rng, diversity_state)
        candidates = [
            self.teacher.generate_candidate(scenario, user, sample_idx=index)
            for index in range(self.best_of_n)
        ]
        winner, report = self._select_best(scenario, user, candidates)
        if winner is None:
            return None, report
        record = DatasetRecord(
            messages=[
                {"role": "system", "content": scenario.system_prompt},
                {"role": "user", "content": user.user_message},
                {"role": "assistant", "content": winner.content},
            ],
            reasoning_content=winner.reasoning_content,
            _meta={
                "pipeline": "B",
                "scenario": scenario.name,
                "user_bundle": dataclass_to_dict(user),
                "teacher_scores": winner.scores,
                "quality_report": asdict(report),
                "best_of_n": self.best_of_n,
            },
        )
        return record, report


def _should_use_pipeline_a(index: int, used_a: int, mix_ratio: float) -> bool:
    desired_a_after_this_item = round((index + 1) * mix_ratio)
    return used_a < desired_a_after_this_item


def build_dataset(
    input_path: str | Path,
    seed: int = 7,
    best_of_n: int = 4,
    mix_ratio: float = 0.5,
    min_gain_delta: float = 0.08,
) -> BuildResult:
    scenarios = load_scenarios(input_path)
    rng = Random(seed)
    mentor = MentorBackend()
    teacher = TeacherBackend(seed=seed)
    target = TargetModelBackend(seed=seed + 17)
    pipeline_a = PipelineA(mentor=mentor, teacher=teacher)
    pipeline_b = PipelineB(mentor=mentor, teacher=teacher, best_of_n=best_of_n)
    diversity_state = DiversityState()

    records: list[DatasetRecord] = []
    rejected: list[dict[str, Any]] = []
    used_a = 0
    used_b = 0

    for index, scenario in enumerate(scenarios):
        use_a = _should_use_pipeline_a(index, used_a, mix_ratio)
        pipeline = pipeline_a if use_a else pipeline_b
        record, quality_report = pipeline.run(scenario, rng, diversity_state)
        if record is None:
            rejected.append(
                {
                    "scenario": scenario.name,
                    "stage": "quality_funnel",
                    "pipeline": "A" if use_a else "B",
                    "reasons": quality_report.reasons,
                    "scores": quality_report.scores,
                }
            )
            continue

        gain_report = evaluate_teacher_gain(
            scenario=scenario,
            record=record,
            target_backend=target,
            min_gain_delta=min_gain_delta,
        )
        record._meta["gain_report"] = gain_report
        if not gain_report["accepted"]:
            rejected.append(
                {
                    "scenario": scenario.name,
                    "stage": "target_model_loop",
                    "pipeline": record._meta["pipeline"],
                    "reasons": gain_report["reasons"],
                    "scores": {
                        "teacher_total": gain_report["teacher_total"],
                        "target_total": gain_report["target_total"],
                        "delta": gain_report["delta"],
                    },
                }
            )
            continue

        if use_a:
            used_a += 1
        else:
            used_b += 1
        records.append(record)

    batch_report = {
        "input_scenarios": len(scenarios),
        "accepted_records": len(records),
        "rejected_records": len(rejected),
        "pipeline_usage": {"A": used_a, "B": used_b},
        "diversity": summarize_diversity(records),
        "rejections": rejected,
    }
    return BuildResult(records=records, batch_report=batch_report)
