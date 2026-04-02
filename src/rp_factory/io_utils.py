from __future__ import annotations

import json
from pathlib import Path

from .models import DatasetRecord, Scenario


def load_scenarios(path: str | Path) -> list[Scenario]:
    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("场景文件必须是 JSON 数组")
    return [Scenario.from_dict(item) for item in payload]


def write_jsonl(path: str | Path, records: list[DatasetRecord]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
