from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.src.teacher import (
    JsonlTeacherStore,
    compare_action_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare student and teacher actions")
    parser.add_argument("--student-jsonl", required=True)
    parser.add_argument("--teacher-jsonl", required=True)
    args = parser.parse_args()
    student_records = []
    with Path(args.student_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                student_records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{args.student_jsonl}:{line_number}: {exc}"
                ) from exc
    metrics = compare_action_predictions(
        student_records,
        JsonlTeacherStore.from_path(args.teacher_jsonl),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
