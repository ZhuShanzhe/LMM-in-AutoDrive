from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from structured_command_parser.scripts.evaluate_parser import (
    matches_expected,
    summarize_result,
)
from structured_command_parser.src.rule_parser import RuleIntentParser


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = MODULE_ROOT / "data" / "organized"
ALL_ACTIONS = {
    "ADJUST_SPEED",
    "AVOID",
    "CANCEL",
    "CHANGE_LANE",
    "EMERGENCY_BRAKE",
    "KEEP_LANE",
    "OVERTAKE",
    "PULL_OVER",
    "RESUME",
    "SET_SPEED",
    "STOP",
    "TURN",
    "YIELD",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in materialized),
        encoding="utf-8",
    )
    return len(materialized)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def expected_of(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("expected") or row.get("proposed_expected") or {}


def action_list(expected: dict[str, Any]) -> list[str]:
    return list(expected.get("actions") or expected.get("actions_unordered") or [])


def normalized_row(
    row: dict[str, Any],
    *,
    dataset: str,
    label_quality: str,
    split_role: str,
) -> dict[str, Any]:
    expected = expected_of(row)
    metadata = row.get("metadata") or {}
    return {
        "sample_id": row["sample_id"],
        "dataset": dataset,
        "source": row.get("source", dataset),
        "source_split": row.get("source_split") or metadata.get("split", ""),
        "source_ref": row.get("source_ref", ""),
        "text_zh": row.get("text_zh") or row.get("text") or "",
        "text_en": row.get("text_en") or "",
        "expected": expected,
        "label_quality": label_quality,
        "translation_status": row.get("translation_status", "NOT_AVAILABLE"),
        "review_status": row.get("review_status")
        or metadata.get("review_status", "REQUIRES_HUMAN_REVIEW"),
        "split_role": split_role,
        "metadata": metadata,
    }


def review_row(row: dict[str, Any], rule_confirmed: bool) -> dict[str, Any]:
    expected = row["expected"]
    return {
        "sample_id": row["sample_id"],
        "dataset": row["dataset"],
        "text_zh": row["text_zh"],
        "text_en": row["text_en"],
        "proposed_status": expected.get("status", "VALID"),
        "proposed_actions": json.dumps(action_list(expected), ensure_ascii=False),
        "proposed_directions": json.dumps(
            expected.get("directions", []), ensure_ascii=False
        ),
        "rule_confirmed": "YES" if rule_confirmed else "NO",
        "translation_status": row["translation_status"],
        "review_status": row["review_status"],
        "approved_text_zh": "",
        "approved_text_en": "",
        "approved_status": "",
        "approved_actions": "",
        "approved_directions": "",
        "reviewer_1": "",
        "reviewer_2": "",
        "review_decision": "",
        "notes": "",
    }


def csv_view_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = row["expected"]
    return {
        "sample_id": row["sample_id"],
        "dataset": row["dataset"],
        "split_role": row["split_role"],
        "text_zh": row["text_zh"],
        "text_en": row["text_en"],
        "status": expected.get("status", "VALID"),
        "actions": json.dumps(action_list(expected), ensure_ascii=False),
        "directions": json.dumps(expected.get("directions", []), ensure_ascii=False),
        "label_quality": row["label_quality"],
        "translation_status": row["translation_status"],
        "review_status": row["review_status"],
        "source_ref": row["source_ref"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    data_root = MODULE_ROOT / "data"
    processed = data_root / "processed"

    golden = [
        normalized_row(
            row,
            dataset="local_golden_regression",
            label_quality="DEVELOPMENT_REGRESSION",
            split_role="development_regression",
        )
        for row in load_jsonl(MODULE_ROOT / "tests" / "fixtures" / "golden_commands.jsonl")
    ]
    chinese_dev = [
        normalized_row(
            row,
            dataset="chinese_diverse_dev",
            label_quality="CURATED_SYNTHETIC_REVIEW_REQUIRED",
            split_role="development",
        )
        for row in load_jsonl(processed / "chinese_diverse_dev.jsonl")
    ]
    legacy_holdout = [
        normalized_row(
            row,
            dataset="chinese_diverse_legacy_holdout",
            label_quality="LEGACY_HOLDOUT_REVIEW_REQUIRED",
            split_role="legacy_evaluation_only",
        )
        for row in load_jsonl(processed / "chinese_diverse_holdout.jsonl")
    ]
    simlingo = [
        normalized_row(
            row,
            dataset="simlingo",
            label_quality="SOURCE_MODE_MAPPED_REVIEW_REQUIRED",
            split_role="bootstrap_candidate",
        )
        for row in load_jsonl(processed / "simlingo_candidates_zh.jsonl")
    ]
    talk2car = [
        normalized_row(
            row,
            dataset="talk2car",
            label_quality="HEURISTIC_LABEL_REVIEW_REQUIRED",
            split_role="bootstrap_candidate",
        )
        for row in load_jsonl(processed / "talk2car_review_queue_zh.jsonl")
    ]

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    write_jsonl(output / "01_development" / "golden_regression_30.jsonl", golden)
    write_jsonl(output / "01_development" / "chinese_dev_80.jsonl", chinese_dev)
    write_jsonl(
        output / "02_legacy_evaluation" / "legacy_holdout_40.jsonl",
        legacy_holdout,
    )
    write_jsonl(output / "03_bilingual_candidates" / "simlingo_327.jsonl", simlingo)
    write_jsonl(output / "03_bilingual_candidates" / "talk2car_300.jsonl", talk2car)

    rule_parser = RuleIntentParser()
    bilingual = simlingo + talk2car
    rule_confirmed: list[dict[str, Any]] = []
    review_priority: list[dict[str, Any]] = []
    confirmation: dict[str, bool] = {}
    for row in bilingual:
        intent = rule_parser.parse(row["text_zh"])
        confirmed = intent is not None and matches_expected(
            summarize_result(intent), row["expected"]
        )
        confirmation[row["sample_id"]] = confirmed
        target = rule_confirmed if confirmed else review_priority
        target.append({**row, "rule_confirmation": "MATCH" if confirmed else "NO_MATCH"})

    write_jsonl(
        output / "04_model_inputs" / "english_parser_rule_confirmed.jsonl",
        rule_confirmed,
    )
    write_jsonl(
        output / "04_model_inputs" / "english_parser_review_required.jsonl",
        review_priority,
    )
    write_jsonl(
        output / "04_model_inputs" / "llm_terminology_rule_mining.jsonl",
        golden + chinese_dev + bilingual,
    )

    review_rows = [
        review_row(row, confirmation[row["sample_id"]]) for row in bilingual
    ]
    review_fields = list(review_rows[0])
    write_csv(output / "05_human_review" / "bilingual_review_627.csv", review_rows, review_fields)

    all_rows = golden + chinese_dev + legacy_holdout + bilingual
    view_rows = [csv_view_row(row) for row in all_rows]
    write_csv(output / "all_samples_777.csv", view_rows, list(view_rows[0]))
    write_csv(
        output / "03_bilingual_candidates" / "bilingual_candidates_627.csv",
        [csv_view_row(row) for row in bilingual],
        list(view_rows[0]),
    )

    runtime = output / "06_runtime_reference"
    runtime.mkdir(parents=True)
    shutil.copy2(
        MODULE_ROOT / "configs" / "translation_glossary.json",
        runtime / "translation_glossary.json",
    )
    shutil.copy2(
        MODULE_ROOT / "configs" / "semantic_prototypes.jsonl",
        runtime / "semantic_prototypes_328.jsonl",
    )

    action_counts = Counter(
        action for row in bilingual for action in action_list(row["expected"])
    )
    summary = {
        "schema": "organized-command-corpus-v1",
        "total_view_rows": len(all_rows),
        "development_regression": len(golden),
        "chinese_development": len(chinese_dev),
        "legacy_evaluation": len(legacy_holdout),
        "bilingual_candidates": len(bilingual),
        "rule_confirmed_bootstrap": len(rule_confirmed),
        "review_priority": len(review_priority),
        "human_verified_bilingual": 0,
        "bilingual_action_counts": dict(sorted(action_counts.items())),
        "missing_bilingual_actions": sorted(ALL_ACTIONS - set(action_counts)),
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    catalog = [
        {
            "path": str(path.relative_to(output)),
            "size_bytes": path.stat().st_size,
            "format": path.suffix.lstrip("."),
            "purpose": "直接查看或按 README 指定用途使用",
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    write_csv(output / "dataset_catalog.csv", catalog, list(catalog[0]))

    readme = f"""# 结构化驾驶指令数据门户

本目录由 `scripts/organize_command_datasets.py` 生成。原始数据保持在 `../external/` 和 `../processed/`，这里仅存放体积很小、可直接查看的统一视图。

## 是否足够

| 目标 | 当前结论 | 依据 |
|---|---|---|
| LLM 离线抽取中英术语表和解析规则 | 足够启动 | {len(bilingual)} 条中英候选、{len(golden) + len(chinese_dev)} 条中文开发样本和现有术语表 |
| 英文小模型基线训练 | 有条件足够 | {len(rule_confirmed)} 条规则一致样本可作 bootstrap，另有 {len(review_priority)} 条需优先审核 |
| 正式模型训练与 95% 指标验收 | 不足 | 当前人工双人确认的中英金标准为 0 条 |

## 目录

```text
organized/
├── README.md
├── all_samples_777.csv                 # Excel/VS Code 直接查看全部样本
├── dataset_catalog.csv                 # 文件索引
├── dataset_summary.json                # 数量与动作覆盖统计
├── 01_development/                     # 已参与开发的 30+80 条样本
├── 02_legacy_evaluation/               # 已使用过的 40 条旧测试，仅回归
├── 03_bilingual_candidates/            # SimLingo 327 + Talk2Car 300
├── 04_model_inputs/
│   ├── english_parser_rule_confirmed.jsonl
│   ├── english_parser_review_required.jsonl
│   └── llm_terminology_rule_mining.jsonl
├── 05_human_review/
│   └── bilingual_review_627.csv        # 双人审核工作表
└── 06_runtime_reference/               # 当前术语表和语义原型快照
```

## 推荐使用顺序

1. 将 `04_model_inputs/llm_terminology_rule_mining.jsonl` 交给大模型，提取候选术语、同义词、否定规则、动作顺序规则和槽位规则。
2. 人工合并结果到 `06_runtime_reference/translation_glossary.json` 的工作副本，不直接覆盖运行时配置。
3. 从 `05_human_review/bilingual_review_627.csv` 开始双人审核，优先处理 `rule_confirmed=NO`。
4. 只有 `review_decision=APPROVED` 且两位审核者一致的记录才能进入正式 `train/dev/test`。
5. 按来源场景分组切分，禁止同一句或近义改写跨越 train/test。

## 数据限制

- `rule_confirmed` 只表示中文规则输出与候选标签一致，不等于人工金标准。
- Talk2Car 标签是启发式建议，已经发现个别动作映射错误。
- SimLingo 标签来自 source mode 映射，适合工程启动，但覆盖集中在设速、变道和碰撞拒绝。
- 现有双语候选缺少的动作：{', '.join(summary['missing_bilingual_actions']) or '无'}。
- `02_legacy_evaluation` 已被多次使用，不能再次宣称为独立冻结测试集。

## 重新生成

```bash
cd /root/autodl-tmp/LMM-in-AutoDrive
conda activate /root/autodl-tmp/conda_envs/command_parser
PYTHONPATH=. python -m structured_command_parser.scripts.organize_command_datasets
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
