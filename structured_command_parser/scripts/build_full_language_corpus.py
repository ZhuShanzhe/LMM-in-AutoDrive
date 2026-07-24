from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from structured_command_parser.scripts.prepare_external_commands import (
    expected_for_simlingo,
    propose_talk2car_expected,
)


MODULE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = MODULE_ROOT / "data"
CORPUS_ROOT = DATA_ROOT / "corpus"
RAW_ROOT = CORPUS_ROOT / "raw"
PROCESSED_ROOT = CORPUS_ROOT / "processed"
DEFERRED_ROOT = CORPUS_ROOT / "deferred_translation"
MANIFEST_ROOT = CORPUS_ROOT / "manifests"


def normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_talk2car() -> list[dict[str, Any]]:
    command_root = RAW_ROOT / "talk2car" / "commands"
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        path = command_root / f"{split}_commands.json"
        commands = json.loads(path.read_text(encoding="utf-8"))["commands"]
        for command in commands:
            text = " ".join(str(command["command"]).split())
            token = str(command["command_token"])
            rows.append(
                {
                    "sample_id": f"talk2car-{split}-{token}",
                    "source": "Talk2Car",
                    "source_split": split,
                    "source_ref": token,
                    "text_en": text,
                    "text_zh": "",
                    "translation_status": "PENDING_LLM_TRANSLATION_AND_HUMAN_REVIEW",
                    "review_status": "REQUIRES_HUMAN_REVIEW",
                    "proposed_expected": propose_talk2car_expected(text),
                    "metadata": {
                        "object_type": command.get("obj_name"),
                        "box_token": command.get("box_token"),
                        "sample_token": command.get("sample_token"),
                    },
                }
            )
    return rows


def archive_documents(path: Path) -> Iterator[tuple[str, Any]]:
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json.gz"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            try:
                yield member.name, json.loads(gzip.decompress(extracted.read()))
            except (gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
                continue


def build_simlingo_dreamer() -> tuple[list[dict[str, Any]], dict[str, int]]:
    archive_root = RAW_ROOT / "simlingo" / "dreamer_archives"
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    for archive_path in sorted(archive_root.glob("dreamer_*.tar.gz")):
        source_split = "validation" if "validation" in archive_path.name else "training"
        for member_name, document in archive_documents(archive_path):
            if not isinstance(document, dict):
                continue
            stats["members"] += 1
            for category, candidates in document.items():
                if not isinstance(candidates, list):
                    continue
                for candidate_index, candidate in enumerate(candidates):
                    if not isinstance(candidate, dict):
                        continue
                    mode = str(candidate.get("mode") or category)
                    instructions = candidate.get("dreamer_instruction") or []
                    if isinstance(instructions, str):
                        instructions = [instructions]
                    for instruction_index, instruction in enumerate(instructions):
                        text = " ".join(str(instruction).split())
                        if not text:
                            continue
                        key = (normalized(text), mode)
                        stats["instruction_occurrences"] += 1
                        if key in unique:
                            unique[key]["metadata"]["occurrences"] += 1
                            continue
                        candidate_for_label = {
                            **candidate,
                            "mode": mode,
                            "instruction": text,
                        }
                        proposed = expected_for_simlingo(candidate_for_label)
                        digest = hashlib.sha1(
                            f"{mode}\0{text}".encode("utf-8")
                        ).hexdigest()[:16]
                        unique[key] = {
                            "sample_id": f"simlingo-dreamer-{digest}",
                            "source": "SimLingo-Dreamer",
                            "source_split": source_split,
                            "source_ref": (
                                f"{archive_path.name}:{member_name}"
                                f"#{candidate_index}:{instruction_index}"
                            ),
                            "text_en": text,
                            "text_zh": "",
                            "translation_status": "PENDING_LLM_TRANSLATION_AND_HUMAN_REVIEW",
                            "review_status": "REQUIRES_HUMAN_REVIEW",
                            "proposed_expected": proposed,
                            "metadata": {
                                "mode": mode,
                                "allowed": candidate.get("allowed"),
                                "safe_to_execute": candidate.get("safe_to_execute"),
                                "route_reasoning": candidate.get("route_reasoning"),
                                "occurrences": 1,
                            },
                        }
    rows = sorted(unique.values(), key=lambda row: (row["metadata"]["mode"], row["text_en"]))
    stats["unique_instructions"] = len(rows)
    stats["mapped_labels"] = sum(row["proposed_expected"] is not None for row in rows)
    return rows, dict(stats)


def find_commentary(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        commentary = value.get("commentary")
        if isinstance(commentary, str) and commentary.strip():
            yield value
            return
        for child in value.values():
            yield from find_commentary(child)
    elif isinstance(value, list):
        for child in value:
            yield from find_commentary(child)


def build_simlingo_commentary() -> tuple[list[dict[str, Any]], dict[str, int]]:
    archive_root = RAW_ROOT / "simlingo" / "commentary_archives"
    unique: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    for archive_path in sorted(archive_root.glob("commentary_*.tar.gz")):
        source_split = "validation" if "validation" in archive_path.name else "training"
        for member_name, document in archive_documents(archive_path):
            stats["members"] += 1
            for item in find_commentary(document):
                text = " ".join(str(item["commentary"]).split())
                key = normalized(text)
                stats["commentary_occurrences"] += 1
                if key in unique:
                    unique[key]["metadata"]["occurrences"] += 1
                    continue
                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
                unique[key] = {
                    "sample_id": f"simlingo-commentary-{digest}",
                    "source": "SimLingo-Commentary",
                    "source_split": source_split,
                    "source_ref": f"{archive_path.name}:{member_name}",
                    "text_en": text,
                    "commentary_template": item.get("commentary_template"),
                    "use": "TERMINOLOGY_AND_RULE_MINING_ONLY",
                    "metadata": {
                        "scenario_name": item.get("scenario_name"),
                        "cause_object": item.get("cause_object_string"),
                        "occurrences": 1,
                    },
                }
    rows = sorted(unique.values(), key=lambda row: row["text_en"])
    stats["unique_commentary"] = len(rows)
    return rows, dict(stats)


def copy_legacy_bilingual_seed() -> int:
    old_processed = DATA_ROOT / "processed"
    target = DEFERRED_ROOT / "legacy_bilingual_seed_627.jsonl"
    inputs = [
        old_processed / "simlingo_candidates_zh.jsonl",
        old_processed / "talk2car_review_queue_zh.jsonl",
        CORPUS_ROOT / "review_required" / "bilingual_seed_627.jsonl",
        target,
    ]
    rows: list[dict[str, Any]] = []
    for path in inputs:
        if path.is_file():
            rows.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    if not rows and target.is_file():
        rows.extend(
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return write_jsonl(target, rows)


def write_readmes(summary: dict[str, Any]) -> None:
    (CORPUS_ROOT / "verified").mkdir(parents=True, exist_ok=True)
    (CORPUS_ROOT / "model_ready").mkdir(parents=True, exist_ok=True)
    (CORPUS_ROOT / "verified" / "README.md").write_text(
        "# 英文解析金标准\n\n初始为空。后续只存放人工确认的英文指令到意图JSON数据。\n",
        encoding="utf-8",
    )
    (CORPUS_ROOT / "model_ready" / "README.md").write_text(
        "# 英文小模型就绪数据\n\n初始为空。后续从 verified 按来源、场景和模板分组切分train/dev/test。\n",
        encoding="utf-8",
    )
    readme = f"""# 驾驶语言全量语料库

当前阶段只使用英文语料归纳术语和解析规则，不进行中文翻译。

## 当前规模

- Talk2Car 全量英文命令：{summary['talk2car_rows']} 条。
- SimLingo Dreamer 去重指令：{summary['dreamer_rows']} 条。
- SimLingo Commentary 去重文本：{summary['commentary_rows']} 条，仅用于术语和规则挖掘。
- 延后使用的旧中英机器翻译种子：{summary['deferred_bilingual_seed_rows']} 条。
- 延后使用的历史中文开发/旧冻结样本：80/40 条。
- 已人工双人验证：0 条。

## 目录

```text
corpus/
├── README.md
├── raw/
│   ├── talk2car/commands/             # train/val/test 官方命令JSON
│   └── simlingo/
│       ├── dreamer_archives/          # 40个指令标签归档
│       ├── commentary_archives/       # 37个驾驶描述归档
│       └── metadata/                  # 官方README和许可证
├── processed/
│   ├── talk2car_all.jsonl
│   ├── simlingo_dreamer_unique.jsonl
│   └── simlingo_commentary_unique.jsonl
├── knowledge_mining/                  # 英文代表样本、GPT批次和候选知识
├── deferred_translation/              # 旧双语种子和历史中文回归样本
├── verified/                          # 人工双人确认后放入
├── model_ready/                       # 后续按场景切分train/dev/test
└── manifests/                         # 下载、处理和审计统计
```

## 当前工作流

1. 从 processed 构建分层英文代表样本和GPT批次。
2. GPT只归纳英文术语候选与解析规则候选，不翻译中文、不逐条生成最终标签。
3. Commentary只用于上下文术语和语义边界，不作为乘客命令训练样本。
4. 人工确认候选术语和规则后，再生成英文伪标签并抽取金标准。
5. 中文到英文翻译属于后续阶段，deferred_translation 当前不参与处理。
"""
    (CORPUS_ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    talk2car = build_talk2car()
    dreamer, dreamer_stats = build_simlingo_dreamer()
    commentary, commentary_stats = build_simlingo_commentary()
    write_jsonl(PROCESSED_ROOT / "talk2car_all.jsonl", talk2car)
    write_jsonl(PROCESSED_ROOT / "simlingo_dreamer_unique.jsonl", dreamer)
    write_jsonl(PROCESSED_ROOT / "simlingo_commentary_unique.jsonl", commentary)

    bilingual_seed_rows = copy_legacy_bilingual_seed()

    summary = {
        "schema": "full-driving-language-corpus-v2",
        "talk2car_rows": len(talk2car),
        "dreamer_rows": len(dreamer),
        "commentary_rows": len(commentary),
        "deferred_bilingual_seed_rows": bilingual_seed_rows,
        "dreamer_stats": dreamer_stats,
        "commentary_stats": commentary_stats,
    }
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    (MANIFEST_ROOT / "corpus_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readmes(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
