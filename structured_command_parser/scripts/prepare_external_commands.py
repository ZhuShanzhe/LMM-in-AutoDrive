from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random
import re
import tarfile
from typing import Any, Iterator

from ..src.intent_boundaries import classify_english_braking


MODULE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = MODULE_ROOT / "data"
DEFAULT_SIMLINGO = DATA_ROOT / "external" / "simlingo" / (
    "dreamer_simlingo_validation_1_scenario_routes_validation_"
    "random_weather_seed_2_balanced_150_chunk_001.tar.gz"
)
DEFAULT_TALK2CAR = (
    DATA_ROOT
    / "external"
    / "talk2car"
    / "data"
    / "commands"
    / "val_commands.json"
)
DEFAULT_OUTPUT = DATA_ROOT / "processed"

SIMLINGO_QUOTAS = {
    "target_speed": 100,
    "faster": 60,
    "faster_factor": 60,
    "slower": 60,
    "slower_factor": 60,
    "stop": 100,
    "lane_change": 100,
    "crash": 80,
}

CRASH_TERMS = (
    "crash",
    "collide",
    "collision",
    "impact",
    "hit ",
    "ram ",
)

MODE_TERMS = {
    "faster": ("accelerat", "speed up", "drive faster", "increase", "boost", "gas"),
    "faster_factor": ("accelerat", "speed up", "drive faster", "increase", "boost", "gas"),
    "slower": ("slow", "brake", "decelerat", "reduce", "ease off", "cut down"),
    "slower_factor": ("slow", "brake", "decelerat", "reduce", "ease off", "cut down"),
    "stop": ("stop", "halt", "cease", "standstill", "hold position"),
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_simlingo_candidates(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json.gz"):
                continue
            file = archive.extractfile(member)
            if file is None:
                continue
            document = json.loads(gzip.decompress(file.read()))
            for category, candidates in document.items():
                if not isinstance(candidates, list):
                    continue
                for candidate_index, candidate in enumerate(candidates):
                    mode = str(candidate.get("mode") or category)
                    instructions = candidate.get("dreamer_instruction") or []
                    if isinstance(instructions, str):
                        instructions = [instructions]
                    for instruction_index, instruction in enumerate(instructions):
                        source_ref = f"{member.name}#{candidate_index}:{instruction_index}"
                        yield source_ref, {**candidate, "mode": mode, "instruction": instruction}


def expected_for_simlingo(candidate: dict[str, Any]) -> dict[str, Any] | None:
    mode = candidate["mode"]
    instruction = candidate["instruction"].casefold()
    mentions_crash = any(term in instruction for term in CRASH_TERMS)
    if mode == "crash" and not mentions_crash:
        return None
    if mode != "crash" and mentions_crash:
        return None
    if mode in MODE_TERMS and not any(term in instruction for term in MODE_TERMS[mode]):
        return None
    if mode == "target_speed":
        speed_match = re.search(
            r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/h|m/s)", instruction
        )
        if speed_match is None:
            return None
        target_speed = float(speed_match.group("value"))
        if speed_match.group("unit") == "km/h":
            target_speed /= 3.6
        if not 0.1 <= target_speed <= 40:
            return None
        return {
            "status": "VALID",
            "category": "BASIC_CONTROL",
            "actions": ["SET_SPEED"],
            "target_speed_mps": [round(float(target_speed), 3)],
        }
    if mode in {"faster", "faster_factor", "slower", "slower_factor"}:
        braking_boundary = classify_english_braking(instruction)
        if mode.startswith("slower") and braking_boundary and (
            braking_boundary.action == "EMERGENCY_BRAKE"
        ):
            return {
                "status": "VALID",
                "category": "EMERGENCY_RESPONSE",
                "urgency": "EMERGENCY",
                "actions": ["EMERGENCY_BRAKE"],
            }
        return {
            "status": "VALID",
            "category": "BASIC_CONTROL",
            "urgency": (
                braking_boundary.urgency
                if mode.startswith("slower") and braking_boundary
                else "NORMAL"
            ),
            "actions": ["ADJUST_SPEED"],
        }
    if mode == "stop":
        braking_boundary = classify_english_braking(instruction)
        if braking_boundary and braking_boundary.action == "EMERGENCY_BRAKE":
            return {
                "status": "VALID",
                "category": "EMERGENCY_RESPONSE",
                "urgency": "EMERGENCY",
                "actions": ["EMERGENCY_BRAKE"],
            }
        return {
            "status": "VALID",
            "category": "BASIC_CONTROL",
            "urgency": braking_boundary.urgency if braking_boundary else "NORMAL",
            "actions": ["STOP"],
        }
    if mode == "lane_change":
        has_left = bool(re.search(r"\bleft\b", instruction))
        has_right = bool(re.search(r"\bright\b", instruction))
        if has_left == has_right:
            return None
        return {
            "status": "VALID",
            "category": "BASIC_CONTROL",
            "actions": ["CHANGE_LANE"],
            "directions": ["LEFT" if has_left else "RIGHT"],
        }
    if mode == "crash":
        return {
            "status": "UNSUPPORTED",
            "category": "BASIC_CONTROL",
            "actions": [],
        }
    return None


def prepare_simlingo(path: Path) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_instructions: set[str] = set()
    for source_ref, candidate in iter_simlingo_candidates(path):
        mode = candidate["mode"]
        if mode not in SIMLINGO_QUOTAS or counts[mode] >= SIMLINGO_QUOTAS[mode]:
            continue
        instruction = " ".join(str(candidate["instruction"]).split())
        normalized = instruction.casefold()
        if not instruction or normalized in seen_instructions:
            continue
        expected = expected_for_simlingo(candidate)
        if expected is None:
            continue
        seen_instructions.add(normalized)
        counts[mode] += 1
        selected.append(
            {
                "sample_id": "",
                "parser_path": "EXTERNAL",
                "source": "SimLingo-Dreamer",
                "source_split": "validation",
                "source_ref": source_ref,
                "text_en": instruction,
                "text_zh": None,
                "text": None,
                "translation_status": "PENDING_MACHINE_TRANSLATION",
                "review_status": "REQUIRES_HUMAN_REVIEW",
                "expected": expected,
                "metadata": {
                    "mode": mode,
                    "allowed": bool(candidate.get("allowed")),
                    "safe_to_execute": bool(candidate.get("safe_to_execute")),
                    "route_reasoning": candidate.get("route_reasoning"),
                },
            }
        )
        if all(counts[mode] >= quota for mode, quota in SIMLINGO_QUOTAS.items()):
            break

    selected.sort(key=lambda row: (row["metadata"]["mode"], row["text_en"]))
    for index, row in enumerate(selected, start=1):
        row["sample_id"] = f"simlingo-{index:04d}"
    return selected


def propose_talk2car_expected(command: str) -> dict[str, Any]:
    text = command.casefold()
    actions: list[str] = []
    directions: list[str] = []
    if any(token in text for token in ("let ", "allow ", "yield", "give way")):
        actions.append("YIELD")
    if any(token in text for token in ("park", "pull over", "pick up", "drop off")):
        actions.append("PULL_OVER")
    if "lane" in text and any(token in text for token in ("switch", "change", "merge")):
        actions.append("CHANGE_LANE")
        if re.search(r"\bleft\b", text):
            directions.append("LEFT")
        elif re.search(r"\bright\b", text):
            directions.append("RIGHT")
    elif "turn" in text:
        actions.append("TURN")
        if re.search(r"\bleft\b", text):
            directions.append("LEFT")
        elif re.search(r"\bright\b", text):
            directions.append("RIGHT")
    if any(token in text for token in ("stop", "halt")) and "PULL_OVER" not in actions:
        actions.append("STOP")
    if any(token in text for token in ("avoid", "go around", "drive around")):
        actions.append("AVOID")
    if not actions:
        actions.append("RESUME")
    category = "NAVIGATION" if any(action in {"TURN", "PULL_OVER"} for action in actions) else "BASIC_CONTROL"
    proposed: dict[str, Any] = {
        "status": "VALID",
        "category": category,
        "actions": actions,
    }
    if directions:
        proposed["directions"] = directions
    return proposed


def prepare_talk2car(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    commands = json.loads(path.read_text(encoding="utf-8"))["commands"]
    random.Random(seed).shuffle(commands)
    selected = commands[:count]
    rows = []
    for index, command in enumerate(selected, start=1):
        rows.append(
            {
                "sample_id": f"talk2car-{index:04d}",
                "source": "Talk2Car",
                "source_split": "validation",
                "source_ref": command["command_token"],
                "text_en": " ".join(command["command"].split()),
                "text_zh": None,
                "translation_status": "PENDING_MACHINE_TRANSLATION",
                "review_status": "REQUIRES_HUMAN_REVIEW",
                "proposed_expected": propose_talk2car_expected(command["command"]),
                "metadata": {
                    "object_type": command.get("obj_name"),
                    "box_token": command.get("box_token"),
                    "sample_token": command.get("sample_token"),
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare external driving commands")
    parser.add_argument("--simlingo", type=Path, default=DEFAULT_SIMLINGO)
    parser.add_argument("--talk2car", type=Path, default=DEFAULT_TALK2CAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--talk2car-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    simlingo = prepare_simlingo(args.simlingo)
    talk2car = prepare_talk2car(args.talk2car, args.talk2car_count, args.seed)
    write_jsonl(args.output / "simlingo_candidates_en.jsonl", simlingo)
    write_jsonl(args.output / "talk2car_review_queue_en.jsonl", talk2car)
    print(f"simlingo_candidates: {len(simlingo)}")
    print(
        "simlingo_modes: "
        + json.dumps(
            dict(Counter(row["metadata"]["mode"] for row in simlingo)),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print(f"talk2car_review_queue: {len(talk2car)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
