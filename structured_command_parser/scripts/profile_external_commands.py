from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import tarfile


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMLINGO = MODULE_ROOT / "data" / "external" / "simlingo" / (
    "dreamer_simlingo_validation_1_scenario_routes_validation_"
    "random_weather_seed_2_balanced_150_chunk_001.tar.gz"
)
DEFAULT_TALK2CAR = (
    MODULE_ROOT
    / "data"
    / "external"
    / "talk2car"
    / "data"
    / "commands"
    / "val_commands.json"
)


def profile_simlingo(path: Path, max_members: int | None) -> dict[str, object]:
    mode_counts: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    members_read = 0

    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json.gz"):
                continue
            file = archive.extractfile(member)
            if file is None:
                continue
            document = json.loads(gzip.decompress(file.read()))
            members_read += 1
            for category, candidates in document.items():
                if not isinstance(candidates, list):
                    continue
                for candidate in candidates:
                    mode = str(candidate.get("mode") or category)
                    safe = bool(candidate.get("safe_to_execute"))
                    mode_counts[mode] += 1
                    safety_counts[f"{mode}|{'safe' if safe else 'unsafe'}"] += 1
                    instructions = candidate.get("dreamer_instruction") or []
                    if isinstance(instructions, str):
                        instructions = [instructions]
                    for instruction in instructions:
                        if instruction not in examples[mode] and len(examples[mode]) < 3:
                            examples[mode].append(instruction)
            if max_members is not None and members_read >= max_members:
                break

    return {
        "members_read": members_read,
        "candidate_count": sum(mode_counts.values()),
        "mode_counts": dict(mode_counts.most_common()),
        "safety_counts": dict(safety_counts.most_common()),
        "examples": dict(sorted(examples.items())),
    }


def profile_talk2car(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    commands = document["commands"]
    object_counts = Counter(command.get("obj_name", "UNKNOWN") for command in commands)
    return {
        "command_count": len(commands),
        "object_counts": dict(object_counts.most_common()),
        "examples": [command["command"] for command in commands[:5]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile downloaded command data")
    parser.add_argument("--simlingo", type=Path, default=DEFAULT_SIMLINGO)
    parser.add_argument("--talk2car", type=Path, default=DEFAULT_TALK2CAR)
    parser.add_argument("--max-simlingo-members", type=int, default=1000)
    args = parser.parse_args()

    result = {
        "simlingo": profile_simlingo(args.simlingo, args.max_simlingo_members),
        "talk2car": profile_talk2car(args.talk2car),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
