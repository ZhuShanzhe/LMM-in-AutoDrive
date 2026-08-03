from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from structured_command_parser.src.schema_tools import (  # noqa: E402
    schema_errors,
    semantic_errors,
)


EXAMPLES_DIR = ROOT / "examples"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    failed = False
    example_paths = sorted(EXAMPLES_DIR.glob("*.json"))

    for path in example_paths:
        document = load_json(path)
        errors = schema_errors(document)
        errors.extend(semantic_errors(document))
        if errors:
            failed = True
            print(f"FAIL {path.name}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {path.name}")

    if failed:
        return 1

    print(f"Validated {len(example_paths)} example files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
