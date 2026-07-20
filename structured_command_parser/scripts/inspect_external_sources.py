from __future__ import annotations

import argparse

from huggingface_hub import HfApi


def format_size(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect external dataset files")
    parser.add_argument("--repo", default="RenzKa/simlingo")
    parser.add_argument("--pattern", default="dreamer")
    args = parser.parse_args()

    api = HfApi()
    pattern = args.pattern.casefold()
    matched = []
    for entry in api.list_repo_tree(
        args.repo,
        repo_type="dataset",
        recursive=True,
        expand=True,
    ):
        path = getattr(entry, "path", "")
        if pattern in path.casefold():
            matched.append((path, getattr(entry, "size", None)))

    total = sum(size or 0 for _, size in matched)
    print(f"repo: {args.repo}")
    print(f"pattern: {args.pattern}")
    print(f"files: {len(matched)}")
    print(f"total: {format_size(total)}")
    for path, size in matched:
        print(f"{format_size(size):>12}  {path}")


if __name__ == "__main__":
    main()
