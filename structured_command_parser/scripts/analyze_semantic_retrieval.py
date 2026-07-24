from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from structured_command_parser.src.semantic_parser import (
    DEFAULT_PROTOTYPES_PATH,
    SemanticIntentParser,
    _label_key,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prototypes", type=Path, default=DEFAULT_PROTOTYPES_PATH)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[10, 20, 30])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70])
    args = parser.parse_args()

    semantic = SemanticIntentParser(
        args.model,
        prototypes_path=args.prototypes,
        similarity_threshold=0.0,
        top_k=max(args.top_k),
    )
    semantic.load()
    embeddings = semantic.prototype_embeddings
    labels = [row["label_key"] for row in semantic.prototypes]
    similarities = embeddings @ embeddings.T
    similarities.fill_diagonal_(-2.0)

    best: tuple[float, int, float] | None = None
    predictions: dict[tuple[int, float], list[tuple[str, float]]] = {}
    for top_k in args.top_k:
        values, indices = similarities.topk(min(top_k, len(labels) - 1), dim=1)
        for temperature in args.temperatures:
            output: list[tuple[str, float]] = []
            for row_values, row_indices in zip(values.tolist(), indices.tolist()):
                best_score = float(row_values[0])
                votes: dict[str, float] = {}
                for score, index in zip(row_values, row_indices):
                    label = labels[index]
                    weight = math.exp((float(score) - best_score) * temperature)
                    votes[label] = votes.get(label, 0.0) + weight
                output.append((max(votes, key=votes.get), best_score))
            predictions[(top_k, temperature)] = output
            correct = sum(predicted == expected for (predicted, _), expected in zip(output, labels))
            accuracy = correct / len(labels)
            print(f"top_k={top_k} temperature={temperature:g} exact={accuracy:.2%}")
            if best is None or accuracy > best[0]:
                best = (accuracy, top_k, temperature)

    assert best is not None
    accuracy, top_k, temperature = best
    output = predictions[(top_k, temperature)]
    print(f"best: top_k={top_k} temperature={temperature:g} exact={accuracy:.2%}")
    for threshold in args.thresholds:
        accepted = [
            (predicted, expected)
            for (predicted, score), expected in zip(output, labels)
            if score >= threshold
        ]
        correct = sum(predicted == expected for predicted, expected in accepted)
        rate = correct / len(accepted) if accepted else 0.0
        print(
            f"threshold={threshold:.2f} coverage={len(accepted) / len(labels):.2%} "
            f"accepted_exact={rate:.2%} ({correct}/{len(accepted)})"
        )

    confusion = Counter(
        (expected, predicted)
        for (predicted, _), expected in zip(output, labels)
        if predicted != expected
    )
    for (expected, predicted), count in confusion.most_common(12):
        print(
            "confusion: "
            + json.dumps(
                {
                    "count": count,
                    "expected": json.loads(expected),
                    "predicted": json.loads(predicted),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
