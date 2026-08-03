from __future__ import annotations

import json
import math
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from .factory import make_document
from .intent_boundaries import classify_chinese_braking
from .normalizer import normalize_text
from .rule_parser import RuleIntentParser


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOTYPES_PATH = MODULE_ROOT / "configs" / "semantic_prototypes.jsonl"
_SPEED = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/h|m/s)", re.I)


def _label_key(expected: dict[str, Any]) -> str:
    actions = expected.get("actions")
    unordered = actions is None
    actions = actions or expected.get("actions_unordered", [])
    payload = {
        "status": expected.get("status", "VALID"),
        "actions": sorted(actions) if unordered else actions,
        "unordered": unordered,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class SemanticIntentParser:
    """Low-latency semantic retrieval over reviewed intent prototypes."""

    def __init__(
        self,
        model_path: str,
        *,
        prototypes_path: Path = DEFAULT_PROTOTYPES_PATH,
        similarity_threshold: float = 0.58,
        top_k: int = 7,
        max_length: int = 64,
        device: str = "cpu",
        cpu_threads: int = 1,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        if cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive")
        self.model_path = model_path
        self.prototypes_path = prototypes_path
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.max_length = max_length
        self.device_name = device
        self.cpu_threads = cpu_threads
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: Any = None
        self.prototype_embeddings: Any = None
        self.prototypes = self._load_prototypes(prototypes_path)

    @staticmethod
    def _load_prototypes(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f"Semantic prototype file not found: {path}")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError("Semantic prototype file is empty")
        for row in rows:
            if not isinstance(row.get("text"), str) or not isinstance(
                row.get("expected"), dict
            ):
                raise ValueError("Each semantic prototype needs text and expected")
            row["label_key"] = _label_key(row["expected"])
            row["weight"] = float(row.get("weight", 1.0))
            if not 0.0 < row["weight"] <= 1.0:
                raise ValueError("Semantic prototype weight must be in (0, 1]")
        return rows

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Semantic parser dependencies are missing. Install requirements-model.txt."
            ) from error

        use_cuda = self.device_name == "cuda" or (
            self.device_name == "auto" and torch.cuda.is_available()
        )
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA semantic parsing requested but CUDA is unavailable")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if self.device.type == "cpu":
            torch.set_num_threads(self.cpu_threads)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            dtype=dtype,
        ).to(self.device).eval()
        texts = [row["text"] for row in self.prototypes]
        self.prototype_embeddings = self._encode(texts, batch_size=64).cpu()
        for _ in range(10):
            self._encode([texts[0]], batch_size=1)

    def parse(
        self,
        raw_text: str,
        *,
        modality: str = "TEXT",
        request_id: str | None = None,
    ) -> dict[str, Any] | None:
        self.load()
        started = perf_counter()
        normalized = normalize_text(raw_text)
        query = self._encode([normalized], batch_size=1).cpu()[0]
        similarities = self.prototype_embeddings @ query
        count = min(self.top_k, len(self.prototypes))
        values, indices = similarities.topk(count)
        best_score = float(values[0])
        if best_score < self.similarity_threshold:
            return None

        votes: dict[str, float] = {}
        best_by_label: dict[str, tuple[float, int]] = {}
        for value, index in zip(values.tolist(), indices.tolist()):
            label = self.prototypes[index]["label_key"]
            weight = (
                math.exp((float(value) - best_score) * 10.0)
                * self.prototypes[index]["weight"]
            )
            votes[label] = votes.get(label, 0.0) + weight
            if label not in best_by_label or value > best_by_label[label][0]:
                best_by_label[label] = (float(value), int(index))
        winning_label = max(votes, key=votes.get)
        score, prototype_index = best_by_label[winning_label]
        expected = self.prototypes[prototype_index]["expected"]
        latency_ms = (perf_counter() - started) * 1000
        return self._make_intent(
            raw_text,
            normalized,
            modality,
            request_id,
            expected,
            score,
            latency_ms,
        )

    def _encode(self, texts: list[str], *, batch_size: int) -> Any:
        import torch
        import torch.nn.functional as functional

        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = self.tokenizer(
                    texts[start : start + batch_size],
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                cls_embedding = self.model(**inputs).last_hidden_state[:, 0]
                embeddings.append(
                    functional.normalize(cls_embedding.float(), p=2, dim=1)
                )
        return torch.cat(embeddings, dim=0)

    def _make_intent(
        self,
        raw_text: str,
        normalized: str,
        modality: str,
        request_id: str | None,
        expected: dict[str, Any],
        score: float,
        latency_ms: float,
    ) -> dict[str, Any]:
        status = expected.get("status", "VALID")
        actions = list(
            expected.get("actions") or expected.get("actions_unordered") or []
        )
        directions = self._directions(normalized)
        if not directions:
            directions = list(expected.get("directions", []))

        if status != "VALID":
            return make_document(
                raw_text=raw_text,
                normalized_text=normalized,
                modality=modality,
                category=expected.get("category", "META_CONTROL"),
                urgency="NORMAL",
                steps=[],
                status=status,
                method="HYBRID",
                model=Path(self.model_path).name,
                confidence=score,
                latency_ms=latency_ms,
                request_id=request_id,
                missing_slots=["intent.steps"]
                if status == "NEEDS_CLARIFICATION"
                else [],
                warnings=["轻量语义模型根据已审核原型完成安全分类。"],
                clarification_question="请补充明确的驾驶动作、方向或目标。"
                if status == "NEEDS_CLARIFICATION"
                else None,
            )

        commands: list[dict[str, Any]] = []
        direction_index = 0
        speeds = list(_SPEED.finditer(normalized))
        speed_index = 0
        target_type, target_relation = self._target(normalized)
        for action in actions:
            command: dict[str, Any] = {"action": action}
            if action in {"CHANGE_LANE", "TURN"}:
                if direction_index >= len(directions):
                    return make_document(
                        raw_text=raw_text,
                        normalized_text=normalized,
                        modality=modality,
                        category="NAVIGATION",
                        urgency="NORMAL",
                        steps=[],
                        status="NEEDS_CLARIFICATION",
                        method="HYBRID",
                        model=Path(self.model_path).name,
                        confidence=score,
                        latency_ms=latency_ms,
                        request_id=request_id,
                        missing_slots=["intent.steps[0].parameters.direction"],
                        warnings=["语义匹配到方向动作，但原文没有明确方向。"],
                        clarification_question="请说明向左、向右还是直行。",
                    )
                command["direction"] = directions[direction_index]
                direction_index += 1
                if action == "CHANGE_LANE":
                    command["lane_count"] = 1
            elif action == "SET_SPEED":
                if speed_index >= len(speeds):
                    command["action"] = "ADJUST_SPEED"
                    command["change"] = self._speed_change(normalized)
                else:
                    speed = speeds[speed_index]
                    speed_index += 1
                    value = float(speed.group("value"))
                    unit = speed.group("unit")
                    command.update(
                        target_speed_mps=round(
                            value / 3.6 if unit.lower() == "km/h" else value, 3
                        ),
                        source_value=value,
                        source_unit=unit,
                    )
            elif action == "ADJUST_SPEED":
                command["change"] = self._speed_change(normalized)
            elif action in {"YIELD", "AVOID", "OVERTAKE"}:
                command.update(
                    target_type=target_type,
                    target_relation=target_relation,
                )
            commands.append(command)

        braking_boundary = classify_chinese_braking(normalized)
        if braking_boundary:
            candidate_actions = {"STOP", "EMERGENCY_BRAKE"}
            if braking_boundary.action == "ADJUST_SPEED":
                candidate_actions.add("ADJUST_SPEED")
            for command in commands:
                if command.get("action") not in candidate_actions:
                    continue
                command["action"] = braking_boundary.action
                if braking_boundary.action == "ADJUST_SPEED":
                    command["change"] = "DECREASE"
                else:
                    command.pop("change", None)

        actions = [str(command["action"]) for command in commands]
        steps = RuleIntentParser._expand_fast_commands(commands)
        urgency = (
            braking_boundary.urgency
            if braking_boundary
            else str(expected.get("urgency") or "NORMAL")
        )
        category = expected.get("category") or self._category(actions, urgency)
        if braking_boundary and braking_boundary.action != "EMERGENCY_BRAKE":
            if set(actions) <= {"STOP", "ADJUST_SPEED"}:
                category = "BASIC_CONTROL"
        return make_document(
            raw_text=raw_text,
            normalized_text=normalized,
            modality=modality,
            category=category,
            urgency=urgency,
            steps=steps,
            status="VALID",
            method="HYBRID",
            model=Path(self.model_path).name,
            confidence=score,
            latency_ms=latency_ms,
            request_id=request_id,
            warnings=["轻量语义模型根据已审核原型解析。"],
            driving_style="CONSERVATIVE"
            if re.search(r"安全|减速|避|让|施工|雨|危险", normalized)
            else "NORMAL",
        )

    @staticmethod
    def _directions(text: str) -> list[str]:
        matches: list[tuple[int, str]] = []
        patterns = {
            "LEFT": r"左转|左拐|向左|左侧|左边",
            "RIGHT": r"右转|右拐|向右|右侧|右边",
            "STRAIGHT": r"直行|直走",
        }
        for direction, pattern in patterns.items():
            matches.extend((match.start(), direction) for match in re.finditer(pattern, text))
        return [direction for _, direction in sorted(matches)]

    @staticmethod
    def _speed_change(text: str) -> str:
        return (
            "DECREASE"
            if re.search(r"减|降|慢|刹车|安全车速|缓行", text)
            else "INCREASE"
        )

    @staticmethod
    def _target(text: str) -> tuple[str, str]:
        target_type = "OBSTACLE"
        relation = "AHEAD"
        if re.search(r"行人|人群|小孩|老人", text):
            target_type = "PEDESTRIAN"
            if re.search(r"横穿|过街|斑马线", text):
                relation = "AHEAD_CROSSING"
        elif re.search(r"骑行|自行车|电动车", text):
            target_type = "CYCLIST"
        elif re.search(r"锥桶|路锥", text):
            target_type = "TRAFFIC_CONE"
        elif re.search(r"施工", text):
            target_type = "CONSTRUCTION_ZONE"
        elif re.search(r"慢车|低速车", text):
            target_type = "SLOW_VEHICLE"
        elif re.search(r"车|公交|救护", text):
            target_type = "VEHICLE"
        if "左侧" in text or "左边" in text:
            relation = "LEFT"
        elif "右侧" in text or "右边" in text:
            relation = "RIGHT"
        return target_type, relation

    @staticmethod
    def _category(actions: list[str], urgency: str) -> str:
        if urgency == "EMERGENCY" or "EMERGENCY_BRAKE" in actions:
            return "EMERGENCY_RESPONSE"
        if any(action in {"YIELD", "PULL_OVER", "OVERTAKE", "AVOID"} for action in actions):
            return "COMPLEX_OBSTACLE_AVOIDANCE"
        if "TURN" in actions:
            return "NAVIGATION"
        if set(actions) <= {"CANCEL", "RESUME"}:
            return "META_CONTROL"
        return "BASIC_CONTROL"
