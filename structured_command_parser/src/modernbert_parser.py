from __future__ import annotations

from pathlib import Path
import json
import re
from time import perf_counter
from typing import Any

import torch
from transformers import AutoTokenizer

from .compositional_frame import (
    decode_token_spans,
    enrich_commands_with_frame,
    extract_semantic_frame,
)
from .english_parser import QwenEnglishIntentParser
from .factory import make_document
from .llm_parser import QwenIntentParser
from .modernbert_labels import (
    ACTION_LABELS,
    CATEGORY_LABELS,
    CHANGE_LABELS,
    DIRECTION_LABELS,
    STATUS_LABELS,
    URGENCY_LABELS,
)
from .modernbert_model import ModernBertDrivingModel
from .schema_tools import semantic_errors
from .semantic_decomposer import decompose_atomic_actions
from .speed_slots import restore_source_target_speeds
from .semantic_normalizer import (
    filter_suppressed_actions,
    normalize_semantics,
)


ACTION_PATTERNS = {
    "KEEP_LANE": r"\b(?:keep|maintain|stay|remain).{0,20}\blane\b",
    "SET_SPEED": (
        r"\b(?:set|maintain|hold|keep|drive at|cruise at|speed up to|accelerate to|"
        r"slow down to|decelerate to|reduce(?: (?:the )?speed)? to)"
        r".{0,24}\b(?:km/h|kmph|kph|m/s|"
        r"kilometers? per hour|kilometres? per hour|"
        r"meters? per second|metres? per second)\b"
    ),
    "ADJUST_SPEED": r"\b(?:accelerate|speed up|slow down|decelerate|brake|reduce speed)\b",
    "STOP": r"\b(?:stop|halt|standstill|cease all movement)\b",
    "WAIT": r"\b(?:wait|hold position|stay put)\b",
    "FOLLOW": r"\b(?:follow|trail|stay behind|catch up)\b",
    "APPROACH": r"\b(?:approach|move closer|drive closer|get closer)\b",
    "NAVIGATE_TO": r"\b(?:go to|drive to|head to|navigate to|take me to)\b",
    "CHANGE_LANE": r"\b(?:change|switch|shift|move|transition).{0,18}\blane\b",
    "MERGE": r"\bmerge\b",
    "TURN": r"\bturn\b|\bmake a (?:left|right)\b",
    "U_TURN": r"\bu[- ]?turn\b|\bturn around\b",
    "PROCEED": r"\b(?:proceed|go forward|go ahead|continue straight|drive through)\b",
    "YIELD": r"\b(?:yield|give way|let .+ pass|allow .+ to pass)\b",
    "PULL_OVER": r"\b(?:pull over|pull up|roadside)\b",
    "PARK": r"\bpark(?:ing)?\b",
    "OVERTAKE": r"\b(?:overtake|get past)\b",
    "PASS_BY": r"\b(?:drive past|go past|pass by)\b",
    "AVOID": r"\b(?:avoid|go around|drive around|steer clear|maneuver around)\b",
    "REVERSE": r"\b(?:reverse|back up)\b",
    "ENTER_AREA": r"\benter\b",
    "EXIT_AREA": r"\b(?:exit|leave)\b",
    "EMERGENCY_BRAKE": r"\b(?:emergency brake|slam .+ brake|brake hard|hard brak|full braking)\b",
    "RESUME": r"\b(?:resume|return to the original)\b",
    "CANCEL": r"\b(?:cancel|abort|revoke|never mind|changed my mind|ignore what i just asked)\b",
}


class ModernBertEnglishIntentParser:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        max_length: int = 96,
        action_threshold: float = 0.5,
        direction_threshold: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.device_name = device
        self.max_length = max_length
        self.action_threshold = action_threshold
        self.direction_threshold = direction_threshold
        self.action_thresholds = {
            action: action_threshold for action in ACTION_LABELS
        }
        self.direction_thresholds = {
            direction: direction_threshold for direction in DIRECTION_LABELS
        }
        self.device: torch.device | None = None
        self.tokenizer: Any = None
        self.model: ModernBertDrivingModel | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("ModernBERT parser requested CUDA but CUDA is unavailable")
        self.device = torch.device(self.device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = ModernBertDrivingModel.from_pretrained(
            self.model_path, dtype=dtype
        ).to(self.device)
        inference_config = Path(self.model_path) / "inference_config.json"
        if inference_config.is_file():
            config = json.loads(inference_config.read_text(encoding="utf-8"))
            self.action_thresholds.update(config.get("action_thresholds", {}))
            self.direction_thresholds.update(config.get("direction_thresholds", {}))
        self.model.eval()

    def warmup(self) -> None:
        self.load()
        self.parse("Keep the current lane.", request_id="modernbert-warmup")

    @torch.inference_mode()
    def parse(
        self,
        text: str,
        *,
        modality: str = "TEXT",
        request_id: str | None = None,
        source_text: str | None = None,
        source_language: str | None = None,
    ) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("English command cannot be empty")
        normalization = normalize_semantics(text, source_text=source_text)
        normalized = normalization["normalized_text"]
        self.load()
        assert self.model is not None and self.tokenizer is not None and self.device is not None
        started = perf_counter()
        encoded = self.tokenizer(
            normalized,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(**encoded)
        probabilities = {
            "actions": torch.sigmoid(logits["actions"])[0].float().cpu(),
            "directions": torch.sigmoid(logits["directions"])[0].float().cpu(),
            "status": torch.softmax(logits["status"], dim=-1)[0].float().cpu(),
            "category": torch.softmax(logits["category"], dim=-1)[0].float().cpu(),
            "urgency": torch.softmax(logits["urgency"], dim=-1)[0].float().cpu(),
            "change": torch.softmax(logits["change"], dim=-1)[0].float().cpu(),
        }
        payload = self._payload(normalized, probabilities)
        payload["commands"] = filter_suppressed_actions(
            payload.get("commands", []),
            normalization["suppressed_intents"],
        )
        predicted_spans = None
        if self.model.semantic_head_loaded:
            semantic_probabilities = torch.softmax(
                logits["semantic_tags"], dim=-1
            )[0].float().cpu()
            predicted_spans = decode_token_spans(
                normalized,
                offsets,
                semantic_probabilities,
            )
        frame = extract_semantic_frame(
            normalized,
            predicted_spans=predicted_spans,
        )
        payload["commands"] = enrich_commands_with_frame(
            payload.get("commands", []),
            frame,
        )
        atomic_commands = (
            [dict(command) for command in payload["commands"]]
            if payload.get("decomposition_used")
            else None
        )
        payload.pop("decomposition_used", None)
        payload = QwenEnglishIntentParser._normalize_payload(payload, normalized)
        if atomic_commands is not None and payload.get("status") == "VALID":
            payload["commands"] = atomic_commands
        if source_text is not None and payload.get("status") == "VALID":
            payload["commands"], source_speed_restored = (
                restore_source_target_speeds(
                    payload.get("commands", []),
                    source_text,
                )
            )
            if source_speed_restored:
                payload.setdefault("warnings", []).append(
                    "Explicit target speed restored from the source-language command."
                )
        status = payload.pop("status")
        missing_slots = payload.pop("missing_slots", [])
        warnings = payload.pop("warnings", [])
        warnings.extend(normalization["warnings"])
        clarification_question = payload.pop("clarification_question", None)
        if not self.model.semantic_head_loaded and frame["entities"]:
            warnings.append(
                "Semantic token head unavailable; deterministic entity-span fallback used."
            )
        steps = QwenIntentParser._expand_commands(payload.get("commands", []))
        if (
            normalization["requires_confirmation"]
            or normalization["unresolved_references"]
        ):
            status = "NEEDS_CLARIFICATION"
            steps = []
            missing_slots = [
                *missing_slots,
                *(
                    ["normalization.asr_confirmation"]
                    if normalization["requires_confirmation"]
                    else []
                ),
                *normalization["unresolved_references"],
            ]
            clarification_question = (
                "Please confirm the uncertain direction or identify the referenced target."
            )
        contract_errors = semantic_errors(
            {
                "intent": {
                    "entities": frame["entities"],
                    "steps": steps,
                    "urgency": payload["urgency"],
                },
                "parse_result": {"status": status},
            }
        )
        if contract_errors:
            status = "NEEDS_CLARIFICATION"
            steps = []
            missing_slots = ["intent.steps.required_parameters"]
            warnings.extend(
                f"Classifier prediction withheld: {error}" for error in contract_errors
            )
            clarification_question = (
                "Please provide the missing direction, target, speed, duration, or condition."
            )
        latency_ms = (perf_counter() - started) * 1000
        confidence = self._confidence(probabilities, status)
        return make_document(
            raw_text=source_text or text,
            normalized_text=normalized,
            modality=modality,
            category=payload["category"],
            urgency=payload["urgency"],
            steps=steps,
            status=status,
            method="HYBRID",
            model=Path(self.model_path).name,
            confidence=confidence,
            latency_ms=latency_ms,
            request_id=request_id,
            missing_slots=missing_slots,
            warnings=warnings,
            clarification_question=clarification_question,
            driving_style=payload.get("driving_style", "NORMAL"),
            max_speed_mps=payload.get("max_speed_mps"),
            language="en-US",
            entities=frame["entities"],
            normalization_edits=normalization["edits"],
            unresolved_references=normalization["unresolved_references"],
            suppressed_intents=normalization["suppressed_intents"],
            translated_text=text if source_text is not None else None,
            source_language=source_language,
        )

    def _payload(self, text: str, probabilities: dict[str, torch.Tensor]) -> dict[str, Any]:
        action_scores = probabilities["actions"]
        decomposed_commands = decompose_atomic_actions(text)
        status_index = int(probabilities["status"].argmax())
        status = STATUS_LABELS[status_index]
        if decomposed_commands:
            status = "VALID"
            actions = []
        else:
            actions = [
                action
                for index, action in enumerate(ACTION_LABELS)
                if float(action_scores[index]) >= self.action_thresholds[action]
            ]
        if status == "VALID" and not actions:
            best_index = int(action_scores.argmax())
            if not decomposed_commands and float(action_scores[best_index]) >= 0.2:
                actions = [ACTION_LABELS[best_index]]
        if status == "VALID" and not decomposed_commands:
            for action, pattern in ACTION_PATTERNS.items():
                if action not in actions and re.search(pattern, text.casefold()):
                    actions.append(action)
        actions.sort(key=lambda action: self._surface_position(text, action, action_scores))

        direction_scores = probabilities["directions"]
        directions = [
            direction
            for index, direction in enumerate(DIRECTION_LABELS)
            if float(direction_scores[index]) >= self.direction_thresholds[direction]
        ]
        if not directions:
            if re.search(r"\bleft(?:-hand)?\b", text, re.IGNORECASE):
                directions.append("LEFT")
            elif re.search(r"\bright(?:-hand)?\b", text, re.IGNORECASE):
                directions.append("RIGHT")
            elif re.search(
                r"\b(?:straight|forward|ahead)\b", text, re.IGNORECASE
            ):
                directions.append("STRAIGHT")
        change = CHANGE_LABELS[int(probabilities["change"].argmax())]
        commands: list[dict[str, Any]] = list(decomposed_commands)
        if not commands:
            for action in actions:
                command: dict[str, Any] = {"action": action}
                if action in {"CHANGE_LANE", "MERGE", "TURN", "PROCEED"} and directions:
                    command["direction"] = directions.pop(0)
                if action == "ADJUST_SPEED" and change != "NONE":
                    command["change"] = change
                commands.append(command)
        if status != "VALID":
            commands = []
        return {
            "commands": commands,
            "decomposition_used": bool(decomposed_commands),
            "status": status,
            "category": CATEGORY_LABELS[int(probabilities["category"].argmax())],
            "urgency": URGENCY_LABELS[int(probabilities["urgency"].argmax())],
            "driving_style": "NORMAL",
            "missing_slots": [] if status == "VALID" else ["intent.steps"],
            "warnings": [],
        }

    @staticmethod
    def _surface_position(text: str, action: str, scores: torch.Tensor) -> tuple[int, float]:
        match = re.search(ACTION_PATTERNS[action], text.casefold())
        index = ACTION_LABELS.index(action)
        return (match.start() if match else 1_000_000, -float(scores[index]))

    @staticmethod
    def _confidence(probabilities: dict[str, torch.Tensor], status: str) -> float:
        status_confidence = float(probabilities["status"].max())
        action_confidence = float(probabilities["actions"].max())
        if status != "VALID":
            return status_confidence
        return 0.5 * status_confidence + 0.5 * action_confidence
