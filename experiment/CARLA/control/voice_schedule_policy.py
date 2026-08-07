"""Route-progress intent activation with parser or explicit-intent ingress."""

import copy
import time

import torch

from control.protocol import normalize_intent


class VoiceSchedulePolicy:
    """Emit configured commands in route order until a learned policy is wired in.

    The policy deliberately consumes a small, plain context contract from the
    scenario.  A future VLA policy can replace this class without changing the
    scenario, controller, video logger, or completion checks.
    """

    def __init__(
        self,
        commands,
        default_speed_kmh=40.0,
        parser_model_path=None,
        parser_device="cuda",
        prefer_configured_execution=False,
    ):
        self.commands = sorted((dict(item) for item in commands), key=self._activation_distance)
        self.default_speed_kmh = float(default_speed_kmh)
        self.context = {}
        self._active_index = None
        self.emitted_command_ids = set()
        self._parser = None
        self._parser_model_path = parser_model_path
        self._parser_device = parser_device
        self._prefer_configured_execution = bool(prefer_configured_execution)
        self._parsed_commands = {}
        self._parse_telemetry = {}
        self._completed_command_ids = set()
        self._completed_at_s = {}
        self._success_hold_s = 1.5

    def warmup(self):
        if not self._parser_model_path:
            return
        from structured_command_parser.src.modernbert_service import ModernBertCommandService

        self._parser = ModernBertCommandService(
            self._parser_model_path,
            device=self._parser_device,
        )
        self._parser.warmup()

    def resume_to(self, progress_m):
        """Restore schedule state before a deterministic route checkpoint."""
        progress_m = float(progress_m)
        for index, command in enumerate(self.commands):
            if self._activation_distance(command) >= progress_m:
                break
            self._active_index = index
            self.emitted_command_ids.add(command["id"])
            self._completed_command_ids.add(command["id"])
            # A resumed checkpoint represents work completed before the
            # current recording. Do not replay the SUCCESS hold indefinitely
            # when this process has no original completion timestamp.
            self._completed_at_s[command["id"]] = -self._success_hold_s

    def set_context(self, context):
        self.context = dict(context or {})

    def decide(self, world_state):
        del world_state
        progress_m = float(self.context.get("progress_m", 0.0))
        self._activate_due_commands(progress_m)
        command = self._active_command()
        if command is None:
            return self._base_intent(progress_m)

        if self._is_success_hold_complete(command):
            return self._waiting_intent(progress_m)

        intent = self._command_intent(command)
        action = intent["action"]
        intent.update({
            "command_id": command["id"],
            # The displayed text is the user utterance / ASR result.  The
            # parser's English normalization remains an internal input.
            "voice_text": command.get("voice_text", ""),
            "structured_command": command.get("structured_command", {}),
            "command_phase": self._command_phase(command),
            "audio_file": command.get("audio_file"),
        })
        if action in (
            "turn_left", "turn_right", "decelerate", "accelerate", "keep_lane",
        ):
            target = self.context.get("turn_route_target" if action.startswith("turn_") else "route_target")
            target = target or self.context.get("route_target")
            if target is not None:
                intent["target_location"] = {
                    "x": target["x"],
                    "y": target["y"],
                    "z": target.get("z", 0.0),
                }
                if "yaw" in target:
                    intent["target_location"]["yaw"] = target["yaw"]
                reference = self.context.get("route_reference")
                if isinstance(reference, dict):
                    intent["target_location"]["reference"] = {
                        key: reference[key]
                        for key in ("x", "y", "z", "yaw")
                        if key in reference
                    }
                intent["route_target_trusted"] = not action.startswith("turn_")
        return intent

    def _command_intent(self, command):
        manual_intent = command.get("driving_intent")
        if manual_intent is not None:
            if not isinstance(manual_intent, dict):
                raise ValueError("driving_intent must be an object for command {0}".format(command["id"]))
            driving_intent = copy.deepcopy(manual_intent)
            parsed = normalize_intent(driving_intent, self.default_speed_kmh)
            parsed["reason"] = "manual_driving_intent"
            parsed["driving_intent"] = driving_intent
            result = driving_intent.get("parse_result", {})
            self._parse_telemetry = {
                "command_id": command["id"],
                "status": result.get("status"),
                "confidence": result.get("confidence"),
                "latency_ms": result.get("latency_ms"),
                "model": result.get("model") or "manual",
                "step_count": len(driving_intent.get("intent", {}).get("steps", [])),
                "source": "manual_driving_intent",
            }
            return parsed
        if self._parser is None:
            return {
                "action": str(command.get("action", "keep_lane")),
                "target_speed_kmh": float(
                    command.get("target_speed_kmh", self.default_speed_kmh)
                ),
                "reason": "configured_voice_schedule",
            }
        command_id = command["id"]
        parsed = self._parsed_command(command)
        if not self._prefer_configured_execution:
            return dict(parsed)

        # The current model is still being corrected for compound speed
        # expressions.  For the deterministic basic-scene demonstration, keep
        # its real parse trace but execute the reviewed intent contract below.
        configured = self._configured_driving_intent(command)
        execution = normalize_intent(configured, self.default_speed_kmh)
        execution["reason"] = "configured_execution_after_modernbert"
        execution["driving_intent"] = configured
        self._parse_telemetry["execution_source"] = "configured_driving_intent"
        self._parse_telemetry["raw_parser_action"] = parsed.get("action")
        self._parse_telemetry["raw_parser_target_speed_kmh"] = parsed.get(
            "target_speed_kmh"
        )
        return execution

    def _parsed_command(self, command):
        command_id = command["id"]
        if command_id not in self._parsed_commands:
            english_text = command.get("parser_text_en")
            if not english_text:
                raise ValueError("parser_text_en is required for command {0}".format(command_id))
            started = time.perf_counter()
            driving_intent = self._parser.parse_text(
                english_text,
                request_id="basic-5km-{0}".format(command_id),
                modality="TEXT",
                source_text=command.get("voice_text") or None,
                source_language="zh-CN" if command.get("voice_text") else None,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            parsed = normalize_intent(driving_intent, self.default_speed_kmh)
            parsed["reason"] = "modernbert_driving_intent"
            parsed["driving_intent"] = driving_intent
            self._parsed_commands[command_id] = parsed
            result = driving_intent.get("parse_result", {})
            self._parse_telemetry = {
                "command_id": command_id,
                "status": result.get("status"),
                "confidence": result.get("confidence"),
                "latency_ms": round(float(result.get("latency_ms", elapsed_ms)), 3),
                "model": result.get("model"),
                "step_count": len(driving_intent.get("intent", {}).get("steps", [])),
                "source": "modernbert",
            }
        return self._parsed_commands[command_id]

    @staticmethod
    def _configured_driving_intent(command):
        """Compile the reviewed schedule into the stable DrivingIntent contract."""
        action = str(command.get("action", "keep_lane")).lower()
        target_speed_mps = round(float(command.get("target_speed_kmh", 40.0)) / 3.6, 6)
        parser_action = "SET_SPEED"
        parameters = {"target_speed_mps": target_speed_mps}
        completion = {"type": "TARGET_SPEED_REACHED"}
        on_blocked = "WAIT_FOR_SAFE"
        if action in {"lane_change_left", "lane_change_right"}:
            parser_action = "CHANGE_LANE"
            parameters = {"direction": action.removeprefix("lane_change_").upper()}
            completion = {"type": "LANE_CHANGE_COMPLETED"}
        elif action in {"turn_left", "turn_right"}:
            parser_action = "TURN"
            parameters = {
                "direction": action.removeprefix("turn_").upper(),
                "target_speed_mps": target_speed_mps,
            }
            completion = {"type": "JUNCTION_EXITED"}
        elif action == "stop":
            parser_action = "STOP"
            parameters = {}
            completion = {"type": "VEHICLE_STOPPED"}
            on_blocked = "SAFE_STOP"
        return {
            "schema_version": "1.1.0",
            "request_id": "basic-5km-{}-configured".format(command["id"]),
            "input": {
                "modality": "TEXT",
                "language": "zh-CN",
                "raw_text": command.get("voice_text", ""),
                "normalized_text": command.get("parser_text_en", ""),
            },
            "intent": {
                "category": "BASIC_CONTROL",
                "urgency": "NORMAL",
                "steps": [{
                    "step_id": "step_1",
                    "action": parser_action,
                    "parameters": parameters,
                    "trigger": {"type": "IMMEDIATE"},
                    "depends_on": [],
                    "preconditions": [],
                    "on_blocked": on_blocked,
                    "completion": completion,
                }],
                "constraints": {
                    "safety_first": True,
                    "obey_traffic_rules": True,
                    "driving_style": "NORMAL",
                },
            },
            "parse_result": {
                "status": "VALID",
                "method": "CONFIGURED_EXECUTION",
                "model": "reviewed-basic-scene-contract",
                "confidence": 1.0,
                "missing_slots": [],
                "warnings": ["modernbert_result_logged_separately"],
                "latency_ms": 0.0,
            },
        }

    def telemetry(self):
        command = self._active_command()
        phase = self._command_phase(command) if command is not None else "WAITING"
        return {
            "progress_m": round(float(self.context.get("progress_m", 0.0)), 3),
            "active_command_id": (
                command.get("id") if command is not None and phase != "WAITING" else None
            ),
            "emitted_command_ids": sorted(self.emitted_command_ids),
            "completed_command_ids": sorted(self._completed_command_ids),
            "command_count": len(self.commands),
            "parser": dict(self._parse_telemetry),
            "parser_enabled": self._parser is not None,
            "command_presentation": {
                "phase": phase,
                "command_id": (
                    command.get("id")
                    if command is not None and phase != "WAITING"
                    else None
                ),
                "voice_text": (
                    command.get("voice_text", "")
                    if command is not None and phase != "WAITING"
                    else ""
                ),
                "audio_file": (
                    command.get("audio_file")
                    if command is not None and phase != "WAITING"
                    else None
                ),
            },
        }

    def encode_intent_tokens(self, text):
        """Return the active ModernBERT backbone features for the VLA adapter.

        The adapter is trained on the parser backbone's last hidden state.
        Reuse the already-warmed parser model rather than loading a second
        ModernBERT copy in the CARLA process.
        """
        if self._parser is None:
            raise RuntimeError("ModernBERT parser must be enabled for VLA tokens")
        parser = self._parser.parser
        parser.load()
        encoded = parser.tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=parser.max_length,
        )
        encoded = {name: tensor.to(parser.device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            tokens = parser.model.backbone(**encoded).last_hidden_state
        return tokens.detach().float().cpu(), encoded["attention_mask"].detach().bool().cpu()

    def _activate_due_commands(self, progress_m):
        for index, command in enumerate(self.commands):
            if progress_m >= self._activation_distance(command):
                if index > 0:
                    previous = self.commands[index - 1]
                    if (
                        previous.get("requires_completion", False)
                        and previous["id"] not in self._completed_command_ids
                    ):
                        break
                self._active_index = index
                self.emitted_command_ids.add(command["id"])
            else:
                break

    def mark_completed(self, command_id):
        if command_id:
            command_id = str(command_id)
            if command_id not in self._completed_command_ids:
                self._completed_command_ids.add(command_id)
                self._completed_at_s[command_id] = float(
                    self.context.get("simulation_time_s", 0.0)
                )

    def _active_command(self):
        if self._active_index is None:
            return None
        return self.commands[self._active_index]

    def _command_phase(self, command):
        if command is None:
            return "WAITING"
        command_id = command["id"]
        if command_id not in self._completed_command_ids:
            return "EXECUTING"
        return "SUCCESS" if not self._is_success_hold_complete(command) else "WAITING"

    def _is_success_hold_complete(self, command):
        if command is None:
            return False
        completed_at = self._completed_at_s.get(command["id"])
        if completed_at is None:
            return False
        now_s = float(self.context.get("simulation_time_s", 0.0))
        return now_s - completed_at >= self._success_hold_s

    def _waiting_intent(self, progress_m):
        """Keep cruising while retaining per-frame scene safety supervision."""
        active = self._active_command()
        speed = (
            float(active.get("target_speed_kmh", self.default_speed_kmh))
            if active is not None
            else self.default_speed_kmh
        )
        intent = self._base_intent(progress_m)
        if (
            active is not None
            and str(active.get("action", "")).startswith("lane_change_")
            and not bool(active.get("return_to_route", False))
        ):
            # The fixed route still represents the pre-change lane. During
            # the waiting interval, let the controller hold the lane reported
            # by CARLA instead of pulling back toward that stale route point.
            intent["route_target_trusted"] = False
        intent.update({
            "target_speed_kmh": speed,
            "reason": "waiting_for_next_voice_command",
            "command_phase": "WAITING",
        })
        # Waiting used to return a flat controller intent without a
        # DrivingIntent. ScheduledSceneBridgePolicy therefore bypassed
        # scene_understanding between utterances, exactly when a lead vehicle
        # may slow down. Keep an internal cruise plan alive so every frame is
        # still safety-gated through control_decision.json. This is not a
        # displayed or counted voice command.
        monitor_id = "{}__continuous_cruise".format(
            active["id"] if active is not None else "route"
        )
        monitor_command = {
            "id": monitor_id,
            "action": "keep_lane",
            "target_speed_kmh": speed,
            "voice_text": "",
            "parser_text_en": "Keep the current lane at {:.1f} km/h.".format(speed),
        }
        intent.update({
            "command_id": monitor_id,
            "driving_intent": self._configured_driving_intent(monitor_command),
            "continuous_safety_monitor": True,
        })
        return intent

    def _base_intent(self, progress_m):
        intent = {
            "action": "keep_lane",
            "target_speed_kmh": self.default_speed_kmh,
            "reason": "temporary_route_following",
            "voice_text": "",
            "structured_command": {},
        }
        target = self.context.get("route_target")
        if target is not None:
            intent["target_location"] = {
                "x": target["x"],
                "y": target["y"],
                "z": target.get("z", 0.0),
            }
            if "yaw" in target:
                intent["target_location"]["yaw"] = target["yaw"]
            reference = self.context.get("route_reference")
            if isinstance(reference, dict):
                intent["target_location"]["reference"] = {
                    key: reference[key]
                    for key in ("x", "y", "z", "yaw")
                    if key in reference
                }
            intent["route_target_trusted"] = True
        return intent

    @staticmethod
    def _activation_distance(command):
        """Use a later execution point when an utterance refers to a future junction."""
        return float(command.get("activate_at_m", command.get("announce_at_m", 0.0)))
