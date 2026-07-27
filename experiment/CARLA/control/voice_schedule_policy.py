"""Route-progress voice policy with optional real DrivingIntent parsing."""

import time

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
    ):
        self.commands = sorted((dict(item) for item in commands), key=self._activation_distance)
        self.default_speed_kmh = float(default_speed_kmh)
        self.context = {}
        self._active_index = None
        self.emitted_command_ids = set()
        self._parser = None
        self._parser_model_path = parser_model_path
        self._parser_device = parser_device
        self._parsed_commands = {}
        self._parse_telemetry = {}

    def warmup(self):
        if not self._parser_model_path:
            return
        from structured_command_parser.src.modernbert_service import ModernBertCommandService

        self._parser = ModernBertCommandService(
            self._parser_model_path,
            device=self._parser_device,
        )
        self._parser.warmup()

    def set_context(self, context):
        self.context = dict(context or {})

    def decide(self, world_state):
        del world_state
        progress_m = float(self.context.get("progress_m", 0.0))
        self._activate_due_commands(progress_m)
        command = self._active_command()
        if command is None:
            return self._base_intent(progress_m)

        intent = self._command_intent(command)
        action = intent["action"]
        intent.update({
            "command_id": command["id"],
            "voice_text": (
                command.get("parser_text_en", "")
                if self._parser is not None
                else command.get("voice_text", "")
            ),
            "structured_command": command.get("structured_command", {}),
        })
        if action in ("turn_left", "turn_right", "decelerate", "accelerate"):
            target = self.context.get("turn_route_target" if action.startswith("turn_") else "route_target")
            target = target or self.context.get("route_target")
            if target is not None:
                intent["target_location"] = {
                    "x": target["x"],
                    "y": target["y"],
                    "z": target.get("z", 0.0),
                }
        return intent

    def _command_intent(self, command):
        if self._parser is None:
            return {
                "action": str(command.get("action", "keep_lane")),
                "target_speed_kmh": float(
                    command.get("target_speed_kmh", self.default_speed_kmh)
                ),
                "reason": "configured_voice_schedule",
            }
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
            # Route geometry and the commanded numeric speed remain execution
            # parameters; the model supplies the semantic action and audit data.
            parsed["target_speed_kmh"] = float(
                command.get("target_speed_kmh", parsed["target_speed_kmh"])
            )
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
            }
        return dict(self._parsed_commands[command_id])

    def telemetry(self):
        command = self._active_command()
        return {
            "progress_m": round(float(self.context.get("progress_m", 0.0)), 3),
            "active_command_id": command.get("id") if command else None,
            "emitted_command_ids": sorted(self.emitted_command_ids),
            "command_count": len(self.commands),
            "parser": dict(self._parse_telemetry),
            "parser_enabled": self._parser is not None,
        }

    def _activate_due_commands(self, progress_m):
        for index, command in enumerate(self.commands):
            if progress_m >= self._activation_distance(command):
                self._active_index = index
                self.emitted_command_ids.add(command["id"])
            else:
                break

    def _active_command(self):
        if self._active_index is None:
            return None
        return self.commands[self._active_index]

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
        return intent

    @staticmethod
    def _activation_distance(command):
        """Use a later execution point when an utterance refers to a future junction."""
        return float(command.get("activate_at_m", command.get("announce_at_m", 0.0)))
