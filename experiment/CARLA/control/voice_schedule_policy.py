"""Temporary route-progress policy for the basic voice-control demonstration."""


class VoiceSchedulePolicy:
    """Emit configured commands in route order until a learned policy is wired in.

    The policy deliberately consumes a small, plain context contract from the
    scenario.  A future VLA policy can replace this class without changing the
    scenario, controller, video logger, or completion checks.
    """

    def __init__(self, commands, default_speed_kmh=40.0):
        self.commands = sorted((dict(item) for item in commands), key=self._announce_distance)
        self.default_speed_kmh = float(default_speed_kmh)
        self.context = {}
        self._active_index = None
        self.emitted_command_ids = set()

    def set_context(self, context):
        self.context = dict(context or {})

    def decide(self, world_state):
        del world_state
        progress_m = float(self.context.get("progress_m", 0.0))
        self._activate_due_commands(progress_m)
        command = self._active_command()
        if command is None:
            return self._base_intent(progress_m)

        action = str(command.get("action", "keep_lane"))
        intent = {
            "action": action,
            "target_speed_kmh": float(command.get("target_speed_kmh", self.default_speed_kmh)),
            "reason": "temporary_voice_schedule",
            "command_id": command["id"],
            "voice_text": command.get("voice_text", ""),
            "structured_command": command.get("structured_command", {}),
        }
        if action not in ("lane_change_left", "lane_change_right"):
            target = self.context.get("route_target")
            if target is not None:
                intent["target_location"] = {
                    "x": target["x"],
                    "y": target["y"],
                    "z": target.get("z", 0.0),
                }
        return intent

    def telemetry(self):
        command = self._active_command()
        return {
            "progress_m": round(float(self.context.get("progress_m", 0.0)), 3),
            "active_command_id": command.get("id") if command else None,
            "emitted_command_ids": sorted(self.emitted_command_ids),
            "command_count": len(self.commands),
        }

    def _activate_due_commands(self, progress_m):
        for index, command in enumerate(self.commands):
            if progress_m >= self._announce_distance(command):
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
    def _announce_distance(command):
        return float(command.get("announce_at_m", 0.0))
