"""Run the temporary external following-policy process for CARLA demos."""

import argparse
import json
import os
import time

from control.placeholder_following_policy import (
    PlaceholderFollowingPolicy,
    atomic_write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Temporary JSON following-policy service")
    parser.add_argument("--world-state", required=True, help="Runner-written world-state JSON path")
    parser.add_argument("--decision-output", required=True, help="ControlDecision JSON path to update")
    parser.add_argument("--target-speed-kmh", type=float, default=25.0)
    parser.add_argument("--poll-ms", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    poll_s = max(0.001, args.poll_ms / 1000.0)
    last_frame = None
    policy = PlaceholderFollowingPolicy(args.target_speed_kmh)
    atomic_write_json(
        args.decision_output,
        policy.decide({}, "startup"),
    )
    try:
        while True:
            try:
                with open(args.world_state, "r", encoding="utf-8") as handle:
                    snapshot = json.load(handle)
                frame_id = snapshot.get("frame_id")
                if frame_id != last_frame:
                    decision = policy.decide(
                        snapshot.get("world_state", {}),
                        frame_id,
                    )
                    atomic_write_json(args.decision_output, decision)
                    last_frame = frame_id
            except (IOError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            time.sleep(poll_s)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    main()
