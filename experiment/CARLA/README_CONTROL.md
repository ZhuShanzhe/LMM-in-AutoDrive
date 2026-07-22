# CARLA Control and Evaluation

This layer connects a high-level driving decision to CARLA and produces
reproducible experiment records. It is intentionally independent of the
scenario implementations in `scenarios/`.

## Decision-to-control interface

The controller accepts either the flat action dict below or the
`structured_command_parser` `DrivingIntent` JSON. `DrivingIntent` is reduced to
one conservative control action in `control/protocol.py`; multi-step execution
still belongs in the later decision/FSM layer.

For a flat dict, the required field is `action`. Optional fields shown below
have safe defaults.

```json
{
  "action": "keep_lane",
  "target_speed_kmh": 25,
  "target_lane": null,
  "target_location": null,
  "emergency": false,
  "reason": "clear road",
  "request_id": "frame-001"
}
```

Supported actions: `keep_lane`, `accelerate`, `decelerate`, `stop`,
`emergency_brake`, `lane_change_left`, `lane_change_right`, `turn_left`, and
`turn_right`.

For `DrivingIntent`, `parse_result.status != VALID` maps to `stop` so invalid
or unsupported model output is not sent to vehicle control. The first
actionable `intent.steps[]` item is mapped as follows:

| DrivingIntent action | Control action |
| --- | --- |
| `KEEP_LANE` | `keep_lane` |
| `SET_SPEED` | `keep_lane` with `target_speed_mps * 3.6` |
| `ADJUST_SPEED` | `accelerate`, `decelerate`, or `keep_lane` |
| `CHANGE_LANE(LEFT/RIGHT)` | `lane_change_left` / `lane_change_right` |
| `TURN(LEFT/RIGHT/STRAIGHT)` | `turn_left` / `turn_right` / `keep_lane` |
| `YIELD`, `PULL_OVER`, `AVOID`, `OVERTAKE` | lane change if direction exists, otherwise `decelerate` |
| `STOP`, `EMERGENCY_BRAKE` | `stop` / `emergency_brake` |

The default `pid` controller supports longitudinal control, lane keeping, and
basic lane-change steering. For junction turns, use `--controller behavior`
or `--controller basic` and provide `target_location` from the planner.

## External decision JSON boundary

`run_control_experiment.py` can read an external decision document on every
simulation tick. This is a Python-version-independent boundary for the later
parser, scene-understanding, risk, or FSM process: it can atomically replace a
JSON file without importing CARLA into that process.

The document may be either a `DrivingIntent` or a flattened
`control_decision.json`. The latter should use one of the nine control actions
listed above and may include `frame_id`, `risk_level`, and other diagnostic
fields; unsupported extra fields are preserved by the producer but ignored by
the CARLA controller.

```powershell
python run_control_experiment.py emergency_brake --duration-s 20 `
  --decision-source json_file `
  --decision-json examples\decisions\emergency_brake.json
```

The included JSON file is only a temporary emergency-braking placeholder. A
missing, malformed, or unsupported external document always becomes a safe
`stop` command rather than allowing the ego vehicle to continue a stale action.
The default `--decision-source rule` remains available for baseline tests.

## Run an experiment

CARLA `0.9.15` on this Windows machine ships a Python 3.7 API package. Start
the CARLA server first, then use a Python 3.7 environment and run from this
directory:

```powershell
python run_control_experiment.py straight_driving --duration-s 25 --goal-distance-m 60 --stop-when-goal-reached
python run_control_experiment.py emergency_brake --duration-s 25
python run_control_experiment.py pedestrian_crossing --duration-s 25
python run_control_experiment.py emergency_brake --duration-s 25 --record-images --record-every-n 2 --camera-width 1920 --camera-height 1080
```

Set `CARLA_ROOT` if CARLA is not installed at
`D:\CARLA\carla-0-9-15-windows\WindowsNoEditor`.

## Outputs

Each run writes to `outputs/runs/<scenario>_<time>/`:

- `run_manifest.json`: fixed parameters and controller selection.
- `frames.jsonl`: one record per simulation tick with action, control values,
  ego state, scenario status, collision/lane events, and latency.
- `metrics.json` and `metrics.csv`: task completion, collisions, lane
  invasions, speeding, distance, scenario result, and decision/control latency
  statistics.

At control/evaluation level, `task_completed` means the requested distance was
reached (when supplied) with no collision, lane invasion, or speeding event.

With `--record-images`, `camera_frames/` holds front-camera evidence at the
default 1920x1080 resolution. Convert
the saved images into a demonstration video with:

```powershell
python frames_to_video.py --frames outputs\runs\<run>\camera_frames --output outputs\runs\<run>\demo.mp4 --fps 10
```

The script uses `ffmpeg` and automatically locates the bundled executable in
the local `carla37` conda environment.

For higher frame rate, bypass PNG files and stream the camera directly to an
H.264 video:

```powershell
python run_control_experiment.py emergency_brake --duration-s 10 --fixed-delta-s 0.0333333333 --camera-width 1920 --camera-height 1080 --video-output outputs\runs\emergency_30fps.mp4 --video-fps 30 --ffmpeg C:\path\to\ffmpeg.exe
```

`RuleDecisionPolicy` in `run_control_experiment.py` is a temporary safety rule
used to validate the closed loop. Replace its `decide(world_state)` call with
the LLM/decision module without changing the control or evaluation interfaces.
