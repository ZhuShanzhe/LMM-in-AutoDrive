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

## Run an experiment

The integration environment is Linux with CARLA `0.9.16` and Python
`3.12.13`. The CARLA server and Python API must use the same version. Install
the API in the data-disk environment and verify it before starting a run:

```bash
conda activate /root/autodl-tmp/conda_envs/command_parser
export CARLA_ROOT=/root/autodl-tmp/CARLA_0.9.16
python -m pip install "$CARLA_ROOT"/PythonAPI/carla/dist/carla-0.9.16-*.whl
python -c "from importlib.metadata import version; print(version('carla'))"
```

Start the Linux server with `-RenderOffScreen` as documented in `README.md`.
Then run from `experiment/CARLA`:

```bash
python run_control_experiment.py straight_driving --duration-s 25 --goal-distance-m 60 --stop-when-goal-reached
python run_control_experiment.py emergency_brake --duration-s 25
python run_control_experiment.py pedestrian_crossing --duration-s 25
python run_control_experiment.py emergency_brake --duration-s 25 --record-images --record-every-n 2 --camera-width 1920 --camera-height 1080
```

Set `CARLA_ROOT` if CARLA is not installed at
`/root/autodl-tmp/CARLA_0.9.16`.

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

```bash
apt-get update
apt-get install -y ffmpeg fonts-noto-cjk
python frames_to_video.py \
  --frames outputs/runs/<run>/camera_frames \
  --output outputs/runs/<run>/demo.mp4 \
  --fps 10
```

The script uses `ffmpeg` from `PATH`; use `--ffmpeg /path/to/ffmpeg` only for a
non-standard installation.

For higher frame rate, bypass PNG files and stream the camera directly to an
H.264 video:

```bash
python run_control_experiment.py emergency_brake \
  --duration-s 10 \
  --fixed-delta-s 0.0333333333 \
  --camera-width 1920 \
  --camera-height 1080 \
  --video-output outputs/runs/emergency_30fps.mp4 \
  --video-fps 30
```

`RuleDecisionPolicy` in `run_control_experiment.py` is a temporary safety rule
used to validate the closed loop. Replace its `decide(world_state)` call with
the LLM/decision module without changing the control or evaluation interfaces.
