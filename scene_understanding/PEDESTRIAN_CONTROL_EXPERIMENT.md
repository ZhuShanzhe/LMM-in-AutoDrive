# Pedestrian Control Experiment

`run_pedestrian_control_experiment` is the closed-loop CARLA 0.9.16 test for
the first step of the structured command “yield to a crossing pedestrian, then
change lane and overtake.” It composes existing team modules without copying or
modifying their implementations:

- the pedestrian-crossing scenario creates the ego vehicle and walker;
- the scene-understanding collector produces a validated `WorldState` each
  synchronous frame;
- semantic alignment and risk assessment consume that frame;
- the control-plan executor emits one JSON `ControlDecision`;
- the team PID controller applies that decision to the real CARLA ego vehicle;
- measured CARLA state produces `StepFeedback` for the active plan step.

## Completion semantics

The experiment reports `step_1` as `COMPLETED` only when all of the following
conditions hold:

1. a pedestrian was previously observed in `crossing_ego_path`;
2. that pedestrian is no longer crossing and has reached the far side
   (`relative_position_ego_m.lateral <= -2.5` metres);
3. ego speed has fallen by at least 3 m/s from the measured initial speed;
4. the collision sensor has reported no collision.

Object disappearance alone is never accepted as completion. A collision,
timeout, or premature terminal plan produces `FAILED` feedback. Completion is
checked before the `SAFE_STOP` missing-target policy so that the first correctly
cleared frame is recorded as success rather than as a false blockage.

## Run

CARLA must already be running, and the world must not contain stale vehicles,
walkers, or sensors.

```bash
python -m scene_understanding.scripts.run_pedestrian_control_experiment \
  --driving-intent inputs/driving_intent.json \
  --initial-state inputs/control_plan_state.json \
  --scenario-root experiment/VAD/CARLA \
  --control-root path/to/carla_control_reference \
  --output-dir outputs/pedestrian_control_experiment
```

The output directory is intentionally required to be new. It contains:

- `timeline.jsonl`: per-frame measurements, decision and applied control;
- `step_feedback.json`: the physical outcome of `step_1`;
- `control_plan_state.json`: persisted state after applying feedback;
- `control_decision.json`: decision for the newly active or terminal step;
- `semantic_alignment.json` and `risk_assessment.json`: final frame contracts;
- `summary.json`: concise experiment result;
- `error.json`: present only when an unexpected exception occurs.

A successful pedestrian sub-experiment normally leaves the complete command
plan `ACTIVE` at `step_2`. If the standalone pedestrian scene contains no slow
vehicle, `step_2` correctly remains `WAITING`; the experiment does not invent a
vehicle or claim that later steps were executed.
