# CARLA Control Handoff

This document fixes the process boundary between the command parser, scene-understanding service, and the CARLA runner. It does not prescribe a scenario implementation or a model runtime.

## Stable Data Flow

```text
DrivingIntent + WorldState(frame N)
        -> semantic_alignment.json + risk_assessment.json
        -> control_decision.json(frame N)
        -> CARLA JsonFileDecisionPolicy
        -> controller action for a permitted frame window
        -> execution-feedback service
        -> step_feedback.json after objective completion evidence
```

The CARLA capture bridge writes frame-aligned `WorldState` and camera/projection bundles under a run's `scene_understanding/` directory. A decision producer must use the same `frame_id` in its `ControlDecision`; do not bind a result from one frame to a different `WorldState`.

## Producer Contract

1. Read the checked-in schemas in `schemas/` and emit a valid `ControlDecision`.
2. Set `frame_id` to the source `WorldState.frame_id`.
3. Write `control_decision.json` atomically: write a temporary file in the destination directory, then replace the destination file.
4. Preserve `request_id`, `source_step_id`, `source_step_action`, risk fields, and blocked reasons for the run log.
5. For multi-step commands, advance `control_plan_state.json` only from explicit `step_feedback.json`; never infer completion from having emitted an action.

`scripts/run_execution_feedback.py` is the reference feedback producer. It
only writes a terminal event when the current `ControlDecision` matches the
current `WorldState.frame_id` and is still the active plan step. The normal
case requires `READY`; the sole exception is a `TARGET_CLEARED` step with a
same-frame, alignment-only `no_matching_entity` block, because a correctly
cleared target is no longer an `AHEAD` or `AHEAD_CROSSING` match.
It handles measured target speed, full stop, duration, junction exit and
location completion directly. For the parser's complex examples it also
requires the same grounded actor to demonstrate a pedestrian clearance or an
overtake rear clearance, and requires a map-derived target lane ID to remain
stable across consecutive frames for `LANE_CHANGE_COMPLETED`.

For a `TARGET_CLEARED` step, the executor keeps an alignment-only miss in a
safe waiting state rather than terminally blocking the plan. The emitted
controller decision remains `stop`; completion still requires the feedback
service to observe the originally grounded actor and prove the clearance.

`scripts/build_control_decision.py` and `scripts/advance_control_plan.py` already use atomic JSON replacement and are the reference producers.

## CARLA Consumer Contract

The runner already accepts a `ControlDecision` through its JSON-file decision source:

```text
--decision-source json_file --decision-json <control_decision.json>
```

Optional `--decision-max-age-frames N` enables a frame guard. The runner safely stops if the decision has no frame ID, refers to a future frame, or is older than `N` frames. Leave it unset only while validating legacy decision producers; set it after measuring the actual decision delay.

On every tick the runner exposes `frame_id` and `simulation_frame` in its external world-state envelope. The JSON-file policy records the accepted decision frame and age in the per-frame policy telemetry.

## Safety Boundary

- A missing, malformed, future, or stale external decision produces `stop`; it must not leave the previous command active.
- Collision, TTC, traffic-light state, and lane-change safety remain deterministic inputs to the risk module. A delayed VLM result cannot override them.
- The VLM/VLA service may enrich semantics asynchronously, but only a fresh, schema-valid `ControlDecision` can reach the controller.

## Integration Readiness Check

Before a producer is connected to CARLA, verify all of the following:

1. Its output validates against `schemas/control_decision.schema.json`.
2. Its `frame_id` exactly matches the `WorldState` used for alignment and risk assessment.
3. It atomically replaces the configured decision path.
4. It can tolerate a CARLA safe-stop response when its output is absent or stale.
5. It emits `StepFeedback` after an action has objectively completed, failed, or been skipped.
6. It keeps the feedback tracker for the active step, so a new request or
   active step cannot inherit an earlier target observation or stable-frame count.
