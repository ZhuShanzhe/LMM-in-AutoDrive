# Control Decision JSON Interface

This interface joins the command parser, scene-understanding outputs, and the
CARLA controller without directly coupling their Python packages.

## Inputs and output

The command consumes four JSON documents from the same request and WorldState
frame:

- `driving_intent.json`: structured user command from the command parser;
- `world_state.json`: metric CARLA snapshot;
- `semantic_alignment.json`: intent targets aligned to actors or lanes;
- `risk_assessment.json`: deterministic object and lane-change safety result.

It writes one `control_decision.json` accepted by the control branch's
`control.protocol.normalize_intent` function:

```bash
python -m scene_understanding.scripts.build_control_decision \
  --driving-intent inputs/driving_intent.json \
  --world-state inputs/world_state.json \
  --semantic-alignment outputs/semantic_alignment.json \
  --risk-assessment outputs/risk_assessment.json \
  --output outputs/control_decision.json
```

The schema is `scene_understanding/schemas/control_decision.schema.json`; a complete document is
available at `scene_understanding/schemas/examples/control_decision.example.json`.

## Deterministic safety rules

1. `request_id`, `frame_id`, and parser status must agree across all inputs.
2. A parser result other than `VALID` becomes a safe `stop` fallback.
3. Explicit `STOP` and `EMERGENCY_BRAKE` commands are preserved.
4. A risk recommendation of `emergency_brake` or `decelerate` overrides an
   ordinary command.
5. A required but unmatched target follows the step's `on_blocked` policy:
   `WAIT_FOR_SAFE` decelerates, `SKIP_STEP` keeps lane, and other or missing
   policies stop.
6. Lane change is emitted only when the corresponding lane judgment is safe.
7. A turn without a planner-provided target location is blocked rather than
   steering blindly.
8. In a sequenced plan, an aligned `OVERTAKE` step without another lane-change
   direction maps to `accelerate`; emergency braking, risk deceleration and
   target-alignment gates still take priority.

## Current execution scope

The adapter always emits one flat controller decision. It selects the first
intent step by default for backward compatibility, while the stateful control
plan executor supplies `source_step_id` to evaluate the current active step.
That executor advances dependencies only after explicit, frame-matched step
feedback; see `CONTROL_PLAN_EXECUTION.md`.

Compatibility is tested against the team controller's
`control.protocol.normalize_intent` interface. This JSON adapter itself has no
CARLA runtime dependency; the closed-loop experiments use the team's CARLA
0.9.16 runtime and controller package.
