# Overtake Control Experiment

`run_overtake_control_experiment` resumes the `ACTIVE step_3` state produced by
the successful lane-change experiment. Ego starts in the passing lane and a
grounded slower vehicle starts ahead in the adjacent lane. The plan emits
`accelerate` only while semantic alignment succeeds and risk permits it; the
existing emergency-brake and deceleration rules retain higher priority.

The experiment reports `COMPLETED` only when:

1. the matched slow vehicle was first measured ahead of ego;
2. the same vehicle remains in `WorldState`;
3. its ego-frame longitudinal position is at least 8 m behind ego;
4. ego remains within 0.9 m of the passing-lane centre;
5. ego heading alignment with that lane is at least 0.95;
6. clearance and lane conditions remain true for five consecutive frames;
7. no collision occurs.

The applied acceleration is capped at 40 km/h. The cap modifies only an
ordinary `accelerate` action and never weakens `decelerate`, `stop`, or
`emergency_brake` decisions.

## Run

```bash
python -m scene_understanding.scripts.run_overtake_control_experiment \
  --driving-intent inputs/driving_intent.json \
  --initial-state inputs/control_plan_state_after_lane_change.json \
  --scenario-root experiment/VAD/CARLA \
  --control-root path/to/carla_control_reference \
  --spawn-index 60 \
  --output-dir outputs/overtake_control_experiment
```

The new output directory contains the per-frame timeline, measured feedback,
final alignment and risk, terminal plan state, controller decision and summary.
Only measured rear clearance can complete the final step; object disappearance
or elapsed time cannot.
