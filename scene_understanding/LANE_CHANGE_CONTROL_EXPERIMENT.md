# Lane-Change Control Experiment

`run_lane_change_control_experiment` resumes an `ACTIVE step_2` plan after the
pedestrian-yield experiment. It uses a CARLA map spawn point with a legal,
same-direction left adjacent lane, creates one slower vehicle ahead, and sends
the safety-gated JSON decision to the team PID controller.

The experiment does not claim that seeing a lane marking or issuing a steering
command completes the maneuver. `step_2` receives `COMPLETED` feedback only
after all of these CARLA measurements agree:

1. semantic alignment has grounded the intended slow vehicle;
2. the slow vehicle remains in the current `WorldState`;
3. ego `lane_id` equals the selected left-lane ID;
4. ego is within 0.9 m of the target lane centre;
5. ego heading has at least 0.95 alignment with the target lane;
6. those geometric conditions remain true for five consecutive frames;
7. no collision has occurred.

Crossing a lane marking is expected in a legal lane change, so lane-invasion
events are retained as evidence but are not by themselves treated as failure.
CARLA/OpenDRIVE may assign a new `road_id` at a connected road segment during
the maneuver, so `road_id` is retained as evidence but is not pinned to the
pre-scan value. Lane ID, lane-centre offset and heading must still agree.
After the map reports the target lane, the experiment supplies the controller
with an explicit point ahead in that lane while measurements stabilize. This
prevents a repeated `lane_change_left` action from requesting another lane.

## Run

CARLA must already be running with no stale vehicles, walkers, or sensors.

```bash
python -m scene_understanding.scripts.run_lane_change_control_experiment \
  --driving-intent inputs/driving_intent.json \
  --initial-state inputs/control_plan_state_after_pedestrian.json \
  --scenario-root experiment/VAD/CARLA \
  --control-root path/to/carla_control_reference \
  --spawn-index 1 \
  --output-dir outputs/lane_change_control_experiment
```

The output directory is intentionally required to be new. It contains a
per-frame `timeline.jsonl`, measured `step_feedback.json`, the resulting plan
state and decision, final alignment and risk contracts, and `summary.json`.

A successful run completes only `step_2` and activates `step_3`. Overtaking is
a separate physical maneuver and must receive its own measured feedback; this
experiment never marks it complete automatically.
