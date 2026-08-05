# Scene 2 Town05 compliance and chain review (2026-08-05)

## Review conclusion

The `hhx` Scene 2 environment is suitable for the competition after the route
anchor and compound-command interface corrections in `main`. The environment
contains three deterministic variants for every required event, so later model
evaluation can compare the same command against different actor behaviour
without the evaluator manually editing the scene.

The long-run result below validates the environment and CARLA preview executor.
It is not reported as a formal VLA command-accuracy result: the current Scene 2
runner exposes an external-control boundary, but this run used CARLA
`BehaviorAgent` to exercise the route and events.

## Requirement mapping

| Requirement | Implementation and evidence | Result |
| --- | --- | --- |
| Cloudy dusk / low light | Town05 weather preset and camera configuration | Pass |
| Urban secondary road, intersections and bus stop | Official `Town05_Opt`, 8,000.9 m audited route | Pass |
| Mixed traffic | 70 traffic vehicles and 24 ambient walkers configured | Pass |
| Text + four-view RGB + LiDAR + vehicle state | exact-frame sensor suite and `MultimodalFrameBundle` contract | Interface pass; short CARLA evidence test reported below |
| 15 compound commands | route-command preflight plus all-command executable-step tests | Pass |
| Slow vehicle | event variants `steady_18`, `steady_22`, `steady_20` | Pass |
| Crosswalk pedestrian | event variants `left_to_right`, `right_to_left`, `stop_and_go` | Pass |
| Bus-stop passengers | `normal_boarding`, `busy_stop`, `cautious_alighting` | Pass |
| Slow cyclist | `slow_12`, `steady_14`, `fast_16` | Pass |
| Route length at least 8 km | generated route 8,000.9 m | Pass |
| Collision = 0 | long-run safety audit | Pass (0) |
| Restricted-line invasions <= 1 | marking-aware audit | Pass (0) |

## Corrections made during integration

1. Corrected the documented Town05 route anchors to spawn indices 244 and 109
   and aligned the four event trigger/actor positions with the runtime config.
2. Normalized compact compound steps such as `SET_SPEED:8.33mps`,
   `TURN:RIGHT`, `CHECK:CROSSWALK_CLEAR` and
   `CHANGE_LANE:RETURN_WHEN_SAFE` into typed VLA/control-plan parameters.
3. Added sequential dependencies and explicit blocked-step policies so later
   model versions can reuse the same command plan without special-case parsing.
4. Added safe control-decision mappings for `WAIT`, `CHECK`, `CONFIRM`,
   `COMPLETE`, `PROCEED`, `PASS_BY` and `KEEP_SAFE_DISTANCE`.
5. Disabled uncommanded preview-agent tailgating lane changes and retained
   marking-aware collision/lane audit output.

## Dynamic validation

### 8 km variant-0 diagnostic

- Output: `experiment/CARLA/outputs/scene2_town05_variant0_fastdiag_20260805`
- Route progress: 8,000.913/8,000.913 m (complete)
- Commands announced: 15/15
- Event variants: `steady_22`, `stop_and_go`, `busy_stop`, `steady_14`
- All four event states: `RESOLVED`
- Collision events: 0
- Lane invasions: 0
- Restricted-line invasions: 0
- Route-command mismatches: 0
- Dynamic actors: 70 traffic vehicles and 21 ambient walkers spawned
- Measurable environment checks: 6/7 pass; the only false check is exact
  multimodal ratio because this fast 8 km run intentionally used `--no-video`

Traffic-light and traffic-flow queues caused intermittent stops. The ego
vehicle recovered without restarting the control loop, which is useful
stress-testing evidence rather than a simulator failure.

### Exact-frame multimodal diagnostic

- Output: `experiment/CARLA/outputs/scene2_town05_variant1_multimodal_20260805`
- Four RGB cameras and LiDAR were synchronized by exact CARLA frame ID.
- Result: 125/125 expected bundles complete, exact completion ratio 1.0,
  zero incomplete bundles and no adjacent-frame filling
- Sensor files: 129 frames for each RGB view and 129 LiDAR frames
- Evidence video: 500 frames, zero dropped frames
- Ground truth: 500 records, including 471 `OBSERVED` records

## Variant coverage

All three variant indices pass deterministic selection and configuration tests.
Variant 0 was exercised dynamically over the complete route. The other two
indices are intentionally retained as held-out evaluation episodes; their
actor parameters are different but their trigger locations, command schedule
and scoring interfaces remain identical.

## Model and chain deficiencies exposed by this review

1. The Scene 2 runner's `--competition-run` correctly requires external ego
   control and complete multimodal evidence, but the repository does not yet
   provide a dedicated Scene 2 online VLA launcher that writes
   `VLAActionProposal`, safety-gate decisions and step feedback for all 15
   commands. Therefore no formal 96% command-accuracy or 98.5% semantic-
   alignment claim is made from the preview run.
2. Scene 3 VLA training is safety-skewed and does not cover the full Scene 2
   compound-action inventory. A future Scene 2 checkpoint should train on the
   normalized multi-step plans, especially lane-return, intersection chaining
   and wait/confirm transitions.
3. The perception audit shows 0/12 pedestrian-presence recall in the rainy-
   night capture. Until mixed-domain retraining fixes this, simulator ground
   truth and deterministic safety gates remain necessary for testing.

## Automated regression

The integrated repository passed 446 pytest cases and 169 subtests. This
includes route-command alignment, three-variant determinism, exact-frame bundle
contracts, all 15 normalized command plans, control safety, VLA interfaces and
Scene 3 audit tests.
