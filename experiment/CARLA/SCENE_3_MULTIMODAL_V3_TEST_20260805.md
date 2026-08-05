# Scene 3 multimodal VLA closed-loop test (2026-08-05)

## Conclusion

The final strict r6 run completes the official 6 km rainy-night route and all
seven scheduled events. It records zero collisions, zero restricted-line
invasions and zero invalid-lane samples. The online chain is text command,
four-view raw RGB plus vehicle/environment state, VLA proposal, deterministic
safety gate, route PID and CARLA vehicle control. Audio is not used.

The result is a system-level pass, not a model-only pass. The safety/liveness
layer changes 648 of 26,254 model proposals, including 502 repeated
`cleared_worker_resume` corrections after the temporary worker has cleared.
This remaining temporal counterfactual weakness must be reported rather than
credited to the VLA.

## Reproducible configuration

- Map: official `Town05_Opt`
- Generated route: 6,000.969 m
- Weather: rainy night, wet road, fog density 35
- Wet-road friction: 0.68; effective ego speed cap: 32 km/h
- Controller: online `vla-route-pid`
- VLA config: `lightweight_vla_adapter/configs/scene3_multimodal_v3.json`
- Checkpoint package path:
  `models/lightweight_vla_adapter/scene3_multimodal_v3/model.pt`
- Inputs: scheduled raw text, four-view RGB, vehicle state and environment state
- Inference: CUDA FP16, one decision every two simulation frames
- Ground truth: one record every four simulation frames
- Video: H.264, 1280x720, 20 fps, HUD overlay; no audio

Run artifacts are intentionally outside Git and use repository-relative paths:

`experiment/CARLA/outputs/scene3_multimodal_v3_full_r6_20260805`

## Final strict 6 km result

| Metric | r6 result | Acceptance |
| --- | ---: | --- |
| Process exit status | 0 | Pass |
| Route | 6,000.969/6,000 m | Pass |
| Scheduled events | 7/7 resolved | Pass |
| Collisions | 0 | Pass |
| Lane-invasion callbacks | 168, all `Broken` | Informational |
| Restricted-line invasions | 0 | Pass |
| Invalid-lane samples | 0 | Pass |
| VLA decisions | 26,254 | Recorded |
| Model proposals accepted unchanged | 25,606 (97.53%) | Recorded |
| Safety/canonical overrides | 648 (2.47%) | See limitations |
| Fallback decisions | 0 | Pass |
| High-risk response latency | mean 20.117 ms, p95/max 22.654 ms | Pass (<120 ms) |
| All-decision sensor-to-response | mean 23.091 ms, p95 26.993 ms | Recorded |
| Video frames | 52,546; dropped 0 | Pass |
| Video duration/size | 2,627.3 s / 4,671,853,387 bytes | Recorded |
| Ground-truth records | 13,127 | Recorded |

All 26,254 parser outputs are `VALID` and all control decisions are `READY`.
The risk distribution is 26,221 low, 25 medium and 8 high. The eight high-risk
decisions are all below 120 ms. There are 36 all-chain latency outliers above
120 ms, but every one is low-risk and attributable to camera waiting and/or
occasional inference jitter; the overall within-120-ms rate is 99.8629%.

Ground-truth evidence includes 1,857 `OBSERVED`, 26 `PARTIAL`, zero `PROXY` and
11,244 `SCHEDULE_ONLY` records. Every one of the seven event contracts has
observed evidence and the summary records `model_output_used=true` and
`training_teacher_force_control=false`.

## Video exposure audit

The r6 MP4 contains 52,546 H.264 frames at 1280x720 and 20 fps, with a measured
duration of 2,627.3 seconds. Sampling every 600 frames (88 samples) gives:

- mean luma: 116.029
- maximum sample mean luma: 124.432
- mean pixels at or above 250: 0.07935%
- maximum pixels at or above 250 in one sample: 0.31879%
- mean pixels equal to 255: 0.01513%

These values do not indicate the large-area highlight clipping seen in the old
overexposed capture. The cause of the old problem was stacked manual camera
gain (`exposure_compensation=+3`, ISO 1600, shutter 80 and f/1.4) combined with
night lighting, wet-road specular reflections, bloom and lens flare. The final
camera uses automatic histogram exposure, compensation 0 and restrained
bloom/lens flare.

## Route-centering failure history and correction

The earlier r4 strict run completed the route and all events with zero
collisions, but correctly failed because it contained 38 `Solid` lane-marking
callbacks. Those callbacks formed periodic clusters at tight Town05 turns and
visual inspection showed real corner cutting.

The first short-horizon correction (r5) eliminated restricted-line events in a
1.16 km route-PID diagnostic, but its 2.8 m minimum lookahead was too aggressive
on ordinary road. The formal r5 VLA run collided with a static guardrail near
3.5 km and became geometrically stuck; it was interrupted and remains a failure
artifact.

The final correction uses an adaptive horizon: normal-road lookahead is 5-7 m;
only a moving vehicle approaching a junction or road transition shortens to
3.2 m. The junction speed cap is 9 km/h, the transition preview window is 40 m,
and lateral PID gains are reduced to Kp 1.60 and Kd 0.14. Evidence before r6:

| Diagnostic | Progress | Collision | Restricted line | Invalid lane |
| --- | ---: | ---: | ---: | ---: |
| `scene3_route_pid_turn_diag_r5c_20260805` | 844.737 m | 0 | 0 | 0 |
| `scene3_route_pid_guardrail_diag_r5c_20260805` | 5,208.039 m | 0 | 0 | 0 |

The second diagnostic passes the former 3.5 km guardrail point and contains 159
lane callbacks, all legal broken markings. Formal r6 then confirms the same
fix under the online VLA, safety gate, ground-truth and video workload.

## Model and chain limitations

1. The model proposes 553 emergency brakes, while only 51 remain in final
   control. Exactly 502 are cleared-worker liveness corrections. More temporal
   counterfactual training is needed for risk disappearance and action release.
2. The safety/canonical layer modifies 2.47% of proposals. Zero collisions and
   violations are therefore properties of the complete architecture, not proof
   that the checkpoint alone is safe.
3. Offline v3 evaluation remains weaker on emergency action recall (38.65%)
   than its overall action accuracy (93.16%) and visual-risk accuracy (89.35%).
4. Thirty-six low-risk decisions exceed 120 ms because camera waits or rare
   inference jitter increase end-to-end latency. High-risk timing passes, but
   a production system should isolate capture and inference scheduling.
5. The selected VLA consumes raw views directly, but the separately evaluated
   detector is not fused into this controller. This limits interpretable
   cross-checking and remains important because the general detector produced
   no pedestrian-positive frames in the rainy-night audit.
6. The conservative 9 km/h junction cap substantially increases video/runtime.
   It is suitable for the zero-violation submission gate but leaves efficiency
   headroom for later controller tuning.

## Regression status

The focused route-controller suite passes 46/46 tests. The final non-audio
project regression passes 450 pytest cases and 169 subtests using isolated
pytest imports across the structured parser, lightweight VLA, scene
understanding and CARLA suites.

A bare repository-root `pytest` still has five pre-existing collection errors:
four audio tests assume their own directory is on `PYTHONPATH`, and the legacy
CARLA folder and current CARLA test folder contain modules with the same test
basename. Audio is outside this text-only evaluation, and `--import-mode=importlib`
eliminates the duplicate-module ambiguity for the relevant project suites.
