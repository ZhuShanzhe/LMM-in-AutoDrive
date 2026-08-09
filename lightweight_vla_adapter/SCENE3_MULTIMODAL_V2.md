# Scene 3 raw-multiview VLA v2

This version removes the CARLA structured-BEV proxy from the deployed Scene 3
policy.  The learned adapter consumes four synchronized RGB cameras, text
tokens, dynamic vehicle state, and explicit weather/environment state.  The
candidate-entity and LiDAR inputs remain optional extension points for the
other two tracks.

## Deployment contract

The released checkpoint must be built with
`configs/scene3_multimodal_v2.json`.  It has `require_raw_camera=true` and
`use_structured_bev=false`; inference fails closed if the four-view bundle is
missing.  It also sets `use_candidate_entities=false`, so CARLA actor-truth
candidate tokens cannot enter the learned policy.  The camera order is
`front,left,right,rear`, each RGB tensor is
`[3,224,224]`, and all four images in a decision originate from one CARLA
frame.  Environment and ego-state vectors have 12 and 8 elements,
respectively.

The CARLA controller may still count nearby actors for the independent safety
audit.  When candidate entities are disabled, it zeros the candidate tensors
and masks before inference and passes no entity IDs to the decoder.  Runtime
logs separate `candidate_count` (actual VLA input) from
`safety_observation_candidate_count` (safety-layer diagnostics).

The deterministic layer is not a second semantic policy.  It may intervene
only for imminent collision risk, an occupied target lane, an invalid
proposal, or the physical speed bound.  Low-risk model mistakes are preserved
in logs and evaluation metrics.

## Model weights in the Docker submission

Weights are intentionally not committed.  Copy the trained directory into the
image, for example `/models/scene3_multimodal_v2`, then set:

```bash
export VLA_CHECKPOINT=/models/scene3_multimodal_v2/model.pt
export MODERNBERT_MODEL=/models/modernbert-drive-command-compositional
```

All repository arguments below are relative to the repository root.  No
Windows path or server-specific repository path is embedded in source code.

## Rebuild the training set

The converter accepts an existing complete CARLA capture with four view
folders, frame truth, and vehicle state.  Generated image references are
relative to the converted dataset directory.

```bash
python lightweight_vla_adapter/scripts/convert_scene3_capture.py \
  --capture-dir experiment/CARLA/outputs/scene3_model_capture \
  --runtime-config experiment/CARLA/configs/scene_3_emergency_6km_runtime.json \
  --parser-model "$MODERNBERT_MODEL" \
  --output-dir experiment/CARLA/outputs/scene3_model_capture/vla_multimodal_dataset_v2 \
  --device cuda
```

An online run can create the same schema with
`--vla-record-training-data <directory>`.  Use the checked-in
`configs/scene3_legacy_collection.json` only for expert-controlled collection;
it is marked `teacher_force_control=true` and must never be used for final
inference.

## Train and evaluate

```bash
python lightweight_vla_adapter/scripts/train_scene3_multimodal.py \
  --dataset experiment/CARLA/outputs/scene3_model_capture/vla_multimodal_dataset_v2 \
  --config lightweight_vla_adapter/configs/scene3_multimodal_v2.json \
  --initialize-from "$LEGACY_VLA_CHECKPOINT" \
  --output-dir outputs/scene3_multimodal_v2 \
  --epochs 20 --batch-size 16

python lightweight_vla_adapter/scripts/evaluate_scene3_modalities.py \
  --dataset experiment/CARLA/outputs/scene3_model_capture/vla_multimodal_dataset_v2 \
  --config lightweight_vla_adapter/configs/scene3_multimodal_v2.json \
  --checkpoint outputs/scene3_multimodal_v2/model.pt \
  --output outputs/scene3_multimodal_v2/modality_ablation.json
```

Training uses ImageNet MobileNetV3-Small initialization, then stores the whole
backbone inside `model.pt`; deployed inference never downloads weights.

## Scene 3 closed loop

At the official `fixed_delta_seconds=0.05`, the default decision interval is
two frames (10 Hz).  The inference cameras run at the same rate, independently
of the chase camera used for video.

```bash
python experiment/CARLA/run_emergency_response_6km.py \
  --host 127.0.0.1 --port 2000 --duration 0 \
  --fixed-delta-seconds 0.05 \
  --ego-controller vla-route-pid \
  --vla-config lightweight_vla_adapter/configs/scene3_multimodal_v2.json \
  --vla-decision-every-n 2 \
  --output-dir experiment/CARLA/outputs/scene3_multimodal_v2_full \
  --presentation-lighting official-rainy-night \
  --camera-mode chase-only \
  --record-ground-truth --ground-truth-every-n 4 \
  --video-output experiment/CARLA/outputs/scene3_multimodal_v2_full/full_6km.mp4 \
  --camera-width 1280 --camera-height 720 --video-fps 20 \
  --record-every-n 1 --video-overlay --require-complete-scene
```

## Current data limitation

The recovered complete training capture is a single official rainy-night run.
It is sufficient to supervise text actions and image-based event risk, but it
does not establish broad weather generalization.  Vehicle and environment
tokens are explicit learned inputs; further captures with varied weather and
speed profiles should be used before claiming strong ablation sensitivity for
those two modalities.
