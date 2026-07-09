#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate drivelm_carla

PROJECT="/root/autodl-tmp/LMM-in-AutoDrive"
DRIVELM_CARLA="${PROJECT}/experiments/drivelm/external/DriveLM-CARLA"
DATA="${PROJECT}/experiments/drivelm/data/PDM_Lite_Carla_LB2"
OUT="${PROJECT}/experiments/drivelm/outputs"
KEYFRAMES="${OUT}/carla_sample_keyframes_no_ds_filter.txt"
GRAPH_OUT="${OUT}/carla_sample_vqa_graph_5"
EXAMPLE_OUT="${OUT}/carla_sample_vqa_examples_5"

mkdir -p "${OUT}" "${GRAPH_OUT}" "${EXAMPLE_OUT}"
cd "${DRIVELM_CARLA}"

# The official extract_keyframes.py currently has filter_routes_for_DS=True by
# default with no CLI flag to turn it off. For this smoke test we call main()
# directly so that partial samples can still be processed.
python -c "from types import SimpleNamespace; import vqa_dataset.extract_keyframes as e; args=SimpleNamespace(path_dataset='${DATA}', path_keyframes='${KEYFRAMES}', skip_first_n_frames=10, use_change_in_steer=True, filter_routes_for_DS=False, keyframe_keys=['light_hazard','walker_hazard','stop_sign_hazard']); e.main(args)"

wc -l "${KEYFRAMES}"
sed -n '1,10p' "${KEYFRAMES}"

python vqa_dataset/carla_vqa_generator_main.py \
  --path-keyframes "${KEYFRAMES}" \
  --data-directory "${DATA}" \
  --output-graph-directory "${GRAPH_OUT}" \
  --output-graph-examples-directory "${EXAMPLE_OUT}" \
  --random-subset-count 5 \
  --sample-frame-mode keyframes

find "${GRAPH_OUT}" -type f | sort | sed -n '1,40p'
cat "${GRAPH_OUT}/stats.json"
