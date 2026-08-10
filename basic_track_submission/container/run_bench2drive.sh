#!/usr/bin/env bash
set -euo pipefail

BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:-/workspace/Bench2Drive}"
LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-${BENCH2DRIVE_ROOT}/leaderboard}"
ROUTES="${ROUTES:-${LEADERBOARD_ROOT}/data/drivetransformer_bench2drive_dev10.xml}"
CHECKPOINT_ENDPOINT="${CHECKPOINT_ENDPOINT:-/workspace/outputs/bench2drive_results.json}"
TEAM_AGENT="${TEAM_AGENT:-${LEADERBOARD_ROOT}/team_code/universal_vla_agent.py}"
TEAM_CONFIG="${TEAM_CONFIG:-/workspace/submission/bench2drive/agent_config.json}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
GPU_RANK="${GPU_RANK:-0}"

cd "${BENCH2DRIVE_ROOT}"
exec python3.12 "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --routes="${ROUTES}" \
  --repetitions=1 \
  --track=SENSORS \
  --checkpoint="${CHECKPOINT_ENDPOINT}" \
  --agent="${TEAM_AGENT}" \
  --agent-config="${TEAM_CONFIG}" \
  --debug=0 \
  --resume=False \
  --port="${PORT}" \
  --traffic-manager-port="${TM_PORT}" \
  --gpu-rank="${GPU_RANK}"
