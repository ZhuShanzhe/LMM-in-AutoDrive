#!/usr/bin/env bash
set -euo pipefail

export CARLA_ROOT="${CARLA_ROOT:-/workspace/CARLA_0.9.16}"
export BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT:-/workspace/Bench2Drive}"
export REPOSITORY_ROOT="${REPOSITORY_ROOT:-/workspace/LMM-in-AutoDrive}"
export SCENARIO_RUNNER_ROOT="${SCENARIO_RUNNER_ROOT:-${BENCH2DRIVE_ROOT}/scenario_runner}"
export LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-${BENCH2DRIVE_ROOT}/leaderboard}"
export TEAM_AGENT="${TEAM_AGENT:-${LEADERBOARD_ROOT}/team_code/universal_vla_agent.py}"
export TEAM_CONFIG="${TEAM_CONFIG:-/workspace/submission/bench2drive/agent_config.json}"
export CHALLENGE_TRACK_CODENAME="${CHALLENGE_TRACK_CODENAME:-SENSORS}"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${LEADERBOARD_ROOT}/team_code:${REPOSITORY_ROOT}:${REPOSITORY_ROOT}/experiment/CARLA:${PYTHONPATH:-}"

mkdir -p /workspace/outputs
exec "$@"
