#!/usr/bin/env bash
set -uo pipefail

source /etc/network_turbo >/dev/null 2>&1 || true

PY_ENV="${PY_ENV:-/root/autodl-tmp/conda_envs/command_parser}"
TARGET="${TARGET:-/root/autodl-tmp/datasets/vla_student/simlingo_hf}"
CACHE="${CACHE:-/root/autodl-tmp/huggingface}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$TARGET" "$CACHE"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "DOWNLOAD_ATTEMPT $attempt/$MAX_ATTEMPTS $(date -Is)"
  if "$PY_ENV/bin/hf" download RenzKa/simlingo \
    --repo-type dataset \
    --include \
      "README.md" \
      "LICENSE" \
      "buckets_paths.pkl" \
      "data_*.tar.gz" \
      "dreamer_*.tar.gz" \
      "commentary_*.tar.gz" \
      "drivelm_*.tar.gz" \
    --cache-dir "$CACHE" \
    --local-dir "$TARGET" \
    --max-workers "$MAX_WORKERS"; then
    if "$PY_ENV/bin/python" "$SCRIPT_DIR/audit_simlingo_download.py" \
      --target "$TARGET" \
      --output "$TARGET/download_audit.json" \
      --require-complete; then
      echo "DOWNLOAD_COMPLETE $(date -Is)"
      exit 0
    fi
    echo "DOWNLOAD_COMMAND_RETURNED_INCOMPLETE $(date -Is)"
  fi
  sleep_seconds=$((attempt < 10 ? attempt * 30 : 300))
  echo "DOWNLOAD_RETRY_IN ${sleep_seconds}s $(date -Is)"
  sleep "$sleep_seconds"
done

echo "DOWNLOAD_FAILED_AFTER_RETRIES $(date -Is)" >&2
exit 1
