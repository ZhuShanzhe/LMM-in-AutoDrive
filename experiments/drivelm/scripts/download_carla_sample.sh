#!/usr/bin/env bash
set -euo pipefail

# Minimal sample only. Full PDM-Lite CARLA data is 330+ GB extracted.
# Current AutoDL disk is not large enough for the full dataset.

BASE="/root/autodl-tmp/LMM-in-AutoDrive/experiments/drivelm/data/PDM_Lite_Carla_LB2"
MIRROR="https://hf-mirror.com/datasets/autonomousvision/PDM_Lite_Carla_LB2/resolve/main"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate drivelm_carla

mkdir -p "${BASE}/Town01/data"
cd "${BASE}"

wget -c -O README.md "${MIRROR}/README.md"
wget -c -O Town01/results.zip "${MIRROR}/Town01/results.zip"
wget -c -O Town01/data/ControlLoss.zip "${MIRROR}/Town01/data/ControlLoss.zip"

python - <<'PY'
import pathlib
import zipfile

for z in [pathlib.Path("Town01/results.zip"), pathlib.Path("Town01/data/ControlLoss.zip")]:
    print(z, "is_zip", zipfile.is_zipfile(z), "size_mb", round(z.stat().st_size / 1024 / 1024, 2))
PY

unzip -oq Town01/results.zip -d Town01/
unzip -oq Town01/data/ControlLoss.zip -d Town01/data/ControlLoss

rm -f Town01/results.zip Town01/data/ControlLoss.zip

# The official zip contains ControlLoss/Route*_Rep0. The DriveLM-CARLA scripts
# expect Town01/data/ControlLoss/Route*_Rep0, so normalize the layout.
if [ -d Town01/data/ControlLoss/ControlLoss ]; then
  for route_dir in Town01/data/ControlLoss/ControlLoss/Route*_Rep0; do
    [ -d "${route_dir}" ] || continue
    route_name="$(basename "${route_dir}")"
    if [ ! -e "Town01/data/ControlLoss/${route_name}" ]; then
      mv "${route_dir}" Town01/data/ControlLoss/
    fi
  done
  rmdir Town01/data/ControlLoss/ControlLoss 2>/dev/null || true
fi

find . -maxdepth 4 -type f | sort | sed -n '1,120p'
