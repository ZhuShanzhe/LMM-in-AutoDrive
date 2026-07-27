#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/root/autodl-tmp/datasets/scene_understanding/nuscenes_full}"
ARCHIVES="$DATA_ROOT/archives"
RAW="$DATA_ROOT/raw"
URL_FILE="$DATA_ROOT/nuscenes_trainval_urls.txt"

mkdir -p "$ARCHIVES" "$RAW" "$DATA_ROOT/logs"

cat >"$URL_FILE" <<'EOF'
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval01_blobs.tgz
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval02_blobs.tgz
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_blobs.tgz
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval04_blobs.tgz
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_blobs.tgz
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval06_blobs.tgz
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_blobs.tgz
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval08_blobs.tgz
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval09_blobs.tgz
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval10_blobs.tgz
https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip
EOF

aria2c \
  --input-file="$URL_FILE" \
  --dir="$ARCHIVES" \
  --continue=true \
  --max-concurrent-downloads=3 \
  --max-connection-per-server=8 \
  --split=8 \
  --min-split-size=10M \
  --max-tries=0 \
  --retry-wait=30 \
  --file-allocation=none

for archive in "$ARCHIVES"/*.tgz; do
  tar -tzf "$archive" >/dev/null
  tar -xzf "$archive" -C "$RAW"
done

unzip -q -o "$ARCHIVES/nuScenes-map-expansion-v1.3.zip" -d "$RAW"
echo "nuScenes trainval and map expansion are ready at $RAW"
