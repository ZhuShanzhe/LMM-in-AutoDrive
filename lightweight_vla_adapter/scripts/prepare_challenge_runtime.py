"""Verify the frozen assets and resolve a portable challenge configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def verify_assets(model_root: Path, manifest: dict) -> list[str]:
    if manifest.get('schema_version') != 'challenge_runtime_assets/1.0':
        raise ValueError('Unsupported asset manifest')
    verified = []
    for item in manifest['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Asset path must be relative to model root')
        path = model_root / relative
        if not path.is_file():
            raise FileNotFoundError(f'Missing runtime asset: {path}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != item['bytes'] or digest != item['sha256']:
            raise ValueError(f'Runtime asset checksum mismatch: {path}')
        verified.append(str(relative))
    return verified


def resolve_config(repo_root: Path, model_root: Path, map_path: Path) -> dict:
    config = json.loads((repo_root / 'lightweight_vla_adapter/configs/challenge_signal_generalization.json').read_text())
    static = json.loads(map_path.read_text())
    if (not static.get('heads') or not static.get('opendrive_sha256')
            or static.get('dynamic_signal_state_exported') is not False):
        raise ValueError('Expected static lamp geometry, map hash, and no dynamic state labels')
    config['event_memory_checkpoint'] = str((model_root/'challenge/phase_recovery_v1.pt').resolve())
    config['traffic_signal_observer'].update(
        static_map_path=str(map_path.resolve()),
        state_checkpoint=str((model_root/'challenge/lamp_state_v2.pt').resolve()),
        detector_weights=str((model_root/'pretrained/yolo11s.pt').resolve()),
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, default=REPO_ROOT/'models')
    parser.add_argument('--map', choices=['Town01', 'Town04', 'Town05'], default='Town04')
    parser.add_argument('--static-map', type=Path, help='Static export for another map; checked against CARLA at startup')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((REPO_ROOT/'lightweight_vla_adapter/configs/challenge_assets.json').read_text())
    verified = verify_assets(args.model_root, manifest)
    map_path = args.static_map or REPO_ROOT/f'lightweight_vla_adapter/configs/maps/{args.map}_traffic_heads.json'
    config = resolve_config(REPO_ROOT, args.model_root, map_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps({
        'status': 'assets_verified', 'verified_files': len(verified),
        'config_path': str(args.output.resolve()), 'precision': 'fp32',
        'checkpoint_path': str((args.model_root/'lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/model.pt').resolve()),
        'parser_model_path': str((args.model_root/'modernbert-drive-command-compositional').resolve()),
        'required_cameras': ['front', 'left', 'right', 'rear'], 'enable_lidar': True,
        'note': 'Asset integrity only; this command does not run a driving test.',
    }, indent=2))


if __name__ == '__main__':
    main()
