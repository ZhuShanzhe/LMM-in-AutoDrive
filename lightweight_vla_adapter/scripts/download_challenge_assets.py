"""Download only frozen runtime assets; verify bytes before publishing each file."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']


def source_for(relative, sources):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('Asset path must stay within the model root')
    selected = [s for s in sources if relative.startswith(s['local_prefix'])]
    if len(selected) != 1:
        raise ValueError(f'Expected one pinned source for {relative}')
    source = selected[0]
    if not re.fullmatch(r'[0-9a-f]{40}', source['revision']):
        raise ValueError('A full immutable revision is required')
    remote = source['remote_prefix'] + relative[len(source['local_prefix']):]
    return source, remote


def download_assets(model_root, manifest, sources, fetch):
    if manifest.get('schema_version') != 'challenge_runtime_assets/1.0':
        raise ValueError('Unsupported asset manifest')
    if sources.get('schema_version') != 'challenge_asset_sources/1.0':
        raise ValueError('Unsupported source manifest')
    if sources.get('manifest_version') != manifest.get('version'):
        raise ValueError('Source and runtime manifest versions differ')
    root = Path(model_root).resolve()
    records = []
    seen = set()
    for entry in manifest['files']:
        relative = entry['path']
        if relative in seen:
            raise ValueError('Duplicate asset path')
        seen.add(relative)
        source, remote = source_for(relative, sources['sources'])
        destination = root / relative
        if destination.exists():
            if not matches(destination, entry):
                raise ValueError(f'Existing asset differs; use a clean model root: {destination}')
            status = 'already_verified'
        else:
            if not destination.resolve().is_relative_to(root):
                raise ValueError('Refusing to write outside model root through a symlink')
            cached = Path(fetch(repo_id=source['repo_id'], filename=remote,
                                revision=source['revision'], token=False))
            if not matches(cached, entry):
                raise ValueError(f'Download checksum mismatch: {relative}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.verified-', dir=destination.parent)
            os.close(fd)
            temporary = Path(temporary)
            try:
                shutil.copyfile(cached, temporary)
                if not matches(temporary, entry):
                    raise ValueError(f'Copied asset checksum mismatch: {relative}')
                # Hard-link publication is atomic and cannot overwrite an existing file.
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    if not matches(destination, entry):
                        raise ValueError(f'Concurrent incompatible asset: {destination}')
            finally:
                temporary.unlink(missing_ok=True)
            status = 'downloaded_verified'
        records.append(dict(path=relative, status=status, repo_id=source['repo_id'],
                            revision=source['revision'], sha256=entry['sha256']))
    return dict(status='verified', model_root=str(root), files=len(records),
                total_bytes=sum(e['bytes'] for e in manifest['files']), records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, default=ROOT/'models')
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    configs = ROOT/'lightweight_vla_adapter/configs'
    manifest = json.loads((configs/'challenge_assets.json').read_text())
    sources = json.loads((configs/'challenge_asset_sources.json').read_text())
    print(json.dumps(download_assets(args.model_root, manifest, sources, hf_hub_download), indent=2))


if __name__ == '__main__':
    main()
