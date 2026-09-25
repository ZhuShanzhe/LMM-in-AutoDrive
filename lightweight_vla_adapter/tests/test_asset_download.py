import hashlib
from pathlib import Path

import pytest

from lightweight_vla_adapter.scripts.download_challenge_assets import download_assets, source_for


def fixture(tmp_path):
    source = tmp_path/'download'
    source.write_bytes(b'verified fixture')
    entry = dict(path='challenge/test.pt', bytes=source.stat().st_size,
                 sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    manifest = dict(schema_version='challenge_runtime_assets/1.0', version='v1', files=[entry])
    sources = dict(schema_version='challenge_asset_sources/1.0', manifest_version='v1', sources=[
        dict(local_prefix='challenge/', remote_prefix='challenge/', repo_id='test/fixture', revision='a'*40)])
    return source, manifest, sources


def test_pinned_verified_download_and_resume(tmp_path):
    source, manifest, sources = fixture(tmp_path)
    calls = []
    def fetch(**kwargs):
        calls.append(kwargs)
        return source
    root = tmp_path/'models'
    assert download_assets(root, manifest, sources, fetch)['files'] == 1
    assert calls[0]['revision'] == 'a'*40
    assert calls[0]['token'] is False
    download_assets(root, manifest, sources, fetch)
    assert len(calls) == 1


def test_corrupted_download_is_not_published(tmp_path):
    source, manifest, sources = fixture(tmp_path)
    source.write_bytes(b'bad download')
    with pytest.raises(ValueError, match='Download checksum'):
        download_assets(tmp_path/'models', manifest, sources, lambda **kw: source)
    assert not (tmp_path/'models/challenge/test.pt').exists()


def test_existing_file_never_overwritten(tmp_path):
    source, manifest, sources = fixture(tmp_path)
    target = tmp_path/'models/challenge/test.pt'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'user file')
    with pytest.raises(ValueError, match='Existing asset differs'):
        download_assets(tmp_path/'models', manifest, sources, lambda **kw: source)
    assert target.read_bytes() == b'user file'


@pytest.mark.parametrize('path', ['../escape', '/absolute', 'challenge/../escape', 'challenge\\escape'])
def test_rejects_escaping_paths(path):
    with pytest.raises(ValueError):
        source_for(path, [])


def test_rejects_mutable_revision(tmp_path):
    _, manifest, sources = fixture(tmp_path)
    sources['sources'][0]['revision'] = 'main'
    with pytest.raises(ValueError, match='immutable'):
        source_for(manifest['files'][0]['path'], sources['sources'])
