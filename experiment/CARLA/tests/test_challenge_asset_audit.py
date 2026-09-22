import hashlib
from evaluation.challenge_asset_audit import audit


def test_audit_reports_all_missing_and_mismatched_without_stopping(tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    (good / "model.pt").write_bytes(b"good")
    (bad / "model.pt").write_bytes(b"fake")
    manifest = dict(schema_version="challenge_runtime_assets/1.0", files=[
        dict(path=name, bytes=4, sha256=hashlib.sha256(b"good").hexdigest())
        for name in ("model.pt", "missing.pt")])
    result = audit(manifest, [bad, good])
    assert result["status"] == "NOT_READY"
    assert result["verified_count"] == 1
    assert result["files"][0]["candidates"][0]["status"] == "MISMATCH"
    assert result["files"][0]["verified_path"] == str(good / "model.pt")
    assert result["files"][1]["status"] == "UNAVAILABLE"
