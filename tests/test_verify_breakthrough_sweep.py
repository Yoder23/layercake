from scripts.verify_breakthrough_sweep import verify


def test_breakthrough_sweep_passes_when_all_promoted_gates_pass(tmp_path):
    artifact = tmp_path / "ok.json"
    artifact.write_text(
        '{"status":"PASS","ratios":{"cpu":5.0},"gates":{"quality":true}}',
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "campaign": "test",
        "claim": "test claim",
        "tracks": {"fair_neural": {}, "product_runtime": {}},
        "promoted_gates": [
            {
                "name": "cpu_5x",
                "track": "fair_neural",
                "artifact": "ok.json",
                "checks": [{"path": "ratios.cpu", "op": ">=", "value": 5.0}],
            },
            {
                "name": "quality",
                "track": "product_runtime",
                "artifact": "ok.json",
                "checks": [{"path": "gates.quality", "op": "==", "value": True}],
            },
        ],
    }
    result = verify(manifest, root=tmp_path)
    assert result["status"] == "PASS"
    assert result["blockers"] == []
    assert result["track_status"]["fair_neural"] == "PASS"


def test_breakthrough_sweep_fails_closed_on_missing_artifact_and_weak_ratio(tmp_path):
    weak = tmp_path / "weak.json"
    weak.write_text('{"status":"PASS","ratios":{"gpu":4.99}}', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "campaign": "test",
        "claim": "test claim",
        "tracks": {"fair_neural": {}, "product_runtime": {}},
        "promoted_gates": [
            {
                "name": "gpu_5x",
                "track": "fair_neural",
                "artifact": "weak.json",
                "checks": [{"path": "ratios.gpu", "op": ">=", "value": 5.0}],
            },
            {
                "name": "local_runtime",
                "track": "product_runtime",
                "artifact": "missing.json",
                "checks": [{"path": "status", "op": "==", "value": "PASS"}],
            },
        ],
    }
    result = verify(manifest, root=tmp_path)
    assert result["status"] == "FAIL"
    assert result["blockers"] == ["gpu_5x", "local_runtime"]
    assert result["promoted_gates"]["local_runtime"]["artifact_status"] == "MISSING"
    assert result["promoted_gates"]["gpu_5x"]["checks"][0]["actual"] == 4.99


def test_breakthrough_sweep_fails_closed_on_missing_metric_path(tmp_path):
    artifact = tmp_path / "bad.json"
    artifact.write_text('{"status":"PASS"}', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "campaign": "test",
        "claim": "test claim",
        "tracks": {"fair_neural": {}},
        "promoted_gates": [
            {
                "name": "required_metric",
                "track": "fair_neural",
                "artifact": "bad.json",
                "checks": [{"path": "ratios.cpu", "op": ">=", "value": 5.0}],
            }
        ],
    }
    result = verify(manifest, root=tmp_path)
    check = result["promoted_gates"]["required_metric"]["checks"][0]
    assert result["status"] == "FAIL"
    assert check["passed"] is False
    assert "KeyError" in check["error"]


def test_breakthrough_sweep_includes_required_evidence_hygiene_gates(tmp_path):
    artifact = tmp_path / "hygiene.json"
    artifact.write_text(
        '{"status":"PASS","gates":{"fresh":true}}',
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "campaign": "test",
        "claim": "test claim",
        "tracks": {"fair_neural": {}},
        "promoted_gates": [],
        "evidence_hygiene_gates": [
            {
                "name": "fresh_smoke",
                "artifact": "hygiene.json",
                "checks": [{"path": "gates.fresh", "op": "==", "value": True}],
            },
            {
                "name": "missing_smoke",
                "artifact": "missing.json",
                "checks": [{"path": "status", "op": "==", "value": "PASS"}],
            },
        ],
    }
    result = verify(manifest, root=tmp_path)
    assert result["status"] == "FAIL"
    assert result["evidence_hygiene_status"] == "FAIL"
    assert result["blockers"] == ["missing_smoke"]
    assert result["evidence_hygiene_gates"]["fresh_smoke"]["passed"] is True
