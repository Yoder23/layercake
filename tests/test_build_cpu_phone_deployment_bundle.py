import hashlib
import json

from scripts.build_cpu_phone_deployment_bundle import build_bundle


def test_deployment_bundle_hashes_artifacts_and_stays_open_without_phone_evidence(tmp_path):
    core = tmp_path / "core.pt"
    domain = tmp_path / "domain.pt"
    core.write_bytes(b"core")
    domain.write_bytes(b"domain")
    result = build_bundle(
        core_paths=[core],
        domain_paths=[domain],
        output_path=tmp_path / "bundle.json",
        root=tmp_path,
    )
    assert result["status"] == "OPEN"
    assert result["required_gates"]["all_artifacts_hashed"] is True
    assert result["required_gates"]["real_phone_runtime_evidence"] is False
    assert result["artifacts"]["cores"][0]["sha256"]


def _valid_phone_evidence(core_bytes: bytes) -> dict:
    return {
        "schema_version": 1,
        "real_phone_hardware": True,
        "device": {
            "manufacturer": "Example",
            "model": "Arm Phone",
            "os": "Android",
            "os_version": "1",
            "architecture": "arm64-v8a",
        },
        "runtime": {
            "engine": "native-test-runtime",
            "engine_version": "1",
            "artifact_sha256": hashlib.sha256(core_bytes).hexdigest(),
        },
        "protocol": {
            "identical_prompt_pack": True,
            "prompt_pack_sha256": "a" * 64,
            "prompt_count": 20,
            "warmup_runs": 3,
            "repetitions": 3,
        },
        "layercake": {
            "latency_ms": [10.0] * 20,
            "quality_total": 20,
            "quality_exact": 20,
            "peak_rss_bytes": 100,
            "artifact_bytes": len(core_bytes),
        },
        "baseline": {
            "name": "Local baseline",
            "version": "1",
            "artifact_sha256": "b" * 64,
            "latency_ms": [60.0] * 20,
            "quality_total": 20,
            "quality_exact": 19,
            "peak_rss_bytes": 200,
            "artifact_bytes": len(core_bytes) * 2,
        },
        "sustained": {
            "duration_seconds": 300,
            "requests": 100,
            "layercake_bytes_per_second": 600.0,
            "baseline_bytes_per_second": 100.0,
            "thermal_start_celsius": 30.0,
            "thermal_end_celsius": 38.0,
            "thermal_peak_celsius": 40.0,
            "throttling_detected": False,
            "layercake_battery_start_percent": 100.0,
            "layercake_battery_end_percent": 99.0,
            "baseline_battery_start_percent": 100.0,
            "baseline_battery_end_percent": 98.0,
        },
    }


def test_deployment_bundle_passes_with_complete_real_phone_evidence(tmp_path):
    core = tmp_path / "core.pt"
    domain = tmp_path / "domain.pt"
    evidence = tmp_path / "phone.json"
    core.write_bytes(b"core")
    domain.write_bytes(b"domain")
    evidence.write_text(json.dumps(_valid_phone_evidence(b"core")), encoding="utf-8")
    result = build_bundle(
        core_paths=[core],
        domain_paths=[domain],
        output_path=tmp_path / "bundle.json",
        phone_evidence_path=evidence,
        root=tmp_path,
    )
    assert result["status"] == "PASS"
    assert result["required_gates"]["real_phone_runtime_evidence"] is True
    assert result["phone_evidence"]["metrics"]["generation_speed_ratio"] == 6.0


def test_deployment_bundle_rejects_boolean_only_phone_claim(tmp_path):
    core = tmp_path / "core.pt"
    domain = tmp_path / "domain.pt"
    evidence = tmp_path / "phone.json"
    core.write_bytes(b"core")
    domain.write_bytes(b"domain")
    evidence.write_text(json.dumps({"real_phone_hardware": True}), encoding="utf-8")
    result = build_bundle(
        core_paths=[core],
        domain_paths=[domain],
        output_path=tmp_path / "bundle.json",
        phone_evidence_path=evidence,
        root=tmp_path,
    )
    assert result["status"] == "OPEN"
    assert result["required_gates"]["real_phone_runtime_evidence"] is False
    assert result["phone_evidence"]["failed"]


def test_deployment_bundle_fails_when_artifact_missing(tmp_path):
    result = build_bundle(
        core_paths=[tmp_path / "missing-core.pt"],
        domain_paths=[tmp_path / "missing-domain.pt"],
        output_path=tmp_path / "bundle.json",
        root=tmp_path,
    )
    assert result["status"] == "FAIL"
    assert result["required_gates"]["core_artifacts_present"] is False
    assert result["required_gates"]["portable_domains_present"] is False
