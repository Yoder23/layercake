import json
import subprocess
import sys


def _artifact(*, gen_ratio: float = 2.0) -> dict:
    return {
        "status": "PASS",
        "tier": "1m_vs_50m",
        "layercake": {
            "params": 500_000,
            "general_bpb": 1.5,
            "train": {"elapsed_seconds": 10.0},
            "generation": {"mean_bytes_per_second": gen_ratio * 100.0},
            "qa_quality_mean": 0.8,
        },
        "baseline": {
            "params": 50_000_000,
            "general_bpb": 2.0,
            "train": {"elapsed_seconds": 20.0, "elapsed_total_seconds": 30.0},
            "generation": {"mean_bytes_per_second": 100.0},
            "qa_quality_mean": 0.7,
        },
        "cost_proxy_param_seconds": {
            "layercake": 5_000_000.0,
            "baseline": 1_500_000_000.0,
        },
    }


def test_asymmetric_verifier_passes_winning_artifact(tmp_path):
    artifact = tmp_path / "artifact.json"
    output = tmp_path / "cert.json"
    artifact.write_text(json.dumps(_artifact()), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_asymmetric_scale_ladder.py",
            "--artifact",
            str(artifact),
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"


def test_asymmetric_verifier_fails_generation_regression(tmp_path):
    artifact = tmp_path / "artifact.json"
    output = tmp_path / "cert.json"
    artifact.write_text(json.dumps(_artifact(gen_ratio=0.5)), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_asymmetric_scale_ladder.py",
            "--artifact",
            str(artifact),
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 1
    cert = json.loads(output.read_text(encoding="utf-8"))
    assert cert["gates"]["generation_speed_met"] is False
