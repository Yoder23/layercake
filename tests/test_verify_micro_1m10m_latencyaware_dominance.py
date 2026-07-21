import json
import subprocess
import sys


def _row(scale: str, *, status: str = "PASS", gen_bps: float = 2.0) -> dict:
    return {
        "scale": scale,
        "status": status,
        "layercake": {
            "params": 10,
            "general_bpb": 1.0,
            "train": {"elapsed_seconds": 1.0},
            "generation": {"mean_bytes_per_second": gen_bps},
            "qa_quality_mean": 2.0,
            "selected_model": {"layers": 0},
        },
        "baseline": {
            "params": 100,
            "general_bpb": 2.0,
            "train": {"elapsed_seconds": 2.0, "elapsed_total_seconds": 3.0},
            "generation": {"mean_bytes_per_second": 1.0},
            "qa_quality_mean": 1.0,
        },
        "cost_proxy_param_seconds": {"layercake": 10.0, "baseline": 1000.0},
    }


def test_latencyaware_verifier_passes_complete_dominance_artifacts(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"scales": [_row(scale) for scale in ("1m", "2m", "5m", "10m")]}),
        encoding="utf-8",
    )
    output = tmp_path / "cert.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_micro_1m10m_latencyaware_dominance.py",
            "--artifacts",
            str(artifact),
            "--output",
            str(output),
            "--min-cost-ratio",
            "5",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    cert = json.loads(output.read_text(encoding="utf-8"))
    assert cert["status"] == "PASS"


def test_latencyaware_verifier_fails_generation_regression(tmp_path):
    artifact = tmp_path / "artifact.json"
    rows = [_row(scale) for scale in ("1m", "2m", "5m", "10m")]
    rows[-1] = _row("10m", gen_bps=0.5)
    artifact.write_text(json.dumps({"scales": rows}), encoding="utf-8")
    output = tmp_path / "cert.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_micro_1m10m_latencyaware_dominance.py",
            "--artifacts",
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
    assert cert["status"] == "FAIL"
    assert cert["scales"][-1]["gates"]["generation_faster"] is False
