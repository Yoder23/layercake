import json
import subprocess
import sys


def _metrics(params: int, bpb: float, seconds: float) -> dict:
    return {
        "status": "COMPLETE",
        "latest": {
            "trainable_params": params,
            "eval_bpb": bpb,
            "elapsed_total_seconds": seconds,
        },
    }


def _generation(bps: float, quality: float) -> dict:
    return {
        "metrics": {
            "generation_bytes_per_second": bps,
            "quality_score": quality,
        }
    }


def test_moonshot_verifier_passes_complete_winning_artifacts(tmp_path):
    lc = tmp_path / "lc.json"
    tf = tmp_path / "tf.json"
    lc_gen = tmp_path / "lc_gen.json"
    tf_gen = tmp_path / "tf_gen.json"
    out = tmp_path / "cert.json"
    lc.write_text(json.dumps(_metrics(100_000_000, 1.8, 100.0)), encoding="utf-8")
    tf.write_text(json.dumps(_metrics(520_000_000, 2.1, 550.0)), encoding="utf-8")
    lc_gen.write_text(json.dumps(_generation(2000.0, 0.8)), encoding="utf-8")
    tf_gen.write_text(json.dumps(_generation(1000.0, 0.7)), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_moonshot_100m_vs_500m.py",
            "--layercake-metrics",
            str(lc),
            "--transformer-metrics",
            str(tf),
            "--layercake-generation",
            str(lc_gen),
            "--transformer-generation",
            str(tf_gen),
            "--output",
            str(out),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    cert = json.loads(out.read_text(encoding="utf-8"))
    assert cert["status"] == "PASS"


def test_moonshot_verifier_fails_without_generation_evidence(tmp_path):
    lc = tmp_path / "lc.json"
    tf = tmp_path / "tf.json"
    out = tmp_path / "cert.json"
    lc.write_text(json.dumps(_metrics(100_000_000, 1.8, 100.0)), encoding="utf-8")
    tf.write_text(json.dumps(_metrics(520_000_000, 2.1, 550.0)), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_moonshot_100m_vs_500m.py",
            "--layercake-metrics",
            str(lc),
            "--transformer-metrics",
            str(tf),
            "--output",
            str(out),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 1
    cert = json.loads(out.read_text(encoding="utf-8"))
    assert cert["status"] == "FAIL"
    assert cert["gates"]["generation_evidence_present"] is False
