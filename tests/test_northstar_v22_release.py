import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "breakthrough_equal"
CERTIFICATE = RESULTS / "northstar_v22_release_certificate.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_committed_release_certificate_is_hash_complete():
    certificate = _load(CERTIFICATE)

    assert certificate["status"] == "PASS"
    assert certificate["failed_required"] == []
    assert all(certificate["required_gates"].values())
    for record in certificate["evidence"].values():
        path = ROOT / record["path"]
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]


def test_cpu_gpu_dominance_recomputes_from_raw_artifacts():
    cases = [
        ("northstar_v22_schema_patch_cpu.json", "cpu"),
        ("northstar_v22_relevance_patch_cpu.json", "cpu"),
        ("northstar_v22_schema_patch_cuda_graph_gpu.json", "cuda"),
        ("northstar_v22_relevance_patch_cuda_graph_gpu.json", "cuda"),
    ]
    for filename, device in cases:
        document = _load(RESULTS / filename)
        heldout = document["splits"]["heldout"]["summary"]
        assert document["benchmark_mode"] == "fair_neural"
        assert document["device"] == device
        assert document["environment"]["device_type"] == device
        assert document["layercake_structured_schema_head"] is False
        assert document["layercake_direct_domain_cache"] is False
        assert heldout["layercake"]["exact_json_accuracy"] == 1.0
        assert (
            heldout["layercake"]["exact_json_accuracy"]
            > heldout["transformer"]["exact_json_accuracy"]
        )
        assert heldout["mean_speed_ratio_layercake_over_transformer"] >= 5.0


def test_release_verifier_passes_from_committed_inputs(tmp_path):
    output = tmp_path / "certificate.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_northstar_v22_release.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _load(output)["status"] == "PASS"
