import importlib.util
import json
from pathlib import Path
import sys


def load_module():
    path = Path("scripts/verify_micro_receiver_transfer_dominance.py")
    spec = importlib.util.spec_from_file_location("micro_receiver_transfer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def strict_certificate(path: Path, status: str = "PASS"):
    payload = {
        "status": status,
        "scales": [
            {"scale": "1m", "status": status},
            {"scale": "2m", "status": status},
            {"scale": "5m", "status": status},
            {"scale": "10m", "status": status},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_receiver_transfer_verifier_passes_with_strict_core_certificate(tmp_path):
    module = load_module()
    cert = tmp_path / "strict.json"
    out = tmp_path / "receiver.json"
    strict_certificate(cert, "PASS")

    result = module.verify(
        cert,
        out,
        eval_bytes=1024,
        eval_batches=2,
        generation_bytes=8,
        train_steps=2,
    )

    assert result["status"] == "PASS"
    assert result["required_gates"]["transfer_ppl_ratio_exact"] is True
    assert result["metrics"]["transfer_max_logit_diff"] == 0.0
    assert out.exists()


def test_receiver_transfer_verifier_rejects_failed_strict_certificate(tmp_path):
    module = load_module()
    cert = tmp_path / "strict.json"
    out = tmp_path / "receiver.json"
    strict_certificate(cert, "FAIL")

    result = module.verify(
        cert,
        out,
        eval_bytes=1024,
        eval_batches=2,
        generation_bytes=8,
        train_steps=2,
    )

    assert result["status"] == "FAIL"
    assert result["required_gates"]["strict_core_receiver_scales_pass"] is False
