from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("scripts/verify_production_1m_vs_5m_transfer_dominance.py")
    spec = importlib.util.spec_from_file_location("production_transfer_dominance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _certificate(path: Path, *, status: str = "PASS") -> Path:
    gates = {
        "bpb_non_inferior": True,
        "training_speed_met": True,
        "cpu_generation_5x_met": True,
        "gpu_generation_noninferior": True,
        "cpu_quality_noninferior": True,
        "gpu_quality_noninferior": True,
    }
    if status != "PASS":
        gates["gpu_generation_noninferior"] = False
    path.write_text(
        json.dumps(
            {
                "status": status,
                "gates": gates,
                "ratios": {
                    "cpu_generation_speed_ratio": 10.0,
                    "gpu_generation_speed_ratio": 1.4,
                },
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_combined_transfer_gate_rejects_failed_source_certificate(tmp_path, monkeypatch):
    module = _load_module()
    cert = _certificate(tmp_path / "dominance.json", status="FAIL")
    out = tmp_path / "out.json"

    monkeypatch.setattr(
        module,
        "_exact_checkpoint_transfer",
        lambda *args, **kwargs: {
            "transfer_ppl_ratio": 1.0,
            "transfer_max_logit_diff": 0.0,
            "transfer_max_abi_diff": 0.0,
            "transfer_generation_exact": True,
            "abi_shape": [2, 256, 32],
            "checkpoint_model_params": 982322,
        },
    )

    result = module.verify(
        dominance_certificate=cert,
        checkpoint=tmp_path / "missing.pt",
        output=out,
        device="cpu",
        seed=1234,
        eval_rows=2,
    )

    assert result["status"] == "FAIL"
    assert result["gates"]["source_dominance_certificate_pass"] is False


def test_combined_transfer_gate_passes_exact_transfer_and_source_dominance(tmp_path, monkeypatch):
    module = _load_module()
    cert = _certificate(tmp_path / "dominance.json", status="PASS")
    out = tmp_path / "out.json"

    monkeypatch.setattr(
        module,
        "_exact_checkpoint_transfer",
        lambda *args, **kwargs: {
            "transfer_ppl_ratio": 1.0,
            "transfer_max_logit_diff": 0.0,
            "transfer_max_abi_diff": 0.0,
            "transfer_generation_exact": True,
            "abi_shape": [2, 256, 32],
            "checkpoint_model_params": 982322,
        },
    )

    result = module.verify(
        dominance_certificate=cert,
        checkpoint=tmp_path / "missing.pt",
        output=out,
        device="cpu",
        seed=1234,
        eval_rows=2,
    )

    assert result["status"] == "PASS"
    assert result["gates"]["transfer_ppl_ratio_exact"] is True
    assert result["gates"]["receiver_inherits_cpu_generation_win"] is True
