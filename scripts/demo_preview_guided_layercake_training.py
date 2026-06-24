from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from layercake.causal_byte_models import CausalBytePatchLM
from layercake.rolling.baselines import baseline_training_smoke_loop, parameter_count
from layercake.rolling.registry import ModuleRegistry
from layercake.rolling.reports import append_capability_ledger, write_training_diff_report
from layercake.rolling.rubric import TrainingRubric
from layercake.rolling.trainer import RollingTrainer


def _dataset(path: Path) -> torch.Tensor:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "LayerCake previews the data before training.\n"
            "Byte patch models preserve rollbackable knowledge.\n"
            "Mobile CPU inference needs efficient updates.\n",
            encoding="utf-8",
        )
    raw = path.read_bytes()[:96]
    return torch.tensor([list(raw)], dtype=torch.long)


def _tiny_model() -> CausalBytePatchLM:
    return CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=64,
        continuous_local=True,
        local_decoder="gru",
        local_layers=1,
    )


def _bpb(model, batch):
    model.eval()
    with torch.no_grad():
        logits, _ = model(batch)
        logits = logits[:, :-1]
        targets = batch[:, 1 : 1 + logits.shape[1]]
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
    return float(loss.item() / math.log(2.0)), float(loss.item())


def run_demo(*, smoke=True, output="results/certificates/preview_guided_layercake_demo.json") -> dict:
    torch.manual_seed(2026)
    data_path = Path("data/rolling_preview_smoke.txt")
    batch = _dataset(data_path)
    model = _tiny_model()
    registry = ModuleRegistry()
    registry.register("layercake_core", model)
    trainer = RollingTrainer(
        registry,
        root="artifacts/commits/preview_guided",
        model_family_id="tiny-layercake-byte-patch",
        abi_hash="tiny-layercake-abi-v1",
        input_interface_hash="bytes-v1",
        byte_patch_hash="fixed-2-v1",
    )
    rubric = TrainingRubric(
        rubric_id="preview_guided_layercake_smoke",
        branch="preview",
        max_steps=4 if smoke else 16,
        trainable_modules=["layercake_core"],
        gates=[
            {"type": "max_metric", "name": "bpb_gate", "metric": "bpb", "threshold": 20.0},
            {"type": "quality_per_step", "name": "qps_gate", "min_gain_per_step": -10.0},
            {"type": "protected_capabilities", "name": "protected_capability_firewall"},
        ],
    )
    initial = trainer.create_commit(None, TrainingRubric(rubric_id="preview_initial", branch="preview"), "initial", "passed")
    before_bpb, before_loss = _bpb(model, batch)
    opt = torch.optim.SGD(model.parameters(), lr=0.02)

    def train_step():
        model.train()
        logits, _ = model(batch)
        logits = logits[:, :-1]
        targets = batch[:, 1 : 1 + logits.shape[1]]
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item())

    baseline = baseline_training_smoke_loop(batch, steps=2, target_params=parameter_count(model))
    commit, cert, rollback, preview, syllabus = trainer.run_preview_guided(
        rubric,
        data_path,
        model=model,
        parent_commit=initial,
        train_step=train_step,
        metrics={"parent": {"bpb": before_bpb}, "bpb": before_bpb, "transformer": {"bpb": baseline["after_bpb"]}},
        certificate_path=output,
    )
    after_bpb, _ = _bpb(model, batch)
    bad_rubric = TrainingRubric(
        rubric_id="preview_guided_bad_stage",
        branch="preview",
        max_steps=4,
        trainable_modules=["layercake_core"],
        gates=[{"type": "max_metric", "name": "bad_bpb_gate", "metric": "bpb", "threshold": 0.0}],
    )

    def bad_step():
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10.0)
        return 999.0

    failed, failed_cert, bad_rollback, bad_preview, bad_syllabus = trainer.run_preview_guided(
        bad_rubric,
        data_path,
        model=model,
        parent_commit=commit,
        train_step=bad_step,
        metrics={"parent": {"bpb": after_bpb}, "bpb": after_bpb},
        certificate_path="results/certificates/preview_guided_bad_stage.json",
    )
    report_path = write_training_diff_report(
        commit,
        initial,
        preview=preview,
        syllabus=syllabus,
        metrics_before={"bpb": before_bpb, "loss": before_loss},
        metrics_after={"bpb": after_bpb},
        gate_results=cert.gate_results,
        transformer_baseline=baseline,
        rollback_report=rollback,
        warnings=preview.warnings,
    )
    append_capability_ledger(
        commit_id=commit.commit_id,
        parent_commit_id=initial.commit_id,
        rubric_id=rubric.rubric_id,
        preview_id=preview.preview_id,
        syllabus_id=syllabus.syllabus_id,
        capability="preview_guided_training",
        metric="bpb",
        value=after_bpb,
        threshold=20.0,
        passed=cert.passed,
        delta_from_parent=after_bpb - before_bpb,
        delta_vs_transformer_baseline=after_bpb - baseline["after_bpb"],
    )
    result = {
        "status": "PASS" if cert.passed and bad_rollback else "FAIL",
        "commit_id": commit.commit_id,
        "failed_commit_id": failed.commit_id,
        "preview_path": f"results/previews/{preview.preview_id}.json",
        "syllabus_path": f"results/syllabi/{syllabus.syllabus_id}.json",
        "certificate_path": output,
        "bad_stage_certificate_path": "results/certificates/preview_guided_bad_stage.json",
        "training_diff_report": str(report_path),
        "before_bpb": before_bpb,
        "after_bpb": after_bpb,
        "transformer_baseline_after_bpb": baseline["after_bpb"],
        "rollback_report": bad_rollback,
    }
    Path(output).write_text(json.dumps({**json.loads(Path(output).read_text(encoding="utf-8")), "demo_result": result}, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="results/certificates/preview_guided_layercake_demo.json")
    args = parser.parse_args()
    result = run_demo(smoke=args.smoke, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
