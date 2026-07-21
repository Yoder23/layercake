"""Verify routed-cake convergence and bit-exact isolation of route zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_byte_core_from_config import _build_model


ROOT = Path(__file__).resolve().parents[1]


def verify(base_path: Path, trained_path: Path, metrics_path: Path) -> dict:
    base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
    trained_checkpoint = torch.load(trained_path, map_location="cpu", weights_only=True)
    base_state = base_checkpoint["model"]
    trained_state = trained_checkpoint["model"]
    if base_state.keys() != trained_state.keys():
        raise RuntimeError("base and trained checkpoint keys differ")
    changed = [
        key
        for key in base_state
        if not torch.equal(base_state[key], trained_state[key])
    ]
    isolated = bool(changed) and all(
        key.startswith("core.3.experts.4.") for key in changed
    )

    base_model = _build_model(base_checkpoint["model_config"], torch.device("cpu"))
    trained_model = _build_model(trained_checkpoint["model_config"], torch.device("cpu"))
    base_model.load_state_dict(base_state, strict=True)
    trained_model.load_state_dict(trained_state, strict=True)
    base_model.set_cake_route(0)
    trained_model.set_cake_route(0)
    base_model.eval()
    trained_model.eval()
    generator = torch.Generator(device="cpu").manual_seed(13332)
    inputs = torch.randint(0, 256, (2, 256), generator=generator)
    with torch.inference_mode():
        base_output = base_model(
            inputs,
            return_aux=True,
            return_patch_prediction=True,
        )
        trained_output = trained_model(
            inputs,
            return_aux=True,
            return_patch_prediction=True,
        )
        base_patch = base_model.generate_next_patch(inputs)
        trained_patch = trained_model.generate_next_patch(inputs)
    base_predictions = torch.stack(base_output[3], dim=2)
    trained_predictions = torch.stack(trained_output[3], dim=2)
    route_zero_generation_exact = all(
        (
            torch.equal(base_output[1], trained_output[1]),
            torch.equal(base_predictions, trained_predictions),
            torch.equal(base_patch, trained_patch),
        )
    )

    base_model.set_cake_route(4)
    trained_model.set_cake_route(4)
    with torch.inference_mode():
        base_route4 = base_model.generate_next_patch(inputs)
        trained_route4 = trained_model.generate_next_patch(inputs)
    route_four_changed = not torch.equal(base_route4, trained_route4)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = metrics["history"]
    first_loss = float(history[0]["loss"])
    final_loss = float(history[-1]["loss"])
    routing = metrics["cake_routing"]
    total_parameters = sum(parameter.numel() for parameter in base_model.parameters())
    optimizer_fraction_of_total = int(routing["optimizer_params"]) / total_parameters
    gates = {
        "only_route_four_tensors_changed": isolated,
        "route_zero_abi_bit_exact": torch.equal(base_output[1], trained_output[1]),
        "route_zero_patch_logits_bit_exact": torch.equal(
            base_predictions, trained_predictions
        ),
        "route_zero_generation_bit_exact": torch.equal(base_patch, trained_patch),
        "route_four_behavior_changed": route_four_changed,
        "training_loss_decreased": final_loss < first_loss,
        "sparse_optimizer_enabled": bool(routing["sparse_optimizer"]),
        "optimizer_fraction_of_total_below_12_percent": (
            optimizer_fraction_of_total < 0.12
        ),
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "base_checkpoint": str(base_path.relative_to(ROOT)).replace("\\", "/"),
        "trained_checkpoint": str(trained_path.relative_to(ROOT)).replace("\\", "/"),
        "changed_tensor_count": len(changed),
        "changed_tensor_prefix": "core.3.experts.4.",
        "changed_tensors": changed,
        "training": {
            "first_logged_loss": first_loss,
            "final_logged_loss": final_loss,
            "loss_ratio_final_over_first": final_loss / first_loss,
            "optimizer_parameters": int(routing["optimizer_params"]),
            "logical_total_parameters": total_parameters,
            "optimizer_fraction_of_total": optimizer_fraction_of_total,
        },
        "route_zero_generation_path_bit_exact": route_zero_generation_exact,
        "legacy_next_byte_logits_bit_exact": torch.equal(
            base_output[0], trained_output[0]
        ),
        "legacy_next_byte_decoder_note": (
            "The compatibility-only local decoder reuses experts 1-4, so training "
            "route 4 is intentionally outside the generation-path isolation claim."
        ),
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "runs_experiment/northstar_v23_shared3_routed_tail/migrated_v22.pt",
    )
    parser.add_argument(
        "--trained",
        type=Path,
        default=ROOT / "runs_experiment/northstar_v23_route4_schema_training/latest.pt",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "runs_experiment/northstar_v23_route4_schema_training/training_metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/breakthrough_equal/northstar_v23_route_isolation.json",
    )
    args = parser.parse_args()
    report = verify(args.base.resolve(), args.trained.resolve(), args.metrics.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
