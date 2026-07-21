"""Losslessly migrate the v22 generation path into sparse routed cakes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_byte_core_from_config import _build_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "runs_experiment" / "northstar_v21_semantic_pointer" / "latest.pt"
DEFAULT_OUTPUT = (
    ROOT / "runs_experiment" / "northstar_v23_shared3_routed_tail" / "migrated_v22.pt"
)
DEFAULT_REPORT = (
    ROOT / "results" / "breakthrough_equal" / "northstar_v23_lossless_migration.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _routed_config(source: dict[str, Any]) -> dict[str, Any]:
    config = dict(source)
    config.update(
        {
            "layers": 4,
            "shared_cake_layers": 3,
            "local_decoder": "routed_window_transformer",
            "local_width": int(source["d_model"]),
            "local_layers": 0,
            "routed_cake_experts": 5,
            "default_cake_route": 0,
        }
    )
    return config


def _copy_block(
    source: dict[str, torch.Tensor],
    destination: dict[str, torch.Tensor],
    *,
    source_prefix: str,
    destination_prefix: str,
) -> list[str]:
    copied: list[str] = []
    for suffix in (
        "attn_norm.weight",
        "attn_norm.bias",
        "qkv.weight",
        "qkv.bias",
        "attn_out.weight",
        "attn_out.bias",
        "ffn_norm.weight",
        "ffn_norm.bias",
        "down.weight",
    ):
        destination[destination_prefix + suffix] = source[source_prefix + suffix]
        copied.append(destination_prefix + suffix)
    fused_key = destination_prefix + "gate_up.weight"
    if fused_key in destination:
        destination[fused_key] = torch.cat(
            [
                source[source_prefix + "gate.weight"],
                source[source_prefix + "up.weight"],
            ],
            dim=0,
        )
        copied.append(fused_key)
    else:
        for suffix in ("gate.weight", "up.weight"):
            destination[destination_prefix + suffix] = source[source_prefix + suffix]
            copied.append(destination_prefix + suffix)
    return copied


def migrate(source_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(source_path, map_location="cpu")
    source_config = dict(checkpoint["model_config"])
    source_state = checkpoint["model"]
    routed_config = _routed_config(source_config)
    torch.manual_seed(13321)
    routed_model = _build_model(routed_config, torch.device("cpu"))
    destination_state = routed_model.state_dict()

    copied_matching: list[str] = []
    for key, value in source_state.items():
        if key in destination_state and destination_state[key].shape == value.shape:
            destination_state[key] = value
            copied_matching.append(key)

    copied_route = _copy_block(
        source_state,
        destination_state,
        source_prefix="core.3.",
        destination_prefix="core.3.experts.0.",
    )
    for route in range(1, 5):
        copied_route.extend(
            _copy_block(
                source_state,
                destination_state,
                source_prefix=f"local_core.{route - 1}.",
                destination_prefix=f"core.3.experts.{route}.",
            )
        )

    routed_model.load_state_dict(destination_state, strict=True)
    routed_model.set_cake_route(0)
    routed_model.eval()
    source_model = _build_model(source_config, torch.device("cpu"))
    source_model.load_state_dict(source_state, strict=True)
    source_model.eval()
    generator = torch.Generator(device="cpu").manual_seed(13322)
    inputs = torch.randint(0, 256, (2, 256), generator=generator)
    with torch.inference_mode():
        source_output = source_model(
            inputs,
            return_aux=True,
            return_patch_prediction=True,
        )
        routed_output = routed_model(
            inputs,
            return_aux=True,
            return_patch_prediction=True,
        )
        source_patch = source_model.generate_next_patch(inputs)
        routed_patch = routed_model.generate_next_patch(inputs)
    source_predictions = torch.stack(source_output[3], dim=2)
    routed_predictions = torch.stack(routed_output[3], dim=2)
    context_exact = torch.equal(source_output[1], routed_output[1])
    logits_exact = torch.equal(source_output[0], routed_output[0])
    prediction_exact = torch.equal(source_predictions, routed_predictions)
    patch_exact = torch.equal(source_patch, routed_patch)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    migrated_checkpoint = {
        "step": int(checkpoint.get("step", 0)),
        "model": routed_model.state_dict(),
        "model_config": routed_config,
        "train_config": {
            "cake_route": 0,
            "cake_sparse_optimizer": True,
            "migration_source": str(source_path),
        },
        "migration": {
            "source_sha256": _sha256(source_path),
            "shared_foundation_layers": 3,
            "domain_cake_routes": 5,
            "active_domain_cake_layers": 1,
            "route_zero_generation_path_exact": (
                logits_exact and context_exact and prediction_exact and patch_exact
            ),
        },
    }
    torch.save(migrated_checkpoint, output_path)
    source_metrics_path = source_path.parent / "training_metrics.json"
    if source_metrics_path.exists():
        migrated_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
        migrated_metrics["config_name"] = "northstar_v23_shared3_routed_tail"
        migrated_metrics["model_config"] = routed_config
        migrated_metrics["migration"] = migrated_checkpoint["migration"]
        migrated_metrics["latest"]["trainable_params"] = sum(
            parameter.numel() for parameter in routed_model.parameters()
        )
        (output_path.parent / "training_metrics.json").write_text(
            json.dumps(migrated_metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    report = {
        "schema_version": 1,
        "status": (
            "PASS"
            if logits_exact and context_exact and prediction_exact and patch_exact
            else "FAIL"
        ),
        "source_checkpoint": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": migrated_checkpoint["migration"]["source_sha256"],
        "migrated_checkpoint": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        "source_parameters": sum(
            parameter.numel() for parameter in source_model.parameters()
        ),
        "migrated_parameters": sum(
            parameter.numel() for parameter in routed_model.parameters()
        ),
        "matching_tensors_copied": len(copied_matching),
        "routed_block_tensors_copied": len(copied_route),
        "verification": {
            "next_byte_logits_bit_exact": logits_exact,
            "context_abi_bit_exact": context_exact,
            "patch_prediction_logits_bit_exact": prediction_exact,
            "generated_patch_bit_exact": patch_exact,
            "patch_prediction_max_abs_diff": float(
                (source_predictions - routed_predictions).abs().max().item()
            ),
            "next_byte_logits_max_abs_diff": float(
                (source_output[0] - routed_output[0]).abs().max().item()
            ),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = migrate(args.source.resolve(), args.output.resolve(), args.report.resolve())
    print(json.dumps({"status": report["status"], "report": str(args.report)}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
