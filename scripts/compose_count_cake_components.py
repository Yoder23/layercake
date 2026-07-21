"""Compose a sparse CountCake state and shape-compatible neural host losslessly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle, save_count_cake_bundle  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-bundle", required=True)
    parser.add_argument("--neural-bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    count_path = Path(args.count_bundle)
    neural_path = Path(args.neural_bundle)
    output = Path(args.output)
    count_model, count_manifest = load_count_cake_bundle(count_path)
    neural_model, neural_manifest = load_count_cake_bundle(neural_path)
    structural_fields = (
        "patch_size",
        "chunking_mode",
        "prediction_start",
        "d_byte",
        "d_model",
        "d_abi",
        "local_width",
        "local_recurrent",
        "local_continuous",
        "confidence_gate",
        "patch_layers",
        "patch_core_type",
        "patch_attention_heads",
        "local_decoder",
        "local_layers",
        "local_gru_layers",
        "local_rank",
        "byte_head",
        "gate_hidden_width",
    )
    mismatch = {
        field: (
            count_manifest["model"].get(field),
            neural_manifest["model"].get(field),
        )
        for field in structural_fields
        if count_manifest["model"].get(field)
        != neural_manifest["model"].get(field)
    }
    if mismatch:
        raise ValueError(f"neural host structures differ: {mismatch}")
    neural_state = {
        name: tensor
        for name, tensor in neural_model.state_dict().items()
        if not name.startswith("count_cake.")
    }
    incompatible = count_model.load_state_dict(neural_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith("count_cake.")
    ]
    if missing or unexpected:
        raise ValueError(
            f"component composition failed: missing={missing}, unexpected={unexpected}"
        )
    metadata = {
        "composition": {
            "kind": "lossless_sparse_state_plus_neural_host",
            "count_source": {
                "path": str(count_path),
                "sha256": _sha256(count_path),
                "manifest_parameters": count_manifest["parameters"],
            },
            "neural_source": {
                "path": str(neural_path),
                "sha256": _sha256(neural_path),
                "manifest_parameters": neural_manifest["parameters"],
            },
            "copied_neural_tensors": len(neural_state),
            "conversion": "none",
        },
        "count_source_metadata": count_manifest.get("metadata", {}),
        "neural_source_metadata": neural_manifest.get("metadata", {}),
    }
    saved = save_count_cake_bundle(count_model, output, metadata=metadata)
    report = {
        "format": "layercake-component-composition/1",
        "status": "COMPLETE",
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "parameters": saved["parameters"],
        },
        **metadata["composition"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
