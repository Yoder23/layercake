from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_number(payload: dict[str, Any], *path: str) -> float:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return float(current)


def build_certificate(
    *,
    layercake_train: Path,
    transformer_train: Path,
    cpu_generation: Path,
    gpu_generation: Path,
    min_speed_ratio: float,
    max_bpb_ratio: float,
) -> dict[str, Any]:
    layercake = _load(layercake_train)
    transformer = _load(transformer_train)
    cpu = _load(cpu_generation)
    gpu = _load(gpu_generation)

    layercake_bpb = _get_number(layercake, "general", "bpb")
    transformer_bpb = _get_number(transformer, "general", "bpb")
    cpu_ratio = _get_number(cpu, "speed_ratio")
    gpu_ratio = _get_number(gpu, "speed_ratio")
    bpb_ratio = layercake_bpb / max(transformer_bpb, 1e-12)

    gates = {
        "layercake_training_complete": layercake.get("status") == "TRAINED",
        "same_size_parameter_window": (
            0.95
            <= _get_number(layercake, "parameters")
            / max(_get_number(transformer, "parameters"), 1e-12)
            <= 1.05
        ),
        "lm_bpb_noninferior": bpb_ratio <= max_bpb_ratio,
        "cpu_generation_5x": cpu_ratio >= min_speed_ratio,
        "gpu_generation_5x": gpu_ratio >= min_speed_ratio,
        "raw_cpu_generation_present": bool(cpu.get("layercake", {}).get("hex"))
        and bool(cpu.get("bpe", {}).get("hex")),
        "raw_gpu_generation_present": bool(gpu.get("layercake", {}).get("hex"))
        and bool(gpu.get("bpe", {}).get("hex")),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    return {
        "status": status,
        "candidate": "same-data LayerCake speed plus BPB candidate",
        "artifacts": {
            "layercake_train": str(layercake_train).replace("\\", "/"),
            "transformer_train": str(transformer_train).replace("\\", "/"),
            "cpu_generation": str(cpu_generation).replace("\\", "/"),
            "gpu_generation": str(gpu_generation).replace("\\", "/"),
        },
        "thresholds": {
            "min_speed_ratio": min_speed_ratio,
            "max_bpb_ratio": max_bpb_ratio,
        },
        "metrics": {
            "layercake_bpb": layercake_bpb,
            "transformer_bpb": transformer_bpb,
            "cpu_generation_speed_ratio": cpu_ratio,
            "gpu_generation_speed_ratio": gpu_ratio,
        },
        "ratios": {
            "heldout_bpb_ratio_layercake_over_transformer": bpb_ratio,
            "cpu_generation_speed_ratio_layercake_over_transformer": cpu_ratio,
            "gpu_generation_speed_ratio_layercake_over_transformer": gpu_ratio,
        },
        "gates": gates,
        "scope": (
            "Aggregates raw same-data training/eval BPB and direct CPU/GPU "
            "generation artifacts. This certificate does not cover task "
            "relevance, schema/action quality, prefill, footprint, domain "
            "migration, phone runtime, or local runtime comparison."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a same-data LayerCake speed plus BPB candidate."
    )
    parser.add_argument("--layercake-train", required=True, type=Path)
    parser.add_argument("--transformer-train", required=True, type=Path)
    parser.add_argument("--cpu-generation", required=True, type=Path)
    parser.add_argument("--gpu-generation", required=True, type=Path)
    parser.add_argument("--min-speed-ratio", type=float, default=5.0)
    parser.add_argument("--max-bpb-ratio", type=float, default=1.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    certificate = build_certificate(
        layercake_train=args.layercake_train,
        transformer_train=args.transformer_train,
        cpu_generation=args.cpu_generation,
        gpu_generation=args.gpu_generation,
        min_speed_ratio=args.min_speed_ratio,
        max_bpb_ratio=args.max_bpb_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(certificate, indent=2, sort_keys=True))
    if certificate["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
