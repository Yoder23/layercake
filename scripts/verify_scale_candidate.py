from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layercake.northstar import NorthStarMetrics


def _load(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_training_bytes(payload: dict) -> float:
    if "training_bytes" in payload:
        return float(payload["training_bytes"])
    if "estimated_total_training_bytes" in payload:
        return float(payload["estimated_total_training_bytes"])
    if "train_bytes" in payload:
        return float(payload["train_bytes"])
    return math.nan


def _extract_bpb(payload: dict) -> float:
    if "heldout_bpb" in payload:
        return float(payload["heldout_bpb"])
    return float(payload.get("general", {}).get("bpb", math.nan))


def _build_metrics(values: dict, args: argparse.Namespace) -> dict:
    required = set(NorthStarMetrics.__dataclass_fields__.keys())
    if required.issubset(values.keys()):
        return {key: values[key] for key in required}

    baseline = _load(args.baseline)
    transfer = _load(args.transfer)
    baseline_domain_bpb = args.baseline_domain_bpb
    if baseline_domain_bpb is None:
        baseline_domain_bpb = baseline.get("python_domain", {}).get("bpb", math.nan)

    return {
        "parameters": int(values.get("parameters", 0)),
        "baseline_parameters": int(baseline.get("parameters", 0)),
        "heldout_bpb": _extract_bpb(values),
        "baseline_heldout_bpb": _extract_bpb(baseline),
        "training_bytes": _extract_training_bytes(values),
        "baseline_training_bytes": _extract_training_bytes(baseline),
        "training_seconds": float(values.get("elapsed_seconds", math.nan)),
        "baseline_training_seconds": float(baseline.get("elapsed_seconds", math.nan)),
        "mobile_prefill_ratio": float(args.mobile_prefill_ratio),
        "mobile_generation_ratio": float(args.mobile_generation_ratio),
        "desktop_prefill_ratio": float(args.desktop_prefill_ratio),
        "desktop_generation_ratio": float(args.desktop_generation_ratio),
        "gpu_prefill_ratio": float(args.gpu_prefill_ratio),
        "gpu_generation_ratio": float(args.gpu_generation_ratio),
        "migration_ppl_ratio": float(transfer.get("ppl_ratio", args.migration_ppl_ratio)),
        "migration_max_logit_diff": float(transfer.get("max_logit_diff", args.migration_max_logit_diff)),
        "migrated_domain_bpb": float(
            transfer.get("target", {}).get(
                "bpb",
                values.get("python_domain", {}).get("bpb", math.nan),
            )
        ),
        "baseline_domain_bpb": float(baseline_domain_bpb),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--transfer")
    parser.add_argument("--baseline-domain-bpb", type=float)
    parser.add_argument("--mobile-prefill-ratio", type=float, default=1.0)
    parser.add_argument("--mobile-generation-ratio", type=float, default=1.0)
    parser.add_argument("--desktop-prefill-ratio", type=float, default=1.0)
    parser.add_argument("--desktop-generation-ratio", type=float, default=1.0)
    parser.add_argument("--gpu-prefill-ratio", type=float, default=1.0)
    parser.add_argument("--gpu-generation-ratio", type=float, default=1.0)
    parser.add_argument("--migration-ppl-ratio", type=float, default=1.0)
    parser.add_argument("--migration-max-logit-diff", type=float, default=0.0)
    args = parser.parse_args()
    values = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    metrics = _build_metrics(values, args)
    certificate = NorthStarMetrics(**metrics).certificate()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(certificate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(certificate, indent=2))
    if certificate["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
