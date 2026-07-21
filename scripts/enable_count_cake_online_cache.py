"""Attach an explicit bounded causal-memory recipe to a trained bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle, save_count_cake_bundle  # noqa: E402


def _specs(value: str) -> tuple[tuple[int, float], ...]:
    if not value:
        return ()
    return tuple(
        (int(item.split(":", 1)[0]), float(item.split(":", 1)[1]))
        for item in value.split(",")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="immutable trained source")
    parser.add_argument("--output", required=True, help="new configured bundle")
    parser.add_argument("--metrics", help="optional source metrics to copy beside output")
    parser.add_argument("--output-metrics")
    parser.add_argument("--backoff-strengths", required=True)
    parser.add_argument("--online-cache-specs", required=True)
    parser.add_argument("--recent-cache-specs", default="")
    parser.add_argument("--normalized-cache-specs", default="")
    parser.add_argument("--cache-window", type=int, required=True)
    parser.add_argument(
        "--cache-normalization",
        choices=("casefold", "classes"),
        default="classes",
    )
    args = parser.parse_args()
    if bool(args.metrics) != bool(args.output_metrics):
        raise ValueError("--metrics and --output-metrics must be supplied together")

    source_path = Path(args.bundle)
    output_path = Path(args.output)
    model, manifest = load_count_cake_bundle(source_path)
    model.count_cake.backoff_strengths = tuple(
        float(value) for value in args.backoff_strengths.split(",")
    )
    if len(model.count_cake.backoff_strengths) < model.count_cake.max_order:
        raise ValueError("backoff recipe is shorter than the trained count cake")
    model.online_cache_specs = _specs(args.online_cache_specs)
    model.online_cache_window = int(args.cache_window)
    model.recent_cache_specs = _specs(args.recent_cache_specs)
    model.normalized_cache_specs = _specs(args.normalized_cache_specs)
    model.cache_normalization = args.cache_normalization
    model._new_causal_cache()

    metadata = dict(manifest.get("metadata", {}))
    metadata["runtime_upgrade"] = {
        "kind": "bounded_strictly_causal_composite_byte_memory",
        "exact_specs": [list(spec) for spec in model.online_cache_specs],
        "recent_specs": [list(spec) for spec in model.recent_cache_specs],
        "normalized_specs": [
            list(spec) for spec in model.normalized_cache_specs
        ],
        "normalization": model.cache_normalization,
        "window": model.online_cache_window,
        "learned_parameters_added": 0,
        "selection": "fit on frozen validation split; certification split untouched",
    }
    saved = save_count_cake_bundle(model, output_path, metadata=metadata)
    payload = output_path.read_bytes()
    artifact = {
        "path": str(output_path),
        "format": saved["format"],
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if args.metrics:
        metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
        metrics.setdefault("config", {})["causal_memory"] = metadata[
            "runtime_upgrade"
        ]
        metrics["artifact"] = artifact
        Path(args.output_metrics).write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
