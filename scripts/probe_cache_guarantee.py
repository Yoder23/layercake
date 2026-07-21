"""Measure exact argmax certificates from bounded causal-memory stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402
from layercake.count_cake_cpu import CountCakeCPUDecoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--patches", type=int, default=20)
    args = parser.parse_args()
    model, _ = load_count_cake_bundle(args.bundle, device="cpu")
    prompt = Path(args.prompt).read_bytes()[:1024]
    rows = torch.tensor(list(prompt), dtype=torch.long).reshape(1, -1)
    reference_state = model.begin_cached_generation(rows)
    reference = CountCakeCPUDecoder(model).generate_cached(
        reference_state, patches=args.patches
    )[0].tolist()
    state = model.begin_cached_generation(rows)
    memory = state["online_cache"]
    history = state["online_history"]
    certified = 0
    certified_equal = 0
    longest = 0
    run = 0
    margins = []
    for observed in reference:
        lower = np.zeros(256, dtype=np.float64)
        upper = np.ones(256, dtype=np.float64)
        if memory.recent is not None:
            for recent, (order, strength) in zip(
                memory.recent._recent, memory.recent.specs
            ):
                if len(history) < order:
                    continue
                match = recent.get(bytes(history[-order:]))
                if (
                    match is None
                    or memory.recent.position - match[1] > memory.recent.window
                ):
                    continue
                delta = np.zeros(256)
                delta[int(match[0])] = 1.0
                lower = (delta + strength * lower) / (1.0 + strength)
                upper = (delta + strength * upper) / (1.0 + strength)
        if memory.normalized is not None:
            for table, (order, strength) in zip(
                memory.normalized._counts, memory.normalized.specs
            ):
                context = memory.normalized._context(history, order)
                continuations = table.get(context)
                if not continuations:
                    continue
                counts = np.zeros(256)
                for target, count in continuations.items():
                    counts[int(target)] = int(count)
                total = counts.sum()
                lower = (counts + strength * lower) / (total + strength)
                upper = (counts + strength * upper) / (total + strength)
        candidate = int(lower.argmax())
        competitor = float(np.max(np.delete(upper, candidate)))
        margin = float(lower[candidate] - competitor)
        is_certified = margin > 0.0
        if is_certified:
            certified += 1
            certified_equal += int(candidate == observed)
            margins.append(margin)
            run += 1
            longest = max(longest, run)
        else:
            run = 0
        memory.update(history, int(observed))
        history.append(int(observed))
        if len(history) > memory.max_order:
            del history[: len(history) - memory.max_order]
    report = {
        "bytes": len(reference),
        "certified": certified,
        "certified_fraction": certified / len(reference),
        "certified_equal": certified_equal,
        "longest_certified_run": longest,
        "minimum_positive_margin": min(margins) if margins else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
