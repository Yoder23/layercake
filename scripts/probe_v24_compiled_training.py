"""Paired compile probe for the matched 1M v24 production candidates.

This is an architecture-development probe, not a release benchmark.  It uses
the exact generation-aligned LayerCake loss and the transformer's next-token
loss, gives both models the same raw-byte exposure, and applies torch.compile
to both training graphs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_northstar_training_speed import (  # noqa: E402
    _benchmark_model,
    _summary,
)
from scripts.train_bpe_transformer_from_config import (  # noqa: E402
    BPETokenTransformerLM,
)
from scripts.train_byte_core_from_config import _build_model  # noqa: E402


BYTES_PER_TOKEN = 3.146


class LayerCakeTrainingGraph(torch.nn.Module):
    def __init__(
        self,
        max_patches: int = 64,
        patch_size: int = 32,
        d_byte: int = 24,
        d_model: int = 168,
        layers: int = 2,
        heads: int = 4,
        generation_width: int = 176,
        generation_heads: int = 4,
        persistent_context: bool = False,
        prediction_mode: str = "radix_recurrent",
        ngram_buckets: int = 0,
        generation_layers: int = 2,
    ) -> None:
        super().__init__()
        self.model = _build_model(
            {
                "patch_size": patch_size,
                "d_byte": d_byte,
                "d_model": d_model,
                "d_abi": 64,
                "layers": layers,
                "heads": heads,
                "max_patches": max_patches,
                "continuous_local": False,
                "direct_global_context": False,
                "local_decoder": "patch_generator_only",
                "local_layers": 0,
                "local_width": d_model,
                "dropout": 0.0,
                "modern_blocks": True,
                "fused_attention": True,
                "patch_prediction": True,
                "patch_prediction_mode": prediction_mode,
                "patch_prediction_context": "global",
                "patch_generation_width": generation_width,
                "patch_generation_heads": generation_heads,
                "patch_generation_bytes": patch_size,
                "patch_generation_persistent_context": persistent_context,
                "patch_generation_ngram_buckets": ngram_buckets,
                "patch_generation_layers": generation_layers,
            },
            torch.device("cpu"),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.model.domain_cake_patch_predictions(rows, loss_only=True)


class TransformerTrainingGraph(torch.nn.Module):
    def __init__(
        self,
        max_len: int = 512,
        d_model: int = 92,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.model = BPETokenTransformerLM(
            vocab_size=4096,
            d_model=d_model,
            layers=layers,
            heads=heads,
            max_len=max_len,
            ff_mult=4,
            dropout=0.0,
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        logits = self.model(rows[:, :-1])
        return torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), rows[:, 1:].flatten()
        )


def _compiled(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return torch.compile(
        module.to(device),
        mode="reduce-overhead",
        fullgraph=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--layercake-batch-size", type=int)
    parser.add_argument("--transformer-batch-size", type=int)
    parser.add_argument("--raw-bytes", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--transformer-d-model", type=int, default=92)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--layercake-d-byte", type=int, default=24)
    parser.add_argument("--layercake-patch-size", type=int, default=32)
    parser.add_argument("--layercake-d-model", type=int, default=168)
    parser.add_argument("--layercake-layers", type=int, default=2)
    parser.add_argument("--layercake-heads", type=int, default=4)
    parser.add_argument("--layercake-generation-width", type=int, default=176)
    parser.add_argument("--layercake-generation-heads", type=int, default=4)
    parser.add_argument("--layercake-ngram-buckets", type=int, default=0)
    parser.add_argument("--layercake-generation-layers", type=int, default=2)
    parser.add_argument(
        "--layercake-persistent-context",
        action="store_true",
    )
    parser.add_argument(
        "--layercake-mode",
        choices=[
            "radix_causal",
            "radix_cumsum",
            "radix_cumsum_hash",
            "radix_depthwise_hash",
            "radix_conv",
            "radix_dilated_conv",
            "radix_hash",
            "radix_grouped_recurrent_hash",
            "radix_prefix",
            "radix_low_rank_recurrent_hash",
            "radix_recurrent",
            "radix_recurrent_conditional_hash",
            "radix_recurrent_hash",
            "radix_rotary_hash",
            "radix_scan",
            "radix_scan_hash",
            "radix_simple_recurrent_hash",
            "radix_window",
        ],
        default="radix_recurrent",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Use torch.compile for both candidates (disabled by default).",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("this probe requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(int(args.cpu_threads))
    token_count = round(args.raw_bytes / BYTES_PER_TOKEN)
    max_patches = max(
        64,
        (args.raw_bytes + 2 * args.layercake_patch_size - 1)
        // args.layercake_patch_size,
    )
    layercake_batch_size = args.layercake_batch_size or args.batch_size
    transformer_batch_size = args.transformer_batch_size or args.batch_size
    layercake_rows = []
    transformer_rows = []
    for repeat in range(args.repeats):
        order = ("layercake", "transformer")
        if repeat % 2:
            order = tuple(reversed(order))
        for name in order:
            if name == "layercake":
                row = _benchmark_model(
                    name=name,
                    build=lambda device: (
                        _compiled(
                            LayerCakeTrainingGraph(
                                max_patches=max_patches,
                                patch_size=args.layercake_patch_size,
                                d_byte=args.layercake_d_byte,
                                d_model=args.layercake_d_model,
                                layers=args.layercake_layers,
                                heads=args.layercake_heads,
                                generation_width=args.layercake_generation_width,
                                generation_heads=args.layercake_generation_heads,
                                persistent_context=args.layercake_persistent_context,
                                prediction_mode=args.layercake_mode,
                                ngram_buckets=args.layercake_ngram_buckets,
                                generation_layers=args.layercake_generation_layers,
                            ),
                            device,
                        )
                        if args.compile_model
                        else LayerCakeTrainingGraph(
                            max_patches=max_patches,
                            patch_size=args.layercake_patch_size,
                            d_byte=args.layercake_d_byte,
                            d_model=args.layercake_d_model,
                            layers=args.layercake_layers,
                            heads=args.layercake_heads,
                            generation_width=args.layercake_generation_width,
                            generation_heads=args.layercake_generation_heads,
                            persistent_context=args.layercake_persistent_context,
                            prediction_mode=args.layercake_mode,
                            ngram_buckets=args.layercake_ngram_buckets,
                            generation_layers=args.layercake_generation_layers,
                        ).to(device)
                    ),
                    loss_fn=lambda model, rows: model(rows),
                    shape=(
                        layercake_batch_size,
                        args.raw_bytes + args.layercake_patch_size,
                    ),
                    logical_bytes_per_step=float(
                        layercake_batch_size * args.raw_bytes
                    ),
                    token_high=256,
                    device=device,
                    warmup_steps=args.warmup_steps,
                    measured_steps=args.measured_steps,
                    repeat=repeat,
                )
                layercake_rows.append(row)
            else:
                row = _benchmark_model(
                    name=name,
                    build=lambda device: (
                        _compiled(
                            TransformerTrainingGraph(
                                max_len=token_count,
                                d_model=args.transformer_d_model,
                                layers=args.transformer_layers,
                                heads=args.transformer_heads,
                            ),
                            device,
                        )
                        if args.compile_model
                        else TransformerTrainingGraph(
                            max_len=token_count,
                            d_model=args.transformer_d_model,
                            layers=args.transformer_layers,
                            heads=args.transformer_heads,
                        ).to(device)
                    ),
                    loss_fn=lambda model, rows: model(rows),
                    shape=(transformer_batch_size, token_count + 1),
                    logical_bytes_per_step=float(
                        transformer_batch_size * token_count * BYTES_PER_TOKEN
                    ),
                    token_high=4096,
                    device=device,
                    warmup_steps=args.warmup_steps,
                    measured_steps=args.measured_steps,
                    repeat=repeat,
                )
                transformer_rows.append(row)

    ratios = [
        layercake["logical_bytes_per_second"]
        / transformer["logical_bytes_per_second"]
        for layercake, transformer in zip(layercake_rows, transformer_rows)
    ]
    print(
        json.dumps(
            {
                "compiled": bool(args.compile_model),
                "device": str(device),
                "cpu_threads": (
                    int(args.cpu_threads) if device.type == "cpu" else None
                ),
                "batch_size": int(args.batch_size),
                "layercake_batch_size": int(layercake_batch_size),
                "transformer_batch_size": int(transformer_batch_size),
                "raw_bytes": int(args.raw_bytes),
                "transformer_d_model": int(args.transformer_d_model),
                "transformer_layers": int(args.transformer_layers),
                "transformer_heads": int(args.transformer_heads),
                "layercake_d_model": int(args.layercake_d_model),
                "layercake_patch_size": int(args.layercake_patch_size),
                "layercake_layers": int(args.layercake_layers),
                "layercake_generation_width": int(
                    args.layercake_generation_width
                ),
                "layercake_generation_heads": int(
                    args.layercake_generation_heads
                ),
                "layercake_persistent_context": bool(
                    args.layercake_persistent_context
                ),
                "layercake_mode": str(args.layercake_mode),
                "layercake_ngram_buckets": int(args.layercake_ngram_buckets),
                "layercake_generation_layers": int(
                    args.layercake_generation_layers
                ),
                "layercake": _summary(layercake_rows),
                "transformer": _summary(transformer_rows),
                "speedup_values": ratios,
                "speedup_median": statistics.median(ratios),
                "speedup_minimum": min(ratios),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
