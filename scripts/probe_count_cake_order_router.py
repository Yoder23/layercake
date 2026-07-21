"""Fit a causal router over frozen CountCake order-stage probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402
from scripts.optimize_count_cake_backoff_fast import _count_statistics  # noqa: E402
from scripts.probe_top_continuation_cake import (  # noqa: E402
    _context_query,
    _top_tables,
)


class CausalOrderRouter(nn.Module):
    def __init__(
        self,
        max_order: int,
        hidden_width: int,
        byte_embedding_width: int,
        candidate_embedding_width: int,
        semantic_width: int,
        extra_experts: int = 0,
    ) -> None:
        super().__init__()
        self.max_order = int(max_order)
        self.byte_embedding_width = int(byte_embedding_width)
        self.candidate_embedding_width = int(candidate_embedding_width)
        self.semantic_width = int(semantic_width)
        self.byte_embedding = nn.Embedding(256, self.byte_embedding_width)
        self.candidate_embedding = nn.Embedding(
            257, self.candidate_embedding_width
        )
        self.network = nn.Sequential(
            nn.Linear(
                self.max_order
                * (
                    3
                    + self.byte_embedding_width
                    + self.candidate_embedding_width
                )
                + self.semantic_width,
                hidden_width,
            ),
            nn.SiLU(),
            nn.Linear(hidden_width, self.max_order + 1 + int(extra_experts)),
        )

    def forward(
        self,
        features: torch.Tensor,
        byte_context: torch.Tensor,
        candidate_context: torch.Tensor,
        candidate_confidence: torch.Tensor,
        semantic_features: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.byte_embedding(byte_context).flatten(-2)
        candidate_embedded = self.candidate_embedding(
            candidate_context
        ).flatten(-2)
        return self.network(
            torch.cat(
                [
                    features,
                    candidate_confidence,
                    embedded,
                    candidate_embedded,
                    semantic_features,
                ],
                dim=-1,
            )
        )


def _prepare_split(
    model,
    path: Path,
    seq_len: int,
    semantic_projection: torch.Tensor,
    top_tables: list[tuple[torch.Tensor, torch.Tensor]],
    max_bytes: int | None = None,
    include_neural_expert: bool = False,
) -> dict:
    payload = path.read_bytes()
    if max_bytes is not None and len(payload) > max_bytes:
        segment_count = min(16, max(1, max_bytes // seq_len))
        segment_bytes = max_bytes // segment_count
        maximum_start = len(payload) - segment_bytes
        starts = np.linspace(
            0, maximum_start, num=segment_count, dtype=np.int64
        )
        payload = b"".join(
            payload[int(start) : int(start) + segment_bytes]
            for start in starts
        )
    row_count = len(payload) // seq_len
    rows = torch.frombuffer(
        bytearray(payload[: row_count * seq_len]), dtype=torch.uint8
    ).reshape(row_count, seq_len).to(device="cuda", dtype=torch.long)
    with torch.inference_mode():
        statistics = _count_statistics(
            model.count_cake, rows, model.prediction_start
        )
    probability = statistics["unigram"]
    stages = [probability.astype(np.float32)]
    feature_parts = []
    candidates = []
    candidate_confidence = []
    for count, total, strength in zip(
        statistics["counts"],
        statistics["totals"],
        model.count_cake.backoff_strengths,
    ):
        probability = (count + strength * probability) / (total + strength)
        stages.append(probability.astype(np.float32))
        feature_parts.append(
            np.stack(
                [
                    np.log1p(total).astype(np.float32) / 16.0,
                    (total > 0).astype(np.float32),
                ],
                axis=-1,
            )
        )
    targets = rows[:, model.prediction_start :]
    for order, (top_target, top_count) in enumerate(top_tables, start=1):
        query = _context_query(
            model.count_cake,
            rows,
            model.prediction_start,
            order,
        )
        context_keys = getattr(model.count_cake, f"context_keys_{order}")
        positions = torch.searchsorted(context_keys, query)
        safe = positions.clamp(max=context_keys.numel() - 1)
        found = (positions < context_keys.numel()) & (
            context_keys[safe] == query
        )
        total = getattr(
            model.count_cake, f"context_totals_{order}"
        )[safe]
        candidates.append(
            torch.where(
                found,
                top_target[safe],
                torch.full_like(query, 256),
            )
            .to(torch.int16)
            .cpu()
            .numpy()
        )
        candidate_confidence.append(
            torch.where(
                found,
                top_count[safe] / total.clamp_min(1.0),
                torch.zeros_like(total),
            )
            .to(torch.float16)
            .cpu()
            .numpy()
        )
    stage_probability = np.stack(stages, axis=-1).reshape(
        -1, model.count_cake.max_order + 1
    )
    features = np.concatenate(feature_parts, axis=-1).reshape(
        -1, model.count_cake.max_order * 2
    )
    byte_context = torch.stack(
        [
            rows[
                :,
                model.prediction_start - model.count_cake.max_order + offset :
                rows.shape[1] - model.count_cake.max_order + offset,
            ]
            for offset in range(model.count_cake.max_order)
        ],
        dim=-1,
    ).reshape(-1, model.count_cake.max_order).to(torch.uint8).cpu().numpy()
    semantic_chunks = []
    neural_probability_chunks = []
    with torch.inference_mode():
        for start in range(0, row_count, 128):
            batch = rows[start : start + 128]
            if model.chunking_mode == "delimiter":
                neural_log_probability, neural_hidden = (
                    model._dynamic_neural_log_probs(batch)
                )
            else:
                context = model._patch_context(batch)
                targets = batch[:, model.prediction_start :].reshape(
                    batch.shape[0], -1, model.patch_size
                )
                neural_log_probability, neural_hidden = model._neural_log_probs(
                    context, targets, rows=batch
                )
            if include_neural_expert:
                neural_probability_chunks.append(
                    neural_log_probability.reshape(-1).exp().cpu().numpy()
                )
            semantic_chunks.append(
                (neural_hidden @ semantic_projection)
                .reshape(-1, semantic_projection.shape[1])
                .to(torch.float16)
                .cpu()
                .numpy()
            )
    semantic_features = np.concatenate(semantic_chunks, axis=0)
    if include_neural_expert:
        stage_probability = np.concatenate(
            [
                stage_probability,
                np.concatenate(neural_probability_chunks, axis=0).reshape(-1, 1),
            ],
            axis=-1,
        )
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": row_count,
        "predicted_per_row": seq_len - model.prediction_start,
        "features": features,
        "byte_context": byte_context,
        "candidate_context": np.stack(candidates, axis=-1).reshape(
            -1, model.count_cake.max_order
        ),
        "candidate_confidence": np.stack(
            candidate_confidence, axis=-1
        ).reshape(-1, model.count_cake.max_order),
        "semantic_features": semantic_features,
        "stage_probability": stage_probability,
    }


@torch.inference_mode()
def _evaluate(
    router: CausalOrderRouter,
    features: np.ndarray,
    byte_context: np.ndarray,
    candidate_context: np.ndarray,
    candidate_confidence: np.ndarray,
    semantic_features: np.ndarray,
    probability: np.ndarray,
    *,
    batch_size: int,
) -> float:
    total_nll = 0.0
    total = 0
    for start in range(0, features.shape[0], batch_size):
        feature_batch = torch.from_numpy(features[start : start + batch_size]).to(
            device="cuda"
        )
        probability_batch = torch.from_numpy(
            probability[start : start + batch_size]
        ).to(device="cuda")
        context_batch = torch.from_numpy(
            byte_context[start : start + batch_size]
        ).to(device="cuda", dtype=torch.long)
        semantic_batch = torch.from_numpy(
            semantic_features[start : start + batch_size]
        ).to(device="cuda", dtype=torch.float32)
        candidate_batch = torch.from_numpy(
            candidate_context[start : start + batch_size]
        ).to(device="cuda", dtype=torch.long)
        confidence_batch = torch.from_numpy(
            candidate_confidence[start : start + batch_size]
        ).to(device="cuda", dtype=torch.float32)
        log_weight = F.log_softmax(
            router(
                feature_batch,
                context_batch,
                candidate_batch,
                confidence_batch,
                semantic_batch,
            ),
            dim=-1,
        )
        log_probability = probability_batch.clamp_min(1e-30).log()
        nll = -torch.logsumexp(log_weight + log_probability, dim=-1)
        total_nll += float(nll.sum())
        total += nll.numel()
    return total_nll / total / math.log(2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument(
        "--fit-data",
        action="append",
        help="training-only split(s); when set, --data is never used for fitting",
    )
    parser.add_argument("--max-fit-bytes", type=int, default=4_000_000)
    parser.add_argument("--include-neural-expert", action="store_true")
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--fit-rows-per-split", type=int, default=250)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--byte-embedding-width", type=int, default=8)
    parser.add_argument("--candidate-embedding-width", type=int, default=8)
    parser.add_argument("--semantic-width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=24001)
    parser.add_argument("--output", required=True)
    parser.add_argument("--router", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the order-router probe requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    model, manifest = load_count_cake_bundle(args.bundle, device="cuda")
    model.eval()
    projection_generator = torch.Generator(device="cuda").manual_seed(
        args.seed + 101
    )
    semantic_projection = torch.randn(
        model.mixture_gate.in_features,
        args.semantic_width,
        generator=projection_generator,
        device="cuda",
    ) / math.sqrt(max(args.semantic_width, 1))
    top_tables = [
        _top_tables(model.count_cake, order)
        for order in range(1, model.count_cake.max_order + 1)
    ]
    splits = [
        _prepare_split(
            model,
            Path(path),
            args.seq_len,
            semantic_projection,
            top_tables,
            include_neural_expert=args.include_neural_expert,
        )
        for path in args.data
    ]
    fit_splits = (
        [
            _prepare_split(
                model,
                Path(path),
                args.seq_len,
                semantic_projection,
                top_tables,
                max_bytes=args.max_fit_bytes,
                include_neural_expert=args.include_neural_expert,
            )
            for path in args.fit_data
        ]
        if args.fit_data
        else splits
    )
    fit_features = []
    fit_contexts = []
    fit_semantic = []
    fit_candidates = []
    fit_candidate_confidence = []
    fit_probabilities = []
    split_boundaries = []
    for split in fit_splits:
        fit_count = min(
            split["features"].shape[0],
            args.fit_rows_per_split * split["predicted_per_row"],
        )
        if fit_count == split["features"].shape[0]:
            raise ValueError("fit rows must leave a held-out suffix")
        split["fit_count"] = fit_count
        fit_features.append(split["features"][:fit_count])
        fit_contexts.append(split["byte_context"][:fit_count])
        fit_semantic.append(split["semantic_features"][:fit_count])
        fit_candidates.append(split["candidate_context"][:fit_count])
        fit_candidate_confidence.append(
            split["candidate_confidence"][:fit_count]
        )
        fit_probabilities.append(split["stage_probability"][:fit_count])
        split_boundaries.append(fit_count)
    if args.fit_data:
        for split in splits:
            split["fit_count"] = 0
    fit_features_np = np.concatenate(fit_features, axis=0)
    fit_context_np = np.concatenate(fit_contexts, axis=0)
    fit_semantic_np = np.concatenate(fit_semantic, axis=0)
    fit_candidate_np = np.concatenate(fit_candidates, axis=0)
    fit_candidate_confidence_np = np.concatenate(
        fit_candidate_confidence, axis=0
    )
    fit_probability_np = np.concatenate(fit_probabilities, axis=0)
    router = CausalOrderRouter(
        model.count_cake.max_order,
        args.hidden_width,
        args.byte_embedding_width,
        args.candidate_embedding_width,
        args.semantic_width,
        extra_experts=int(args.include_neural_expert),
    ).to(device="cuda")
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.lr, weight_decay=0.001
    )
    generator = torch.Generator().manual_seed(args.seed + 17)
    train_started = time.perf_counter()
    history = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            fit_features_np.shape[0],
            (min(args.batch_size, fit_features_np.shape[0]),),
            generator=generator,
        ).numpy()
        features = torch.from_numpy(fit_features_np[indices]).to(device="cuda")
        byte_context = torch.from_numpy(fit_context_np[indices]).to(
            device="cuda", dtype=torch.long
        )
        semantic_features = torch.from_numpy(fit_semantic_np[indices]).to(
            device="cuda", dtype=torch.float32
        )
        candidate_context = torch.from_numpy(
            fit_candidate_np[indices]
        ).to(device="cuda", dtype=torch.long)
        candidate_confidence = torch.from_numpy(
            fit_candidate_confidence_np[indices]
        ).to(device="cuda", dtype=torch.float32)
        probability = torch.from_numpy(fit_probability_np[indices]).to(
            device="cuda"
        )
        optimizer.zero_grad(set_to_none=True)
        log_weight = F.log_softmax(
            router(
                features,
                byte_context,
                candidate_context,
                candidate_confidence,
                semantic_features,
            ),
            dim=-1,
        )
        loss = -torch.logsumexp(
            log_weight + probability.clamp_min(1e-30).log(), dim=-1
        ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            event = {
                "step": step,
                "loss_bpb": float(loss.detach() / math.log(2.0)),
                "elapsed_seconds": time.perf_counter() - train_started,
            }
            history.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
    torch.cuda.synchronize()
    fit_bpb = _evaluate(
        router,
        fit_features_np,
        fit_context_np,
        fit_candidate_np,
        fit_candidate_confidence_np,
        fit_semantic_np,
        fit_probability_np,
        batch_size=args.batch_size,
    )
    split_reports = []
    for split in splits:
        boundary = split["fit_count"]
        heldout_probability = split["stage_probability"][boundary:]
        heldout_features = split["features"][boundary:]
        heldout_context = split["byte_context"][boundary:]
        heldout_semantic = split["semantic_features"][boundary:]
        heldout_candidates = split["candidate_context"][boundary:]
        heldout_candidate_confidence = split[
            "candidate_confidence"
        ][boundary:]
        final_stage_bpb = float(
            -np.log(heldout_probability[:, -1]).mean() / math.log(2.0)
        )
        oracle_bpb = float(
            -np.log(heldout_probability.max(axis=-1)).mean() / math.log(2.0)
        )
        split_reports.append(
            {
                "path": split["path"],
                "bytes": split["bytes"],
                "sha256": split["sha256"],
                "fit_scored_bytes": boundary,
                "heldout_scored_bytes": int(heldout_features.shape[0]),
                "final_stage_bpb": final_stage_bpb,
                "target_aware_oracle_bpb": oracle_bpb,
                "router_bpb": _evaluate(
                    router,
                    heldout_features,
                    heldout_context,
                    heldout_candidates,
                    heldout_candidate_confidence,
                    heldout_semantic,
                    heldout_probability,
                    batch_size=args.batch_size,
                ),
            }
        )
    arrays = {
        name: parameter.detach().cpu().numpy()
        for name, parameter in router.state_dict().items()
    }
    router_path = Path(args.router)
    router_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(router_path, **arrays)
    report = {
        "format": "layercake-count-cake-order-router-probe/1",
        "status": "COMPLETE",
        "warning": "architecture-selection result; not a release certificate",
        "fit_protocol": {
            "fit_data": args.fit_data or args.data,
            "evaluation_data_used_for_fit": not bool(args.fit_data),
            "max_fit_bytes": args.max_fit_bytes if args.fit_data else None,
        },
        "bundle": {
            "path": args.bundle,
            "parameters": manifest["parameters"],
            "max_order": model.count_cake.max_order,
        },
        "router": {
            "path": str(router_path),
            "parameters": sum(p.numel() for p in router.parameters()),
            "hidden_width": args.hidden_width,
            "byte_embedding_width": args.byte_embedding_width,
            "candidate_embedding_width": args.candidate_embedding_width,
            "semantic_width": args.semantic_width,
            "input_features": model.count_cake.max_order
            * (
                3
                + args.byte_embedding_width
                + args.candidate_embedding_width
            )
            + args.semantic_width,
            "include_neural_expert": args.include_neural_expert,
        },
        "training": {
            "fit_bpb": fit_bpb,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "history": history,
            "seconds": time.perf_counter() - train_started,
        },
        "heldout_suffixes": split_reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
