"""Screen a normalized hashed bit-tree cake over raw bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def _parse_orders(value: str) -> tuple[int, ...]:
    orders = tuple(sorted({int(item) for item in value.split(",")}))
    if not orders or orders[0] <= 0:
        raise ValueError("orders must be positive")
    return orders


def _context_hash(
    data: np.ndarray,
    starts: np.ndarray,
    order: int,
    mask: int,
) -> np.ndarray:
    context = np.full(starts.shape, order * 0x9E3779B1, dtype=np.uint64)
    for offset in range(order):
        context *= np.uint64(257)
        context += data[starts - order + offset].astype(np.uint64) + 1
    return np.bitwise_and(context, np.uint64(mask)).astype(np.int64)


def _node_ids(
    context: np.ndarray,
    targets: np.ndarray,
    depth: int,
    mask: int,
) -> np.ndarray:
    prefix = (
        np.zeros_like(targets, dtype=np.int64)
        if depth == 0
        else targets.astype(np.int64) >> (8 - depth)
    )
    marker = (1 << depth) + prefix
    mixed = (
        context.astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93)
        + marker.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)
        + np.uint64(depth * 0x85EBCA6B)
    )
    return np.bitwise_and(mixed, np.uint64(mask)).astype(np.int64)


def _train_tables(
    data: np.ndarray,
    *,
    orders: tuple[int, ...],
    buckets: int,
    chunk_bytes: int,
) -> tuple[list[np.ndarray], int]:
    mask = buckets - 1
    maximum_order = max(orders)
    tables = [np.zeros((8, buckets, 2), dtype=np.uint32) for _ in orders]
    trained = 0
    for begin in range(maximum_order, data.size, chunk_bytes):
        end = min(begin + chunk_bytes, data.size)
        starts = np.arange(begin, end, dtype=np.int64)
        targets = data[starts]
        for table, order in zip(tables, orders):
            context = _context_hash(data, starts, order, mask)
            for depth in range(8):
                nodes = _node_ids(context, targets, depth, mask)
                bits = (targets >> (7 - depth)) & 1
                combined = nodes * 2 + bits.astype(np.int64)
                counts = np.bincount(combined, minlength=buckets * 2)
                table[depth] += counts.reshape(buckets, 2).astype(np.uint32)
        trained += end - begin
        print(json.dumps({"trained_bytes": trained}), flush=True)
    return tables, trained


def _evaluate(
    data: np.ndarray,
    *,
    orders: tuple[int, ...],
    tables: list[np.ndarray],
    strengths: tuple[float, ...],
    chunk_bytes: int,
) -> dict:
    buckets = tables[0].shape[1]
    mask = buckets - 1
    maximum_order = max(orders)
    nll = np.zeros(len(strengths), dtype=np.float64)
    scored = 0
    for begin in range(maximum_order, data.size, chunk_bytes):
        end = min(begin + chunk_bytes, data.size)
        starts = np.arange(begin, end, dtype=np.int64)
        targets = data[starts]
        contexts = [
            _context_hash(data, starts, order, mask) for order in orders
        ]
        for depth in range(8):
            bits = ((targets >> (7 - depth)) & 1).astype(np.int64)
            count_pairs = []
            for table, context in zip(tables, contexts):
                nodes = _node_ids(context, targets, depth, mask)
                count_pairs.append(table[depth, nodes].astype(np.float64))
            for strength_index, strength in enumerate(strengths):
                probability = np.full(targets.shape, 0.5, dtype=np.float64)
                for pair in count_pairs:
                    total = pair[:, 0] + pair[:, 1]
                    probability = (
                        pair[:, 1] + strength * probability
                    ) / (total + strength)
                observed = np.where(bits == 1, probability, 1.0 - probability)
                nll[strength_index] -= np.log2(np.clip(observed, 1e-30, None)).sum()
        scored += targets.size
    return {
        "scored_bytes": scored,
        "by_strength": {
            str(strength): float(value / scored)
            for strength, value in zip(strengths, nll)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--orders", default="1,2,3,4,6,8")
    parser.add_argument("--buckets", type=int, default=262144)
    parser.add_argument("--max-train-bytes", type=int, default=0)
    parser.add_argument("--chunk-bytes", type=int, default=2_000_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.buckets <= 0 or args.buckets & (args.buckets - 1):
        raise ValueError("buckets must be a power of two")
    orders = _parse_orders(args.orders)
    started = time.perf_counter()
    train_payload = Path(args.train).read_bytes()
    if args.max_train_bytes > 0:
        train_payload = train_payload[: args.max_train_bytes]
    eval_payload = Path(args.eval).read_bytes()
    train = np.frombuffer(train_payload, dtype=np.uint8)
    evaluation = np.frombuffer(eval_payload, dtype=np.uint8)
    train_started = time.perf_counter()
    tables, trained = _train_tables(
        train,
        orders=orders,
        buckets=args.buckets,
        chunk_bytes=args.chunk_bytes,
    )
    training_seconds = time.perf_counter() - train_started
    strengths = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
    quality = _evaluate(
        evaluation,
        orders=orders,
        tables=tables,
        strengths=strengths,
        chunk_bytes=args.chunk_bytes,
    )
    best_strength, best_bpb = min(
        quality["by_strength"].items(), key=lambda item: item[1]
    )
    report = {
        "format": "layercake-hashed-bit-cake-probe/1",
        "status": "COMPLETE",
        "warning": "validation-selected architecture probe; not release evidence",
        "architecture": {
            "input_output": "raw bytes",
            "tokenizer": None,
            "orders": list(orders),
            "buckets_per_order": args.buckets,
            "logical_count_parameters": len(orders) * 8 * args.buckets * 2,
            "normalized_bit_tree": True,
            "generation_valid": True,
        },
        "train": {
            "path": args.train,
            "bytes": len(train_payload),
            "sha256": hashlib.sha256(train_payload).hexdigest(),
            "scored_bytes": trained,
            "seconds": training_seconds,
        },
        "eval": {
            "path": args.eval,
            "bytes": len(eval_payload),
            "sha256": hashlib.sha256(eval_payload).hexdigest(),
            **quality,
            "best_strength": float(best_strength),
            "best_bpb": best_bpb,
        },
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
