"""Bounded, fail-closed architecture search over real held-out bytes."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from layercake.models.baseline_transformer import (
    BytePairTokenizer,
    ModernBPETransformer,
    TransformerConfig,
)
from layercake.models.foundation import FoundationConfig, LayerCakeFoundation
from layercake.models.foundation_v2 import FoundationV2Config, LayerCakeFoundationV2
from .data import ByteCorpus, sha256_file
from .foundation import _config


REQUIRED_ABLATIONS = (
    "existing_layercake_foundation",
    "strong_byte_transformer",
    "strong_bpe_transformer",
    "dense_layercake",
    "sparse_layercake",
    "local_path_only",
    "global_path_only",
    "no_routed_experts",
    "oracle_route",
    "learned_route",
    "fixed_patches",
    "multiscale_patches",
    "incremental_vs_full_context",
    "selected_final_architecture",
)


def _v2_config(ablation: str) -> FoundationV2Config:
    return FoundationV2Config(
        d_byte=20, d_local=32, d_global=48, local_layers=1, local_kernel=5,
        fast_patch_size=4, slow_patch_size=16, global_layers=1,
        routed_experts=24, expert_expansion=3, abi_width=32, ablation=ablation,
    )


def _candidate_model(name: str, seed: int):
    torch.manual_seed(seed)
    if name == "existing_layercake_foundation":
        model = LayerCakeFoundation(FoundationConfig(
            d_byte=20, d_model=48, recurrent_layers=1, local_kernel=5,
            routed_experts=8, expert_expansion=3, abi_width=32,
        ))
        return model, "v1", seed % 8, None
    if name in {"strong_byte_transformer", "strong_bpe_transformer"}:
        merges = 0 if name == "strong_byte_transformer" else 64
        tokenizer = None
        model = ModernBPETransformer(TransformerConfig(
            vocab_size=256 + merges, width=64, layers=3, heads=4,
            max_tokens=512, expansion=3,
        ))
        return model, "transformer", None, tokenizer
    mapping = {
        "dense_layercake": "dense", "sparse_layercake": "sparse",
        "local_path_only": "local_only", "global_path_only": "global_only",
        "no_routed_experts": "no_routed_experts", "oracle_route": "oracle_route",
        "learned_route": "learned_route", "fixed_patches": "fixed_patches",
        "multiscale_patches": "multiscale_patches",
        "incremental_vs_full_context": "selected",
        "selected_final_architecture": "selected",
    }
    model = LayerCakeFoundationV2(_v2_config(mapping[name]))
    route = None if name == "learned_route" else seed % model.config.routed_experts
    return model, "v2", route, None


def _train_byte_model(model, kind: str, route, train: ByteCorpus, *, config: dict, seed: int, device):
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    curve = []
    started = time.perf_counter()
    for step, row in enumerate(train.batches(
        batch_size=int(config["batch_size"]), sequence_bytes=int(config["sequence_bytes"]),
        steps=int(config["steps"]), seed=seed, device=device,
    ), start=1):
        inputs, targets = row[:, :-1], row[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        if kind == "v2":
            logits = model(inputs, route=route)
        else:
            if route is not None:
                model.set_route(route)
            logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, int(config["steps"]) // 4) == 0:
            curve.append({"step": step, "loss": float(loss.detach())})
    return curve, time.perf_counter() - started


def _train_bpe(model, corpus: ByteCorpus, *, config: dict, seed: int, device):
    tokenizer_sample = bytes(np_value for np_value in corpus.data[:200_000])
    tokenizer = BytePairTokenizer.train(tokenizer_sample, merge_count=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    curve = []
    started = time.perf_counter()
    for step, row in enumerate(corpus.batches(
        batch_size=int(config["batch_size"]), sequence_bytes=int(config["sequence_bytes"]),
        steps=int(config["steps"]), seed=seed, device="cpu",
    ), start=1):
        token_rows = [tokenizer.encode(bytes(values.tolist())) for values in row]
        length = min(min(map(len, token_rows)), model.config.max_tokens)
        tokens = torch.tensor([values[:length] for values in token_rows], device=device)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, int(config["steps"]) // 4) == 0:
            curve.append({"step": step, "loss": float(loss.detach())})
    return tokenizer, curve, time.perf_counter() - started


@torch.inference_mode()
def _evaluate(model, kind, route, corpus, config, device, tokenizer=None):
    model.eval()
    losses = []
    raw_bytes = 0
    for row in corpus.fixed_batches(
        batch_size=int(config["evaluation_batch_size"]),
        sequence_bytes=int(config["sequence_bytes"]),
        batches=int(config["evaluation_batches"]), device="cpu" if tokenizer else device,
    ):
        if tokenizer is not None:
            token_rows = [tokenizer.encode(bytes(values.tolist())) for values in row]
            length = min(min(map(len, token_rows)), model.config.max_tokens)
            tokens = torch.tensor([values[:length] for values in token_rows], device=device)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            logits = model(inputs)
            loss_sum = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")
            covered = sum(len(tokenizer.decode(values[:length])) for values in token_rows)
            losses.append(float(loss_sum))
            raw_bytes += covered
        else:
            row = row.to(device)
            inputs, targets = row[:, :-1], row[:, 1:]
            if kind == "v2":
                logits = model(inputs, route=route)
            else:
                if kind == "v1" and route is not None:
                    model.set_route(route)
                logits = model(inputs)
            losses.append(float(F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="sum"
            )))
            raw_bytes += targets.numel()
    return sum(losses) / max(raw_bytes, 1) / 0.6931471805599453


def _latency(model, kind, route, device) -> float:
    length = 128
    maximum = model.config.max_tokens if kind == "transformer" else length
    inputs = torch.arange(min(length, maximum), device=device)[None] % (
        model.config.vocab_size if kind == "transformer" else 256
    )
    for _ in range(2):
        model(inputs) if kind != "v2" else model(inputs, route=route)
    samples = []
    for _ in range(5):
        started = time.perf_counter_ns()
        model(inputs) if kind != "v2" else model(inputs, route=route)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return statistics.median(samples)


def _pareto(rows: list[dict]) -> list[str]:
    frontier = []
    for row in rows:
        dominated = any(
            other is not row
            and other["mean_bpb"] <= row["mean_bpb"]
            and other["median_forward_ms"] <= row["median_forward_ms"]
            and other["active_parameters"] <= row["active_parameters"]
            and (
                other["mean_bpb"] < row["mean_bpb"]
                or other["median_forward_ms"] < row["median_forward_ms"]
                or other["active_parameters"] < row["active_parameters"]
            )
            for other in rows
        )
        if not dominated:
            frontier.append(row["candidate"])
    return frontier


def run_architecture_search(config_path: str | Path, output_path: str | Path) -> dict:
    config = _config(config_path)
    budget = config["budget"]
    candidates = list(config.get("candidates", REQUIRED_ABLATIONS))
    if tuple(candidates) != REQUIRED_ABLATIONS:
        raise ValueError("search must include every locked required ablation in order")
    if len(candidates) > int(budget["maximum_candidates"]):
        raise ValueError("candidate budget exceeded")
    train = ByteCorpus(config["data"]["train"])
    selection = ByteCorpus(config["data"]["architecture_selection"])
    device = torch.device("cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    raw_rows = []
    summary = []
    for candidate in candidates:
        candidate_rows = []
        for seed in config["seeds"]:
            if time.perf_counter() - started > float(budget["maximum_wall_seconds"]):
                raise RuntimeError("architecture search exhausted its locked wall-time budget")
            model, kind, route, _ = _candidate_model(candidate, int(seed))
            if candidate == "strong_bpe_transformer":
                tokenizer, curve, training_seconds = _train_bpe(
                    model.to(device), train, config=config["training"], seed=int(seed), device=device
                )
            else:
                tokenizer = None
                curve, training_seconds = _train_byte_model(
                    model, kind, route, train, config=config["training"], seed=int(seed), device=device
                )
            bpb = _evaluate(
                model, kind, route, selection, config["training"], device, tokenizer=tokenizer
            )
            latency = _latency(model, kind, route, device)
            total = sum(parameter.numel() for parameter in model.parameters())
            if kind == "v2":
                parameter_report = model.parameter_report(route or 0)
                if candidate == "dense_layercake":
                    active = total
                elif candidate == "no_routed_experts":
                    inactive = sum(
                        parameter.numel() for parameter in model.experts.parameters()
                    )
                    active = total - inactive
                else:
                    active = int(parameter_report["active_parameters"])
                active_fraction = active / total
            elif kind == "v1":
                parameter_report = model.parameter_report(route or 0)
                active = int(parameter_report["active_parameters_per_homogeneous_batch"])
                active_fraction = float(parameter_report["active_fraction"])
            else:
                active = total
                active_fraction = 1.0
            row = {
                "candidate": candidate, "seed": int(seed), "kind": kind,
                "architecture_selection_bpb": bpb, "training_seconds": training_seconds,
                "forward_milliseconds": latency, "total_parameters": total,
                "active_parameters": active, "active_fraction": active_fraction,
                "curve": curve, "test_accessed": False,
            }
            candidate_rows.append(row)
            raw_rows.append(row)
        summary.append({
            "candidate": candidate,
            "mean_bpb": statistics.fmean(row["architecture_selection_bpb"] for row in candidate_rows),
            "median_forward_ms": statistics.median(row["forward_milliseconds"] for row in candidate_rows),
            "total_parameters": int(statistics.median(row["total_parameters"] for row in candidate_rows)),
            "active_parameters": int(statistics.median(row["active_parameters"] for row in candidate_rows)),
            "active_fraction": statistics.fmean(row["active_fraction"] for row in candidate_rows),
        })
    byte_baseline = next(row for row in summary if row["candidate"] == "strong_byte_transformer")
    eligible = [
        row for row in summary
        if "transformer" not in row["candidate"]
        and row["active_fraction"] <= float(config["promotion"]["maximum_active_fraction"])
        and row["mean_bpb"] <= byte_baseline["mean_bpb"] * float(config["promotion"]["maximum_bpb_ratio"])
    ]
    selected = min(eligible, key=lambda row: (row["mean_bpb"], row["median_forward_ms"])) if eligible else None
    evidence = {
        "format": "layercake-architecture-search/2",
        "status": "PASS" if selected else "FAIL",
        "selected_candidate": selected,
        "selection_split_only": True,
        "final_test_accessed": False,
        "budget": budget,
        "promotion": config["promotion"],
        "data": {
            "train_sha256": sha256_file(config["data"]["train"]),
            "architecture_selection_sha256": sha256_file(config["data"]["architecture_selection"]),
        },
        "raw_runs": raw_rows,
        "summary": summary,
        "pareto_frontier": _pareto(summary),
        "failed_runs": [],
        "wall_seconds": time.perf_counter() - started,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence
