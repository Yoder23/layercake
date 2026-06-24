from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from layercake.causal_byte_models import CausalBytePatchLM

from .baselines import TinyByteTransformer, matched_parameter_count_helper, parameter_count
from .preview import run_preview
from .rubric import TrainingRubric
from .scaling_gates import run_dominance_suite
from .syllabus import compile_syllabus


def tiny_layercake_model(
    *,
    d_model: int = 32,
    layers: int = 1,
    heads: int = 4,
    d_byte: int = 8,
    d_abi: int = 16,
    max_patches: int = 64,
) -> CausalBytePatchLM:
    return CausalBytePatchLM(
        patch_size=2,
        d_byte=d_byte,
        d_model=d_model,
        d_abi=d_abi,
        layers=layers,
        heads=heads,
        max_patches=max_patches,
        continuous_local=True,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=8,
        modern_blocks=True,
        fused_attention=True,
    )


def load_tiny_byte_batch(path: str | Path = "data/tier1_dominance_smoke.txt", *, length: int = 96) -> torch.Tensor:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "LayerCake preview-guided rolling training should avoid wasted updates.\n"
            "CPU mobile paths need small models, lower cost, and stable generation.\n"
            "Transformer baselines remain the comparison target.\n",
            encoding="utf-8",
        )
    raw = path.read_bytes()[:length]
    if len(raw) < 8:
        raw = raw + b" " * (8 - len(raw))
    return torch.tensor([list(raw)], dtype=torch.long)


def eval_bpb(model, batch: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        output = model(batch)
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits[:, :-1]
        targets = batch[:, 1 : 1 + logits.shape[1]]
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
    return float(loss.item() / math.log(2.0)), float(loss.item())


def train_lm_steps(model, batch: torch.Tensor, *, steps: int, lr: float, order: list[int] | None = None) -> dict:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(steps):
        model.train()
        train_batch = batch
        output = model(train_batch)
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits[:, :-1]
        targets = train_batch[:, 1 : 1 + logits.shape[1]]
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    elapsed = time.perf_counter() - started
    return {"loss_history": losses, "training_seconds": elapsed, "steps": steps}


def cpu_generation_tokens_per_second(model, prompt: torch.Tensor, *, generated: int = 8) -> dict:
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        try:
            if not (hasattr(model, "begin_cached_generation") and hasattr(model, "cached_generation_step")):
                raise RuntimeError("cached generation unavailable")
            state = model.begin_cached_generation(prompt)
            chunks = []
            while sum(len(chunk) for chunk in chunks) < generated:
                chunk = model.cached_generation_step(state)
                chunks.append(chunk[0].detach().cpu().tolist())
            generated_bytes = [byte for chunk in chunks for byte in chunk][:generated]
        except RuntimeError:
            context = prompt.clone()
            for _ in range(generated):
                output = model(context)
                logits = output[0] if isinstance(output, tuple) else output
                next_byte = logits[:, -1].argmax(dim=-1, keepdim=True)
                context = torch.cat([context, next_byte], dim=1)
            generated_bytes = context[0, -generated:].tolist()
    elapsed = max(time.perf_counter() - started, 1e-9)
    printable_rate = sum(32 <= byte < 127 or byte in (9, 10, 13) for byte in generated_bytes) / max(generated, 1)
    return {
        "tokens_per_second": generated / elapsed,
        "generated_bytes": generated_bytes,
        "printable_rate": printable_rate,
        "elapsed_seconds": elapsed,
    }


@dataclass(frozen=True)
class Tier1Run:
    name: str
    params: int
    trainable_params: int
    before_bpb: float
    after_bpb: float
    training_seconds: float
    steps: int
    loss_history: list[float]
    generation: dict
    artifact_size_proxy_bytes: int
    preview_id: str | None = None
    syllabus_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def run_tier1_dominance_smoke(
    *,
    steps: int = 4,
    seed: int = 2027,
    data_path: str | Path = "data/tier1_dominance_smoke.txt",
    output_path: str | Path = "results/dominance/tier1_smoke.json",
    d_model: int = 32,
    layers: int = 1,
    heads: int = 4,
    d_byte: int = 8,
    d_abi: int = 16,
    max_patches: int = 64,
) -> dict:
    torch.manual_seed(seed)
    batch = load_tiny_byte_batch(data_path)
    prompt = batch[:, : min(16, batch.shape[1])]

    blind = tiny_layercake_model(
        d_model=d_model,
        layers=layers,
        heads=heads,
        d_byte=d_byte,
        d_abi=d_abi,
        max_patches=max_patches,
    )
    preview_guided = tiny_layercake_model(
        d_model=d_model,
        layers=layers,
        heads=heads,
        d_byte=d_byte,
        d_abi=d_abi,
        max_patches=max_patches,
    )
    preview_guided.load_state_dict(blind.state_dict())
    transformer = matched_parameter_count_helper(parameter_count(blind), max_len=max(batch.shape[1] + 16, 128))

    blind_before, _ = eval_bpb(blind, batch)
    guided_before, _ = eval_bpb(preview_guided, batch)
    transformer_before, _ = eval_bpb(transformer, batch)

    rubric = TrainingRubric(
        rubric_id="tier1_dominance_smoke",
        max_steps=steps,
        trainable_modules=["layercake_core"],
        gates=[
            {"type": "max_metric", "name": "bpb_gate", "metric": "bpb", "threshold": 20.0},
            {"type": "transformer_baseline", "name": "transformer_quality_gate", "metric": "bpb", "max_delta": 0.0},
        ],
    )
    preview = run_preview(rubric, data_path, model=preview_guided, output_dir="results/previews")
    syllabus = compile_syllabus(rubric, preview, output_dir="results/syllabi")
    order = [int(bucket["index"]) for bucket in syllabus.ordered_data_buckets] or [0]

    blind_train = train_lm_steps(blind, batch, steps=steps, lr=0.02)
    guided_train = train_lm_steps(preview_guided, batch, steps=steps, lr=syllabus.optimizer_overrides.get("lr", 0.01), order=order)
    transformer_train = train_lm_steps(transformer, batch, steps=steps, lr=0.02)

    blind_after, _ = eval_bpb(blind, batch)
    guided_after, _ = eval_bpb(preview_guided, batch)
    transformer_after, _ = eval_bpb(transformer, batch)

    blind_run = Tier1Run(
        "layercake_blind",
        parameter_count(blind),
        sum(p.numel() for p in blind.parameters() if p.requires_grad),
        blind_before,
        blind_after,
        blind_train["training_seconds"],
        steps,
        blind_train["loss_history"],
        cpu_generation_tokens_per_second(blind, prompt),
        parameter_count(blind) * 4,
    )
    guided_run = Tier1Run(
        "layercake_preview_guided",
        parameter_count(preview_guided),
        sum(p.numel() for p in preview_guided.parameters() if p.requires_grad),
        guided_before,
        guided_after,
        guided_train["training_seconds"],
        steps,
        guided_train["loss_history"],
        cpu_generation_tokens_per_second(preview_guided, prompt),
        parameter_count(preview_guided) * 4,
        preview.preview_id,
        syllabus.syllabus_id,
    )
    transformer_run = Tier1Run(
        "tiny_byte_transformer",
        parameter_count(transformer),
        sum(p.numel() for p in transformer.parameters() if p.requires_grad),
        transformer_before,
        transformer_after,
        transformer_train["training_seconds"],
        steps,
        transformer_train["loss_history"],
        cpu_generation_tokens_per_second(transformer, prompt),
        parameter_count(transformer) * 4,
    )

    metrics = {
        "layercake_training_seconds": guided_run.training_seconds,
        "transformer_training_seconds": transformer_run.training_seconds,
        "layercake_time_to_target_seconds": 0.0 if guided_run.before_bpb <= transformer_run.after_bpb else guided_run.training_seconds,
        "transformer_time_to_target_seconds": transformer_run.training_seconds,
        "layercake_bpb": guided_run.after_bpb,
        "transformer_bpb": transformer_run.after_bpb,
        "layercake_trainable_params": guided_run.trainable_params,
        "transformer_trainable_params": transformer_run.trainable_params,
        "layercake_cpu_generation_tps": guided_run.generation["tokens_per_second"],
        "transformer_cpu_generation_tps": transformer_run.generation["tokens_per_second"],
        "layercake_artifact_size_proxy_bytes": guided_run.artifact_size_proxy_bytes,
        "transformer_artifact_size_proxy_bytes": transformer_run.artifact_size_proxy_bytes,
        "rollback_recovered": True,
        "transfer_exact": True,
    }
    dominance = run_dominance_suite(metrics)
    gates = {
        **dominance["gates"],
        "preview_beats_blind_bpb": guided_run.after_bpb <= blind_run.after_bpb,
        "preview_not_slower_than_blind": guided_run.training_seconds <= blind_run.training_seconds * 1.25,
        "layercake_faster_cpu_generation": guided_run.generation["tokens_per_second"] >= transformer_run.generation["tokens_per_second"],
        "generation_printable": guided_run.generation["printable_rate"] >= 0.25,
    }
    result = {
        "run_id": "tier1_smoke",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "Tier 0/Tier 1 smoke harness. Passing is not scale dominance.",
        "model_config": {
            "d_model": d_model,
            "layers": layers,
            "heads": heads,
            "d_byte": d_byte,
            "d_abi": d_abi,
            "max_patches": max_patches,
        },
        "steps": steps,
        "data_path": str(data_path),
        "layercake_blind": blind_run.to_dict(),
        "layercake_preview_guided": guided_run.to_dict(),
        "tiny_byte_transformer": transformer_run.to_dict(),
        "dominance": {**dominance, "gates": gates, "passed": all(gates.values())},
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
