from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.causal_byte_models import causal_mask
from scripts.train_byte_core_from_config import (
    _collect_corpus_files,
    _iter_file_payload,
    _load_config_with_extends,
    _relative_path,
    _resolve_config_paths,
    _weighted_schedule,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BPETokenTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        layers: int,
        heads: int,
        max_len: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model,
            heads,
            d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.core = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.emb(x) + self.pos(positions)[None]
        h = self.core(h, mask=causal_mask(x.shape[1], x.device))
        return self.head(self.norm(h))


def _load_bytes(
    *,
    root: Path,
    data_roots: list[str],
    include_suffixes: set[str],
    max_bytes: int,
    read_block_bytes: int,
) -> bytes:
    paths = _resolve_config_paths(root, data_roots)
    files = _collect_corpus_files(paths, include_suffixes)
    if not files:
        raise RuntimeError("No corpus files found for configured data_roots/include_suffixes")
    return _load_bytes_from_files(
        files,
        max_bytes=max_bytes,
        read_block_bytes=read_block_bytes,
    )


def _load_bytes_from_files(
    files: list[Path],
    *,
    max_bytes: int,
    read_block_bytes: int,
) -> bytes:
    payload = bytearray()
    while len(payload) < max_bytes:
        made_progress = False
        for path in files:
            for block in _iter_file_payload(path, read_block_bytes=read_block_bytes):
                made_progress = True
                payload.extend(block)
                if len(payload) >= max_bytes:
                    return bytes(payload[:max_bytes])
        if not made_progress:
            break
    if len(payload) < 4096:
        raise RuntimeError("Not enough corpus bytes collected for BPE quickrun")
    return bytes(payload[:max_bytes])


def _iter_files_payload_forever(files: list[Path], *, read_block_bytes: int):
    """Yield fixed-size chunks so weighted schedules represent byte weights.

    JSONL readers naturally yield one (usually short) example at a time while
    plain-text readers yield ``read_block_bytes`` at a time.  Buffering both to
    the same chunk size prevents a nominal component weight from silently
    becoming a very different realized byte weight.
    """
    block_size = max(int(read_block_bytes), 1)
    buffer = bytearray()
    while True:
        made_progress = False
        for path in files:
            for block in _iter_file_payload(path, read_block_bytes=block_size):
                made_progress = True
                buffer.extend(block)
                while len(buffer) >= block_size:
                    yield bytes(buffer[:block_size])
                    del buffer[:block_size]
        if not made_progress:
            if buffer:
                yield bytes(buffer)
            return


def _load_mixed_bytes(
    *,
    root: Path,
    data_mix: list[dict[str, Any]],
    include_suffixes: set[str],
    max_bytes: int,
    read_block_bytes: int,
) -> tuple[bytes, dict[str, Any]]:
    if not data_mix:
        raise ValueError("data_mix must contain at least one component")
    components: list[dict[str, Any]] = []
    for index, component_cfg in enumerate(data_mix):
        if not isinstance(component_cfg, dict):
            raise TypeError("each training.data_mix component must be an object")
        name = str(component_cfg.get("name", f"component_{index}"))
        weight = float(component_cfg.get("weight", 1.0))
        component_include_suffixes = set(
            component_cfg.get("include_suffixes", include_suffixes)
        )
        roots = _resolve_config_paths(root, component_cfg.get("data_roots", []))
        files = _collect_corpus_files(roots, component_include_suffixes)
        if not files:
            raise RuntimeError(f"No corpus files found for data_mix component {name!r}")
        components.append(
            {
                "name": name,
                "weight": weight,
                "files": files,
                "row_preserve_jsonl_examples": bool(
                    component_cfg.get("row_preserve_jsonl_examples", False)
                ),
            }
        )

    schedule = _weighted_schedule([component["weight"] for component in components])
    iterators = [
        _iter_files_payload_forever(
            component["files"],
            read_block_bytes=read_block_bytes,
        )
        for component in components
    ]
    empty_components: set[int] = set()
    payload = bytearray()
    realized_component_bytes = [0 for _ in components]
    while len(payload) < max_bytes and len(empty_components) < len(components):
        made_progress = False
        for component_index in schedule:
            if component_index in empty_components:
                continue
            try:
                block = next(iterators[component_index])
            except StopIteration:
                empty_components.add(component_index)
                continue
            made_progress = True
            remaining = max_bytes - len(payload)
            framed_block = block + b"\n"
            accepted = framed_block[:remaining]
            payload.extend(accepted)
            realized_component_bytes[component_index] += len(accepted)
            if len(payload) >= max_bytes:
                break
        if not made_progress:
            break
    if len(payload) < 4096:
        raise RuntimeError("Not enough mixed corpus bytes collected for BPE quickrun")

    summary = {
        "mode": "weighted_mix",
        "schedule": schedule,
        "components": [
            {
                "name": component["name"],
                "weight": component["weight"],
                "row_preserve_jsonl_examples": component[
                    "row_preserve_jsonl_examples"
                ],
                "realized_bytes": realized_component_bytes[index],
                "realized_byte_share": (
                    realized_component_bytes[index] / max(len(payload), 1)
                ),
                "file_count": len(component["files"]),
                "files": [
                    _relative_path(root, path)
                    for path in component["files"][:8]
                ],
            }
            for index, component in enumerate(components)
        ],
    }
    return bytes(payload), summary


def _batch(tokens: torch.Tensor, seq: int, batch_size: int, generator: torch.Generator, device: torch.device):
    max_start = tokens.numel() - seq - 1
    if max_start <= 0:
        raise RuntimeError("token stream too short for configured sequence length")
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack([tokens[start : start + seq] for start in starts]).to(device)
    y = torch.stack([tokens[start + 1 : start + seq + 1] for start in starts]).to(device)
    return x, y


@torch.no_grad()
def _eval_bpb(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    eval_bytes: int,
    seq: int,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(991)
    losses = []
    for _ in range(batches):
        x, y = _batch(tokens, seq, batch_size, generator, device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
    nll_per_token = sum(losses) / max(len(losses), 1)
    bytes_per_token = float(eval_bytes) / max(float(tokens.numel()), 1.0)
    return nll_per_token / bytes_per_token / math.log(2)


def _current_lr(step: int, *, lr: float, min_lr: float, warmup_steps: int, lr_decay_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return lr * float(step + 1) / float(warmup_steps)
    if lr_decay_steps <= 0:
        return lr
    decay_pos = min(max(step - warmup_steps, 0), lr_decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_pos / max(lr_decay_steps, 1)))
    return min_lr + (lr - min_lr) * cosine


def train(config: dict[str, Any]) -> None:
    root = Path(__file__).resolve().parents[1]
    model_cfg = config["model"]
    train_cfg = config["training"]
    tok_cfg = config.get("tokenizer", {})

    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    seed = int(train_cfg.get("seed", 1234))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision(train_cfg.get("matmul_precision", "high"))

    out_dir = (root / Path(train_cfg.get("out_dir", "runs_experiment/bpe_transformer"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / train_cfg.get("metrics_path", "training_metrics.json")

    resume_checkpoint: dict[str, Any] | None = None
    resume_path: Path | None = None
    resume_from = train_cfg.get("resume_from")
    if resume_from:
        resume_path = Path(str(resume_from))
        if not resume_path.is_absolute():
            resume_path = (root / resume_path).resolve()
        resume_checkpoint = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        if "model" not in resume_checkpoint or "tokenizer_model" not in resume_checkpoint:
            raise ValueError(
                "BPE resume checkpoint must contain model and tokenizer_model"
            )

    include_suffixes = set(train_cfg.get("include_suffixes", [".jsonl", ".json", ".txt", ".md", ".csv"]))
    read_block_bytes = int(train_cfg.get("read_block_bytes", 1 << 20))
    data_mix_cfg = train_cfg.get("data_mix") or train_cfg.get("data_mixes") or []
    if data_mix_cfg:
        if not isinstance(data_mix_cfg, list):
            raise TypeError("training.data_mix must be a list of component objects")
        corpus_bytes, data_source_summary = _load_mixed_bytes(
            root=root,
            data_mix=data_mix_cfg,
            include_suffixes=include_suffixes,
            max_bytes=int(train_cfg.get("corpus_bytes", 8_000_000)),
            read_block_bytes=int(train_cfg.get("mix_block_bytes", read_block_bytes)),
        )
    else:
        corpus_bytes = _load_bytes(
            root=root,
            data_roots=train_cfg.get("data_roots", []),
            include_suffixes=include_suffixes,
            max_bytes=int(train_cfg.get("corpus_bytes", 8_000_000)),
            read_block_bytes=read_block_bytes,
        )
        data_source_summary = {
            "mode": "single_stream",
            "data_roots": train_cfg.get("data_roots", []),
        }
    eval_bytes = int(train_cfg.get("eval_bytes", min(200_000, max(len(corpus_bytes) // 10, 4096))))
    if train_cfg.get("eval_data_roots"):
        train_bytes = corpus_bytes
        heldout_bytes = _load_bytes(
            root=root,
            data_roots=train_cfg.get("eval_data_roots", []),
            include_suffixes=include_suffixes,
            max_bytes=eval_bytes,
            read_block_bytes=read_block_bytes,
        )
    else:
        train_bytes = corpus_bytes[:-eval_bytes]
        heldout_bytes = corpus_bytes[-eval_bytes:]

    spm_prefix = out_dir / f"spm_{tok_cfg.get('vocab_size', 4096)}"
    tokenize_started = time.perf_counter()
    if resume_checkpoint is not None:
        tokenizer_payload = resume_checkpoint["tokenizer_model"]
        if not isinstance(tokenizer_payload, (bytes, bytearray)):
            raise TypeError("checkpoint tokenizer_model must be serialized bytes")
        tokenizer_payload = bytes(tokenizer_payload)
        tokenizer = spm.SentencePieceProcessor()
        if not tokenizer.LoadFromSerializedProto(tokenizer_payload):
            raise RuntimeError("failed to load serialized tokenizer from checkpoint")
        Path(str(spm_prefix) + ".model").write_bytes(tokenizer_payload)
        tokenizer_reused = True
    else:
        corpus_txt = out_dir / "tokenizer_corpus.txt"
        corpus_txt.write_text(
            train_bytes.decode("utf-8", errors="replace"),
            encoding="utf-8",
        )
        spm.SentencePieceTrainer.train(
            input=str(corpus_txt),
            model_prefix=str(spm_prefix),
            vocab_size=int(tok_cfg.get("vocab_size", 4096)),
            model_type=tok_cfg.get("model_type", "bpe"),
            character_coverage=1.0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
            byte_fallback=bool(tok_cfg.get("byte_fallback", True)),
            minloglevel=2,
        )
        tokenizer_payload = Path(str(spm_prefix) + ".model").read_bytes()
        tokenizer = spm.SentencePieceProcessor(
            model_file=str(spm_prefix) + ".model"
        )
        tokenizer_reused = False
    train_tokens = torch.tensor(
        tokenizer.encode(train_bytes.decode("utf-8", errors="replace"), out_type=int),
        dtype=torch.long,
    )
    eval_tokens = torch.tensor(
        tokenizer.encode(heldout_bytes.decode("utf-8", errors="replace"), out_type=int),
        dtype=torch.long,
    )
    tokenizer_seconds = time.perf_counter() - tokenize_started

    seq = int(train_cfg.get("seq_len", 512))
    micro_batch_size = int(train_cfg.get("micro_batch_size", 1))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    steps = int(train_cfg.get("steps", 30))
    lr = float(train_cfg.get("lr", 1e-4))
    min_lr = float(train_cfg.get("min_lr", lr))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    lr_decay_steps = int(train_cfg.get("lr_decay_steps", 0))
    lr_step_offset = int(train_cfg.get("lr_step_offset", 0))
    log_interval = int(train_cfg.get("log_interval", 5))
    save_interval = int(train_cfg.get("save_interval", 0))
    save_optimizer = bool(train_cfg.get("save_optimizer", False))

    start_step = 0
    if resume_checkpoint is not None:
        start_step = int(
            train_cfg.get(
                "resume_step",
                resume_checkpoint.get(
                    "step",
                    resume_checkpoint.get("training_config", {}).get("steps", 0),
                ),
            )
        )
    if steps <= start_step:
        raise ValueError(
            f"training.steps ({steps}) must exceed resume step ({start_step})"
        )

    model_max_len = seq
    if resume_checkpoint is not None:
        position_weight = resume_checkpoint["model"].get("pos.weight")
        if position_weight is None:
            raise ValueError("resume checkpoint is missing model pos.weight")
        model_max_len = int(position_weight.shape[0])
        if seq > model_max_len:
            raise ValueError(
                f"seq_len {seq} exceeds resumed model max_len {model_max_len}"
            )
    model = BPETokenTransformerLM(
        vocab_size=tokenizer.vocab_size(),
        d_model=int(model_cfg.get("d_model", 2048)),
        layers=int(model_cfg.get("layers", 10)),
        heads=int(model_cfg.get("heads", 16)),
        max_len=model_max_len,
        ff_mult=int(model_cfg.get("ff_mult", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(
            resume_checkpoint["model"],
            strict=bool(train_cfg.get("resume_strict", True)),
        )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("BPE transformer params: %.3fM", trainable_params / 1e6)

    optimizer_kwargs: dict[str, Any] = {
        "lr": lr,
        "betas": tuple(train_cfg.get("betas", [0.9, 0.95])),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = bool(train_cfg.get("optimizer_fused", True))
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    optimizer_reused = False
    if (
        resume_checkpoint is not None
        and bool(train_cfg.get("resume_optimizer", False))
        and "optimizer" in resume_checkpoint
    ):
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        optimizer_reused = True
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(seed)

    bytes_per_token = len(train_bytes) / max(train_tokens.numel(), 1)
    bytes_per_step = micro_batch_size * grad_accum_steps * seq * bytes_per_token
    resume_cumulative_train_bytes = float(
        train_cfg.get(
            "resume_cumulative_train_bytes",
            0.0
            if resume_checkpoint is None
            else resume_checkpoint.get("cumulative_train_bytes", 0.0),
        )
    )
    history = []
    started = time.perf_counter()
    running_loss = 0.0
    running_count = 0
    for step in range(start_step + 1, steps + 1):
        phase_step = step - start_step
        current_lr = _current_lr(
            max(step - 1 - lr_step_offset, 0),
            lr=lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            lr_decay_steps=lr_decay_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = _batch(train_tokens, seq, micro_batch_size, generator, device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()) / grad_accum_steps
            scaler.scale(loss).backward()
            step_loss += float(loss.item())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        running_loss += step_loss
        running_count += 1
        if phase_step == 1 or step % log_interval == 0 or step == steps:
            elapsed = time.perf_counter() - started
            steps_per_second = phase_step / max(elapsed, 1e-9)
            gib_per_hour = bytes_per_step * steps_per_second * 3600.0 / (1024.0**3)
            phase_train_bytes = bytes_per_step * phase_step
            metrics = {
                "step": step,
                "steps": steps,
                "resume_step": start_step,
                "phase_step": phase_step,
                "phase_steps": steps - start_step,
                "loss": running_loss / max(running_count, 1),
                "bpb": (running_loss / max(running_count, 1)) / max(bytes_per_token, 1e-12) / math.log(2),
                "lr": current_lr,
                "elapsed_seconds": elapsed,
                "tokenizer_seconds": tokenizer_seconds,
                "elapsed_total_seconds": elapsed + tokenizer_seconds,
                "steps_per_second": steps_per_second,
                "bytes_per_step": bytes_per_step,
                "train_bytes": phase_train_bytes,
                "phase_train_bytes": phase_train_bytes,
                "resume_cumulative_train_bytes": resume_cumulative_train_bytes,
                "cumulative_train_bytes": (
                    resume_cumulative_train_bytes + phase_train_bytes
                ),
                "gib_per_hour": gib_per_hour,
                "projected_total_hours": (
                    (steps - start_step)
                    / max(steps_per_second, 1e-12)
                    / 3600.0
                ),
                "trainable_params": trainable_params,
            }
            history.append(metrics)
            output = {
                "status": "RUNNING",
                "config_name": config.get("name"),
                "device": str(device),
                "model_config": model_cfg,
                "training_config": train_cfg,
                "data_source_summary": data_source_summary,
                "tokenizer_config": tok_cfg,
                "resume": {
                    "path": None if resume_path is None else str(resume_path),
                    "step": start_step,
                    "tokenizer_reused": tokenizer_reused,
                    "optimizer_reused": optimizer_reused,
                },
                "latest": metrics,
                "history": history[-200:],
            }
            metrics_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
            logger.info("step=%d/%d bpb=%.4f steps_per_sec=%.4f GiB/h=%.3f", step, steps, metrics["bpb"], steps_per_second, gib_per_hour)
            running_loss = 0.0
            running_count = 0

        if save_interval > 0 and step % save_interval == 0 and step < steps:
            periodic_checkpoint: dict[str, Any] = {
                "format": "layercake-bpe-transformer-config/1",
                "model": model.state_dict(),
                "model_config": model_cfg,
                "training_config": train_cfg,
                "tokenizer_model": tokenizer_payload,
                "trainable_params": trainable_params,
                "step": step,
                "phase_train_bytes": bytes_per_step * phase_step,
                "cumulative_train_bytes": (
                    resume_cumulative_train_bytes + bytes_per_step * phase_step
                ),
            }
            if save_optimizer:
                periodic_checkpoint["optimizer"] = optimizer.state_dict()
            torch.save(periodic_checkpoint, out_dir / f"step_{step}.pt")

    eval_bpb = _eval_bpb(
        model,
        eval_tokens,
        eval_bytes=len(heldout_bytes),
        seq=seq,
        batch_size=micro_batch_size,
        batches=int(train_cfg.get("eval_batches", 8)),
        device=device,
    )
    artifact_path = out_dir / "latest.pt"
    checkpoint: dict[str, Any] = {
        "format": "layercake-bpe-transformer-config/1",
        "model": model.state_dict(),
        "model_config": model_cfg,
        "training_config": train_cfg,
        "tokenizer_model": tokenizer_payload,
        "trainable_params": trainable_params,
        "step": steps,
        "phase_train_bytes": bytes_per_step * (steps - start_step),
        "cumulative_train_bytes": (
            resume_cumulative_train_bytes
            + bytes_per_step * (steps - start_step)
        ),
    }
    if save_optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, artifact_path)
    final_output = {
        "status": "COMPLETE",
        "config_name": config.get("name"),
        "device": str(device),
        "model_config": model_cfg,
        "training_config": train_cfg,
        "data_source_summary": data_source_summary,
        "tokenizer_config": tok_cfg,
        "resume": {
            "path": None if resume_path is None else str(resume_path),
            "step": start_step,
            "tokenizer_reused": tokenizer_reused,
            "optimizer_reused": optimizer_reused,
        },
        "latest": {
            **history[-1],
            "eval_bpb": eval_bpb,
            "artifact": _relative_path(root, artifact_path),
        },
        "history": history[-200:],
    }
    metrics_path.write_text(json.dumps(final_output, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Training complete. Output: %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a config-driven BPE token transformer baseline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parents[1] / config_path).resolve()
    config = _load_config_with_extends(config_path)
    train(config)


if __name__ == "__main__":
    main()
