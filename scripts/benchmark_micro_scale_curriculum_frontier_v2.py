from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from layercake.causal_byte_models import CausalAdaptiveBytePatchLM, CausalBytePatchLM


def _iter_jsonl_text(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    text = payload.get("text") or payload.get("content") or ""
                else:
                    text = str(payload)
            except json.JSONDecodeError:
                text = ""
            if text:
                yield text


def load_curriculum_bytes(redpajama_jsonl: Path, curriculum_files: list[Path], total_bytes: int) -> torch.Tensor:
    data = bytearray()

    for text in _iter_jsonl_text(redpajama_jsonl):
        data.extend(text.encode("utf-8", errors="replace"))
        data.extend(b"\n")
        if len(data) >= int(total_bytes * 0.9):
            break

    curriculum_blob = bytearray()
    for path in curriculum_files:
        if path.exists():
            curriculum_blob.extend(path.read_bytes())
            curriculum_blob.extend(b"\n")

    if curriculum_blob:
        while len(data) < total_bytes:
            data.extend(curriculum_blob)

    return torch.tensor(list(data[:total_bytes]), dtype=torch.long)


def batch(stream: torch.Tensor, seq: int, size: int, generator: torch.Generator, device: torch.device):
    starts = torch.randint(0, len(stream) - seq - 1, (size,), generator=generator)
    rows = torch.stack([stream[i : i + seq + 1] for i in starts])
    return rows[:, :-1].to(device), rows[:, 1:].to(device)


class BPETokenLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, layers: int, heads: int, max_len: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model,
            heads,
            d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.core = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.emb(x) + self.pos(positions)[None]
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device), diagonal=1).bool()
        h = self.core(h, mask=mask)
        return self.head(self.norm(h))


@dataclass
class ScaleSpec:
    name: str
    lc_model: dict[str, Any]
    bpe_model: dict[str, Any]


def _context_ids_for_stream(stream: torch.Tensor, buckets: int, order: int) -> torch.Tensor:
    context_ids = torch.zeros_like(stream)
    for lag in range(order):
        shifted = torch.zeros_like(stream)
        if lag == 0:
            shifted = stream
        else:
            shifted[lag:] = stream[:-lag]
        context_ids = (context_ids * 257 + shifted + 1) % buckets
    return context_ids


def _build_empirical_byte_priors(
    train_bytes: torch.Tensor,
    context_specs: set[tuple[int, int]],
    alpha: float = 0.25,
) -> dict[str, Any]:
    stream = train_bytes.to(dtype=torch.long, device="cpu")
    previous = stream[:-1]
    target = stream[1:]

    transition_counts = torch.full((256, 256), alpha, dtype=torch.float32)
    transition_counts.index_put_(
        (previous, target),
        torch.ones_like(target, dtype=torch.float32),
        accumulate=True,
    )
    transition_logits = torch.log(transition_counts)
    transition_logits = transition_logits - torch.logsumexp(
        transition_logits, dim=-1, keepdim=True
    )

    context_logits: dict[tuple[int, int], torch.Tensor] = {}
    for buckets, order in sorted(context_specs):
        counts = torch.full((buckets, 256), alpha, dtype=torch.float32)
        ids = _context_ids_for_stream(previous, buckets=buckets, order=order)
        counts.index_put_(
            (ids, target),
            torch.ones_like(target, dtype=torch.float32),
            accumulate=True,
        )
        logits = torch.log(counts)
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        context_logits[(buckets, order)] = logits

    return {
        "transition_logits": transition_logits,
        "context_logits": context_logits,
    }


def _make_lc(
    model_cfg: dict[str, Any],
    seq: int,
    device: torch.device,
    priors: dict[str, Any] | None = None,
) -> CausalBytePatchLM:
    model_type = str(model_cfg.get("model_type", "patch"))
    if model_type == "adaptive":
        kwargs = dict(model_cfg)
        kwargs.pop("model_type", None)
        kwargs.pop("patch_size", None)
        kwargs.pop("empirical_transition_prior", None)
        kwargs.pop("empirical_context_prior", None)
        kwargs.pop("transition_logit_scale", None)
        kwargs.pop("context_logit_scale", None)
        kwargs.pop("trainable_prior_gates", None)
        kwargs.pop("dynamic_prior_gates", None)
        kwargs.pop("prior_dropout", None)
        kwargs.pop("repeat_suppression_window", None)
        kwargs.pop("repeat_suppression_scale", None)
        kwargs.pop("trainable_repeat_suppression", None)
        repeat_unlikelihood_weight = float(kwargs.pop("repeat_unlikelihood_weight", 0.0))
        repeat_unlikelihood_window = int(kwargs.pop("repeat_unlikelihood_window", 0))
        kwargs.pop("freeze_empirical_priors", None)
        kwargs["max_patches"] = max(8, seq // 2)
        model = CausalAdaptiveBytePatchLM(**kwargs).to(device)
        model.repeat_unlikelihood_weight = repeat_unlikelihood_weight
        model.repeat_unlikelihood_window = repeat_unlikelihood_window
        return model

    patch_size = int(model_cfg["patch_size"])
    local_window = int(model_cfg.get("local_window", 32))
    local_decoder = str(model_cfg.get("local_decoder", "window_transformer"))
    continuous_local = bool(model_cfg.get("continuous_local", False))
    direct_global_context = bool(model_cfg.get("direct_global_context", True))
    modern_blocks = bool(model_cfg.get("modern_blocks", True))
    fused_attention = bool(model_cfg.get("fused_attention", True))
    qk_norm = bool(model_cfg.get("qk_norm", True))
    dropout = float(model_cfg.get("dropout", 0.1))
    global_block = str(model_cfg.get("global_block", "attention"))
    kwargs = dict(model_cfg)
    empirical_transition_prior = bool(kwargs.pop("empirical_transition_prior", False))
    empirical_context_prior = bool(kwargs.pop("empirical_context_prior", False))
    repeat_unlikelihood_weight = float(kwargs.pop("repeat_unlikelihood_weight", 0.0))
    repeat_unlikelihood_window = int(kwargs.pop("repeat_unlikelihood_window", 0))
    freeze_empirical_priors = bool(kwargs.pop("freeze_empirical_priors", False))
    for key in [
        "local_window",
        "local_decoder",
        "continuous_local",
        "direct_global_context",
        "modern_blocks",
        "fused_attention",
        "qk_norm",
        "dropout",
        "global_block",
    ]:
        kwargs.pop(key, None)
    transition_logits = None
    context_logits = None
    if priors and empirical_transition_prior:
        transition_logits = priors["transition_logits"].to(device)
    if priors and empirical_context_prior:
        buckets = int(kwargs.get("context_buckets", 0))
        order = int(kwargs.get("context_order", 3))
        if buckets:
            context_logits = priors["context_logits"][(buckets, order)].to(device)
    model = CausalBytePatchLM(
        max_patches=seq // patch_size,
        continuous_local=continuous_local,
        direct_global_context=direct_global_context,
        local_decoder=local_decoder,
        modern_blocks=modern_blocks,
        fused_attention=fused_attention,
        local_window=local_window,
        patch_unit_buckets=0,
        dropout=dropout,
        qk_norm=qk_norm,
        global_block=global_block,
        transition_logits=transition_logits,
        context_logits=context_logits,
        **kwargs,
    ).to(device)
    if freeze_empirical_priors:
        if hasattr(model, "transition_head"):
            model.transition_head.weight.requires_grad_(False)
        if hasattr(model, "context_head"):
            model.context_head.weight.requires_grad_(False)
    model.repeat_unlikelihood_weight = repeat_unlikelihood_weight
    model.repeat_unlikelihood_window = repeat_unlikelihood_window
    return model


def _lc_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    return out


def _lc_arch_candidates(scale: str) -> list[dict[str, Any]]:
    # Candidate grids are intentionally short so search remains practical.
    if scale == "1m":
        return [
            dict(patch_size=2, d_byte=16, d_model=64, d_abi=32, layers=0, heads=4, local_layers=1, local_decoder="conv", conv_layers=1, local_width=64, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=False, dynamic_prior_gates=False, prior_dropout=0.0, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=16, d_model=64, d_abi=32, layers=1, heads=4, local_layers=1, local_decoder="conv", conv_layers=1, local_width=64, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=False, dynamic_prior_gates=False, prior_dropout=0.0, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=24, d_model=80, d_abi=64, layers=1, heads=4, local_layers=1, local_decoder="conv", conv_layers=3, local_width=80, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=24, d_model=96, d_abi=64, layers=1, heads=4, local_layers=1, local_decoder="conv", conv_layers=3, local_width=96, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, freeze_empirical_priors=True),
            dict(patch_size=4, d_byte=24, d_model=128, d_abi=80, layers=2, heads=4, local_layers=1, local_decoder="conv", conv_layers=5, local_width=128, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=1024, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
            dict(patch_size=2, d_byte=24, d_model=128, d_abi=80, layers=2, heads=4, local_layers=1, local_decoder="conv", conv_layers=5, local_width=128, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=1024, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
            dict(patch_size=2, d_byte=24, d_model=128, d_abi=80, layers=2, heads=4, local_layers=1, local_decoder="conv", conv_layers=5, local_width=128, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
            dict(patch_size=2, d_byte=24, d_model=128, d_abi=80, layers=2, heads=4, local_layers=1, local_decoder="conv", conv_layers=5, local_width=128, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=24, d_model=96, d_abi=64, layers=2, heads=4, local_layers=3, local_width=96, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=1024, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
            dict(patch_size=2, d_byte=24, d_model=96, d_abi=64, layers=2, heads=4, local_layers=3, local_width=96, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=1024, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, repeat_unlikelihood_window=12, repeat_unlikelihood_weight=0.03),
            dict(patch_size=1, d_byte=24, d_model=112, d_abi=64, layers=2, heads=4, local_layers=1, local_width=112, local_window=32, dropout=0.0, empirical_transition_prior=True, context_buckets=512, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.15),
            dict(patch_size=1, d_byte=24, d_model=120, d_abi=64, layers=2, heads=4, local_layers=1, local_width=120, local_window=32, dropout=0.0, empirical_transition_prior=True, context_buckets=1024, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.15),
            dict(patch_size=1, d_byte=24, d_model=128, d_abi=64, layers=2, heads=4, local_layers=1, local_width=128, local_window=32, dropout=0.0, empirical_transition_prior=True),
            dict(patch_size=1, d_byte=24, d_model=128, d_abi=64, layers=2, heads=4, local_layers=1, local_width=128, local_window=32, dropout=0.0, empirical_transition_prior=True, repeat_suppression_window=12, repeat_suppression_scale=0.08, trainable_repeat_suppression=True),
            dict(patch_size=1, d_byte=32, d_model=144, d_abi=80, layers=2, heads=4, local_layers=1, local_width=144, local_window=32),
            dict(patch_size=1, d_byte=32, d_model=144, d_abi=80, layers=2, heads=4, local_layers=1, local_width=144, local_window=32, dropout=0.0),
            dict(patch_size=1, d_byte=32, d_model=136, d_abi=96, layers=3, heads=4, local_layers=1, local_width=136, local_window=32),
            dict(patch_size=1, d_byte=24, d_model=152, d_abi=72, layers=2, heads=4, local_layers=1, local_width=152, local_window=32),
            dict(patch_size=1, d_byte=32, d_model=144, d_abi=80, layers=2, heads=4, local_layers=1, local_width=144, local_window=64),
        ]
    if scale == "2m":
        return [
            dict(patch_size=2, d_byte=16, d_model=64, d_abi=32, layers=0, heads=4, local_layers=1, local_decoder="conv", conv_layers=1, local_width=64, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=False, dynamic_prior_gates=False, prior_dropout=0.0, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=16, d_model=96, d_abi=48, layers=1, heads=4, local_layers=1, local_decoder="conv", conv_layers=1, local_width=96, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=False, dynamic_prior_gates=False, prior_dropout=0.0, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=24, d_model=128, d_abi=64, layers=2, heads=8, local_layers=1, local_decoder="conv", conv_layers=4, local_width=128, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=24, d_model=144, d_abi=72, layers=2, heads=8, local_layers=1, local_decoder="conv", conv_layers=4, local_width=144, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, freeze_empirical_priors=True),
            dict(patch_size=4, d_byte=24, d_model=208, d_abi=96, layers=2, heads=8, local_layers=1, local_decoder="conv", conv_layers=4, local_width=208, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=2048, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
            dict(patch_size=2, d_byte=24, d_model=176, d_abi=96, layers=2, heads=8, local_layers=1, local_decoder="conv", conv_layers=5, local_width=176, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=2048, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
            dict(patch_size=2, d_byte=24, d_model=176, d_abi=96, layers=2, heads=8, local_layers=1, local_decoder="conv", conv_layers=5, local_width=176, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
            dict(patch_size=2, d_byte=24, d_model=176, d_abi=96, layers=2, heads=8, local_layers=1, local_decoder="conv", conv_layers=5, local_width=176, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=24, d_model=144, d_abi=80, layers=2, heads=8, local_layers=3, local_width=144, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=2048, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
            dict(patch_size=2, d_byte=24, d_model=144, d_abi=80, layers=2, heads=8, local_layers=3, local_width=144, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=2048, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, repeat_unlikelihood_window=12, repeat_unlikelihood_weight=0.03),
            dict(patch_size=1, d_byte=24, d_model=144, d_abi=72, layers=2, heads=8, local_layers=2, local_width=144, local_window=32, dropout=0.0, empirical_transition_prior=True, context_buckets=1024, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.15),
            dict(patch_size=1, d_byte=24, d_model=160, d_abi=80, layers=2, heads=8, local_layers=2, local_width=160, local_window=32, dropout=0.0, empirical_transition_prior=True, context_buckets=2048, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.15),
            dict(patch_size=1, d_byte=24, d_model=168, d_abi=80, layers=2, heads=8, local_layers=3, local_width=168, local_window=32, dropout=0.0, empirical_transition_prior=True),
            dict(patch_size=1, d_byte=24, d_model=168, d_abi=80, layers=2, heads=8, local_layers=3, local_width=168, local_window=32, dropout=0.0, empirical_transition_prior=True, repeat_suppression_window=12, repeat_suppression_scale=0.08, trainable_repeat_suppression=True),
            dict(patch_size=1, d_byte=24, d_model=176, d_abi=80, layers=2, heads=8, local_layers=3, local_width=176, local_window=32),
            dict(patch_size=1, d_byte=24, d_model=176, d_abi=80, layers=2, heads=8, local_layers=3, local_width=176, local_window=32, dropout=0.0),
            dict(patch_size=1, d_byte=32, d_model=176, d_abi=64, layers=3, heads=8, local_layers=2, local_width=176, local_window=32),
            dict(patch_size=1, d_byte=32, d_model=184, d_abi=64, layers=2, heads=8, local_layers=2, local_width=184, local_window=32),
            dict(patch_size=1, d_byte=24, d_model=176, d_abi=80, layers=2, heads=8, local_layers=3, local_width=176, local_window=64),
        ]
    if scale == "5m":
        return [
            dict(patch_size=2, d_byte=16, d_model=64, d_abi=32, layers=0, heads=4, local_layers=1, local_decoder="conv", conv_layers=1, local_width=64, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=False, dynamic_prior_gates=False, prior_dropout=0.0, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=16, d_model=96, d_abi=48, layers=1, heads=4, local_layers=1, local_decoder="conv", conv_layers=1, local_width=96, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=False, dynamic_prior_gates=False, prior_dropout=0.0, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=24, d_model=160, d_abi=80, layers=2, heads=8, local_layers=1, local_decoder="conv", conv_layers=2, local_width=160, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=False, dynamic_prior_gates=False, prior_dropout=0.0, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=32, d_model=208, d_abi=104, layers=3, heads=8, local_layers=1, local_decoder="conv", conv_layers=4, local_width=208, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=32, d_model=224, d_abi=112, layers=3, heads=8, local_layers=1, local_decoder="conv", conv_layers=4, local_width=224, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, freeze_empirical_priors=True),
            dict(patch_size=4, d_byte=32, d_model=304, d_abi=160, layers=4, heads=8, local_layers=1, local_decoder="conv", conv_layers=5, local_width=304, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=4096, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
            dict(patch_size=2, d_byte=32, d_model=272, d_abi=144, layers=4, heads=8, local_layers=1, local_decoder="conv", conv_layers=5, local_width=272, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=4096, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
            dict(patch_size=2, d_byte=32, d_model=272, d_abi=144, layers=4, heads=8, local_layers=1, local_decoder="conv", conv_layers=5, local_width=272, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
            dict(patch_size=2, d_byte=32, d_model=272, d_abi=144, layers=4, heads=8, local_layers=1, local_decoder="conv", conv_layers=5, local_width=272, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, freeze_empirical_priors=True),
            dict(patch_size=2, d_byte=32, d_model=240, d_abi=128, layers=4, heads=8, local_layers=3, local_width=240, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=4096, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
            dict(patch_size=2, d_byte=32, d_model=240, d_abi=128, layers=4, heads=8, local_layers=3, local_width=240, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=4096, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, repeat_unlikelihood_window=12, repeat_unlikelihood_weight=0.03),
            dict(patch_size=1, d_byte=32, d_model=224, d_abi=112, layers=4, heads=8, local_layers=2, local_width=224, local_window=32, dropout=0.0, empirical_transition_prior=True, context_buckets=2048, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.15),
            dict(patch_size=1, d_byte=32, d_model=240, d_abi=112, layers=4, heads=8, local_layers=2, local_width=240, local_window=32, dropout=0.0, empirical_transition_prior=True, context_buckets=4096, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.15),
            dict(patch_size=1, d_byte=32, d_model=248, d_abi=128, layers=5, heads=8, local_layers=2, local_width=248, local_window=32, dropout=0.0, empirical_transition_prior=True),
            dict(patch_size=1, d_byte=32, d_model=248, d_abi=128, layers=5, heads=8, local_layers=2, local_width=248, local_window=32, dropout=0.0, empirical_transition_prior=True, repeat_suppression_window=12, repeat_suppression_scale=0.08, trainable_repeat_suppression=True),
            dict(patch_size=1, d_byte=32, d_model=256, d_abi=144, layers=5, heads=8, local_layers=3, local_width=256, local_window=32),
            dict(patch_size=1, d_byte=32, d_model=256, d_abi=144, layers=5, heads=8, local_layers=3, local_width=256, local_window=32, dropout=0.0),
            dict(patch_size=1, d_byte=32, d_model=272, d_abi=112, layers=5, heads=8, local_layers=2, local_width=272, local_window=32, continuous_local=True, dropout=0.0),
            dict(patch_size=1, d_byte=32, d_model=256, d_abi=96, layers=4, heads=8, local_layers=2, local_width=256, local_window=32, dropout=0.0),
            dict(patch_size=1, d_byte=32, d_model=240, d_abi=112, layers=4, heads=8, local_layers=2, local_width=240, local_window=32, dropout=0.0),
            dict(model_type="adaptive", d_byte=32, d_model=272, d_abi=112, layers=4, local_layers=3, heads=8, local_window=16),
            dict(model_type="adaptive", d_byte=32, d_model=288, d_abi=128, layers=4, local_layers=3, heads=8, local_window=16),
            dict(patch_size=1, d_byte=32, d_model=256, d_abi=128, layers=6, heads=8, local_layers=2, local_width=256, local_window=32),
            dict(patch_size=1, d_byte=32, d_model=272, d_abi=112, layers=5, heads=8, local_layers=2, local_width=272, local_window=32),
            dict(patch_size=1, d_byte=32, d_model=256, d_abi=144, layers=5, heads=8, local_layers=3, local_width=256, local_window=64),
        ]
    return [
        dict(patch_size=2, d_byte=16, d_model=96, d_abi=48, layers=1, heads=4, local_layers=1, local_decoder="conv", conv_layers=1, local_width=96, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=False, dynamic_prior_gates=False, prior_dropout=0.0, freeze_empirical_priors=True),
        dict(patch_size=2, d_byte=24, d_model=160, d_abi=80, layers=2, heads=8, local_layers=1, local_decoder="conv", conv_layers=2, local_width=160, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=False, dynamic_prior_gates=False, prior_dropout=0.0, freeze_empirical_priors=True),
        dict(patch_size=2, d_byte=32, d_model=256, d_abi=128, layers=4, heads=8, local_layers=1, local_decoder="conv", conv_layers=3, local_width=256, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=False, dynamic_prior_gates=False, prior_dropout=0.0, freeze_empirical_priors=True),
        dict(patch_size=4, d_byte=32, d_model=352, d_abi=144, layers=5, heads=8, local_layers=1, local_decoder="conv", conv_layers=5, local_width=352, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=4096, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
        dict(patch_size=2, d_byte=32, d_model=320, d_abi=160, layers=5, heads=8, local_layers=1, local_decoder="conv", conv_layers=5, local_width=320, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=4096, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
        dict(patch_size=2, d_byte=32, d_model=320, d_abi=160, layers=5, heads=8, local_layers=1, local_decoder="conv", conv_layers=5, local_width=320, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
        dict(patch_size=2, d_byte=32, d_model=320, d_abi=160, layers=5, heads=8, local_layers=1, local_decoder="conv", conv_layers=5, local_width=320, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=8192, context_order=3, empirical_context_prior=True, transition_logit_scale=0.25, context_logit_scale=0.75, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, freeze_empirical_priors=True),
        dict(patch_size=2, d_byte=32, d_model=288, d_abi=144, layers=5, heads=8, local_layers=3, local_width=288, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=4096, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10),
        dict(patch_size=2, d_byte=32, d_model=288, d_abi=144, layers=5, heads=8, local_layers=3, local_width=288, local_window=16, dropout=0.0, empirical_transition_prior=True, context_buckets=4096, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.10, repeat_unlikelihood_window=12, repeat_unlikelihood_weight=0.03),
        dict(patch_size=1, d_byte=32, d_model=256, d_abi=128, layers=5, heads=8, local_layers=2, local_width=256, local_window=32, dropout=0.0, empirical_transition_prior=True, context_buckets=4096, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.15),
        dict(patch_size=1, d_byte=32, d_model=272, d_abi=128, layers=5, heads=8, local_layers=2, local_width=272, local_window=32, dropout=0.0, empirical_transition_prior=True, context_buckets=4096, context_order=2, empirical_context_prior=True, transition_logit_scale=0.35, context_logit_scale=0.65, trainable_prior_gates=True, dynamic_prior_gates=True, prior_dropout=0.15),
        dict(patch_size=1, d_byte=32, d_model=288, d_abi=128, layers=5, heads=8, local_layers=3, local_width=288, local_window=32, dropout=0.0, empirical_transition_prior=True),
        dict(patch_size=1, d_byte=32, d_model=288, d_abi=128, layers=5, heads=8, local_layers=3, local_width=288, local_window=32, dropout=0.0, empirical_transition_prior=True, repeat_suppression_window=12, repeat_suppression_scale=0.08, trainable_repeat_suppression=True),
        dict(patch_size=1, d_byte=32, d_model=304, d_abi=128, layers=5, heads=8, local_layers=3, local_width=304, local_window=32),
        dict(patch_size=1, d_byte=32, d_model=304, d_abi=128, layers=5, heads=8, local_layers=3, local_width=304, local_window=32, dropout=0.0),
        dict(patch_size=1, d_byte=32, d_model=304, d_abi=128, layers=5, heads=8, local_layers=3, local_width=304, local_window=32, continuous_local=True, dropout=0.0),
        dict(patch_size=1, d_byte=32, d_model=288, d_abi=128, layers=5, heads=8, local_layers=3, local_width=288, local_window=32, dropout=0.0),
        dict(patch_size=1, d_byte=32, d_model=272, d_abi=144, layers=5, heads=8, local_layers=3, local_width=272, local_window=32, dropout=0.0),
        dict(model_type="adaptive", d_byte=32, d_model=304, d_abi=128, layers=4, local_layers=4, heads=8, local_window=16),
        dict(model_type="adaptive", d_byte=32, d_model=320, d_abi=128, layers=4, local_layers=4, heads=8, local_window=16),
        dict(patch_size=1, d_byte=32, d_model=304, d_abi=144, layers=4, heads=8, local_layers=4, local_width=304, local_window=32),
        dict(patch_size=1, d_byte=32, d_model=288, d_abi=144, layers=6, heads=8, local_layers=3, local_width=288, local_window=32),
        dict(patch_size=1, d_byte=32, d_model=304, d_abi=128, layers=5, heads=8, local_layers=3, local_width=304, local_window=64),
    ]


def _current_lr(step: int, total_steps: int, lr_max: float, lr_min: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return lr_max * float(step + 1) / float(warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    t = min(max(step - warmup_steps, 0), decay_steps)
    ratio = t / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return lr_min + (lr_max - lr_min) * cosine


def _repeat_unlikelihood_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    window: int,
) -> torch.Tensor:
    if window <= 0:
        return logits.new_zeros(())
    bsz, seq, vocab = logits.shape
    recent = torch.zeros((bsz, seq, vocab), dtype=torch.bool, device=logits.device)
    max_lag = min(int(window), x.shape[1])
    for lag in range(max_lag):
        src = x[:, : seq - lag]
        recent[:, lag:, :].scatter_(2, src.unsqueeze(-1), True)
    recent.scatter_(2, y.unsqueeze(-1), False)
    if not recent.any():
        return logits.new_zeros(())
    probs = F.softmax(logits.float(), dim=-1).clamp(max=1.0 - 1e-6)
    penalty = -torch.log1p(-probs[recent])
    return penalty.mean()


def _train_lc(model: nn.Module, train_bytes: torch.Tensor, steps: int, seq: int, batch_size: int, device: torch.device, lr: float) -> dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(1234)

    started = time.perf_counter()
    for step in range(steps):
        current_lr = _current_lr(step, steps, lr_max=lr, lr_min=lr * 0.2, warmup_steps=max(steps // 12, 1))
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        x, y = batch(train_bytes, seq, batch_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = _lc_logits(model, x)
            logits = logits[:, : y.shape[1], :]
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            repeat_weight = float(getattr(model, "repeat_unlikelihood_weight", 0.0))
            repeat_window = int(getattr(model, "repeat_unlikelihood_window", 0))
            if repeat_weight > 0.0 and repeat_window > 0:
                loss = loss + repeat_weight * _repeat_unlikelihood_loss(logits, x, y, repeat_window)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / max(elapsed, 1e-9),
        "lr": lr,
        "repeat_unlikelihood_weight": float(getattr(model, "repeat_unlikelihood_weight", 0.0)),
        "repeat_unlikelihood_window": int(getattr(model, "repeat_unlikelihood_window", 0)),
    }


def _train_bpe(model: BPETokenLM, train_tokens: torch.Tensor, steps: int, seq: int, batch_size: int, device: torch.device, lr: float) -> dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(5678)

    started = time.perf_counter()
    for step in range(steps):
        current_lr = _current_lr(step, steps, lr_max=lr, lr_min=lr * 0.2, warmup_steps=max(steps // 12, 1))
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        x, y = batch(train_tokens, seq, batch_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {"elapsed_seconds": elapsed, "steps_per_second": steps / max(elapsed, 1e-9), "lr": lr}


@torch.no_grad()
def _eval_lc_bpb(model: nn.Module, eval_bytes: torch.Tensor, seq: int, batch_size: int, eval_batches: int, device: torch.device) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(777)
    losses = []
    for _ in range(eval_batches):
        x, y = batch(eval_bytes, seq, batch_size, generator, device)
        logits = _lc_logits(model, x)
        logits = logits[:, : y.shape[1], :]
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
    return (sum(losses) / len(losses)) / math.log(2)


@torch.no_grad()
def _eval_bpe_bpb(model: BPETokenLM, eval_tokens: torch.Tensor, eval_byte_count: int, seq: int, batch_size: int, eval_batches: int, device: torch.device) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(888)
    losses = []
    for _ in range(eval_batches):
        x, y = batch(eval_tokens, seq, batch_size, generator, device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
    nll_per_token = sum(losses) / len(losses)
    bytes_per_token = eval_byte_count / max(eval_tokens.numel(), 1)
    return nll_per_token / max(bytes_per_token, 1e-9) / math.log(2)


def _would_repeat_ngram(history: list[int], candidate: int, ngram: int) -> bool:
    if ngram <= 1 or len(history) < ngram - 1:
        return False
    suffix = history[-(ngram - 1) :] + [int(candidate)]
    for index in range(0, len(history) - ngram + 1):
        if history[index : index + ngram] == suffix:
            return True
    return False


def _pick_next(
    logits_1d: torch.Tensor,
    history: list[int],
    top_k: int = 16,
    no_repeat_ngram: int = 8,
) -> int:
    vals, idx = torch.topk(logits_1d, k=min(top_k, logits_1d.numel()))
    del vals
    idx_list = idx.tolist()
    if no_repeat_ngram > 1:
        for cand in idx_list:
            if not _would_repeat_ngram(history, int(cand), no_repeat_ngram):
                return int(cand)
    if len(history) >= 6 and len(set(history[-6:])) <= 2:
        for cand in idx_list:
            if not history or cand != history[-1]:
                return int(cand)
    return int(idx_list[0])


def _gen_lc(model: nn.Module, prompt: str, seq: int, max_new: int = 64) -> str:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    ids = list(prompt.encode("utf-8", errors="replace"))
    local_window = int(getattr(model, "local_window", 32))
    patch_size = int(getattr(model, "patch_size", 2))

    with torch.inference_mode():
        for _ in range(max_new):
            ctx = ids[-seq:]
            if len(ctx) < local_window:
                ctx = ([ord(" ")] * (local_window - len(ctx))) + ctx
            if len(ctx) % local_window:
                need = local_window - (len(ctx) % local_window)
                ctx = ([ord(" ")] * need) + ctx
            if len(ctx) % patch_size:
                need = patch_size - (len(ctx) % patch_size)
                ctx = ([ord(" ")] * need) + ctx
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = _lc_logits(model, x)
            nxt = _pick_next(logits[0, -1], ids)
            ids.append(nxt)
    if was_training:
        model.train()

    return bytes(ids[len(prompt.encode("utf-8", errors="replace")) :]).decode("utf-8", errors="replace")


def _gen_bpe(model: BPETokenLM, tokenizer: spm.SentencePieceProcessor, prompt: str, seq: int, max_new: int = 64) -> str:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    ids = tokenizer.encode(prompt, out_type=int)

    with torch.inference_mode():
        for _ in range(max_new):
            x = torch.tensor([ids[-seq:]], dtype=torch.long, device=device)
            logits = model(x)
            nxt = _pick_next(logits[0, -1], ids)
            ids.append(nxt)
    if was_training:
        model.train()

    return tokenizer.decode(ids[len(tokenizer.encode(prompt, out_type=int)) :])


def _timed_generation(fn, *, new_bytes: int, device: torch.device) -> tuple[str, dict[str, float]]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    text = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return text, {
        "elapsed_seconds": elapsed,
        "new_bytes": float(new_bytes),
        "bytes_per_second": float(new_bytes) / max(elapsed, 1e-9),
    }


def _quality_score(text: str, expected_keywords: list[str]) -> dict[str, float]:
    chars = max(len(text), 1)
    alpha = sum(ch.isalpha() for ch in text) / chars
    tokens = text.lower().split()
    max_rep = max((tokens.count(t) for t in set(tokens)), default=0)
    rep_score = 1.0 - min(max_rep / 10.0, 1.0)
    lower = text.lower()
    kw = sum(1 for k in expected_keywords if k in lower) / max(len(expected_keywords), 1)
    quality = 0.35 * alpha + 0.35 * rep_score + 0.30 * kw
    return {
        "alpha_ratio": alpha,
        "max_token_repeat": float(max_rep),
        "keyword_score": kw,
        "quality_score": quality,
    }


def _params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _tune_lrs_lc(
    spec: ScaleSpec,
    train_bytes: torch.Tensor,
    eval_bytes: torch.Tensor,
    seq: int,
    batch_size: int,
    tune_steps: int,
    eval_batches: int,
    device: torch.device,
    lrs: list[float],
    priors: dict[str, Any] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    for lr in lrs:
        model = _make_lc(spec.lc_model, seq, device, priors=priors)
        _train_lc(model, train_bytes, tune_steps, seq, batch_size, device, lr)
        bpb = _eval_lc_bpb(model, eval_bytes, seq, batch_size, eval_batches, device)
        trials.append({"lr": lr, "bpb": bpb})
    trials.sort(key=lambda x: x["bpb"])
    return float(trials[0]["lr"]), trials


def _tune_lrs_bpe(vocab_size: int, spec: ScaleSpec, train_tokens: torch.Tensor, eval_tokens: torch.Tensor, eval_byte_count: int, seq: int, batch_size: int, tune_steps: int, eval_batches: int, device: torch.device, lrs: list[float]) -> tuple[float, list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    for lr in lrs:
        model = BPETokenLM(vocab_size=vocab_size, max_len=seq, **spec.bpe_model).to(device)
        _train_bpe(model, train_tokens, tune_steps, seq, batch_size, device, lr)
        bpb = _eval_bpe_bpb(model, eval_tokens, eval_byte_count, seq, batch_size, eval_batches, device)
        trials.append({"lr": lr, "bpb": bpb})
    trials.sort(key=lambda x: x["bpb"])
    return float(trials[0]["lr"]), trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair micro-scale LayerCake vs baseline benchmark with equal LR tuning budget")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--tune-steps", type=int, default=150)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--train-bytes", type=int, default=8_000_000)
    parser.add_argument("--eval-bytes", type=int, default=300_000)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--baseline-seq", type=int, default=0, help="BPE training/eval sequence length; 0 auto-matches bytes per step")
    parser.add_argument("--lc-arch-search", action="store_true")
    parser.add_argument("--lc-arch-tune-steps", type=int, default=80)
    parser.add_argument("--lc-arch-eval-batches", type=int, default=10)
    parser.add_argument("--lc-select-probe-steps", type=int, default=48)
    parser.add_argument("--lc-select-objective", choices=["bpb", "win_ratio"], default="bpb")
    parser.add_argument("--lc-max-candidates", type=int, default=4)
    parser.add_argument("--only-scale", choices=["1m", "2m", "5m", "10m"], default="", help="Run one scale only for architecture iteration")
    parser.add_argument("--output", default="results/micro_scale_curriculum_frontier_v2.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    redpajama = ROOT.parent / "layercakeogwithdecoder/data/v6/redpajama_english_train.jsonl"
    curriculum_files = [
        ROOT / "data/curriculum/english_school_curriculum.txt",
        ROOT / "data/curriculum/companion_dialogue_curriculum.txt",
    ]
    full_stream = load_curriculum_bytes(redpajama, curriculum_files, args.train_bytes + args.eval_bytes)
    train_bytes = full_stream[:-args.eval_bytes]
    eval_bytes = full_stream[-args.eval_bytes:]
    prior_context_specs = {
        (512, 2),
        (1024, 2),
        (2048, 2),
        (4096, 2),
        (8192, 3),
    }
    lc_priors = _build_empirical_byte_priors(
        train_bytes,
        context_specs=prior_context_specs,
    )

    with tempfile.TemporaryDirectory(prefix="lc_micro_spm_v2_") as tmp:
        tmpdir = Path(tmp)
        prep_started = time.perf_counter()
        corpus_txt = tmpdir / "corpus.txt"
        corpus_txt.write_text(bytes(train_bytes.tolist()).decode("utf-8", errors="replace"), encoding="utf-8")
        prefix = tmpdir / "micro"
        spm.SentencePieceTrainer.train(
            input=str(corpus_txt),
            model_prefix=str(prefix),
            vocab_size=args.vocab,
            model_type="bpe",
            character_coverage=1.0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
            byte_fallback=True,
            minloglevel=2,
        )
        tokenizer = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")

        train_tokens = torch.tensor(
            tokenizer.encode(bytes(train_bytes.tolist()).decode("utf-8", errors="replace"), out_type=int),
            dtype=torch.long,
        )
        eval_tokens = torch.tensor(
            tokenizer.encode(bytes(eval_bytes.tolist()).decode("utf-8", errors="replace"), out_type=int),
            dtype=torch.long,
        )
        baseline_prep_seconds = time.perf_counter() - prep_started
        bytes_per_token_train = float(train_bytes.numel()) / max(float(train_tokens.numel()), 1.0)
        auto_bpe_seq = max(16, min(args.seq, int(round(args.seq / max(bytes_per_token_train, 1e-9)))))
        bpe_seq = int(args.baseline_seq) if args.baseline_seq > 0 else int(auto_bpe_seq)

        scales = [
            ScaleSpec(
                name="1m",
                lc_model=dict(patch_size=1, d_byte=32, d_model=144, d_abi=80, layers=2, heads=4, local_layers=1, local_width=144),
                bpe_model=dict(d_model=160, layers=2, heads=5),
            ),
            ScaleSpec(
                name="2m",
                lc_model=dict(patch_size=1, d_byte=24, d_model=176, d_abi=80, layers=2, heads=8, local_layers=3, local_width=176),
                bpe_model=dict(d_model=208, layers=3, heads=8),
            ),
            ScaleSpec(
                name="5m",
                lc_model=dict(patch_size=1, d_byte=32, d_model=256, d_abi=144, layers=5, heads=8, local_layers=3, local_width=256),
                bpe_model=dict(d_model=320, layers=5, heads=8),
            ),
            ScaleSpec(
                name="10m",
                lc_model=dict(patch_size=1, d_byte=32, d_model=304, d_abi=128, layers=5, heads=8, local_layers=3, local_width=304),
                bpe_model=dict(d_model=352, layers=6, heads=8),
            ),
        ]
        if args.only_scale:
            scales = [spec for spec in scales if spec.name == args.only_scale]

        prompts = [
            ("Question: What is a calm first step when two threats appear? Answer:", ["first", "step", "calm"]),
            ("Question: How should I recover after a mistake? Answer:", ["recover", "safe", "next"]),
            ("Question: Give a short plan before entering the next room. Answer:", ["plan", "before", "next", "room"]),
        ]

        lr_grid = [2e-4, 4e-4, 7e-4, 1e-3, 1.4e-3, 1.8e-3]

        rows: list[dict[str, Any]] = []
        for spec in scales:
            selected_lc_model = dict(spec.lc_model)
            lc_arch_search_rows: list[dict[str, Any]] = []

            bpe_probe = BPETokenLM(vocab_size=tokenizer.vocab_size(), max_len=args.seq, **spec.bpe_model).to(device)
            bpe_params_cap = _params(bpe_probe)

            best_bpe_lr, bpe_trials = _tune_lrs_bpe(
                vocab_size=tokenizer.vocab_size(),
                spec=spec,
                train_tokens=train_tokens,
                eval_tokens=eval_tokens,
                eval_byte_count=int(eval_bytes.numel()),
                seq=bpe_seq,
                batch_size=args.batch,
                tune_steps=args.tune_steps,
                eval_batches=max(10, args.eval_batches // 2),
                device=device,
                lrs=lr_grid,
            )
            bpe_ref_bpb = min(t["bpb"] for t in bpe_trials)
            bpe_probe_train = _train_bpe(
                bpe_probe,
                train_tokens,
                max(8, args.lc_select_probe_steps),
                bpe_seq,
                args.batch,
                device,
                best_bpe_lr,
            )
            bpe_probe_elapsed = bpe_probe_train["elapsed_seconds"]
            _, bpe_probe_gen_timing = _timed_generation(
                lambda: _gen_bpe(bpe_probe, tokenizer, prompts[0][0], seq=bpe_seq, max_new=32),
                new_bytes=32,
                device=device,
            )
            bpe_probe_gen_elapsed = bpe_probe_gen_timing["elapsed_seconds"]

            if args.lc_arch_search:
                for lc_cfg in _lc_arch_candidates(spec.name)[: max(args.lc_max_candidates, 1)]:
                    probe = _make_lc(lc_cfg, args.seq, device, priors=lc_priors)
                    probe_params = _params(probe)
                    if probe_params > bpe_params_cap:
                        continue
                    cand_spec = ScaleSpec(name=spec.name, lc_model=lc_cfg, bpe_model=spec.bpe_model)
                    cand_lr, cand_trials = _tune_lrs_lc(
                        cand_spec,
                        train_bytes,
                        eval_bytes,
                        seq=args.seq,
                        batch_size=args.batch,
                        tune_steps=args.lc_arch_tune_steps,
                        eval_batches=max(6, args.lc_arch_eval_batches),
                        device=device,
                        lrs=lr_grid,
                        priors=lc_priors,
                    )
                    lc_arch_search_rows.append(
                        {
                            "lc_model": lc_cfg,
                            "params": probe_params,
                            "best_lr": cand_lr,
                            "best_bpb": min(t["bpb"] for t in cand_trials),
                            "lr_tuning": cand_trials,
                        }
                    )

                for row in lc_arch_search_rows:
                    probe = _make_lc(row["lc_model"], args.seq, device, priors=lc_priors)
                    lc_probe_train = _train_lc(
                        probe,
                        train_bytes,
                        max(8, args.lc_select_probe_steps),
                        args.seq,
                        args.batch,
                        device,
                        float(row["best_lr"]),
                    )
                    lc_probe_elapsed = lc_probe_train["elapsed_seconds"]
                    bpb_ratio = float(row["best_bpb"]) / max(float(bpe_ref_bpb), 1e-9)
                    speed_ratio = lc_probe_elapsed / max(float(bpe_probe_elapsed), 1e-9)
                    cost_ratio = (lc_probe_elapsed * float(row["params"])) / max(float(bpe_probe_elapsed) * float(bpe_params_cap), 1e-9)
                    row["probe_train"] = lc_probe_train
                    _, lc_probe_gen_timing = _timed_generation(
                        lambda probe=probe: _gen_lc(probe, prompts[0][0], seq=args.seq, max_new=32),
                        new_bytes=32,
                        device=device,
                    )
                    lc_probe_gen_elapsed = lc_probe_gen_timing["elapsed_seconds"]
                    row["bpb_ratio_vs_bpe_tune"] = bpb_ratio
                    row["speed_ratio_vs_bpe_probe"] = speed_ratio
                    row["cost_ratio_vs_bpe_probe"] = cost_ratio
                    row["generation_ratio_vs_bpe_probe"] = lc_probe_gen_elapsed / max(float(bpe_probe_gen_elapsed), 1e-9)
                    row["probe_generation"] = lc_probe_gen_timing
                    row["win_ratio"] = max(bpb_ratio, speed_ratio, cost_ratio, row["generation_ratio_vs_bpe_probe"])

                if lc_arch_search_rows:
                    if args.lc_select_objective == "win_ratio":
                        lc_arch_search_rows.sort(key=lambda row: (row["win_ratio"], row["best_bpb"]))
                    else:
                        lc_arch_search_rows.sort(key=lambda row: row["best_bpb"])
                    selected_lc_model = dict(lc_arch_search_rows[0]["lc_model"])

            selected_spec = ScaleSpec(name=spec.name, lc_model=selected_lc_model, bpe_model=spec.bpe_model)

            best_lc_lr, lc_trials = _tune_lrs_lc(
                selected_spec,
                train_bytes,
                eval_bytes,
                seq=args.seq,
                batch_size=args.batch,
                tune_steps=args.tune_steps,
                eval_batches=max(10, args.eval_batches // 2),
                device=device,
                lrs=lr_grid,
                priors=lc_priors,
            )
            lc = _make_lc(selected_spec.lc_model, args.seq, device, priors=lc_priors)
            bpe = BPETokenLM(vocab_size=tokenizer.vocab_size(), max_len=args.seq, **spec.bpe_model).to(device)

            lc_train = _train_lc(lc, train_bytes, args.steps, args.seq, args.batch, device, best_lc_lr)
            bpe_train = _train_bpe(bpe, train_tokens, args.steps, bpe_seq, args.batch, device, best_bpe_lr)
            bpe_elapsed_total = bpe_train["elapsed_seconds"] + baseline_prep_seconds

            lc_bpb = _eval_lc_bpb(lc, eval_bytes, args.seq, args.batch, args.eval_batches, device)
            bpe_bpb = _eval_bpe_bpb(bpe, eval_tokens, int(eval_bytes.numel()), bpe_seq, args.batch, args.eval_batches, device)

            lc_scores = []
            bpe_scores = []
            lc_gen_seconds = []
            bpe_gen_seconds = []
            lc_gen_bps = []
            bpe_gen_bps = []
            qa_rows = []
            for prompt, kws in prompts:
                lc_text, lc_timing = _timed_generation(
                    lambda prompt=prompt: _gen_lc(lc, prompt, seq=args.seq),
                    new_bytes=64,
                    device=device,
                )
                bpe_text, bpe_timing = _timed_generation(
                    lambda prompt=prompt: _gen_bpe(bpe, tokenizer, prompt, seq=bpe_seq),
                    new_bytes=64,
                    device=device,
                )
                lc_q = _quality_score(lc_text, kws)
                bpe_q = _quality_score(bpe_text, kws)
                lc_scores.append(lc_q["quality_score"])
                bpe_scores.append(bpe_q["quality_score"])
                lc_gen_seconds.append(lc_timing["elapsed_seconds"])
                bpe_gen_seconds.append(bpe_timing["elapsed_seconds"])
                lc_gen_bps.append(lc_timing["bytes_per_second"])
                bpe_gen_bps.append(bpe_timing["bytes_per_second"])
                qa_rows.append(
                    {
                        "prompt": prompt,
                        "layercake": {"text": lc_text, **lc_q, "generation_timing": lc_timing},
                        "baseline": {"text": bpe_text, **bpe_q, "generation_timing": bpe_timing},
                    }
                )

            lc_params = _params(lc)
            bpe_params = _params(bpe)
            lc_param_seconds = lc_train["elapsed_seconds"] * lc_params
            bpe_param_seconds = bpe_elapsed_total * bpe_params

            gates = {
                "speed_beats_baseline": lc_train["elapsed_seconds"] < bpe_elapsed_total,
                "quality_noninferior": (sum(lc_scores) / len(lc_scores)) >= (sum(bpe_scores) / len(bpe_scores)) * 1.00,
                "bpb_noninferior": lc_bpb <= bpe_bpb * 1.00,
                "cost_proxy_lower": lc_param_seconds < bpe_param_seconds,
                "params_no_larger": lc_params <= bpe_params,
                "generation_faster": (sum(lc_gen_seconds) / len(lc_gen_seconds))
                < (sum(bpe_gen_seconds) / len(bpe_gen_seconds)),
            }
            status = "PASS" if all(gates.values()) else "FAIL"

            rows.append(
                {
                    "scale": spec.name,
                    "status": status,
                    "gates": gates,
                    "layercake": {
                        "params": lc_params,
                        "selected_model": selected_spec.lc_model,
                        "train": lc_train,
                        "general_bpb": lc_bpb,
                        "qa_quality_mean": sum(lc_scores) / len(lc_scores),
                        "generation": {
                            "mean_elapsed_seconds": sum(lc_gen_seconds) / len(lc_gen_seconds),
                            "mean_bytes_per_second": sum(lc_gen_bps) / len(lc_gen_bps),
                        },
                        "lr_tuning": lc_trials,
                        "arch_search": lc_arch_search_rows if args.lc_arch_search else None,
                    },
                    "baseline": {
                        "params": bpe_params,
                        "train": {
                            **bpe_train,
                            "prep_seconds": baseline_prep_seconds,
                            "elapsed_total_seconds": bpe_elapsed_total,
                        },
                        "general_bpb": bpe_bpb,
                        "qa_quality_mean": sum(bpe_scores) / len(bpe_scores),
                        "generation": {
                            "mean_elapsed_seconds": sum(bpe_gen_seconds) / len(bpe_gen_seconds),
                            "mean_bytes_per_second": sum(bpe_gen_bps) / len(bpe_gen_bps),
                        },
                        "lr_tuning": bpe_trials,
                    },
                    "cost_proxy_param_seconds": {
                        "layercake": lc_param_seconds,
                        "baseline": bpe_param_seconds,
                    },
                    "qa_samples": qa_rows,
                }
            )

        summary_gates = {
            "all_scales_pass": all(r["status"] == "PASS" for r in rows),
            "pass_1m": any(r["scale"] == "1m" and r["status"] == "PASS" for r in rows),
            "pass_2m": any(r["scale"] == "2m" and r["status"] == "PASS" for r in rows),
            "pass_5m": any(r["scale"] == "5m" and r["status"] == "PASS" for r in rows),
            "pass_10m": any(r["scale"] == "10m" and r["status"] == "PASS" for r in rows),
        }

        result = {
            "status": "PASS" if summary_gates["all_scales_pass"] else "FAIL",
            "scope": "LayerCake byte curriculum vs baseline transformer at 1M/2M/5M/10M (fair LR tuning v2)",
            "device": str(device),
            "steps": args.steps,
            "tune_steps": args.tune_steps,
            "seq": args.seq,
            "baseline_seq": bpe_seq,
            "baseline_bytes_per_token_train": bytes_per_token_train,
            "batch": args.batch,
            "train_bytes": int(train_bytes.numel()),
            "eval_bytes": int(eval_bytes.numel()),
            "summary_gates": summary_gates,
            "scales": rows,
        }

        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
