from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class TinyByteTransformer(nn.Module):
    def __init__(self, d_model=32, layers=1, heads=4, max_len=128):
        super().__init__()
        self.emb = nn.Embedding(256, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(d_model, heads, d_model * 2, batch_first=True, norm_first=True)
        self.core = nn.TransformerEncoder(block, layers)
        self.head = nn.Linear(d_model, 256)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device).clamp_max(self.pos.num_embeddings - 1)
        h = self.emb(x) + self.pos(positions)[None]
        mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float("-inf"), device=x.device), diagonal=1)
        return self.head(self.core(h, mask=mask))


TinyBPETransformer = TinyByteTransformer


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def matched_parameter_count_helper(target_params: int, *, max_len=128) -> TinyByteTransformer:
    best = TinyByteTransformer(16, 1, 4, max_len)
    best_params = parameter_count(best)
    best_delta = abs(best_params - target_params)
    for layers in (1, 2, 3, 4, 6, 8):
        for width in (16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384):
            heads = 8 if width % 8 == 0 else 4 if width % 4 == 0 else 2
            candidate = TinyByteTransformer(width, layers, heads, max_len)
            params = parameter_count(candidate)
            delta = abs(params - target_params)
            candidate_is_valid = params >= target_params
            best_is_valid = best_params >= target_params
            if (
                (candidate_is_valid and not best_is_valid)
                or (candidate_is_valid == best_is_valid and delta < best_delta)
            ):
                best, best_delta, best_params = candidate, delta, params
    return best


def baseline_bpb(model: nn.Module, batch: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(batch)[:, :-1]
        targets = batch[:, 1 : 1 + logits.shape[1]]
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
    return float(loss.item() / math.log(2.0))


def matched_training_step_helper(model: nn.Module, batch: torch.Tensor, *, lr=0.01) -> float:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    logits = model(batch)[:, :-1]
    targets = batch[:, 1 : 1 + logits.shape[1]]
    loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.item())


def baseline_training_smoke_loop(batch: torch.Tensor, *, steps=2, target_params=10000) -> dict:
    model = matched_parameter_count_helper(target_params, max_len=batch.shape[1])
    before = baseline_bpb(model, batch)
    losses = [matched_training_step_helper(model, batch) for _ in range(steps)]
    after = baseline_bpb(model, batch)
    return {
        "model": "TinyByteTransformer",
        "params": parameter_count(model),
        "before_bpb": before,
        "after_bpb": after,
        "loss_history": losses,
    }
