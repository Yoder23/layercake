"""Transfer metrics that separate copy fidelity from generation quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TransferResult:
    source_size: str
    source_seed: int
    source_input_mode: str
    target_size: str
    target_seed: int
    target_input_mode: str
    abi_version: str
    brick_type: str
    weight_max_diff: float
    function_max_diff: float
    generation_equal: bool | None
    domain_ppl_source: float | None
    domain_ppl_target: float | None
    general_ppl_source: float | None
    general_ppl_target: float | None
    degradation_ratio: float | None
    abi_drift: float | None
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def state_dict_max_diff(a: nn.Module, b: nn.Module) -> float:
    a_state, b_state = a.state_dict(), b.state_dict()
    if a_state.keys() != b_state.keys():
        raise ValueError("state dict key mismatch")
    return max(
        ((a_state[key].detach().cpu() - b_state[key].detach().cpu()).abs().max().item()
         for key in a_state),
        default=0.0,
    )


def perplexity_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("expected logits [B,T,V] and targets [B,T]")
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return math.exp(min(loss.item(), 30.0))


def ppl_degradation_ratio(source_ppl: float, target_ppl: float) -> float:
    if source_ppl <= 0:
        raise ValueError("source perplexity must be positive")
    return target_ppl / source_ppl


def classify_transfer(
    *,
    weight_max_diff: float,
    function_max_diff: float,
    degradation_ratio: float | None,
    max_ppl_ratio: float = 1.05,
) -> str:
    if weight_max_diff != 0.0:
        return "L0_FAIL"
    if function_max_diff != 0.0:
        return "L1_FAIL"
    if degradation_ratio is None:
        return "L1_PASS_UNMEASURED_GENERATION"
    return "PPL_PASS" if degradation_ratio <= max_ppl_ratio else "PPL_REGRESSION"
