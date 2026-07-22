"""Compact local semantic router used after the lexical fast path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

import torch
from torch import nn


DOMAINS = ("python", "mathematics", "biomedical", "actions", "game")
_INJECTION = re.compile(
    r"\b(ignore|bypass|override|disregard)\b.{0,40}\b(router|route|cake|specialist)\b",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"(['\"]).*?\1")


class HashedNgramEncoder:
    def __init__(self, width: int = 1024, minimum: int = 2, maximum: int = 5):
        self.width = int(width)
        self.minimum = int(minimum)
        self.maximum = int(maximum)

    def encode(self, prompts: list[str]) -> torch.Tensor:
        output = torch.zeros(len(prompts), self.width)
        for row, prompt in enumerate(prompts):
            normalized = " ".join(_QUOTED.sub(" ", prompt.casefold()).split())
            padded = f"  {normalized}  "
            for size in range(self.minimum, self.maximum + 1):
                for index in range(max(0, len(padded) - size + 1)):
                    ngram = padded[index:index + size].encode("utf-8")
                    digest = hashlib.blake2b(ngram, digest_size=8, person=b"lc-route").digest()
                    bucket = int.from_bytes(digest, "little") % self.width
                    output[row, bucket] += 1.0
            norm = output[row].norm()
            if norm:
                output[row] /= norm
        return output


@dataclass(frozen=True)
class LearnedRoute:
    selected: tuple[str, ...]
    confidence: float
    abstained: bool
    probabilities: dict[str, float]
    reason: str


class CompactSemanticRouter(nn.Module):
    def __init__(self, *, feature_width: int = 1024, hidden_width: int = 64):
        super().__init__()
        self.encoder = HashedNgramEncoder(feature_width)
        self.network = nn.Sequential(
            nn.Linear(feature_width, hidden_width), nn.SiLU(),
            nn.Linear(hidden_width, len(DOMAINS)),
        )

    def forward(self, prompts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        return self.network(self.encoder.encode(prompts).to(device))

    @torch.inference_mode()
    def route(
        self,
        prompt: str,
        *,
        installed: set[str],
        top_k: int = 2,
        threshold: float = 0.18,
        forced: tuple[str, ...] | None = None,
    ) -> LearnedRoute:
        if forced is not None:
            allowed = tuple(domain for domain in forced if domain in installed)
            return LearnedRoute(allowed, 1.0 if allowed else 0.0, not allowed, {}, "forced")
        if _INJECTION.search(prompt):
            return LearnedRoute((), 0.0, True, {}, "prompt-control-abstention")
        probabilities = torch.sigmoid(self([prompt]))[0].cpu()
        mapping = {domain: float(probabilities[index]) for index, domain in enumerate(DOMAINS)}
        ranked = sorted(mapping, key=mapping.get, reverse=True)
        if ranked and mapping[ranked[0]] >= threshold and ranked[0] not in installed:
            return LearnedRoute((), mapping[ranked[0]], True, mapping, "missing-cake")
        selected = tuple(
            domain for domain in ranked
            if domain in installed and mapping[domain] >= threshold
        )[:max(1, top_k)]
        confidence = mapping[ranked[0]] if ranked else 0.0
        reason = "learned-semantic" if selected else (
            "missing-cake" if confidence >= threshold else "uncertain-core-fallback"
        )
        return LearnedRoute(selected, confidence, not selected, mapping, reason)
