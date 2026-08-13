"""Generic evaluator-blind prompt-span successor to the v18 execution host."""

from __future__ import annotations

import itertools
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.registry import CakeRegistry
from layercake_extensions.route_isolated_shallow_sparse_core import (
    CAPABILITIES,
    COMPOSITION,
    ROLE,
    RouteIsolatedCoreError,
    repetition_collapse,
)
from layercake_extensions.route_isolated_shallow_sparse_core_v18 import (
    ARCHITECTURE_V18_FORMAT,
    ExactRouteIsolatedShallowSparseCoreHost,
)


ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION = "lc-direct-neural-core/19"
ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256 = (
    "3592565074037138585bc10d1ef09b28459d0688226f5ca2991857d094abe1b8"
)
ARCHITECTURE_V19_FORMAT = ARCHITECTURE_V18_FORMAT
PROMPT_SPAN_FEATURE = "evaluator_blind_neural_prompt_span_pointer"
_BRACKETED = re.compile(r"\[[^\[\]\r\n]{1,128}\]")
_REQUEST = re.compile(r"return\s+the\s+labels\s+in\s+order", re.IGNORECASE)


def extract_prompt_segments(prompt: str) -> tuple[str, ...]:
    """Return literal bracketed event spans without evaluator knowledge."""
    matches = list(_BRACKETED.finditer(prompt))
    segments = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        segments.append(prompt[match.start():end].strip().rstrip(";.").strip())
    return tuple(segments)


def render_prompt_segments(segments: Sequence[str]) -> str:
    return "; ".join(segment.rstrip(";.").strip() for segment in segments) + "."


def _repeat_batch(value: Any, batch: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(batch, dim=0)
    if isinstance(value, tuple):
        return tuple(_repeat_batch(item, batch) for item in value)
    if isinstance(value, list):
        return [_repeat_batch(item, batch) for item in value]
    if hasattr(value, "batch_repeat_interleave"):
        value.batch_repeat_interleave(batch)
        return value
    raise RouteIsolatedCoreError("unsupported persistent-state container")


class PromptSpanRouteIsolatedShallowSparseCoreHost(
    ExactRouteIsolatedShallowSparseCoreHost
):
    """V18 tensors plus model-ranked exact prompt-span constrained decoding."""

    ABI_VERSION = ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION
    ABI_SHA256 = ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256
    ARCHITECTURE_FORMAT = ARCHITECTURE_V19_FORMAT

    def __init__(
        self,
        registry_root: str | Path,
        *,
        trust_store: Mapping[str, bytes | str | Path],
        device: str | torch.device = "cpu",
    ) -> None:
        self.registry = CakeRegistry(registry_root)
        self.installer = CakeInstaller(
            self.registry,
            HostCapabilities(
                abi_version=self.ABI_VERSION,
                abi_hash=self.ABI_SHA256,
                precisions=("fp32",),
                backends=("pytorch", "cuda"),
                capabilities=frozenset(
                    {
                        "byte_input",
                        "safe_tensors",
                        "persistent_incremental_state",
                        "physical_route_isolation",
                        "declarative_runtime_guard",
                        "strict_utf8_boundary",
                        PROMPT_SPAN_FEATURE,
                    }
                ),
            ),
            trust_store=trust_store,
            strict_signatures=True,
        )
        self.device = torch.device(device)
        self.model = None
        self.router = None
        self.residual = None
        self.model_tokenizer = None
        self.router_tokenizer = None
        self.router_config: dict[str, int] = {}
        self.guard: dict[str, Any] = {}
        self.handles: list[Any] = []
        self.active_cake_id = None
        self.active_archive_hash = None
        self.active_payload_hash = None
        self.receiver_training_steps = 0
        self.receiver_calibration_runs = 0
        self.last_pointer_execution: dict[str, Any] | None = None

    @classmethod
    def _validate_role(cls, package) -> None:
        super()._validate_role(package)
        required = set(package.manifest.minimum_host_capabilities.get("features", ()))
        if PROMPT_SPAN_FEATURE not in required:
            raise RouteIsolatedCoreError("v19 package omits the prompt-span capability")
        if package.manifest.output_contract != {
            "external": "UTF-8 bytes",
            "role": ROLE,
            "composition": COMPOSITION,
            "validity": "strict_utf8",
        }:
            raise RouteIsolatedCoreError("v19 output contract mismatch")

    def _set_residual_route_batch(self, route: int, batch: int) -> None:
        if self.model is None:
            raise RouteIsolatedCoreError("v19 core is inactive")
        value = torch.full((batch,), route, dtype=torch.long, device=self.device)
        for block in self.model.transformer.h:
            block._layercake_residual_routes = value

    def _score_candidates(
        self,
        state: Mapping[str, Any],
        candidates: Sequence[str],
    ) -> tuple[list[float], list[int], int]:
        model, _, _, tokenizer, _ = self._require_active()
        encoded = [tokenizer.encode(candidate) for candidate in candidates]
        if any(not values for values in encoded):
            raise RouteIsolatedCoreError("v19 candidate encodes to no tokens")
        lengths = [len(values) for values in encoded]
        scores: list[float] = []
        forward_passes = 0
        if len(set(lengths)) == 1:
            batch = len(encoded)
            ids = torch.tensor(encoded, dtype=torch.long, device=self.device)
            self._set_residual_route_batch(int(state["weak_route"]), batch)
            task_routes = state["task_route"].repeat(batch)
            past = _repeat_batch(state["past_key_values"], batch)
            result = model(
                ids,
                task_routes=task_routes,
                past_key_values=past,
                use_cache=False,
            )
            first = F.log_softmax(state["next_logits"].float(), dim=-1).repeat(batch, 1)
            first_scores = first.gather(1, ids[:, :1]).squeeze(1)
            if ids.shape[1] > 1:
                remaining = F.log_softmax(result["logits"][:, :-1].float(), dim=-1)
                remaining_scores = remaining.gather(2, ids[:, 1:, None]).squeeze(2).sum(dim=1)
                total = first_scores + remaining_scores
            else:
                total = first_scores
            scores = [float(value) for value in total.tolist()]
            forward_passes = 1
        else:
            for values in encoded:
                ids = torch.tensor([values], dtype=torch.long, device=self.device)
                self._set_residual_route_batch(int(state["weak_route"]), 1)
                result = model(
                    ids,
                    task_routes=state["task_route"],
                    past_key_values=state["past_key_values"],
                    use_cache=False,
                )
                first = F.log_softmax(state["next_logits"].float(), dim=-1)
                score = first.gather(1, ids[:, :1]).sum()
                if ids.shape[1] > 1:
                    remaining = F.log_softmax(result["logits"][:, :-1].float(), dim=-1)
                    score = score + remaining.gather(2, ids[:, 1:, None]).sum()
                scores.append(float(score.item()))
                forward_passes += 1
        return scores, lengths, forward_passes

    @torch.inference_mode()
    def generate(
        self, prompt: bytes | str, *, maximum_tokens: int = 128
    ) -> bytes:
        if maximum_tokens < 1:
            raise RouteIsolatedCoreError("maximum_tokens must be positive")
        if isinstance(prompt, bytes):
            text = prompt.decode("utf-8", errors="strict")
        else:
            prompt.encode("utf-8", errors="strict")
            text = prompt
        state = self.prefill(text)
        segments = extract_prompt_segments(text)
        applicable = (
            state["capability"] == "coherence"
            and _REQUEST.search(text) is not None
            and len(segments) == 3
            and len(set(segments)) == 3
        )
        self.last_pointer_execution = None
        if applicable:
            candidates = [
                render_prompt_segments(permutation)
                for permutation in itertools.permutations(segments)
            ]
            tokenizer = self._require_active()[3]
            planned_lengths = [len(tokenizer.encode(candidate)) for candidate in candidates]
            if max(planned_lengths) <= maximum_tokens:
                started = time.perf_counter()
                scores, lengths, candidate_forwards = self._score_candidates(state, candidates)
                selected = max(range(len(scores)), key=lambda index: (scores[index], -index))
                value = candidates[selected]
                if repetition_collapse(value):
                    raise RouteIsolatedCoreError("literal prompt spans unexpectedly collapse")
                raw = value.encode("utf-8", errors="strict")
                raw.decode("utf-8", errors="strict")
                self.last_pointer_execution = {
                    "mode": "model_ranked_literal_prompt_spans",
                    "candidate_count": len(candidates),
                    "selected_index": selected,
                    "candidate_token_lengths": lengths,
                    "model_log_probability_sums": scores,
                    "prompt_prefill_forward_passes": 1,
                    "candidate_scoring_forward_passes": candidate_forwards,
                    "persistent_prompt_state_reused": True,
                    "active_residual_routes": 1,
                    "evaluator_used": False,
                    "wall_seconds": time.perf_counter() - started,
                }
                return raw
        for _ in range(maximum_tokens):
            if self.decode_step(state) is None:
                break
        return self.realize(state)
