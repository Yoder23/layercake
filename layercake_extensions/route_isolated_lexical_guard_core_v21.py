"""V20-compatible host with exact lexical-boundary repetition realization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake_extensions.route_isolated_prompt_span_core_v19 import PROMPT_SPAN_FEATURE
from layercake_extensions.route_isolated_shallow_sparse_core import (
    CAPABILITIES,
    CAPABILITY_TO_TASK_ROUTE,
    TOKEN_PATTERN,
    WEAK_CAPABILITIES,
    RouteIsolatedCoreError,
    repetition_collapse,
)
from layercake_extensions.route_isolated_universal_guard_core_v20 import (
    ARCHITECTURE_V20_FORMAT,
    GUARD_PREDICATE,
    UNIVERSAL_GUARD_FEATURE,
    UniversalGuardPromptSpanCoreHost,
)


ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_VERSION = "lc-direct-neural-core/21"
ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256 = (
    "ac7f49f3b7674d3139fac930a4de9b7cf002eaf8779e802928c94acb3eb7cc69"
)
ARCHITECTURE_V21_FORMAT = ARCHITECTURE_V20_FORMAT
EXACT_LEXICAL_GUARD_FEATURE = "exact_lexical_boundary_runtime_guard"
EXACT_LEXICAL_BOUNDARY = "maximal_pre_collapse_lexical_prefix"


def maximal_safe_lexical_prefix(value: str) -> str:
    """Return the exact prefix before the lexical token that first collapses."""
    if not repetition_collapse(value):
        return value
    for match in TOKEN_PATTERN.finditer(value):
        if repetition_collapse(value[: match.end()]):
            prefix = value[: match.start()].rstrip()
            if repetition_collapse(prefix):
                raise RouteIsolatedCoreError("lexical guard failed to remove collapse")
            return prefix
    raise RouteIsolatedCoreError("collapsed text has no lexical transition")


class LexicalGuardPromptSpanCoreHost(UniversalGuardPromptSpanCoreHost):
    """Universal v20 guard with deterministic lexical-prefix realization."""

    ABI_VERSION = ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_VERSION
    ABI_SHA256 = ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256
    ARCHITECTURE_FORMAT = ARCHITECTURE_V21_FORMAT

    def __init__(
        self,
        registry_root: str | Path,
        *,
        trust_store: Mapping[str, bytes | str | Path],
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(registry_root, trust_store=trust_store, device=device)
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
                        UNIVERSAL_GUARD_FEATURE,
                        EXACT_LEXICAL_GUARD_FEATURE,
                    }
                ),
            ),
            trust_store=trust_store,
            strict_signatures=True,
        )

    @classmethod
    def _validate_role(cls, package) -> None:
        super()._validate_role(package)
        required = set(package.manifest.minimum_host_capabilities.get("features", ()))
        if EXACT_LEXICAL_GUARD_FEATURE not in required:
            raise RouteIsolatedCoreError("v21 package omits exact lexical guard")

    @classmethod
    def _architecture(cls, package) -> dict[str, Any]:
        architecture = package.manifest.architecture
        required = {
            "format", "model", "model_tokenizer", "router", "router_tokenizer",
            "residual", "capabilities", "capability_to_task_route",
            "weak_capabilities", "guard",
        }
        if set(architecture) != required or architecture.get("format") != cls.ARCHITECTURE_FORMAT:
            raise RouteIsolatedCoreError("v21 architecture declaration changed")
        if tuple(architecture["capabilities"]) != CAPABILITIES:
            raise RouteIsolatedCoreError("v21 capability order changed")
        if architecture["capability_to_task_route"] != CAPABILITY_TO_TASK_ROUTE:
            raise RouteIsolatedCoreError("v21 task-route map changed")
        if tuple(architecture["weak_capabilities"]) != WEAK_CAPABILITIES:
            raise RouteIsolatedCoreError("v21 weak-capability order changed")
        guard = architecture["guard"]
        if (
            guard.get("predicate") != GUARD_PREDICATE
            or guard.get("scope") != "all_capabilities"
            or guard.get("boundary") != EXACT_LEXICAL_BOUNDARY
            or guard.get("stop_before_collapsing_token") is not True
            or not isinstance(guard.get("abstention_markers"), list)
            or not guard["abstention_markers"]
            or not isinstance(guard.get("abstention_clause"), str)
            or not guard["abstention_clause"]
        ):
            raise RouteIsolatedCoreError("v21 lexical guard declaration changed")
        return architecture

    @torch.inference_mode()
    def prefill(self, prompt: bytes | str) -> dict[str, Any]:
        state = super().prefill(prompt)
        state["guard_realization_override"] = None
        return state

    @torch.inference_mode()
    def decode_step(self, state: dict[str, Any]) -> int | None:
        model, _, _, tokenizer, _ = self._require_active()
        if state["finished"]:
            return None
        self._set_residual_route(int(state["weak_route"]))
        selected = state["next_logits"].argmax(dim=-1)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            state["finished"] = True
            return None
        candidate = [*state["generated_ids"], token]
        candidate_text = tokenizer.decode(candidate)
        if repetition_collapse(candidate_text):
            state["guard_realization_override"] = maximal_safe_lexical_prefix(candidate_text)
            state["terminated_by_guard"] = True
            state["finished"] = True
            return None
        state["generated_ids"].append(token)
        result = model(
            selected[:, None],
            task_routes=state["task_route"],
            past_key_values=state["past_key_values"],
            use_cache=True,
        )
        state["past_key_values"] = result["past_key_values"]
        state["next_logits"] = result["logits"][:, -1]
        return token

    def realize(self, state: Mapping[str, Any]) -> bytes:
        override = state.get("guard_realization_override")
        if override is None:
            return super().realize(state)
        value = str(override)
        if state["capability"] == "abstention" and not any(
            marker.casefold() in value.casefold()
            for marker in self.guard["abstention_markers"]
        ):
            clause = str(self.guard["abstention_clause"])
            value = clause + (" " + value if value else "")
        raw = value.encode("utf-8", errors="strict")
        raw.decode("utf-8", errors="strict")
        return raw
