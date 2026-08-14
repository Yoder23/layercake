"""V19-compatible prompt-span host with the frozen guard on every capability."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake_extensions.route_isolated_prompt_span_core_v19 import (
    ARCHITECTURE_V19_FORMAT,
    PROMPT_SPAN_FEATURE,
    PromptSpanRouteIsolatedShallowSparseCoreHost,
)
from layercake_extensions.route_isolated_shallow_sparse_core import (
    CAPABILITIES,
    CAPABILITY_TO_TASK_ROUTE,
    WEAK_CAPABILITIES,
    RouteIsolatedCoreError,
    repetition_collapse,
)


ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_VERSION = "lc-direct-neural-core/20"
ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_SHA256 = (
    "0f051d114583661b5ea4943ea906f688b103ee82c2bc39cb6b898b1f8198d923"
)
ARCHITECTURE_V20_FORMAT = ARCHITECTURE_V19_FORMAT
UNIVERSAL_GUARD_FEATURE = "universal_declarative_runtime_guard"
GUARD_PREDICATE = (
    "contiguous_1_to_16_token_span_repeated_4_times_or_"
    "fourgram_diversity_below_0.35_at_32_tokens"
)


class UniversalGuardPromptSpanCoreHost(PromptSpanRouteIsolatedShallowSparseCoreHost):
    """The v19 host with its unchanged collapse predicate applied universally."""

    ABI_VERSION = ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_VERSION
    ABI_SHA256 = ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_SHA256
    ARCHITECTURE_FORMAT = ARCHITECTURE_V20_FORMAT

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
        if UNIVERSAL_GUARD_FEATURE not in required:
            raise RouteIsolatedCoreError("v20 package omits the universal guard capability")

    @classmethod
    def _architecture(cls, package) -> dict[str, Any]:
        architecture = package.manifest.architecture
        required = {
            "format",
            "model",
            "model_tokenizer",
            "router",
            "router_tokenizer",
            "residual",
            "capabilities",
            "capability_to_task_route",
            "weak_capabilities",
            "guard",
        }
        if set(architecture) != required or architecture.get("format") != cls.ARCHITECTURE_FORMAT:
            raise RouteIsolatedCoreError("v20 architecture declaration changed")
        if tuple(architecture["capabilities"]) != CAPABILITIES:
            raise RouteIsolatedCoreError("v20 capability order changed")
        if architecture["capability_to_task_route"] != CAPABILITY_TO_TASK_ROUTE:
            raise RouteIsolatedCoreError("v20 task-route map changed")
        if tuple(architecture["weak_capabilities"]) != WEAK_CAPABILITIES:
            raise RouteIsolatedCoreError("v20 weak-capability order changed")
        guard = architecture["guard"]
        if (
            guard.get("predicate") != GUARD_PREDICATE
            or guard.get("scope") != "all_capabilities"
            or guard.get("stop_before_collapsing_token") is not True
            or not isinstance(guard.get("abstention_markers"), list)
            or not guard["abstention_markers"]
            or not isinstance(guard.get("abstention_clause"), str)
            or not guard["abstention_clause"]
        ):
            raise RouteIsolatedCoreError("v20 universal guard declaration changed")
        return architecture

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
        if repetition_collapse(tokenizer.decode(candidate)):
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
