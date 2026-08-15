"""V24-compatible allocation-bounded host with a fifth clarification route."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from layercake.cake.installer import HostCapabilities
from layercake.cake.package import CakePackage
from layercake.models.shallow_sparse_english import (
    ShallowSparseEnglishConfig,
    ShallowSparseEnglishCore,
)
from layercake.portable_domain import state_dict_hash
from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer
from layercake_extensions.route_isolated_allocation_bounded_core_v24 import (
    ALLOCATION_BOUNDED_ADOPTION_FEATURE,
    AllocationBoundedRuntimeResidencyCoreHost,
    _adopt_authenticated_state,
)
from layercake_extensions.route_isolated_format_literal_core_v22 import (
    FORMAT_LITERAL_DECLARATION,
    FORMAT_LITERAL_FEATURE,
)
from layercake_extensions.route_isolated_lexical_guard_core_v21 import (
    EXACT_LEXICAL_BOUNDARY,
    EXACT_LEXICAL_GUARD_FEATURE,
)
from layercake_extensions.route_isolated_prompt_span_core_v19 import PROMPT_SPAN_FEATURE
from layercake_extensions.route_isolated_runtime_residency_core_v23 import (
    SINGLE_PARSE_ACTIVATION_FEATURE,
    _SingleParseCakeInstaller,
)
from layercake_extensions.route_isolated_shallow_sparse_core import (
    CAPABILITIES,
    CAPABILITY_TO_TASK_ROUTE,
    WEAK_CAPABILITIES,
    SparseCapabilityRouter,
    RouteIsolatedCoreError,
    _DeclaredTokenizer,
)
from layercake_extensions.route_isolated_universal_guard_core_v20 import (
    GUARD_PREDICATE,
    UNIVERSAL_GUARD_FEATURE,
)


ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_VERSION = "lc-direct-neural-core/25"
ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256 = (
    "6d0fea2697fc7af74e9f3cfaee591d658b3244acfb10502048a4105c010b6908"
)
ARCHITECTURE_V25_FORMAT = "layercake-route-isolated-shallow-sparse-core/3"
CLARIFICATION_ROUTE_ISOLATION_FEATURE = "clarification_route_isolation"
RESIDUAL_CAPABILITIES_V25 = (*WEAK_CAPABILITIES, "clarification")
CLARIFICATION_ROUTE_V25 = 4


class ClarificationRouteAllocationBoundedCoreHost(
    AllocationBoundedRuntimeResidencyCoreHost
):
    """Adopt one signed five-route payload and isolate clarification on route four."""

    ABI_VERSION = ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_VERSION
    ABI_SHA256 = ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256
    ARCHITECTURE_FORMAT = ARCHITECTURE_V25_FORMAT

    def __init__(
        self,
        registry_root: str | Path,
        *,
        trust_store: Mapping[str, bytes | str | Path],
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(registry_root, trust_store=trust_store, device=device)
        self.installer = _SingleParseCakeInstaller(
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
                        FORMAT_LITERAL_FEATURE,
                        SINGLE_PARSE_ACTIVATION_FEATURE,
                        ALLOCATION_BOUNDED_ADOPTION_FEATURE,
                        CLARIFICATION_ROUTE_ISOLATION_FEATURE,
                    }
                ),
            ),
            trust_store=trust_store,
            strict_signatures=True,
        )

    @classmethod
    def _validate_role(cls, package: CakePackage) -> None:
        super()._validate_role(package)
        required = set(package.manifest.minimum_host_capabilities.get("features", ()))
        if CLARIFICATION_ROUTE_ISOLATION_FEATURE not in required:
            raise RouteIsolatedCoreError("v25 package omits clarification-route isolation")
        architecture = cls._architecture(package)
        residual = architecture["residual"]
        if (
            set(residual) != {"width", "rank", "routes", "reuse"}
            or residual["reuse"] != "before_each_transformer_block"
            or int(residual["width"]) != int(architecture["model"]["width"])
            or int(residual["rank"]) != 16
            or int(residual["routes"]) != len(RESIDUAL_CAPABILITIES_V25)
        ):
            raise RouteIsolatedCoreError("v25 residual geometry changed")

    @classmethod
    def _architecture(cls, package: CakePackage) -> dict[str, Any]:
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
            "format_literal",
        }
        if set(architecture) != required or architecture.get("format") != cls.ARCHITECTURE_FORMAT:
            raise RouteIsolatedCoreError("v25 architecture declaration changed")
        if tuple(architecture["capabilities"]) != CAPABILITIES:
            raise RouteIsolatedCoreError("v25 capability order changed")
        if architecture["capability_to_task_route"] != CAPABILITY_TO_TASK_ROUTE:
            raise RouteIsolatedCoreError("v25 task-route map changed")
        if tuple(architecture["weak_capabilities"]) != RESIDUAL_CAPABILITIES_V25:
            raise RouteIsolatedCoreError("v25 residual-capability order changed")
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
            raise RouteIsolatedCoreError("v25 lexical guard declaration changed")
        if architecture["format_literal"] != FORMAT_LITERAL_DECLARATION:
            raise RouteIsolatedCoreError("v25 format-literal declaration changed")
        return architecture

    def activate(self, source: str | Path) -> dict[str, Any]:
        record, package = self.installer.inspect_install(source, validator=self._validate_role)
        del record
        architecture = self._architecture(package)
        model_config = ShallowSparseEnglishConfig(**architecture["model"])
        model_state = self._namespace(package.tensors, "model.")
        model, model_aliases, model_buffers = _adopt_authenticated_state(
            lambda: ShallowSparseEnglishCore(model_config), model_state
        )
        router_config = architecture["router"]
        if set(router_config) != {
            "vocabulary",
            "character_hash_buckets",
            "character_ngram_minimum",
            "character_ngram_maximum",
            "hash_seed",
            "classes",
        } or int(router_config["classes"]) != len(CAPABILITIES) + 1:
            raise RouteIsolatedCoreError("v25 router configuration changed")
        router_state = self._namespace(package.tensors, "router.")
        router, router_aliases, router_buffers = _adopt_authenticated_state(
            lambda: SparseCapabilityRouter(
                int(router_config["vocabulary"]),
                int(router_config["character_hash_buckets"]),
                int(router_config["classes"]),
            ),
            router_state,
        )
        residual_config = architecture["residual"]
        if (
            set(residual_config) != {"width", "rank", "routes", "reuse"}
            or residual_config["reuse"] != "before_each_transformer_block"
            or int(residual_config["width"]) != model_config.width
            or int(residual_config["rank"]) != 16
            or int(residual_config["routes"]) != len(RESIDUAL_CAPABILITIES_V25)
        ):
            raise RouteIsolatedCoreError("v25 residual geometry changed")
        residual_state = self._namespace(package.tensors, "residual.")
        residual, residual_aliases, residual_buffers = _adopt_authenticated_state(
            lambda: self.RESIDUAL_TYPE(
                int(residual_config["width"]),
                int(residual_config["rank"]),
                int(residual_config["routes"]),
            ),
            residual_state,
        )
        model_tokenizer = architecture["model_tokenizer"]
        if set(model_tokenizer) != {"format", "tokenizers_json", "sha256", "eos_token_id"} or model_tokenizer["format"] != "declarative-tokenizers-json/1":
            raise RouteIsolatedCoreError("v25 model tokenizer declaration changed")
        canonical_tokenizer = json.dumps(
            model_tokenizer["tokenizers_json"], sort_keys=True, separators=(",", ":")
        ).encode()
        if hashlib.sha256(canonical_tokenizer).hexdigest() != model_tokenizer["sha256"]:
            raise RouteIsolatedCoreError("v25 model tokenizer hash mismatch")
        declared = _DeclaredTokenizer(
            model_tokenizer["tokenizers_json"], int(model_tokenizer["eos_token_id"])
        )
        router_tokenizer = Utf8ConcatenativeBpeTokenizer.from_document(
            architecture["router_tokenizer"]
        )
        if router_tokenizer.vocab_size != int(router_config["vocabulary"]):
            raise RouteIsolatedCoreError("v25 router tokenizer vocabulary mismatch")
        adopted_state_hash = state_dict_hash(package.tensors)
        for module in (model, router, residual):
            module.to(self.device).eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for handle in self.handles:
            handle.remove()
        self.model = model
        self.router = router
        self.residual = residual
        self.model_tokenizer = declared
        self.router_tokenizer = router_tokenizer
        self.router_config = {key: int(value) for key, value in router_config.items()}
        self.guard = dict(architecture["guard"])
        self._route_state_key = None
        self._route_state_tensor = None
        self._attach()
        self.active_cake_id = package.manifest.cake_id
        self.active_archive_hash = package.archive_hash
        self.active_payload_hash = package.manifest.tensor_payload_hash
        aliases = model_aliases + router_aliases + residual_aliases
        return {
            "status": "ACTIVE",
            "cake_id": self.active_cake_id,
            "archive_hash": self.active_archive_hash,
            "payload_hash": self.active_payload_hash,
            "state_dict_hash": adopted_state_hash,
            "device": str(self.device),
            "receiver_training_steps": 0,
            "receiver_calibration_runs": 0,
            "authenticated_package_parses": 1,
            "strict_assigned_tensor_count": aliases,
            "authenticated_tensor_count": len(package.tensors),
            "meta_tensors_after_adoption": 0,
            "reconstructed_nonpersistent_buffers": model_buffers + router_buffers + residual_buffers,
        }

    @torch.inference_mode()
    def prefill(self, prompt: bytes | str) -> dict[str, Any]:
        model, _, _, tokenizer, _ = self._require_active()
        if isinstance(prompt, bytes):
            text = prompt.decode("utf-8", errors="strict")
        else:
            prompt.encode("utf-8", errors="strict")
            text = prompt
        capability = self.route(text)
        residual_route = (
            RESIDUAL_CAPABILITIES_V25.index(capability)
            if capability in RESIDUAL_CAPABILITIES_V25
            else -1
        )
        self._set_residual_route(residual_route)
        prompt_ids = tokenizer.encode(text.rstrip() + "\n")
        if not prompt_ids:
            raise RouteIsolatedCoreError("v25 prompt encodes to no tokens")
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        forced = (
            torch.tensor(
                [CAPABILITY_TO_TASK_ROUTE[capability]],
                dtype=torch.long,
                device=self.device,
            )
            if residual_route >= 0
            else None
        )
        result = model(
            ids,
            prompt_lengths=torch.tensor([len(prompt_ids)], dtype=torch.long, device=self.device),
            task_routes=forced,
            use_cache=True,
        )
        return {
            "past_key_values": result["past_key_values"],
            "task_route": result["task_routes"].detach().clone(),
            "capability": capability,
            "weak_route": residual_route,
            "next_logits": result["logits"][:, -1],
            "generated_ids": [],
            "terminated_by_guard": False,
            "finished": False,
            "guard_realization_override": None,
        }
