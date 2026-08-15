"""Isolated v23-compatible host with allocation-bounded tensor adoption."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from layercake.cake.installer import HostCapabilities
from layercake.cake.package import CakePackage
from layercake.models.shallow_sparse_english import (
    ShallowSparseEnglishConfig,
    ShallowSparseEnglishCore,
)
from layercake.portable_domain import state_dict_hash
from layercake_extensions.bpe_direct_neural_core import (
    Utf8ConcatenativeBpeTokenizer,
)
from layercake_extensions.route_isolated_format_literal_core_v22 import (
    FORMAT_LITERAL_FEATURE,
)
from layercake_extensions.route_isolated_lexical_guard_core_v21 import (
    EXACT_LEXICAL_GUARD_FEATURE,
)
from layercake_extensions.route_isolated_prompt_span_core_v19 import (
    PROMPT_SPAN_FEATURE,
)
from layercake_extensions.route_isolated_runtime_residency_core_v23 import (
    SINGLE_PARSE_ACTIVATION_FEATURE,
    RuntimeResidencyFormatLiteralCoreHost,
    _SingleParseCakeInstaller,
)
from layercake_extensions.route_isolated_shallow_sparse_core import (
    CAPABILITIES,
    WEAK_CAPABILITIES,
    RouteIsolatedCoreError,
    SparseCapabilityRouter,
    _DeclaredTokenizer,
)
from layercake_extensions.route_isolated_universal_guard_core_v20 import (
    UNIVERSAL_GUARD_FEATURE,
)


ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_VERSION = "lc-direct-neural-core/24"
ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_SHA256 = (
    "888768a57632eeb7b207616f1ef1331fdf3820b6d408a2ebcc4d741c10099553"
)
ARCHITECTURE_V24_FORMAT = "layercake-route-isolated-shallow-sparse-core/2"
ALLOCATION_BOUNDED_ADOPTION_FEATURE = "allocation_bounded_tensor_adoption"


def _adopt_authenticated_state(
    factory: Callable[[], torch.nn.Module],
    state: Mapping[str, torch.Tensor],
) -> tuple[torch.nn.Module, int, int]:
    """Create no initialized parameters, then strictly adopt authenticated storage."""

    with torch.device("meta"):
        module = factory()
    module.load_state_dict(dict(state), strict=True, assign=True)
    reconstructed = 0
    for name, value in list(module.named_buffers()):
        if value.device.type != "meta":
            continue
        if name.startswith("transformer.h.") and name.endswith(".attn.bias"):
            replacement = torch.tril(
                torch.ones(value.shape[-2:], dtype=torch.bool, device="cpu")
            ).view(value.shape)
        elif name.startswith("transformer.h.") and name.endswith(
            ".attn.masked_bias"
        ):
            replacement = torch.tensor(-1e4, dtype=value.dtype, device="cpu")
        else:
            raise RouteIsolatedCoreError(
                f"v24 cannot reconstruct non-persistent meta buffer: {name}"
            )
        parent = module
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], replacement)
        reconstructed += 1
    values = module.state_dict()
    if any(
        value.device.type == "meta"
        for value in (*module.parameters(), *module.buffers())
    ):
        raise RouteIsolatedCoreError("v24 strict adoption left a meta tensor")
    aliases = sum(
        values[name].untyped_storage().data_ptr()
        == state[name].untyped_storage().data_ptr()
        for name in state
    )
    if aliases != len(state):
        raise RouteIsolatedCoreError("v24 did not adopt authenticated tensor storage")
    return module, aliases, reconstructed


class AllocationBoundedRuntimeResidencyCoreHost(
    RuntimeResidencyFormatLiteralCoreHost
):
    """V24 isolates activation allocation changes from every sealed host."""

    ABI_VERSION = ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_VERSION
    ABI_SHA256 = ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_SHA256
    ARCHITECTURE_FORMAT = ARCHITECTURE_V24_FORMAT

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
        if ALLOCATION_BOUNDED_ADOPTION_FEATURE not in required:
            raise RouteIsolatedCoreError("v24 package omits allocation-bounded adoption")

    def activate(self, source: str | Path) -> dict[str, Any]:
        record, package = self.installer.inspect_install(
            source,
            validator=self._validate_role,
        )
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
            raise RouteIsolatedCoreError("v24 router configuration changed")
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
            or int(residual_config["routes"]) != len(WEAK_CAPABILITIES)
        ):
            raise RouteIsolatedCoreError("v24 residual geometry changed")
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
        if (
            set(model_tokenizer)
            != {"format", "tokenizers_json", "sha256", "eos_token_id"}
            or model_tokenizer["format"] != "declarative-tokenizers-json/1"
        ):
            raise RouteIsolatedCoreError("v24 model tokenizer declaration changed")
        canonical_tokenizer = json.dumps(
            model_tokenizer["tokenizers_json"], sort_keys=True, separators=(",", ":")
        ).encode()
        if hashlib.sha256(canonical_tokenizer).hexdigest() != model_tokenizer["sha256"]:
            raise RouteIsolatedCoreError("v24 model tokenizer hash mismatch")
        declared = _DeclaredTokenizer(
            model_tokenizer["tokenizers_json"], int(model_tokenizer["eos_token_id"])
        )
        router_tokenizer = Utf8ConcatenativeBpeTokenizer.from_document(
            architecture["router_tokenizer"]
        )
        if router_tokenizer.vocab_size != int(router_config["vocabulary"]):
            raise RouteIsolatedCoreError("v24 router tokenizer vocabulary mismatch")
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
            "reconstructed_nonpersistent_buffers": model_buffers
            + router_buffers
            + residual_buffers,
        }
