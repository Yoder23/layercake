"""Isolated v22-compatible host with single-parse activation and route reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import torch

from layercake.cake.installer import CakeInstaller, HostCapabilities, InstallationError
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
    FormatLiteralLexicalGuardCoreHost,
)
from layercake_extensions.route_isolated_lexical_guard_core_v21 import (
    EXACT_LEXICAL_GUARD_FEATURE,
)
from layercake_extensions.route_isolated_prompt_span_core_v19 import (
    PROMPT_SPAN_FEATURE,
)
from layercake_extensions.route_isolated_shallow_sparse_core import (
    CAPABILITIES,
    WEAK_CAPABILITIES,
    SparseCapabilityRouter,
    RouteIsolatedCoreError,
    _DeclaredTokenizer,
)
from layercake_extensions.route_isolated_universal_guard_core_v20 import (
    UNIVERSAL_GUARD_FEATURE,
)


ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_VERSION = "lc-direct-neural-core/23"
ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_SHA256 = (
    "a5ea993fab864ee4730dd5038cf5c54f570491d3c4acf4c1b7aec8220463bc9c"
)
ARCHITECTURE_V23_FORMAT = "layercake-route-isolated-shallow-sparse-core/2"
SINGLE_PARSE_ACTIVATION_FEATURE = "single_authenticated_package_activation"


class _SingleParseCakeInstaller(CakeInstaller):
    """Keep authenticated package ownership inside one inspect/install call."""

    def inspect_install(
        self,
        source: str | Path,
        *,
        validator: Callable[[CakePackage], None],
    ) -> tuple[dict[str, Any], CakePackage]:
        package = self.inspect(source)
        validator(package)
        self._validate_host(package)
        if self.strict_signatures and not package.signed:
            raise InstallationError("single-parse activation requires a signed package")
        manifest = package.manifest
        for dependency in manifest.dependencies:
            if self.registry.get(dependency) is None:
                raise InstallationError(f"missing cake dependency: {dependency}")
        blob = self.registry.store_blob(package.path, package.archive_hash)
        record = {
            "cake_id": manifest.cake_id,
            "name": manifest.name,
            "description": manifest.description,
            "version": manifest.version,
            "cake_type": manifest.cake_type,
            "abi_version": manifest.abi_version,
            "abi_hash": manifest.abi_hash,
            "archive_hash": package.archive_hash,
            "package_hash": manifest.package_hash,
            "tensor_payload_hash": manifest.tensor_payload_hash,
            "signed": package.signed,
            "trusted_local": False,
            "publisher": manifest.publisher,
            "domains": list(manifest.domains),
            "keywords": list(manifest.keywords),
            "permissions": list(manifest.permissions),
            "composition": manifest.output_contract.get(
                "composition", manifest.output_contract.get("combination", "none")
            ),
            "installed_at": time.time(),
            "blob": str(blob),
        }
        previous = self.registry.activate(record)
        return {"status": "UPDATED" if previous else "INSTALLED", **record}, package


class RuntimeResidencyFormatLiteralCoreHost(FormatLiteralLexicalGuardCoreHost):
    """V23 isolates package-residency optimization from every sealed host."""

    ABI_VERSION = ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_VERSION
    ABI_SHA256 = ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_SHA256
    ARCHITECTURE_FORMAT = ARCHITECTURE_V23_FORMAT

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
                    }
                ),
            ),
            trust_store=trust_store,
            strict_signatures=True,
        )
        self._route_state_key: tuple[int, int] | None = None
        self._route_state_tensor: torch.Tensor | None = None

    @classmethod
    def _validate_role(cls, package: CakePackage) -> None:
        super()._validate_role(package)
        required = set(package.manifest.minimum_host_capabilities.get("features", ()))
        if SINGLE_PARSE_ACTIVATION_FEATURE not in required:
            raise RouteIsolatedCoreError("v23 package omits single-parse activation")

    def _assign_residual_route(self, route: int, batch: int) -> None:
        if self.model is None:
            raise RouteIsolatedCoreError("v23 core is inactive")
        key = (int(route), int(batch))
        if (
            self._route_state_key == key
            and self._route_state_tensor is not None
            and all(
                getattr(block, "_layercake_residual_routes", None)
                is self._route_state_tensor
                for block in self.model.transformer.h
            )
        ):
            return
        value = torch.full((batch,), route, dtype=torch.long, device=self.device)
        for block in self.model.transformer.h:
            block._layercake_residual_routes = value
        self._route_state_key = key
        self._route_state_tensor = value

    def _set_residual_route(self, route: int) -> None:
        self._assign_residual_route(route, 1)

    def _set_residual_route_batch(self, route: int, batch: int) -> None:
        self._assign_residual_route(route, batch)

    def activate(self, source: str | Path) -> dict[str, Any]:
        record, package = self.installer.inspect_install(
            source,
            validator=self._validate_role,
        )
        architecture = self._architecture(package)
        model_config = ShallowSparseEnglishConfig(**architecture["model"])
        model = ShallowSparseEnglishCore(model_config)
        model.load_state_dict(self._namespace(package.tensors, "model."), strict=True)
        router_config = architecture["router"]
        if set(router_config) != {
            "vocabulary",
            "character_hash_buckets",
            "character_ngram_minimum",
            "character_ngram_maximum",
            "hash_seed",
            "classes",
        } or int(router_config["classes"]) != len(CAPABILITIES) + 1:
            raise RouteIsolatedCoreError("v23 router configuration changed")
        router = SparseCapabilityRouter(
            int(router_config["vocabulary"]),
            int(router_config["character_hash_buckets"]),
            int(router_config["classes"]),
        )
        router.load_state_dict(self._namespace(package.tensors, "router."), strict=True)
        residual_config = architecture["residual"]
        if (
            set(residual_config) != {"width", "rank", "routes", "reuse"}
            or residual_config["reuse"] != "before_each_transformer_block"
            or int(residual_config["width"]) != model_config.width
            or int(residual_config["rank"]) != 16
            or int(residual_config["routes"]) != len(WEAK_CAPABILITIES)
        ):
            raise RouteIsolatedCoreError("v23 residual geometry changed")
        residual = self.RESIDUAL_TYPE(
            int(residual_config["width"]),
            int(residual_config["rank"]),
            int(residual_config["routes"]),
        )
        residual.load_state_dict(
            self._namespace(package.tensors, "residual."), strict=True
        )
        model_tokenizer = architecture["model_tokenizer"]
        if (
            set(model_tokenizer)
            != {"format", "tokenizers_json", "sha256", "eos_token_id"}
            or model_tokenizer["format"] != "declarative-tokenizers-json/1"
        ):
            raise RouteIsolatedCoreError("v23 model tokenizer declaration changed")
        canonical_tokenizer = json.dumps(
            model_tokenizer["tokenizers_json"], sort_keys=True, separators=(",", ":")
        ).encode()
        if hashlib.sha256(canonical_tokenizer).hexdigest() != model_tokenizer["sha256"]:
            raise RouteIsolatedCoreError("v23 model tokenizer hash mismatch")
        declared = _DeclaredTokenizer(
            model_tokenizer["tokenizers_json"], int(model_tokenizer["eos_token_id"])
        )
        router_tokenizer = Utf8ConcatenativeBpeTokenizer.from_document(
            architecture["router_tokenizer"]
        )
        if router_tokenizer.vocab_size != int(router_config["vocabulary"]):
            raise RouteIsolatedCoreError("v23 router tokenizer vocabulary mismatch")
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
        return {
            "status": "ACTIVE",
            "cake_id": self.active_cake_id,
            "archive_hash": self.active_archive_hash,
            "payload_hash": self.active_payload_hash,
            "state_dict_hash": state_dict_hash(package.tensors),
            "device": str(self.device),
            "receiver_training_steps": 0,
            "receiver_calibration_runs": 0,
            "authenticated_package_parses": 1,
        }

    def remove(self) -> dict[str, Any]:
        result = super().remove()
        self._route_state_key = None
        self._route_state_tensor = None
        return result
