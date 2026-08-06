"""Signed direct neural English-core host role for immutable LayerCake artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from layercake.cake.installer import CakeInstaller, HostCapabilities, InstallationError
from layercake.cake.package import CakePackage, load_package
from layercake.cake.registry import CakeRegistry
from layercake.models.portable_decoder import load_cake_module
from layercake.portable_domain import state_dict_hash


DIRECT_NEURAL_CORE_ABI_VERSION = "lc-direct-neural-core/1"
DIRECT_NEURAL_CORE_ABI_SHA256 = "ed32179655dc8991610a1b47e55d8c8ccc80bb7f35f74764cbd54e79bc520213"
DIRECT_NEURAL_CORE_ROLE = "english-core"
DIRECT_NEURAL_CORE_COMPOSITION = "direct_core_only_no_router"


class DirectNeuralCoreError(ValueError):
    """Raised when a package cannot safely occupy the direct core role."""


class DirectNeuralCoreHost:
    """Install and execute one signed self-causal core without receiver learning."""

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
                abi_version=DIRECT_NEURAL_CORE_ABI_VERSION,
                abi_hash=DIRECT_NEURAL_CORE_ABI_SHA256,
                precisions=("fp32",),
                backends=("pytorch", "cuda"),
                capabilities=frozenset(
                    {"byte_input", "safe_tensors", "persistent_incremental_state"}
                ),
            ),
            trust_store=trust_store,
            strict_signatures=True,
        )
        self.device = torch.device(device)
        self.module: nn.Module | None = None
        self.active_cake_id: str | None = None
        self.active_archive_hash: str | None = None
        self.active_payload_hash: str | None = None
        self.receiver_training_steps = 0
        self.receiver_calibration_runs = 0

    @staticmethod
    def _validate_role(package: CakePackage) -> None:
        manifest = package.manifest
        if not package.signed:
            raise DirectNeuralCoreError("direct neural core must be signed")
        if manifest.cake_type != "portable_decoder":
            raise DirectNeuralCoreError("direct neural core requires a portable decoder")
        if manifest.abi_version != DIRECT_NEURAL_CORE_ABI_VERSION or manifest.abi_hash != DIRECT_NEURAL_CORE_ABI_SHA256:
            raise DirectNeuralCoreError("direct neural core ABI identity mismatch")
        if manifest.domains != (DIRECT_NEURAL_CORE_ROLE,):
            raise DirectNeuralCoreError("package is not exclusively marked as the English core")
        if manifest.dependencies:
            raise DirectNeuralCoreError("direct neural core cannot depend on another package")
        if manifest.input_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE}:
            raise DirectNeuralCoreError("direct neural core input contract mismatch")
        if manifest.output_contract != {
            "external": "UTF-8 bytes",
            "role": DIRECT_NEURAL_CORE_ROLE,
            "composition": DIRECT_NEURAL_CORE_COMPOSITION,
        }:
            raise DirectNeuralCoreError("direct neural core output contract mismatch")
        architecture = manifest.architecture
        if architecture.get("name") != "portable_token_plan" or architecture.get("private_representation") != "portable_token_plan_pointer_transformer":
            raise DirectNeuralCoreError("direct neural core must use the self-causal portable token plan")
        required = set(manifest.minimum_host_capabilities.get("features", []))
        if not {"byte_input", "safe_tensors", "persistent_incremental_state"} <= required:
            raise DirectNeuralCoreError("direct neural core host capabilities are incomplete")

    def activate(self, source: str | Path) -> dict[str, Any]:
        inspected = self.installer.inspect(source)
        self._validate_role(inspected)
        record = self.installer.install(source)
        installed = load_package(
            record["blob"],
            trust_store=self.installer.trust_store,
            require_signature=True,
        )
        self._validate_role(installed)
        module = load_cake_module(installed).to(self.device).eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        self.module = module
        self.active_cake_id = installed.manifest.cake_id
        self.active_archive_hash = installed.archive_hash
        self.active_payload_hash = installed.manifest.tensor_payload_hash
        return {
            "status": "ACTIVE",
            "cake_id": self.active_cake_id,
            "archive_hash": self.active_archive_hash,
            "payload_hash": self.active_payload_hash,
            "state_dict_hash": state_dict_hash(module.state_dict()),
            "device": str(self.device),
            "receiver_training_steps": self.receiver_training_steps,
            "receiver_calibration_runs": self.receiver_calibration_runs,
        }

    def _require_active(self) -> nn.Module:
        if self.module is None or self.active_cake_id is None:
            raise DirectNeuralCoreError("no direct neural core is active")
        return self.module

    def prefill(self, prompt: bytes | str):
        module = self._require_active()
        return module.prefill_bytes(prompt)

    def decode_step(self, state):
        module = self._require_active()
        return module.decode_step(state)

    def generate(self, prompt: bytes | str, *, maximum_actions: int | None = None) -> bytes:
        module = self._require_active()
        return module.generate_bytes(prompt, maximum_actions=maximum_actions)

    def verify(self) -> dict[str, Any]:
        if self.active_cake_id is None:
            raise DirectNeuralCoreError("no direct neural core is active")
        result = self.installer.verify(self.active_cake_id)
        if result["archive_hash"] != self.active_archive_hash or result["payload_hash"] != self.active_payload_hash:
            raise DirectNeuralCoreError("active direct neural core identity changed")
        return {**result, "role": DIRECT_NEURAL_CORE_ROLE}

    def remove(self) -> dict[str, Any]:
        if self.active_cake_id is None:
            raise DirectNeuralCoreError("no direct neural core is active")
        cake_id = self.active_cake_id
        result = self.installer.remove(cake_id)
        self.module = None
        self.active_cake_id = None
        self.active_archive_hash = None
        self.active_payload_hash = None
        return result
