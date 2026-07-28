"""Manual, sparse execution host for byte-facing direct neural cakes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from layercake.cake.installer import (
    CakeInstaller,
    HostCapabilities,
)
from layercake.cake.package import load_package
from layercake.cake.registry import CakeRegistry

from .portable_decoder import load_cake_module


@dataclass(frozen=True)
class DirectCakeResult:
    cake_id: str
    output: bytes
    actions: tuple[int, ...]
    prefill_calls: int
    decode_step_calls: int


class DirectCakeHost:
    """Install many direct cakes while executing only an explicit selection.

    This class deliberately has no router. Discovery, ranking, top-k, and
    composition remain separate orchestration concerns.
    """

    def __init__(
        self,
        registry_root: str | Path,
        *,
        abi_version: str,
        abi_hash: str,
        trust_store: Mapping[str, bytes | str | Path],
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        backends = (
            ("pytorch", "cuda")
            if self.device.type == "cuda"
            else ("pytorch",)
        )
        self.registry = CakeRegistry(registry_root)
        self.installer = CakeInstaller(
            self.registry,
            HostCapabilities(
                abi_version=abi_version,
                abi_hash=abi_hash,
                precisions=("fp32",),
                backends=backends,
                capabilities=frozenset(
                    {"byte_input", "safe_tensors", "incremental"}
                ),
            ),
            trust_store=trust_store,
        )
        self._trust_store = dict(trust_store)
        self._models: dict[str, torch.nn.Module] = {}
        self._telemetry: dict[str, dict[str, int]] = {}

    def install(self, source: str | Path) -> dict[str, Any]:
        record = self.installer.install(source)
        cake_id = str(record["cake_id"])
        self._models.pop(cake_id, None)
        self._telemetry.setdefault(
            cake_id,
            {
                "module_load_calls": 0,
                "prefill_calls": 0,
                "decode_step_calls": 0,
            },
        )
        return record

    def remove(self, cake_id: str) -> dict[str, Any]:
        record = self.installer.remove(cake_id)
        self._models.pop(cake_id, None)
        return record

    def installed_ids(self) -> tuple[str, ...]:
        return tuple(row["cake_id"] for row in self.registry.list())

    def _load_selected(self, cake_id: str) -> torch.nn.Module:
        if cake_id in self._models:
            return self._models[cake_id]
        record = self.registry.get(cake_id)
        if record is None:
            raise KeyError(f"direct cake is not installed: {cake_id}")
        package = load_package(
            self.registry.blob_path(record["archive_hash"]),
            trust_store=self._trust_store,
        )
        if (
            package.manifest.cake_type != "portable_decoder"
            or package.manifest.input_contract.get("mode")
            != "direct_selected_portable_decoder"
        ):
            raise ValueError("package is not a direct neural decoder")
        model = load_cake_module(package).to(self.device).eval()
        self._models[cake_id] = model
        self._telemetry[cake_id]["module_load_calls"] += 1
        return model

    @torch.inference_mode()
    def generate(
        self,
        cake_id: str,
        prompt: bytes | str,
        *,
        maximum_actions: int | None = None,
    ) -> DirectCakeResult:
        model = self._load_selected(cake_id)
        if not all(
            hasattr(model, name)
            for name in ("prefill_bytes", "decode_step", "tokenizer")
        ):
            raise TypeError(
                "selected direct cake lacks incremental byte execution"
            )
        counters = self._telemetry[cake_id]
        state = model.prefill_bytes(prompt)
        counters["prefill_calls"] += 1
        limit = (
            model.maximum_target_actions
            if maximum_actions is None
            else min(
                int(maximum_actions), model.maximum_target_actions
            )
        )
        steps = 0
        while not state.complete and steps < limit:
            model.decode_step(state)
            counters["decode_step_calls"] += 1
            steps += 1
        output = model.tokenizer.decode_actions(
            state.generated_actions, state.source_lexemes
        )
        return DirectCakeResult(
            cake_id=cake_id,
            output=output,
            actions=tuple(state.generated_actions),
            prefill_calls=1,
            decode_step_calls=steps,
        )

    def telemetry(self) -> dict[str, dict[str, int]]:
        return {
            cake_id: dict(values)
            for cake_id, values in sorted(self._telemetry.items())
        }

    def reset_telemetry(self) -> None:
        for values in self._telemetry.values():
            for key in values:
                values[key] = 0
