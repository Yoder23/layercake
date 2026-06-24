from __future__ import annotations

from pathlib import Path
import hashlib
import tempfile

import torch
from torch import nn

from .common import file_sha256


class ModuleRegistry:
    def __init__(self) -> None:
        self.modules: dict[str, nn.Module] = {}

    def register(self, name: str, module: nn.Module) -> None:
        self.modules[name] = module

    def list_modules(self) -> list[str]:
        return sorted(self.modules)

    def freeze(self, name: str) -> None:
        for parameter in self.modules[name].parameters():
            parameter.requires_grad_(False)

    def unfreeze(self, name: str) -> None:
        for parameter in self.modules[name].parameters():
            parameter.requires_grad_(True)

    def hash_module(self, name: str) -> str:
        digest = hashlib.sha256()
        state = self.modules[name].state_dict()
        for key in sorted(state):
            tensor = state[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def module_hashes(self) -> dict[str, str]:
        return {name: self.hash_module(name) for name in self.list_modules()}

    def module_hash(self, name: str) -> str:
        return self.hash_module(name)

    def save_module(self, name: str, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.modules[name].state_dict(), path)
        return str(path)

    def load_module(self, name: str, path: str | Path) -> None:
        self.modules[name].load_state_dict(torch.load(path, map_location="cpu"))

    def restore_module(self, name: str, path: str | Path) -> None:
        self.load_module(name, path)

    def restore_module_from_commit(self, name: str, commit) -> None:
        self.load_module(name, commit.artifact_paths[name])

    def compare_module_hashes(self, a, b) -> dict[str, tuple[str | None, str | None]]:
        names = sorted(set(a.module_hashes) | set(b.module_hashes))
        return {
            name: (a.module_hashes.get(name), b.module_hashes.get(name))
            for name in names
            if a.module_hashes.get(name) != b.module_hashes.get(name)
        }

    def trainable_parameter_count(self) -> int:
        return sum(
            p.numel()
            for module in self.modules.values()
            for p in module.parameters()
            if p.requires_grad
        )

    def total_parameter_count(self) -> int:
        return sum(p.numel() for module in self.modules.values() for p in module.parameters())
