"""Serializable state for constant-work autoregressive byte decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors


STATE_FORMAT = "layercake-incremental-state/1"


@dataclass
class IncrementalState:
    """All mutable data required to predict the next byte.

    Tensors are batch-first except ``recurrent_hidden``, which follows the
    PyTorch GRU ``[layers, batch, width]`` convention.  The state deliberately
    contains no retained prompt tensor.
    """

    local_history: torch.Tensor
    incomplete_patch: torch.Tensor
    recurrent_hidden: torch.Tensor
    completed_patch_summary: torch.Tensor
    canonical_state: torch.Tensor
    next_logits: torch.Tensor | None
    generated_bytes: torch.Tensor
    route: int
    model_fingerprint: str
    prompt_bytes: int = 0
    decoded_bytes: int = 0
    sampler: str = "greedy"
    temperature: float = 1.0
    top_k: int = 0
    sampler_seed: int = 0
    sampler_counter: int = 0
    active_cake: str | None = None
    capture_generated: bool = False
    cake_state: torch.Tensor | None = None
    verifier_state: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.local_history.shape[0])

    @property
    def state_bytes(self) -> int:
        tensors = self.tensor_dict()
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())

    def tensor_dict(self) -> dict[str, torch.Tensor]:
        tensors = {
            "local_history": self.local_history,
            "incomplete_patch": self.incomplete_patch,
            "recurrent_hidden": self.recurrent_hidden,
            "completed_patch_summary": self.completed_patch_summary,
            "canonical_state": self.canonical_state,
            "generated_bytes": self.generated_bytes,
        }
        if self.next_logits is not None:
            tensors["next_logits"] = self.next_logits
        if self.cake_state is not None:
            tensors["cake_state"] = self.cake_state
        return {name: tensor.detach().contiguous().cpu() for name, tensor in tensors.items()}


def fingerprint_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def restore_incremental_state(
    payload: bytes,
    *,
    expected_model_fingerprint: str,
    device: torch.device | str,
) -> IncrementalState:
    tensors = load_safetensors(payload)
    # safetensors.torch.load does not expose metadata, so the JSON contract is
    # stored as a uint8 tensor as well as archive metadata.
    metadata_tensor = tensors.pop("__metadata_json", None)
    if metadata_tensor is None:
        raise ValueError("incremental state metadata is missing")
    try:
        metadata = json.loads(bytes(metadata_tensor.tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("incremental state metadata is invalid") from error
    if metadata.get("format") != STATE_FORMAT:
        raise ValueError("unsupported incremental state format")
    if metadata.get("model_fingerprint") != expected_model_fingerprint:
        raise ValueError("incremental state belongs to different model weights")
    required = {
        "local_history", "incomplete_patch", "recurrent_hidden",
        "completed_patch_summary", "canonical_state", "generated_bytes",
    }
    if not required.issubset(tensors):
        raise ValueError("incremental state tensor set is incomplete")
    moved = {name: tensor.to(device) for name, tensor in tensors.items()}
    return IncrementalState(
        local_history=moved["local_history"],
        incomplete_patch=moved["incomplete_patch"],
        recurrent_hidden=moved["recurrent_hidden"],
        completed_patch_summary=moved["completed_patch_summary"],
        canonical_state=moved["canonical_state"],
        next_logits=moved.get("next_logits"),
        generated_bytes=moved["generated_bytes"],
        cake_state=moved.get("cake_state"),
        route=int(metadata["route"]),
        model_fingerprint=str(metadata["model_fingerprint"]),
        prompt_bytes=int(metadata["prompt_bytes"]),
        decoded_bytes=int(metadata["decoded_bytes"]),
        sampler=str(metadata["sampler"]),
        temperature=float(metadata["temperature"]),
        top_k=int(metadata["top_k"]),
        sampler_seed=int(metadata["sampler_seed"]),
        sampler_counter=int(metadata["sampler_counter"]),
        active_cake=metadata.get("active_cake"),
        capture_generated=bool(metadata.get("capture_generated", False)),
        verifier_state=dict(metadata.get("verifier_state", {})),
    )


def serialize_state_with_metadata(state: IncrementalState) -> bytes:
    """Serialize state without pickle or executable object reconstruction."""

    metadata = {
        "format": STATE_FORMAT,
        "route": state.route,
        "model_fingerprint": state.model_fingerprint,
        "prompt_bytes": state.prompt_bytes,
        "decoded_bytes": state.decoded_bytes,
        "sampler": state.sampler,
        "temperature": state.temperature,
        "top_k": state.top_k,
        "sampler_seed": state.sampler_seed,
        "sampler_counter": state.sampler_counter,
        "active_cake": state.active_cake,
        "capture_generated": state.capture_generated,
        "verifier_state": state.verifier_state,
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    tensors = state.tensor_dict()
    tensors["__metadata_json"] = torch.tensor(list(encoded), dtype=torch.uint8)
    return save_safetensors(tensors)
