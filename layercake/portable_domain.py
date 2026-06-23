"""Versioned, core-independent domain payloads for exact transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import torch
from torch import nn

from .canonical_anchors import causal_byte_anchors


PORTABLE_DOMAIN_FORMAT = "layercake-portable-domain/1"
CANONICAL_ANCHOR_VERSION = "lc-causal-byte-anchor/1"


def canonical_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PortableDomainSpec:
    domain_id: str
    feature_width: int = 64
    hidden_width: int = 256
    architecture: str = "anchor_mlp"
    embedding_width: int = 64
    format_version: str = PORTABLE_DOMAIN_FORMAT
    anchor_version: str = CANONICAL_ANCHOR_VERSION
    input_mode: str = "byte"
    byte_vocab_size: int = 256
    quantization: str = "fp32"

    def __post_init__(self) -> None:
        if not self.domain_id:
            raise ValueError("domain_id must be non-empty")
        if self.feature_width <= 0 or self.hidden_width <= 0:
            raise ValueError("decoder widths must be positive")
        if self.architecture not in {"anchor_mlp", "byte_gru"}:
            raise ValueError(f"unsupported decoder architecture: {self.architecture}")
        if self.embedding_width <= 0:
            raise ValueError("embedding_width must be positive")
        if self.format_version != PORTABLE_DOMAIN_FORMAT:
            raise ValueError(f"unsupported format: {self.format_version}")
        if self.anchor_version != CANONICAL_ANCHOR_VERSION:
            raise ValueError(f"unsupported anchor contract: {self.anchor_version}")
        if self.input_mode != "byte" or self.byte_vocab_size != 256:
            raise ValueError("portable domain v1 requires raw 256-value bytes")
        if self.quantization not in {"fp32", "int8_symmetric_per_tensor"}:
            raise ValueError(f"unsupported quantization: {self.quantization}")

    def canonical_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return canonical_json_hash(self.canonical_dict())


class PortableDomainDecoder(nn.Module):
    """Predict next bytes from deterministic causal anchors only.

    Host-core weights are intentionally absent from this path. Two runtimes loading
    the same verified artifact therefore produce identical logits and greedy output.
    """

    def __init__(
        self,
        feature_width: int = 64,
        hidden_width: int = 256,
        architecture: str = "anchor_mlp",
        embedding_width: int = 64,
        *,
        d_abi: int | None = None,
        hidden: int | None = None,
    ):
        super().__init__()
        # Legacy aliases keep early research artifacts loadable.
        self.feature_width = d_abi if d_abi is not None else feature_width
        self.hidden_width = hidden if hidden is not None else hidden_width
        self.architecture = architecture
        self.embedding_width = embedding_width
        if architecture == "anchor_mlp":
            self.decoder = nn.Sequential(
                nn.LayerNorm(self.feature_width),
                nn.Linear(self.feature_width, self.hidden_width),
                nn.GELU(),
                nn.Linear(self.hidden_width, 256),
            )
        elif architecture == "byte_gru":
            self.byte_embedding = nn.Embedding(256, embedding_width)
            self.recurrent = nn.GRU(
                embedding_width + self.feature_width,
                self.hidden_width,
                batch_first=True,
            )
            self.decoder = nn.Sequential(
                nn.LayerNorm(self.hidden_width),
                nn.Linear(self.hidden_width, 256),
            )
        else:
            raise ValueError(f"unsupported decoder architecture: {architecture}")

    @property
    def d_abi(self) -> int:
        return self.feature_width

    @property
    def hidden(self) -> int:
        return self.hidden_width

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        anchors = causal_byte_anchors(byte_ids, self.feature_width)
        if self.architecture == "byte_gru":
            embedded = self.byte_embedding(byte_ids)
            hidden, _ = self.recurrent(torch.cat([embedded, anchors], dim=-1))
            return self.decoder(hidden)
        return self.decoder(anchors)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def quantized_state_hash(quantized_state: Mapping[str, dict]) -> str:
    digest = hashlib.sha256()
    for name in sorted(quantized_state):
        item = quantized_state[name]
        values = item["values"].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(values.numpy().tobytes())
        digest.update(float(item["scale"]).hex().encode("ascii"))
    return digest.hexdigest()


def build_portable_artifact(
    model: PortableDomainDecoder,
    spec: PortableDomainSpec,
    *,
    training: dict | None = None,
    evaluation: dict | None = None,
) -> dict:
    if model.feature_width != spec.feature_width:
        raise ValueError("model feature width does not match artifact spec")
    if model.hidden_width != spec.hidden_width:
        raise ValueError("model hidden width does not match artifact spec")
    if model.architecture != spec.architecture:
        raise ValueError("model architecture does not match artifact spec")
    if model.embedding_width != spec.embedding_width:
        raise ValueError("model embedding width does not match artifact spec")
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    artifact = {
        "format": PORTABLE_DOMAIN_FORMAT,
        "spec": spec.canonical_dict(),
        "spec_hash": spec.hash(),
        "training": training or {},
        "evaluation": evaluation or {},
    }
    if spec.quantization == "fp32":
        artifact["state_dict"] = state
        artifact["payload_hash"] = state_dict_hash(state)
        return artifact
    quantized_state = {}
    for name, tensor in state.items():
        scale = max(tensor.abs().max().item() / 127.0, 1e-12)
        quantized_state[name] = {
            "values": (tensor / scale).round().clamp(-127, 127).to(torch.int8),
            "scale": scale,
        }
    artifact["quantized_state"] = quantized_state
    artifact["payload_hash"] = quantized_state_hash(quantized_state)
    return artifact


def load_portable_artifact(
    artifact: dict, device: torch.device | str = "cpu"
) -> tuple[PortableDomainSpec, PortableDomainDecoder]:
    if artifact.get("format") != PORTABLE_DOMAIN_FORMAT:
        raise ValueError("unsupported portable domain artifact")
    raw_spec = artifact["spec"]
    spec = PortableDomainSpec(**raw_spec)
    if artifact.get("spec_hash") not in {
        spec.hash(),
        canonical_json_hash(raw_spec),
    }:
        raise ValueError("portable domain spec hash mismatch")
    if spec.quantization == "fp32":
        if artifact.get("payload_hash") != state_dict_hash(artifact["state_dict"]):
            raise ValueError("portable domain payload hash mismatch")
        state_dict = artifact["state_dict"]
    else:
        quantized_state = artifact["quantized_state"]
        if artifact.get("payload_hash") != quantized_state_hash(quantized_state):
            raise ValueError("portable domain payload hash mismatch")
        state_dict = {
            name: item["values"].float() * float(item["scale"])
            for name, item in quantized_state.items()
        }
    model = PortableDomainDecoder(
        feature_width=spec.feature_width,
        hidden_width=spec.hidden_width,
        architecture=spec.architecture,
        embedding_width=spec.embedding_width,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return spec, model


def quantize_portable_artifact(artifact: dict) -> dict:
    spec, model = load_portable_artifact(artifact, "cpu")
    if spec.quantization != "fp32":
        raise ValueError("only fp32 portable artifacts can be quantized")
    quantized_spec = PortableDomainSpec(
        **{**spec.canonical_dict(), "quantization": "int8_symmetric_per_tensor"}
    )
    return build_portable_artifact(
        model,
        quantized_spec,
        training=artifact.get("training"),
        evaluation=artifact.get("evaluation"),
    )


def artifact_payload_bytes(artifact: dict) -> int:
    if "state_dict" in artifact:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in artifact["state_dict"].values()
        )
    return sum(
        item["values"].numel() * item["values"].element_size() + 8
        for item in artifact["quantized_state"].values()
    )


class LayerCakeRuntime:
    """Runtime selecting either host-core or exact portable-domain predictions."""

    def __init__(self, core: nn.Module | None = None):
        self.core = core
        self.domains: dict[str, tuple[PortableDomainSpec, PortableDomainDecoder]] = {}

    def install_portable_domain(
        self, artifact: dict, device: torch.device | str = "cpu"
    ) -> str:
        spec, decoder = load_portable_artifact(artifact, device)
        self.domains[spec.domain_id] = (spec, decoder)
        return spec.domain_id

    def logits(self, byte_ids: torch.Tensor, *, domain_id: str | None = None):
        if domain_id is not None:
            try:
                return self.domains[domain_id][1](byte_ids)
            except KeyError as exc:
                raise KeyError(f"portable domain is not installed: {domain_id}") from exc
        if self.core is None:
            raise ValueError("base inference requires a host core")
        output = self.core(byte_ids)
        return output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def generate(
        self,
        prompt: bytes | str | torch.Tensor,
        *,
        max_new_bytes: int,
        domain_id: str,
        context_bytes: int = 256,
    ) -> torch.Tensor:
        if isinstance(prompt, str):
            prompt = prompt.encode("utf-8")
        if isinstance(prompt, bytes):
            prompt = torch.tensor(list(prompt), dtype=torch.long).unsqueeze(0)
        if prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError("generation currently requires one [1, sequence] prompt")
        decoder = self.domains[domain_id][1]
        device = next(decoder.parameters()).device
        generated = prompt.to(device)
        for _ in range(max_new_bytes):
            context = generated[:, -context_bytes:]
            next_byte = decoder(context)[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_byte], dim=1)
        return generated
