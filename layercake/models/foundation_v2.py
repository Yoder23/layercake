"""Quality-oriented sparse byte foundation with exact incremental execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from .canonical_interface import CanonicalByteInterface, CanonicalInterfaceConfig
from .incremental_state import fingerprint_state_dict
from .routed_cakes import Top1RoutedFoundationCakes


@dataclass(frozen=True)
class FoundationV2Config:
    d_byte: int = 48
    d_local: int = 96
    d_global: int = 128
    local_layers: int = 1
    local_kernel: int = 5
    fast_patch_size: int = 4
    slow_patch_size: int = 16
    global_layers: int = 1
    routed_experts: int = 16
    expert_expansion: int = 4
    abi_width: int = 64
    dropout: float = 0.0
    ablation: str = "selected"
    architecture_version: str = "layercake-foundation-v2/1"

    def __post_init__(self) -> None:
        values = (
            self.d_byte, self.d_local, self.d_global, self.local_layers,
            self.local_kernel, self.fast_patch_size, self.slow_patch_size,
            self.global_layers, self.routed_experts, self.expert_expansion, self.abi_width,
        )
        if min(values) <= 0 or self.local_kernel < 2 or self.routed_experts < 2:
            raise ValueError("invalid foundation-v2 configuration")
        if self.slow_patch_size % self.fast_patch_size:
            raise ValueError("slow patch size must be a multiple of fast patch size")
        allowed = {
            "selected", "dense", "sparse", "local_only", "global_only",
            "no_routed_experts", "fixed_patches", "multiscale_patches",
            "oracle_route", "learned_route",
        }
        if self.ablation not in allowed:
            raise ValueError(f"unsupported ablation: {self.ablation}")

    def canonical_dict(self) -> dict:
        return asdict(self)


class LearnedPatchSummary(nn.Module):
    def __init__(self, input_width: int, output_width: int):
        super().__init__()
        self.key = nn.Linear(input_width, max(16, input_width // 2), bias=False)
        self.score = nn.Linear(max(16, input_width // 2), 1, bias=False)
        self.value = nn.Linear(input_width, output_width, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(torch.tanh(self.key(values))), dim=-2)
        return torch.sum(weights * self.value(values), dim=-2)


@dataclass
class FoundationV2State:
    conv_history: torch.Tensor
    local_hidden: torch.Tensor
    fast_buffer: torch.Tensor
    slow_buffer: torch.Tensor
    fast_hidden: torch.Tensor
    slow_hidden: torch.Tensor
    fast_context: torch.Tensor
    slow_context: torch.Tensor
    anchor_state: torch.Tensor
    canonical_state: torch.Tensor
    next_core_logits: torch.Tensor | None
    next_logits: torch.Tensor | None
    generated_bytes: torch.Tensor
    route: int
    model_fingerprint: str
    prompt_bytes: int = 0
    decoded_bytes: int = 0
    sampler_seed: int = 0
    sampler_counter: int = 0
    cake_hidden: torch.Tensor | None = None
    active_cake: str | None = None
    capture_generated: bool = False

    @property
    def batch_size(self) -> int:
        return int(self.conv_history.shape[0])

    @property
    def state_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.tensors().values()
        )

    def tensors(self) -> dict[str, torch.Tensor]:
        values = {
            "conv_history": self.conv_history,
            "local_hidden": self.local_hidden,
            "fast_buffer": self.fast_buffer,
            "slow_buffer": self.slow_buffer,
            "fast_hidden": self.fast_hidden,
            "slow_hidden": self.slow_hidden,
            "fast_context": self.fast_context,
            "slow_context": self.slow_context,
            "anchor_state": self.anchor_state,
            "canonical_state": self.canonical_state,
            "generated_bytes": self.generated_bytes,
        }
        if self.next_core_logits is not None:
            values["next_core_logits"] = self.next_core_logits
        if self.next_logits is not None:
            values["next_logits"] = self.next_logits
        if self.cake_hidden is not None:
            values["cake_hidden"] = self.cake_hidden
        return values


class LayerCakeFoundationV2(nn.Module):
    """Byte-rate recurrent language trunk plus two sparse patch-rate paths."""

    def __init__(self, config: FoundationV2Config | None = None):
        super().__init__()
        self.config = config or FoundationV2Config()
        cfg = self.config
        self.byte_embedding = nn.Embedding(256, cfg.d_byte)
        self.local_in = nn.Linear(cfg.d_byte, cfg.d_local)
        self.local_norm = nn.LayerNorm(cfg.d_local)
        self.local_depthwise = nn.Conv1d(
            cfg.d_local, cfg.d_local, cfg.local_kernel, groups=cfg.d_local
        )
        self.local_mix = nn.Linear(cfg.d_local, cfg.d_local)
        self.local_recurrent = nn.GRU(
            cfg.d_local, cfg.d_local, num_layers=cfg.local_layers,
            batch_first=True, dropout=cfg.dropout if cfg.local_layers > 1 else 0.0,
        )
        self.fast_summary = LearnedPatchSummary(cfg.d_local, cfg.d_global)
        self.slow_summary = LearnedPatchSummary(cfg.d_local, cfg.d_global)
        self.fast_recurrent = nn.GRU(
            cfg.d_global, cfg.d_global, num_layers=cfg.global_layers, batch_first=True
        )
        self.slow_recurrent = nn.GRU(
            cfg.d_global, cfg.d_global, num_layers=cfg.global_layers, batch_first=True
        )
        self.experts = Top1RoutedFoundationCakes(
            cfg.d_global, cfg.routed_experts, expansion=cfg.expert_expansion
        )
        self.local_to_global = nn.Linear(cfg.d_local, cfg.d_global)
        self.output_norm = nn.LayerNorm(cfg.d_global)
        self.output = nn.Linear(cfg.d_global, 256, bias=False)
        self.canonical = CanonicalByteInterface(
            CanonicalInterfaceConfig(width=cfg.abi_width, logit_width=cfg.abi_width // 2)
        )
        self._model_fingerprint: str | None = None
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _pad_patches(values: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, int]:
        pad = (-values.shape[1]) % patch_size
        if pad:
            values = F.pad(values, (0, 0, 0, pad))
        return values.reshape(values.shape[0], -1, patch_size, values.shape[-1]), pad

    @staticmethod
    def _shift_expand(values: torch.Tensor, patch_size: int, length: int) -> torch.Tensor:
        zero = values.new_zeros(values.shape[0], 1, values.shape[-1])
        shifted = torch.cat([zero, values[:, :-1]], dim=1)
        return shifted.repeat_interleave(patch_size, dim=1)[:, :length]

    def set_route(self, route: int | None) -> None:
        self.experts.set_route(route)

    def forward(
        self,
        byte_ids: torch.Tensor,
        *,
        route: int | None = None,
        fusion_cake: nn.Module | None = None,
        return_aux: bool = False,
    ):
        if byte_ids.ndim != 2 or byte_ids.shape[1] == 0:
            raise ValueError("byte_ids must be non-empty [batch, sequence]")
        if route is not None:
            self.experts.set_route(route)
        embedded = self.byte_embedding(byte_ids.long())
        local_input = self.local_in(embedded)
        normalized = self.local_norm(local_input).transpose(1, 2)
        convolved = self.local_depthwise(
            F.pad(normalized, (self.config.local_kernel - 1, 0))
        ).transpose(1, 2)
        mixed = local_input + self.local_mix(F.silu(convolved))
        local, _ = self.local_recurrent(mixed)

        fast_patches, _ = self._pad_patches(local, self.config.fast_patch_size)
        slow_patches, _ = self._pad_patches(local, self.config.slow_patch_size)
        fast_summary = self.fast_summary(fast_patches)
        slow_summary = self.slow_summary(slow_patches)
        fast_global, _ = self.fast_recurrent(fast_summary)
        slow_global, _ = self.slow_recurrent(slow_summary)
        slow_for_fast = self._shift_expand(
            slow_global, self.config.slow_patch_size // self.config.fast_patch_size,
            fast_global.shape[1],
        )
        if self.config.ablation == "fixed_patches":
            slow_for_fast = torch.zeros_like(slow_for_fast)
        expert_input = fast_global + slow_for_fast
        if self.config.ablation == "no_routed_experts":
            routed = expert_input
            balance_loss = expert_input.new_zeros(())
        elif self.config.ablation == "dense":
            routed = torch.stack([expert(expert_input) for expert in self.experts.experts]).mean(dim=0)
            balance_loss = expert_input.new_zeros(())
        else:
            routed, balance_loss = self.experts(expert_input, return_aux_loss=True)
        fast_context = self._shift_expand(routed, self.config.fast_patch_size, byte_ids.shape[1])
        slow_context = self._shift_expand(slow_global, self.config.slow_patch_size, byte_ids.shape[1])
        if self.config.ablation in {"fixed_patches", "local_only"}:
            slow_context = torch.zeros_like(slow_context)
        if self.config.ablation == "local_only":
            fast_context = torch.zeros_like(fast_context)
        local_output = self.local_to_global(local)
        if self.config.ablation == "global_only":
            local_output = torch.zeros_like(local_output)
        core_logits = self.output(
            self.output_norm(local_output + fast_context + slow_context)
        )
        canonical = self.canonical(core_logits, byte_ids)
        logits = core_logits
        cake_hidden = None
        if fusion_cake is not None:
            logits, cake_hidden = fusion_cake(core_logits, canonical, byte_ids)
        if return_aux:
            return logits, {
                "core_logits": core_logits,
                "canonical": canonical,
                "routing_balance_loss": balance_loss,
                "cake_hidden": cake_hidden,
            }
        return logits

    def _fingerprint(self) -> str:
        if self._model_fingerprint is None:
            self._model_fingerprint = fingerprint_state_dict(self.state_dict())
        return self._model_fingerprint

    @staticmethod
    def _cake_fingerprint(cake: nn.Module | None) -> str | None:
        if cake is None:
            return None
        value = getattr(cake, "_layercake_state_fingerprint", None)
        if value is None:
            value = fingerprint_state_dict(cake.state_dict())
            setattr(cake, "_layercake_state_fingerprint", value)
        return str(value)

    def reset_state(
        self, *, batch_size: int = 1, route: int = 0, sampler_seed: int = 0,
        capture_generated: bool = False,
    ) -> FoundationV2State:
        cfg = self.config
        parameter = next(self.parameters())
        zeros = lambda *shape: torch.zeros(*shape, device=parameter.device, dtype=parameter.dtype)
        anchor_width = cfg.abi_width - cfg.abi_width // 2
        return FoundationV2State(
            conv_history=zeros(batch_size, cfg.local_kernel - 1, cfg.d_local),
            local_hidden=zeros(cfg.local_layers, batch_size, cfg.d_local),
            fast_buffer=zeros(batch_size, 0, cfg.d_local),
            slow_buffer=zeros(batch_size, 0, cfg.d_local),
            fast_hidden=zeros(cfg.global_layers, batch_size, cfg.d_global),
            slow_hidden=zeros(cfg.global_layers, batch_size, cfg.d_global),
            fast_context=zeros(batch_size, cfg.d_global),
            slow_context=zeros(batch_size, cfg.d_global),
            anchor_state=zeros(batch_size, anchor_width),
            canonical_state=zeros(batch_size, cfg.abi_width),
            next_core_logits=None,
            next_logits=None,
            generated_bytes=torch.zeros(batch_size, 0, device=parameter.device, dtype=torch.long),
            route=int(route),
            model_fingerprint=self._fingerprint(),
            sampler_seed=int(sampler_seed),
            capture_generated=bool(capture_generated),
        )

    def _consume(
        self,
        state: FoundationV2State,
        byte: torch.Tensor,
        fusion_cake: nn.Module | None,
    ) -> FoundationV2State:
        if byte.ndim != 1:
            byte = byte.flatten()
        embedded = self.byte_embedding(byte.long())[:, None]
        local_input = self.local_in(embedded)
        current_norm = self.local_norm(local_input)
        window = torch.cat([state.conv_history, current_norm], dim=1)
        convolved = self.local_depthwise(window.transpose(1, 2)).transpose(1, 2)
        mixed = local_input + self.local_mix(F.silu(convolved))
        local, local_hidden = self.local_recurrent(mixed, state.local_hidden)
        combined = self.local_to_global(local[:, 0]) + state.fast_context + state.slow_context
        core_logits = self.output(self.output_norm(combined))
        canonical, anchor = self.canonical.step(core_logits, byte, state.anchor_state)
        logits = core_logits
        cake_hidden = state.cake_hidden
        if fusion_cake is not None:
            logits_seq, cake_hidden = fusion_cake(
                core_logits[:, None], canonical[:, None], byte[:, None], cake_hidden
            )
            logits = logits_seq[:, 0]

        state.conv_history = window[:, 1:].detach()
        state.local_hidden = local_hidden.detach()
        state.anchor_state = anchor.detach()
        state.canonical_state = canonical.detach()
        state.next_core_logits = core_logits
        state.next_logits = logits
        state.cake_hidden = None if cake_hidden is None else cake_hidden.detach()
        state.fast_buffer = torch.cat([state.fast_buffer, local], dim=1)
        state.slow_buffer = torch.cat([state.slow_buffer, local], dim=1)
        if state.fast_buffer.shape[1] == self.config.fast_patch_size:
            summary = self.fast_summary(state.fast_buffer)[:, None]
            fast, state.fast_hidden = self.fast_recurrent(summary, state.fast_hidden)
            self.experts.set_route(state.route)
            routed = self.experts(fast + state.slow_context[:, None])
            state.fast_context = routed[:, 0].detach()
            state.fast_hidden = state.fast_hidden.detach()
            state.fast_buffer = state.fast_buffer[:, :0].detach()
        if state.slow_buffer.shape[1] == self.config.slow_patch_size:
            summary = self.slow_summary(state.slow_buffer)[:, None]
            slow, state.slow_hidden = self.slow_recurrent(summary, state.slow_hidden)
            state.slow_context = slow[:, 0].detach()
            state.slow_hidden = state.slow_hidden.detach()
            state.slow_buffer = state.slow_buffer[:, :0].detach()
        return state

    @torch.inference_mode()
    def prefill(
        self,
        prompt: bytes | str | torch.Tensor,
        *,
        route: int = 0,
        fusion_cake: nn.Module | None = None,
        sampler_seed: int = 0,
        capture_generated: bool = False,
    ) -> FoundationV2State:
        if isinstance(prompt, str):
            prompt = prompt.encode("utf-8")
        if isinstance(prompt, bytes):
            prompt = torch.tensor(list(prompt), dtype=torch.long)[None]
        if prompt.ndim != 2 or prompt.shape[1] == 0:
            raise ValueError("prefill requires a non-empty [batch, sequence] prompt")
        state = self.reset_state(
            batch_size=prompt.shape[0], route=route, sampler_seed=sampler_seed,
            capture_generated=capture_generated,
        )
        state.active_cake = self._cake_fingerprint(fusion_cake)
        prompt = prompt.to(state.conv_history.device)
        for index in range(prompt.shape[1]):
            state = self._consume(state, prompt[:, index], fusion_cake)
        state.prompt_bytes = int(prompt.numel())
        return state

    @torch.inference_mode()
    def decode_step(
        self,
        state: FoundationV2State,
        *,
        next_byte: torch.Tensor | None = None,
        fusion_cake: nn.Module | None = None,
    ) -> tuple[torch.Tensor, FoundationV2State]:
        if state.model_fingerprint != self._fingerprint() or state.next_logits is None:
            raise ValueError("invalid or unprefilled state")
        if state.active_cake != self._cake_fingerprint(fusion_cake):
            raise ValueError("incremental state belongs to a different active cake")
        logits = state.next_logits
        selected = logits.argmax(-1) if next_byte is None else next_byte.to(logits.device).long().flatten()
        if state.capture_generated:
            state.generated_bytes = torch.cat([state.generated_bytes, selected[:, None]], dim=1)
        state.decoded_bytes += state.batch_size
        state.sampler_counter += 1
        return logits, self._consume(state, selected, fusion_cake)

    @torch.inference_mode()
    def decode_many(
        self,
        state: FoundationV2State,
        count: int,
        *,
        fusion_cake: nn.Module | None = None,
    ) -> tuple[torch.Tensor, FoundationV2State]:
        rows = []
        for _ in range(count):
            logits, state = self.decode_step(state, fusion_cake=fusion_cake)
            rows.append(logits[:, None])
        empty = state.conv_history.new_zeros(state.batch_size, 0, 256)
        return (torch.cat(rows, dim=1) if rows else empty), state

    def serialize_state(self, state: FoundationV2State) -> bytes:
        metadata = {
            "format": "layercake-foundation-v2-state/1",
            "route": state.route,
            "model_fingerprint": state.model_fingerprint,
            "prompt_bytes": state.prompt_bytes,
            "decoded_bytes": state.decoded_bytes,
            "sampler_seed": state.sampler_seed,
            "sampler_counter": state.sampler_counter,
            "active_cake": state.active_cake,
            "capture_generated": state.capture_generated,
        }
        encoded = json.dumps(metadata, sort_keys=True).encode("utf-8")
        tensors = {name: tensor.detach().contiguous().cpu() for name, tensor in state.tensors().items()}
        tensors["__metadata_json"] = torch.tensor(list(encoded), dtype=torch.uint8)
        return save_safetensors(tensors)

    def restore_state(self, payload: bytes) -> FoundationV2State:
        tensors = load_safetensors(payload)
        raw = tensors.pop("__metadata_json", None)
        if raw is None:
            raise ValueError("state metadata is missing")
        metadata = json.loads(bytes(raw.tolist()).decode("utf-8"))
        if metadata.get("format") != "layercake-foundation-v2-state/1":
            raise ValueError("unsupported state format")
        if metadata.get("model_fingerprint") != self._fingerprint():
            raise ValueError("state belongs to different model weights")
        device = next(self.parameters()).device
        moved = {name: tensor.to(device) for name, tensor in tensors.items()}
        return FoundationV2State(
            conv_history=moved["conv_history"], local_hidden=moved["local_hidden"],
            fast_buffer=moved["fast_buffer"], slow_buffer=moved["slow_buffer"],
            fast_hidden=moved["fast_hidden"], slow_hidden=moved["slow_hidden"],
            fast_context=moved["fast_context"], slow_context=moved["slow_context"],
            anchor_state=moved["anchor_state"], canonical_state=moved["canonical_state"],
            next_core_logits=moved.get("next_core_logits"), next_logits=moved.get("next_logits"),
            generated_bytes=moved["generated_bytes"], cake_hidden=moved.get("cake_hidden"),
            route=int(metadata["route"]), model_fingerprint=str(metadata["model_fingerprint"]),
            prompt_bytes=int(metadata["prompt_bytes"]), decoded_bytes=int(metadata["decoded_bytes"]),
            sampler_seed=int(metadata["sampler_seed"]), sampler_counter=int(metadata["sampler_counter"]),
            active_cake=metadata.get("active_cake"),
            capture_generated=bool(metadata.get("capture_generated", False)),
        )

    def parameter_report(self, route: int = 0) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        experts = sum(parameter.numel() for parameter in self.experts.experts.parameters())
        one = sum(parameter.numel() for parameter in self.experts.experts[route].parameters())
        active = total - experts + one
        return {
            "total_parameters": total,
            "active_parameters": active,
            "active_fraction": active / total,
            "optimizer_resident_parameters_for_pinned_route": active,
            "routed_experts": self.config.routed_experts,
        }


def architecture_hash(config: FoundationV2Config) -> str:
    raw = json.dumps(config.canonical_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
