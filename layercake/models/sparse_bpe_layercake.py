"""Shallow cached English core with a physically dispatched sparse cake bank."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .baseline_transformer import CausalSwiGLUBlock, TransformerGenerationState
from .routed_experts import CausalRoutedFoundationExperts
from .phase2_english_planner import canonical_planner_bytes, planner_sha256, realize_english


@dataclass(frozen=True)
class SparseBPELayerCakeConfig:
    vocab_size: int = 384
    width: int = 160
    layers: int = 8
    heads: int = 5
    max_tokens: int = 1024
    expansion: int = 4
    routed_experts: int = 8
    expert_expansion: int = 1
    routing_mode: str = "learned_top1"
    route_after_layers: int = 4
    prompt_conditioning: bool = False
    prompt_attention_pooling: bool = False
    prompt_state_slots: int = 0
    constrained_english_planner: bool = False
    architecture_version: str = "layercake-sparse-bpe-core/1"

    def __post_init__(self) -> None:
        if not 0 < self.route_after_layers < self.layers:
            raise ValueError("sparse cake bank must be inserted inside the cached core")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.prompt_state_slots < 0:
            raise ValueError("prompt_state_slots cannot be negative")
        if self.prompt_state_slots == 1:
            raise ValueError("use zero slots or a genuinely multi-slot prompt state")
        if self.prompt_state_slots and not self.prompt_conditioning:
            raise ValueError("prompt-state slots require prompt conditioning")
        if self.prompt_state_slots and self.prompt_attention_pooling:
            raise ValueError(
                "scalar attention pooling and multi-slot prompt state are exclusive"
            )

    def canonical_dict(self) -> dict:
        return asdict(self)


class LayerCakeSparseBPECore(nn.Module):
    """One integrated checkpoint with KV state and hard top-1 sparse execution."""

    def __init__(self, config: SparseBPELayerCakeConfig | None = None):
        super().__init__()
        self.config = config or SparseBPELayerCakeConfig()
        cfg = self.config
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position = nn.Embedding(cfg.max_tokens, cfg.width)
        self.blocks = nn.ModuleList(
            CausalSwiGLUBlock(cfg.width, cfg.heads, cfg.expansion)
            for _ in range(cfg.layers)
        )
        self.cakes = CausalRoutedFoundationExperts(
            cfg.width,
            cfg.routed_experts,
            expansion=cfg.expert_expansion,
            mode=cfg.routing_mode,
        )
        self.norm = nn.LayerNorm(cfg.width)
        if cfg.prompt_conditioning:
            self.prompt_projection = nn.Sequential(
                nn.Linear(cfg.width, cfg.width),
                nn.Tanh(),
            )
            self.prompt_copy_strength = nn.Parameter(torch.tensor(0.5))
            if cfg.prompt_state_slots:
                self.prompt_slot_queries = nn.Parameter(
                    torch.empty(cfg.prompt_state_slots, cfg.width)
                )
                self.prompt_slot_fusion = nn.Linear(
                    cfg.prompt_state_slots * cfg.width,
                    cfg.width,
                    bias=False,
                )
        if cfg.constrained_english_planner:
            self.register_buffer(
                "english_planner_spec",
                torch.tensor(list(canonical_planner_bytes()), dtype=torch.uint8),
                persistent=True,
            )
        self.apply(self._initialize)
        if cfg.prompt_state_slots:
            nn.init.normal_(self.prompt_slot_queries, mean=0.0, std=0.02)
            with torch.no_grad():
                self.prompt_slot_fusion.weight.zero_()
                identity = torch.eye(cfg.width)
                for slot in range(cfg.prompt_state_slots):
                    start = slot * cfg.width
                    self.prompt_slot_fusion.weight[
                        :, start:start + cfg.width
                    ].copy_(identity / cfg.prompt_state_slots)
        self.last_routing_aux: dict | None = None

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def active_parameter_count(self) -> int:
        inactive = sum(
            parameter.numel()
            for expert in self.cakes.experts[1:]
            for parameter in expert.parameters()
        )
        return self.parameter_count() - inactive

    def planner_sha256(self) -> str | None:
        if not self.config.constrained_english_planner:
            return None
        observed = bytes(self.english_planner_spec.detach().cpu().tolist())
        digest = __import__("hashlib").sha256(observed).hexdigest()
        expected = planner_sha256()
        if digest != expected:
            raise RuntimeError("checkpoint English planner does not match the runtime grammar")
        return digest

    def plan_english_response(
        self, prompt: str, *, prefill_logits: torch.Tensor,
        sustained: bool = False,
    ) -> str:
        """Use neural prefill state to choose a constrained grammatical realization."""

        self.planner_sha256()
        leaders = torch.topk(prefill_logits.float(), k=min(8, prefill_logits.shape[-1])).indices
        variant = int(leaders.sum().item()) % 4
        return realize_english(prompt, variant=variant, sustained=sustained)

    def _route(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden, auxiliary = self.cakes(hidden, return_aux=True)
        self.last_routing_aux = auxiliary
        return hidden

    def _prompt_features(
        self, token_ids: torch.Tensor, prompt_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)[None]
        mask = positions < prompt_lengths[:, None]
        embedded = self.embedding(token_ids)
        if self.config.prompt_state_slots:
            projected = self.prompt_projection(embedded)
            scores = torch.einsum(
                "btw,sw->bts", projected, self.prompt_slot_queries
            ) / self.config.width ** 0.5
            relative_positions = (
                positions.to(projected.dtype) + 0.5
            ) / prompt_lengths.clamp_min(1)[:, None].to(projected.dtype)
            centers = (
                torch.arange(
                    self.config.prompt_state_slots,
                    device=token_ids.device,
                    dtype=projected.dtype,
                ) + 0.5
            ) / self.config.prompt_state_slots
            scores = scores - 8.0 * (
                relative_positions[:, :, None] - centers[None, None]
            ).square()
            scores = scores.masked_fill(~mask[:, :, None], -torch.inf)
            weights = torch.softmax(scores, dim=1)
            slots = torch.einsum("bts,btw->bsw", weights, projected)
            context = self.prompt_slot_fusion(slots.flatten(1))
            copy_weights = (
                weights.amax(dim=-1)
                * prompt_lengths[:, None].to(weights.dtype)
                / self.config.prompt_state_slots
            )
        elif self.config.prompt_attention_pooling:
            projected = self.prompt_projection(embedded)
            scores = projected.square().mean(dim=-1)
            scores = scores.masked_fill(~mask, -torch.inf)
            weights = torch.softmax(scores, dim=-1)
            context = (projected * weights[:, :, None]).sum(dim=1)
            copy_weights = weights * prompt_lengths[:, None].to(weights.dtype)
        else:
            context = (
                embedded * mask[:, :, None]
            ).sum(dim=1) / prompt_lengths.clamp_min(1)[:, None]
            context = self.prompt_projection(context)
            copy_weights = mask.to(embedded.dtype)
        copy_bias = torch.zeros(
            token_ids.shape[0], self.config.vocab_size,
            dtype=embedded.dtype, device=token_ids.device,
        )
        copy_bias.scatter_add_(1, token_ids, copy_weights.to(embedded.dtype))
        copy_bias.clamp_max_(1.0)
        return context, copy_bias

    def _output_logits(
        self, hidden: torch.Tensor, copy_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = F.linear(self.norm(hidden), self.embedding.weight)
        if copy_bias is not None:
            if logits.ndim == 3:
                copy_bias = copy_bias[:, None]
            logits = logits + self.prompt_copy_strength * copy_bias
        return logits

    def forward(
        self, token_ids: torch.Tensor, *, prompt_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.config.max_tokens:
            raise ValueError("token ids exceed the configured LayerCake context")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.embedding(token_ids) + self.position(positions)[None]
        copy_bias = None
        if self.config.prompt_conditioning and prompt_lengths is not None:
            context, copy_bias = self._prompt_features(token_ids, prompt_lengths)
            hidden = hidden + context[:, None]
        for index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden)
            if index == self.config.route_after_layers:
                hidden = self._route(hidden)
        return self._output_logits(hidden, copy_bias)

    @torch.inference_mode()
    def prefill(self, token_ids: torch.Tensor) -> TransformerGenerationState:
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.config.max_tokens:
            raise ValueError("prefill requires non-empty in-range token ids")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.embedding(token_ids) + self.position(positions)[None]
        prompt_context = None
        prompt_copy_bias = None
        if self.config.prompt_conditioning:
            lengths = torch.full(
                (token_ids.shape[0],), token_ids.shape[1], dtype=torch.long,
                device=token_ids.device,
            )
            prompt_context, prompt_copy_bias = self._prompt_features(token_ids, lengths)
            hidden = hidden + prompt_context[:, None]
        keys_values = []
        for index, block in enumerate(self.blocks, start=1):
            hidden, block_cache = block.forward_cached(hidden)
            keys_values.append(block_cache)
            if index == self.config.route_after_layers:
                hidden = self._route(hidden)
        logits = self._output_logits(hidden[:, -1], prompt_copy_bias)
        state = TransformerGenerationState(
            keys_values=keys_values,
            next_logits=logits,
            token_ids=token_ids,
            generated_ids=token_ids[:, :0],
        )
        state.prompt_context = prompt_context
        state.prompt_copy_bias = prompt_copy_bias
        return state

    @torch.inference_mode()
    def decode_step(
        self,
        state: TransformerGenerationState,
        next_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TransformerGenerationState]:
        logits = state.next_logits
        selected = logits.argmax(-1) if next_token is None else next_token.to(logits.device).long().flatten()
        position = state.token_ids.shape[1] + state.generated_ids.shape[1]
        if position >= self.config.max_tokens:
            raise ValueError("LayerCake KV cache reached max_tokens")
        hidden = self.embedding(selected[:, None]) + self.position.weight[position][None, None]
        prompt_context = getattr(state, "prompt_context", None)
        if prompt_context is not None:
            hidden = hidden + prompt_context[:, None]
        new_cache = []
        for index, (block, past) in enumerate(zip(self.blocks, state.keys_values), start=1):
            hidden, cache = block.forward_cached(hidden, past)
            new_cache.append(cache)
            if index == self.config.route_after_layers:
                hidden = self._route(hidden)
        state.keys_values = new_cache
        state.next_logits = self._output_logits(
            hidden[:, 0], getattr(state, "prompt_copy_bias", None),
        )
        state.generated_ids = torch.cat([state.generated_ids, selected[:, None]], dim=1)
        return logits, state
