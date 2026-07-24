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
    recurrent_prompt_memory: bool = False
    prompt_memory_key_width: int = 32
    hierarchical_prompt_memory: bool = False
    structured_prompt_memory: bool = False
    structured_prompt_roles: int = 6
    prompt_memory_capacity: int = 128
    prompt_memory_chunk_size: int = 8
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
        if self.recurrent_prompt_memory and self.prompt_state_slots < 2:
            raise ValueError(
                "recurrent prompt memory requires a genuinely multi-slot state"
            )
        if self.prompt_memory_key_width <= 0:
            raise ValueError("prompt_memory_key_width must be positive")
        if self.hierarchical_prompt_memory and not self.prompt_conditioning:
            raise ValueError("hierarchical prompt memory requires prompt conditioning")
        if self.structured_prompt_memory and not self.prompt_conditioning:
            raise ValueError("structured prompt memory requires prompt conditioning")
        if self.structured_prompt_memory and self.hierarchical_prompt_memory:
            raise ValueError("structured and legacy hierarchical memory are exclusive")
        if self.structured_prompt_roles < 2:
            raise ValueError("structured prompt memory requires multiple roles")
        if self.prompt_memory_capacity <= 0:
            raise ValueError("prompt memory capacity must be positive")
        if self.prompt_memory_chunk_size <= 0:
            raise ValueError("prompt memory chunk size must be positive")
        if self.prompt_memory_capacity % self.prompt_memory_chunk_size:
            raise ValueError("prompt memory capacity must divide into whole chunks")

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
            if cfg.recurrent_prompt_memory:
                self.prompt_memory_norm = nn.LayerNorm(cfg.width)
                self.prompt_memory_query = nn.Linear(
                    cfg.width, cfg.prompt_memory_key_width, bias=False
                )
                self.prompt_memory_key = nn.Linear(
                    cfg.width, cfg.prompt_memory_key_width, bias=False
                )
                self.prompt_memory_gate = nn.Linear(cfg.width, 1)
                self.prompt_pointer_gate = nn.Linear(cfg.width, 1)
            if cfg.hierarchical_prompt_memory:
                self.hierarchical_memory_norm = nn.LayerNorm(cfg.width)
                self.hierarchical_memory_query = nn.Linear(
                    cfg.width, cfg.prompt_memory_key_width, bias=False
                )
                self.hierarchical_token_key = nn.Linear(
                    cfg.width, cfg.prompt_memory_key_width, bias=False
                )
                self.hierarchical_chunk_key = nn.Linear(
                    cfg.width, cfg.prompt_memory_key_width, bias=False
                )
                self.hierarchical_memory_gate = nn.Linear(cfg.width, 1)
                self.hierarchical_pointer_strength = nn.Parameter(
                    torch.tensor(-2.0)
                )
            if cfg.structured_prompt_memory:
                self.structured_prompt_depthwise = nn.Conv1d(
                    cfg.width,
                    cfg.width,
                    kernel_size=3,
                    padding=1,
                    groups=cfg.width,
                    bias=False,
                )
                self.structured_prompt_pointwise = nn.Linear(
                    cfg.width, cfg.width, bias=False
                )
                self.structured_role_queries = nn.Parameter(
                    torch.empty(cfg.structured_prompt_roles, cfg.width)
                )
                self.structured_memory_norm = nn.LayerNorm(cfg.width)
                self.structured_memory_query = nn.Linear(
                    cfg.width, cfg.prompt_memory_key_width, bias=False
                )
                self.structured_token_key = nn.Linear(
                    cfg.width, cfg.prompt_memory_key_width, bias=False
                )
                self.structured_role_key = nn.Linear(
                    cfg.width, cfg.prompt_memory_key_width, bias=False
                )
                self.structured_memory_gate = nn.Linear(cfg.width, 1)
                self.structured_pointer_strength = nn.Parameter(
                    torch.tensor(1.5)
                )
                self.structured_successor_strength = nn.Parameter(
                    torch.tensor(1.0)
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
        if cfg.recurrent_prompt_memory:
            with torch.no_grad():
                self.prompt_memory_gate.bias.fill_(-2.0)
                self.prompt_pointer_gate.bias.fill_(-3.0)
        if cfg.hierarchical_prompt_memory:
            with torch.no_grad():
                self.hierarchical_memory_gate.bias.fill_(-2.0)
        if cfg.structured_prompt_memory:
            nn.init.normal_(self.structured_role_queries, mean=0.0, std=0.02)
            with torch.no_grad():
                self.structured_memory_gate.bias.fill_(-2.0)
        self.last_routing_aux: dict | None = None
        self.last_prompt_memory_aux: dict | None = None
        self.last_structured_pointer_weights: torch.Tensor | None = None
        self.last_structured_pointer_ids: torch.Tensor | None = None

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)[None]
        mask = positions < prompt_lengths[:, None]
        embedded = self.embedding(token_ids)
        slots = None
        slot_copy_distribution = None
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
            if self.config.recurrent_prompt_memory:
                slot_copy_distribution = torch.zeros(
                    token_ids.shape[0],
                    self.config.prompt_state_slots,
                    self.config.vocab_size,
                    dtype=weights.dtype,
                    device=token_ids.device,
                )
                slot_copy_distribution.scatter_add_(
                    2,
                    token_ids[:, None].expand(
                        -1, self.config.prompt_state_slots, -1
                    ),
                    weights.permute(0, 2, 1),
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
        return context, copy_bias, slots, slot_copy_distribution

    def _apply_recurrent_prompt_memory(
        self,
        hidden: torch.Tensor,
        slots: torch.Tensor,
        slot_copy_distribution: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.prompt_memory_norm(hidden)
        queries = self.prompt_memory_query(normalized)
        keys = self.prompt_memory_key(slots)
        scores = torch.einsum(
            "btm,bsm->bts", queries, keys
        ) / self.config.prompt_memory_key_width ** 0.5
        weights = torch.softmax(scores, dim=-1)
        memory = torch.einsum("bts,bsw->btw", weights, slots)
        gate = torch.sigmoid(self.prompt_memory_gate(normalized))
        hidden = hidden + gate * memory
        pointer_distribution = torch.einsum(
            "bts,bsv->btv", weights, slot_copy_distribution
        )
        self.last_prompt_memory_aux = {
            "mean_gate": gate.detach().mean(),
            "maximum_gate": gate.detach().amax(),
            "slot_activation": weights.detach().mean(dim=(0, 1)),
            "pointer_mass": pointer_distribution.detach().sum(dim=-1).mean(),
        }
        return hidden, pointer_distribution

    def _hierarchical_prompt_features(
        self, token_ids: torch.Tensor, prompt_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build an exactly bounded token/chunk hierarchy from any prompt length."""

        batch = token_ids.shape[0]
        capacity = self.config.prompt_memory_capacity
        positions = torch.arange(capacity, device=token_ids.device)[None]
        retained = prompt_lengths.clamp(min=1, max=capacity)
        # Short prompts retain every token and pad to the fixed ABI capacity.
        # Long prompts are deterministically sampled across the full span.
        indexes = torch.div(
            positions * prompt_lengths[:, None],
            capacity,
            rounding_mode="floor",
        )
        short = prompt_lengths[:, None] <= capacity
        indexes = torch.where(short, positions, indexes)
        indexes = indexes.clamp_max(token_ids.shape[1] - 1)
        memory_ids = token_ids.gather(1, indexes.expand(batch, -1))
        token_mask = positions < retained[:, None]
        memory = self.prompt_projection(self.embedding(memory_ids))
        memory = memory * token_mask[:, :, None]

        chunk_size = self.config.prompt_memory_chunk_size
        chunks = capacity // chunk_size
        chunk_memory = memory.reshape(
            batch, chunks, chunk_size, self.config.width
        )
        chunk_token_mask = token_mask.reshape(batch, chunks, chunk_size)
        chunk_mask = chunk_token_mask.any(dim=-1)
        chunk_sums = chunk_memory.sum(dim=2)
        chunk_counts = chunk_token_mask.sum(dim=-1).clamp_min(1)
        chunk_summaries = chunk_sums / chunk_counts[:, :, None]
        return memory_ids, memory, token_mask, chunk_summaries, chunk_mask

    def _apply_hierarchical_prompt_memory(
        self,
        hidden: torch.Tensor,
        token_memory: torch.Tensor,
        token_mask: torch.Tensor,
        chunk_summaries: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.hierarchical_memory_norm(hidden)
        query = self.hierarchical_memory_query(normalized)
        token_keys = self.hierarchical_token_key(token_memory)
        chunk_keys = self.hierarchical_chunk_key(chunk_summaries)
        scale = self.config.prompt_memory_key_width ** 0.5
        chunk_scores = torch.einsum(
            "btm,bcm->btc", query, chunk_keys
        ) / scale
        chunk_scores = chunk_scores.masked_fill(
            ~chunk_mask[:, None], -torch.inf
        )
        token_scores = torch.einsum(
            "btm,bpm->btp", query, token_keys
        ) / scale
        token_scores = token_scores + chunk_scores.repeat_interleave(
            self.config.prompt_memory_chunk_size, dim=-1
        )
        token_scores = token_scores.masked_fill(
            ~token_mask[:, None], -torch.inf
        )
        weights = torch.softmax(token_scores, dim=-1)
        recalled = torch.einsum("btp,bpw->btw", weights, token_memory)
        gate = torch.sigmoid(self.hierarchical_memory_gate(normalized))
        hidden = hidden + gate * recalled
        self.last_prompt_memory_aux = {
            "mean_gate": gate.detach().mean(),
            "maximum_gate": gate.detach().amax(),
            "slot_activation": torch.softmax(
                chunk_scores.detach(), dim=-1
            ).mean(dim=(0, 1)),
            "pointer_mass": weights.detach().sum(dim=-1).mean(),
        }
        return hidden, weights

    def _structured_prompt_features(
        self, token_ids: torch.Tensor, prompt_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompt tokens once into contextual tokens and role records."""

        batch = token_ids.shape[0]
        capacity = self.config.prompt_memory_capacity
        positions = torch.arange(capacity, device=token_ids.device)[None]
        retained = prompt_lengths.clamp(min=1, max=capacity)
        indexes = torch.div(
            positions * prompt_lengths[:, None],
            capacity,
            rounding_mode="floor",
        )
        indexes = torch.where(
            prompt_lengths[:, None] <= capacity, positions, indexes
        )
        indexes = indexes.clamp_max(token_ids.shape[1] - 1)
        memory_ids = token_ids.gather(1, indexes.expand(batch, -1))
        token_mask = positions < retained[:, None]
        base = self.prompt_projection(self.embedding(memory_ids))
        local = self.structured_prompt_depthwise(
            base.transpose(1, 2)
        ).transpose(1, 2)
        contextual = base + self.structured_prompt_pointwise(F.silu(local))
        contextual = contextual * token_mask[:, :, None]
        role_scores = torch.einsum(
            "bpw,rw->brp", contextual, self.structured_role_queries
        ) / self.config.width ** 0.5
        role_scores = role_scores.masked_fill(
            ~token_mask[:, None], -torch.inf
        )
        role_weights = torch.softmax(role_scores, dim=-1)
        role_memory = torch.einsum(
            "brp,bpw->brw", role_weights, contextual
        )
        return memory_ids, contextual, token_mask, role_memory

    def _apply_structured_prompt_memory(
        self,
        hidden: torch.Tensor,
        input_token_ids: torch.Tensor,
        memory_ids: torch.Tensor,
        token_memory: torch.Tensor,
        token_mask: torch.Tensor,
        role_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read bounded role/token records with an associative successor prior."""

        normalized = self.structured_memory_norm(hidden)
        query = self.structured_memory_query(normalized)
        scale = self.config.prompt_memory_key_width ** 0.5
        role_scores = torch.matmul(
            query, self.structured_role_key(role_memory).transpose(-1, -2)
        ) / scale
        role_weights = torch.softmax(role_scores, dim=-1)
        role_recall = torch.matmul(role_weights, role_memory)
        token_scores = torch.matmul(
            query, self.structured_token_key(token_memory).transpose(-1, -2)
        ) / scale
        matches = input_token_ids[:, :, None] == memory_ids[:, None, :]
        successor = torch.zeros_like(matches)
        successor[:, :, 1:] = matches[:, :, :-1]
        token_scores = token_scores + F.softplus(
            self.structured_successor_strength
        ) * successor.to(token_scores.dtype)
        token_scores = token_scores.masked_fill(
            ~token_mask[:, None], -torch.inf
        )
        token_weights = torch.softmax(token_scores, dim=-1)
        token_recall = torch.matmul(token_weights, token_memory)
        gate = torch.sigmoid(self.structured_memory_gate(normalized))
        hidden = hidden + gate * (role_recall + token_recall)
        self.last_structured_pointer_weights = token_weights
        self.last_structured_pointer_ids = memory_ids
        self.last_prompt_memory_aux = {
            "mean_gate": gate.detach().mean(),
            "maximum_gate": gate.detach().amax(),
            "slot_activation": role_weights.detach().mean(dim=(0, 1)),
            "pointer_mass": token_weights.detach().sum(dim=-1).mean(),
        }
        return hidden, token_weights

    def _output_logits(
        self,
        hidden: torch.Tensor,
        copy_bias: torch.Tensor | None = None,
        pointer_distribution: torch.Tensor | None = None,
        pointer_token_ids: torch.Tensor | None = None,
        pointer_token_weights: torch.Tensor | None = None,
        structured_token_ids: torch.Tensor | None = None,
        structured_token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.norm(hidden)
        logits = F.linear(normalized, self.embedding.weight)
        if copy_bias is not None:
            if logits.ndim == 3:
                copy_bias = copy_bias[:, None]
            logits = logits + self.prompt_copy_strength * copy_bias
        if pointer_distribution is not None:
            pointer_distribution = pointer_distribution.float().clamp_min(0)
            pointer_distribution = pointer_distribution / (
                pointer_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            )
            pointer_gate = torch.sigmoid(
                self.prompt_pointer_gate(normalized).float()
            )
            probabilities = (
                (1.0 - pointer_gate) * torch.softmax(logits.float(), dim=-1)
                + pointer_gate * pointer_distribution
            )
            logits = probabilities.clamp_min(1e-12).log()
        if pointer_token_ids is not None and pointer_token_weights is not None:
            dynamic_bias = torch.zeros_like(logits)
            if logits.ndim == 3:
                indexes = pointer_token_ids[:, None].expand(
                    -1, logits.shape[1], -1
                )
            else:
                indexes = pointer_token_ids
            dynamic_bias.scatter_add_(
                -1, indexes, pointer_token_weights.to(logits.dtype)
            )
            logits = logits + F.softplus(
                self.hierarchical_pointer_strength
            ) * dynamic_bias
        if (
            structured_token_ids is not None
            and structured_token_weights is not None
        ):
            if logits.ndim == 3:
                indexes = structured_token_ids[:, None].expand(
                    -1, logits.shape[1], -1
                )
            else:
                indexes = structured_token_ids
            logits = logits.scatter_add(
                -1,
                indexes,
                F.softplus(self.structured_pointer_strength)
                * structured_token_weights.to(logits.dtype),
            )
        return logits

    def forward(
        self, token_ids: torch.Tensor, *, prompt_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.config.max_tokens:
            raise ValueError("token ids exceed the configured LayerCake context")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.embedding(token_ids) + self.position(positions)[None]
        copy_bias = None
        memory_slots = None
        slot_copy_distribution = None
        pointer_distribution = None
        hierarchical = None
        hierarchical_weights = None
        structured = None
        structured_weights = None
        if self.config.prompt_conditioning and prompt_lengths is not None:
            (
                context,
                copy_bias,
                memory_slots,
                slot_copy_distribution,
            ) = self._prompt_features(token_ids, prompt_lengths)
            hidden = hidden + context[:, None]
            if self.config.hierarchical_prompt_memory:
                hierarchical = self._hierarchical_prompt_features(
                    token_ids, prompt_lengths
                )
            if self.config.structured_prompt_memory:
                structured = self._structured_prompt_features(
                    token_ids, prompt_lengths
                )
        for index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden)
            if index == self.config.route_after_layers:
                hidden = self._route(hidden)
                if (
                    self.config.recurrent_prompt_memory
                    and memory_slots is not None
                    and slot_copy_distribution is not None
                ):
                    hidden, pointer_distribution = (
                        self._apply_recurrent_prompt_memory(
                            hidden, memory_slots, slot_copy_distribution
                        )
                    )
                if (
                    self.config.hierarchical_prompt_memory
                    and hierarchical is not None
                ):
                    (
                        memory_ids,
                        token_memory,
                        token_mask,
                        chunk_summaries,
                        chunk_mask,
                    ) = hierarchical
                    hidden, hierarchical_weights = (
                        self._apply_hierarchical_prompt_memory(
                            hidden,
                            token_memory,
                            token_mask,
                            chunk_summaries,
                            chunk_mask,
                        )
                    )
                if (
                    self.config.structured_prompt_memory
                    and structured is not None
                ):
                    hidden, structured_weights = (
                        self._apply_structured_prompt_memory(
                            hidden,
                            token_ids,
                            structured[0],
                            structured[1],
                            structured[2],
                            structured[3],
                        )
                    )
        return self._output_logits(
            hidden,
            copy_bias,
            pointer_distribution,
            (
                hierarchical[0]
                if hierarchical is not None
                else None
            ),
            hierarchical_weights,
            structured[0] if structured is not None else None,
            structured_weights,
        )

    @torch.inference_mode()
    def prefill(self, token_ids: torch.Tensor) -> TransformerGenerationState:
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.config.max_tokens:
            raise ValueError("prefill requires non-empty in-range token ids")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.embedding(token_ids) + self.position(positions)[None]
        prompt_context = None
        prompt_copy_bias = None
        prompt_memory_slots = None
        prompt_memory_copy_distribution = None
        pointer_distribution = None
        hierarchical = None
        hierarchical_weights = None
        structured = None
        structured_weights = None
        if self.config.prompt_conditioning:
            lengths = torch.full(
                (token_ids.shape[0],), token_ids.shape[1], dtype=torch.long,
                device=token_ids.device,
            )
            (
                prompt_context,
                prompt_copy_bias,
                prompt_memory_slots,
                prompt_memory_copy_distribution,
            ) = self._prompt_features(token_ids, lengths)
            hidden = hidden + prompt_context[:, None]
            if self.config.hierarchical_prompt_memory:
                hierarchical = self._hierarchical_prompt_features(
                    token_ids, lengths
                )
            if self.config.structured_prompt_memory:
                structured = self._structured_prompt_features(
                    token_ids, lengths
                )
        keys_values = []
        for index, block in enumerate(self.blocks, start=1):
            hidden, block_cache = block.forward_cached(hidden)
            keys_values.append(block_cache)
            if index == self.config.route_after_layers:
                hidden = self._route(hidden)
                if (
                    self.config.recurrent_prompt_memory
                    and prompt_memory_slots is not None
                    and prompt_memory_copy_distribution is not None
                ):
                    hidden, pointer_distribution = (
                        self._apply_recurrent_prompt_memory(
                            hidden,
                            prompt_memory_slots,
                            prompt_memory_copy_distribution,
                        )
                    )
                if (
                    self.config.hierarchical_prompt_memory
                    and hierarchical is not None
                ):
                    hidden, hierarchical_weights = (
                        self._apply_hierarchical_prompt_memory(
                            hidden,
                            hierarchical[1],
                            hierarchical[2],
                            hierarchical[3],
                            hierarchical[4],
                        )
                    )
                if (
                    self.config.structured_prompt_memory
                    and structured is not None
                ):
                    hidden, structured_weights = (
                        self._apply_structured_prompt_memory(
                            hidden,
                            token_ids,
                            structured[0],
                            structured[1],
                            structured[2],
                            structured[3],
                        )
                    )
        logits = self._output_logits(
            hidden[:, -1],
            prompt_copy_bias,
            (
                pointer_distribution[:, -1]
                if pointer_distribution is not None
                else None
            ),
            hierarchical[0] if hierarchical is not None else None,
            (
                hierarchical_weights[:, -1]
                if hierarchical_weights is not None
                else None
            ),
            structured[0] if structured is not None else None,
            (
                structured_weights[:, -1]
                if structured_weights is not None
                else None
            ),
        )
        state = TransformerGenerationState(
            keys_values=keys_values,
            next_logits=logits,
            token_ids=token_ids,
            generated_ids=token_ids[:, :0],
        )
        state.prompt_context = prompt_context
        state.prompt_copy_bias = prompt_copy_bias
        state.prompt_memory_slots = prompt_memory_slots
        state.prompt_memory_copy_distribution = (
            prompt_memory_copy_distribution
        )
        state.hierarchical_prompt_memory = hierarchical
        state.structured_prompt_memory = structured
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
        pointer_distribution = None
        hierarchical_weights = None
        structured_weights = None
        new_cache = []
        for index, (block, past) in enumerate(zip(self.blocks, state.keys_values), start=1):
            hidden, cache = block.forward_cached(hidden, past)
            new_cache.append(cache)
            if index == self.config.route_after_layers:
                hidden = self._route(hidden)
                memory_slots = getattr(state, "prompt_memory_slots", None)
                memory_copy = getattr(
                    state, "prompt_memory_copy_distribution", None
                )
                if (
                    self.config.recurrent_prompt_memory
                    and memory_slots is not None
                    and memory_copy is not None
                ):
                    hidden, pointer_distribution = (
                        self._apply_recurrent_prompt_memory(
                            hidden, memory_slots, memory_copy
                        )
                    )
                hierarchical = getattr(
                    state, "hierarchical_prompt_memory", None
                )
                if (
                    self.config.hierarchical_prompt_memory
                    and hierarchical is not None
                ):
                    hidden, hierarchical_weights = (
                        self._apply_hierarchical_prompt_memory(
                            hidden,
                            hierarchical[1],
                            hierarchical[2],
                            hierarchical[3],
                            hierarchical[4],
                        )
                    )
                structured = getattr(
                    state, "structured_prompt_memory", None
                )
                if (
                    self.config.structured_prompt_memory
                    and structured is not None
                ):
                    hidden, structured_weights = (
                        self._apply_structured_prompt_memory(
                            hidden,
                            selected[:, None],
                            structured[0],
                            structured[1],
                            structured[2],
                            structured[3],
                        )
                    )
        state.keys_values = new_cache
        state.next_logits = self._output_logits(
            hidden[:, 0],
            getattr(state, "prompt_copy_bias", None),
            (
                pointer_distribution[:, 0]
                if pointer_distribution is not None
                else None
            ),
            (
                state.hierarchical_prompt_memory[0]
                if self.config.hierarchical_prompt_memory
                else None
            ),
            (
                hierarchical_weights[:, 0]
                if hierarchical_weights is not None
                else None
            ),
            (
                state.structured_prompt_memory[0]
                if self.config.structured_prompt_memory
                else None
            ),
            (
                structured_weights[:, 0]
                if structured_weights is not None
                else None
            ),
        )
        state.generated_ids = torch.cat([state.generated_ids, selected[:, None]], dim=1)
        return logits, state
