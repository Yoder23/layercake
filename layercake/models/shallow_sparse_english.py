"""Three-block English core with physically selected instruction cakes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model


@dataclass(frozen=True)
class ShallowSparseEnglishConfig:
    vocab_size: int = 50257
    width: int = 768
    layers: int = 3
    heads: int = 12
    max_tokens: int = 1024
    task_cakes: int = 10
    task_cake_rank: int = 64
    architecture_version: str = (
        "layercake-shallow-sparse-english/1-three-block-task-cakes"
    )

    def __post_init__(self) -> None:
        if self.layers != 3:
            raise ValueError("the preregistered shallow English core has three blocks")
        if self.width % self.heads:
            raise ValueError("width must divide evenly across attention heads")
        if self.task_cakes != 10 or self.task_cake_rank != 64:
            raise ValueError("the preregistered instruction-cake graph changed")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)


class LowRankInstructionCake(nn.Module):
    """A small nonlinear residual that is called only for its selected route."""

    def __init__(self, width: int, rank: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.up(F.silu(self.down(self.norm(hidden))))


class ShallowSparseEnglishCore(nn.Module):
    """Cached shallow transformer plus one physically dispatched task cake."""

    def __init__(self, config: ShallowSparseEnglishConfig | None = None):
        super().__init__()
        self.config = config or ShallowSparseEnglishConfig()
        cfg = self.config
        transformer_config = GPT2Config(
            vocab_size=cfg.vocab_size,
            n_positions=cfg.max_tokens,
            n_ctx=cfg.max_tokens,
            n_embd=cfg.width,
            n_layer=cfg.layers,
            n_head=cfg.heads,
            n_inner=4 * cfg.width,
            activation_function="gelu_new",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=True,
        )
        self.transformer = GPT2Model(transformer_config)
        self.task_classifier = nn.Linear(cfg.width, cfg.task_cakes)
        self.task_cakes = nn.ModuleList(
            LowRankInstructionCake(cfg.width, cfg.task_cake_rank)
            for _ in range(cfg.task_cakes)
        )
        self.last_task_logits: torch.Tensor | None = None
        self.last_task_routes: torch.Tensor | None = None
        self.last_cake_calls: tuple[int, ...] = ()

    @property
    def output_weight(self) -> torch.Tensor:
        return self.transformer.wte.weight

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def active_parameter_count(self) -> int:
        inactive = sum(
            parameter.numel()
            for cake in self.task_cakes[1:]
            for parameter in cake.parameters()
        )
        return self.parameter_count() - inactive

    def _prompt_summary(
        self,
        hidden: torch.Tensor,
        *,
        prompt_lengths: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, tokens, _ = hidden.shape
        positions = torch.arange(tokens, device=hidden.device)[None]
        if prompt_lengths is not None:
            mask = positions < prompt_lengths[:, None]
        elif attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool)
        else:
            mask = torch.ones(
                batch, tokens, dtype=torch.bool, device=hidden.device
            )
        weights = mask.to(hidden.dtype)
        return (hidden * weights[:, :, None]).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1)

    def _dispatch(
        self, hidden: torch.Tensor, routes: torch.Tensor
    ) -> torch.Tensor:
        if routes.ndim != 1 or routes.shape[0] != hidden.shape[0]:
            raise ValueError("task routes must contain one route per batch row")
        output = torch.empty_like(hidden)
        calls: list[int] = []
        for route, cake in enumerate(self.task_cakes):
            rows = torch.nonzero(routes == route, as_tuple=False).flatten()
            if not rows.numel():
                continue
            selected = hidden.index_select(0, rows)
            output.index_copy_(0, rows, cake(selected))
            calls.append(route)
        self.last_cake_calls = tuple(calls)
        self.last_task_routes = routes.detach()
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        prompt_lengths: torch.Tensor | None = None,
        task_routes: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("input ids must be [batch, tokens]")
        result = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        hidden = result.last_hidden_state
        task_logits = None
        if task_routes is None:
            if past_key_values is not None:
                raise ValueError("cached decode requires the prefill task route")
            summary = self._prompt_summary(
                hidden,
                prompt_lengths=prompt_lengths,
                attention_mask=attention_mask,
            )
            task_logits = self.task_classifier(summary)
            task_routes = task_logits.argmax(dim=-1)
        else:
            task_routes = task_routes.to(hidden.device).long().flatten()
            if past_key_values is None:
                summary = self._prompt_summary(
                    hidden,
                    prompt_lengths=prompt_lengths,
                    attention_mask=attention_mask,
                )
                task_logits = self.task_classifier(summary)
        adapted = self._dispatch(hidden, task_routes)
        logits = F.linear(adapted, self.output_weight)
        self.last_task_logits = task_logits
        return {
            "logits": logits,
            "past_key_values": result.past_key_values,
            "task_logits": task_logits,
            "task_routes": task_routes,
            "hidden": adapted,
        }

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> dict[str, Any]:
        result = self(
            input_ids,
            prompt_lengths=torch.full(
                (input_ids.shape[0],),
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            ),
            use_cache=True,
        )
        return {
            "past_key_values": result["past_key_values"],
            "task_routes": result["task_routes"],
            "next_logits": result["logits"][:, -1],
            "prompt_ids": input_ids,
            "generated_ids": input_ids[:, :0],
        }

    @torch.inference_mode()
    def decode_step(
        self,
        state: dict[str, Any],
        next_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        logits = state["next_logits"]
        selected = (
            logits.argmax(dim=-1)
            if next_token is None
            else next_token.to(logits.device).long().flatten()
        )
        result = self(
            selected[:, None],
            task_routes=state["task_routes"],
            past_key_values=state["past_key_values"],
            use_cache=True,
        )
        state["past_key_values"] = result["past_key_values"]
        state["next_logits"] = result["logits"][:, -1]
        state["generated_ids"] = torch.cat(
            (state["generated_ids"], selected[:, None]), dim=1
        )
        return logits, state

    def physical_sparse_contract(self) -> dict[str, Any]:
        cake_parameters = [
            sum(parameter.numel() for parameter in cake.parameters())
            for cake in self.task_cakes
        ]
        return {
            "installed_task_cakes": len(self.task_cakes),
            "maximum_active_task_cakes_per_sequence": 1,
            "task_cake_parameter_counts": cake_parameters,
            "inactive_cakes_called": 0,
            "dispatch": "batch rows are index-selected into only the chosen module",
        }
