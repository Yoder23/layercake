"""Generic guarded route-isolated shallow-sparse direct English-core host."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage, load_package
from layercake.cake.registry import CakeRegistry
from layercake.models.shallow_sparse_english import (
    ShallowSparseEnglishConfig,
    ShallowSparseEnglishCore,
)
from layercake.portable_domain import state_dict_hash
from layercake_extensions.bpe_direct_neural_core import (
    Utf8ConcatenativeBpeTokenizer,
)


ROUTE_ISOLATED_CORE_ABI_VERSION = "lc-direct-neural-core/17"
ROUTE_ISOLATED_CORE_ABI_SHA256 = (
    "59d69acbe41417fc802e8a1ba559b29c3f64d760e498f4056864945eab367635"
)
ARCHITECTURE_FORMAT = "layercake-route-isolated-shallow-sparse-core/1"
ROLE = "english-core"
COMPOSITION = "direct_core_only_no_router"
CAPABILITIES = (
    "grammar",
    "coherence",
    "prompt_grounding",
    "instruction_following",
    "conversation",
    "supplied_text_summarization",
    "rewriting",
    "email_drafting_from_notes",
    "tone_control",
    "format_control",
    "clarification",
    "abstention",
    "fact_free_reasoning",
    "fluent_realization",
)
CAPABILITY_TO_TASK_ROUTE = {
    "grammar": 0,
    "coherence": 0,
    "fluent_realization": 0,
    "prompt_grounding": 1,
    "instruction_following": 1,
    "conversation": 2,
    "clarification": 2,
    "abstention": 2,
    "supplied_text_summarization": 3,
    "rewriting": 3,
    "email_drafting_from_notes": 4,
    "tone_control": 4,
    "format_control": 4,
    "fact_free_reasoning": 5,
}
WEAK_CAPABILITIES = (
    "abstention",
    "coherence",
    "fluent_realization",
    "tone_control",
)
METADATA_LABEL = "__metadata__"
TOKEN_PATTERN = re.compile(r"[\w]+(?:['’][\w]+)*|[^\w\s]", re.UNICODE)


class RouteIsolatedCoreError(ValueError):
    """Raised when a v17 package or external boundary is invalid."""


def repetition_collapse(output: str) -> bool:
    tokens = TOKEN_PATTERN.findall(output.casefold())
    maximum_width = min(16, len(tokens) // 4)
    for width in range(1, maximum_width + 1):
        for start in range(0, len(tokens) - (4 * width) + 1):
            block = tokens[start : start + width]
            if all(
                tokens[
                    start + repeat * width : start + (repeat + 1) * width
                ]
                == block
                for repeat in range(1, 4)
            ):
                return True
    if len(tokens) >= 32:
        fourgrams = [
            tuple(tokens[index : index + 4])
            for index in range(len(tokens) - 3)
        ]
        if len(set(fourgrams)) / len(fourgrams) < 0.35:
            return True
    return False


class RouteIsolatedResidual(nn.Module):
    """Four disjoint rank-r experts stored in legacy-compatible flat tensors."""

    def __init__(self, width: int, rank: int, routes: int) -> None:
        super().__init__()
        self.width = width
        self.rank = rank
        self.routes = routes
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank * routes, bias=False)
        self.up = nn.Linear(rank * routes, width, bias=False)

    def delta(self, hidden: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden)
        outputs = torch.zeros_like(hidden)
        for route in routes.unique(sorted=True):
            route_id = int(route.item())
            if route_id < 0:
                continue
            if route_id >= self.routes:
                raise RouteIsolatedCoreError("residual route is outside the package contract")
            rows = torch.nonzero(routes == route_id, as_tuple=False).flatten()
            start = route_id * self.rank
            stop = start + self.rank
            selected = normalized.index_select(0, rows)
            low = F.linear(selected, self.down.weight[start:stop])
            delta = F.linear(F.silu(low), self.up.weight[:, start:stop])
            outputs.index_copy_(0, rows, delta)
        return outputs


class SparseCapabilityRouter(nn.Module):
    def __init__(self, vocabulary: int, buckets: int, classes: int) -> None:
        super().__init__()
        self.bpe = nn.EmbeddingBag(vocabulary, classes, mode="mean")
        self.character = nn.EmbeddingBag(buckets, classes, mode="mean")
        self.bias = nn.Parameter(torch.zeros(classes))

    def forward(
        self,
        bpe_ids: torch.Tensor,
        bpe_offsets: torch.Tensor,
        character_ids: torch.Tensor,
        character_offsets: torch.Tensor,
    ) -> torch.Tensor:
        return self.bpe(bpe_ids, bpe_offsets) + self.character(
            character_ids, character_offsets
        ) + self.bias


class _DeclaredTokenizer:
    def __init__(self, document: Mapping[str, Any], eos_token_id: int) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - environment gate
            raise RouteIsolatedCoreError(
                "v17 requires the optional external-tokenizer dependency"
            ) from exc
        raw = json.dumps(document, sort_keys=True, separators=(",", ":"))
        self._tokenizer = Tokenizer.from_str(raw)
        self.eos_token_id = eos_token_id
        if self._tokenizer.get_vocab_size(with_added_tokens=True) <= eos_token_id:
            raise RouteIsolatedCoreError("declared EOS is outside the tokenizer")

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, values: Sequence[int]) -> str:
        return self._tokenizer.decode(list(values), skip_special_tokens=True)


def _fnv1a(data: bytes, seed: int) -> int:
    value = (2166136261 ^ seed) & 0xFFFFFFFF
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _character_features(
    text: str, buckets: int, minimum: int, maximum: int, seed: int
) -> list[int]:
    normalized = " ".join(text.casefold().split())
    bounded = "^" + normalized + "$"
    features = []
    for width in range(minimum, maximum + 1):
        for start in range(max(0, len(bounded) - width + 1)):
            gram = bounded[start : start + width].encode("utf-8")
            features.append(_fnv1a(gram, seed) % buckets)
    if not features:
        features.append(_fnv1a(bounded.encode("utf-8"), seed) % buckets)
    return features


def _semantic_segments(text: str) -> list[str]:
    lines = text.splitlines()
    if len(lines) < 2:
        return [text.strip()]
    first = lines[0].strip()
    remainder = "\n".join(lines[1:]).strip()
    return [part for part in (first, remainder) if part]


def _bag(sequences: Sequence[Sequence[int]], device: torch.device):
    offsets: list[int] = []
    flattened: list[int] = []
    for sequence in sequences:
        if not sequence:
            raise RouteIsolatedCoreError("router feature sequence is empty")
        offsets.append(len(flattened))
        flattened.extend(sequence)
    return (
        torch.tensor(flattened, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
    )


class RouteIsolatedShallowSparseCoreHost:
    """Install and execute one signed v17 English core without receiver learning."""

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
                abi_version=ROUTE_ISOLATED_CORE_ABI_VERSION,
                abi_hash=ROUTE_ISOLATED_CORE_ABI_SHA256,
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
                    }
                ),
            ),
            trust_store=trust_store,
            strict_signatures=True,
        )
        self.device = torch.device(device)
        self.model: ShallowSparseEnglishCore | None = None
        self.router: SparseCapabilityRouter | None = None
        self.residual: RouteIsolatedResidual | None = None
        self.model_tokenizer: _DeclaredTokenizer | None = None
        self.router_tokenizer: Utf8ConcatenativeBpeTokenizer | None = None
        self.router_config: dict[str, int] = {}
        self.guard: dict[str, Any] = {}
        self.handles: list[Any] = []
        self.active_cake_id: str | None = None
        self.active_archive_hash: str | None = None
        self.active_payload_hash: str | None = None
        self.receiver_training_steps = 0
        self.receiver_calibration_runs = 0

    @staticmethod
    def _validate_role(package: CakePackage) -> None:
        manifest = package.manifest
        if not package.signed or manifest.cake_type != "portable_decoder":
            raise RouteIsolatedCoreError("v17 requires a signed portable decoder")
        if (
            manifest.abi_version != ROUTE_ISOLATED_CORE_ABI_VERSION
            or manifest.abi_hash != ROUTE_ISOLATED_CORE_ABI_SHA256
        ):
            raise RouteIsolatedCoreError("v17 ABI identity mismatch")
        if manifest.domains != (ROLE,) or manifest.dependencies:
            raise RouteIsolatedCoreError("v17 package is not an exclusive English core")
        if manifest.input_contract != {
            "external": "UTF-8 bytes",
            "role": ROLE,
            "validity": "strict_utf8",
        }:
            raise RouteIsolatedCoreError("v17 input contract mismatch")
        if manifest.output_contract != {
            "external": "UTF-8 bytes",
            "role": ROLE,
            "composition": COMPOSITION,
            "validity": "strict_utf8",
        }:
            raise RouteIsolatedCoreError("v17 output contract mismatch")
        required = set(manifest.minimum_host_capabilities.get("features", ()))
        expected = {
            "byte_input",
            "safe_tensors",
            "persistent_incremental_state",
            "physical_route_isolation",
            "declarative_runtime_guard",
            "strict_utf8_boundary",
        }
        if not expected <= required:
            raise RouteIsolatedCoreError("v17 host capabilities are incomplete")

    @staticmethod
    def _namespace(
        tensors: Mapping[str, torch.Tensor], prefix: str
    ) -> dict[str, torch.Tensor]:
        values = {
            name[len(prefix) :]: tensor
            for name, tensor in tensors.items()
            if name.startswith(prefix)
        }
        if not values:
            raise RouteIsolatedCoreError(f"missing {prefix} tensor namespace")
        return values

    @staticmethod
    def _architecture(package: CakePackage) -> dict[str, Any]:
        architecture = package.manifest.architecture
        required = {
            "format",
            "model",
            "model_tokenizer",
            "router",
            "router_tokenizer",
            "residual",
            "capabilities",
            "capability_to_task_route",
            "weak_capabilities",
            "guard",
        }
        if set(architecture) != required or architecture.get("format") != ARCHITECTURE_FORMAT:
            raise RouteIsolatedCoreError("v17 architecture declaration changed")
        if tuple(architecture["capabilities"]) != CAPABILITIES:
            raise RouteIsolatedCoreError("v17 capability order changed")
        if architecture["capability_to_task_route"] != CAPABILITY_TO_TASK_ROUTE:
            raise RouteIsolatedCoreError("v17 task-route map changed")
        if tuple(architecture["weak_capabilities"]) != WEAK_CAPABILITIES:
            raise RouteIsolatedCoreError("v17 weak-capability order changed")
        guard = architecture["guard"]
        if (
            guard.get("predicate")
            != "contiguous_1_to_16_token_span_repeated_4_times_or_fourgram_diversity_below_0.35_at_32_tokens"
            or guard.get("scope") != "weak_capabilities_only"
            or guard.get("stop_before_collapsing_token") is not True
            or not isinstance(guard.get("abstention_markers"), list)
            or not guard["abstention_markers"]
            or not isinstance(guard.get("abstention_clause"), str)
            or not guard["abstention_clause"]
        ):
            raise RouteIsolatedCoreError("v17 guard declaration changed")
        return architecture

    def _attach(self) -> None:
        if self.model is None or self.residual is None:
            raise RouteIsolatedCoreError("v17 modules are incomplete")

        def hook(module, args, kwargs):
            hidden = args[0]
            routes = getattr(module, "_layercake_residual_routes", None)
            if routes is None or routes.shape[0] != hidden.shape[0]:
                raise RouteIsolatedCoreError("v17 residual route is absent")
            active = routes.ge(0)
            if not bool(active.any()):
                return args, kwargs
            delta = self.residual.delta(hidden, routes)
            delta = delta * active.to(delta.dtype)[:, None, None]
            return (hidden + delta, *args[1:]), kwargs

        self.handles = [
            block.register_forward_pre_hook(hook, with_kwargs=True)
            for block in self.model.transformer.h
        ]

    def _set_residual_route(self, route: int) -> None:
        if self.model is None:
            raise RouteIsolatedCoreError("v17 core is inactive")
        value = torch.tensor([route], dtype=torch.long, device=self.device)
        for block in self.model.transformer.h:
            block._layercake_residual_routes = value

    def activate(self, source: str | Path) -> dict[str, Any]:
        inspected = self.installer.inspect(source)
        self._validate_role(inspected)
        record = self.installer.install(source)
        package = load_package(
            record["blob"], trust_store=self.installer.trust_store, require_signature=True
        )
        self._validate_role(package)
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
            raise RouteIsolatedCoreError("v17 router configuration changed")
        router = SparseCapabilityRouter(
            int(router_config["vocabulary"]),
            int(router_config["character_hash_buckets"]),
            int(router_config["classes"]),
        )
        router.load_state_dict(self._namespace(package.tensors, "router."), strict=True)
        residual_config = architecture["residual"]
        if set(residual_config) != {"width", "rank", "routes", "reuse"} or residual_config["reuse"] != "before_each_transformer_block":
            raise RouteIsolatedCoreError("v17 residual configuration changed")
        if (
            int(residual_config["width"]) != model_config.width
            or int(residual_config["rank"]) != 16
            or int(residual_config["routes"]) != len(WEAK_CAPABILITIES)
        ):
            raise RouteIsolatedCoreError("v17 residual geometry changed")
        residual = RouteIsolatedResidual(
            int(residual_config["width"]),
            int(residual_config["rank"]),
            int(residual_config["routes"]),
        )
        residual.load_state_dict(
            self._namespace(package.tensors, "residual."), strict=True
        )
        model_tokenizer = architecture["model_tokenizer"]
        if set(model_tokenizer) != {"format", "tokenizers_json", "sha256", "eos_token_id"} or model_tokenizer["format"] != "declarative-tokenizers-json/1":
            raise RouteIsolatedCoreError("v17 model tokenizer declaration changed")
        canonical_tokenizer = json.dumps(
            model_tokenizer["tokenizers_json"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if hashlib.sha256(canonical_tokenizer).hexdigest() != model_tokenizer["sha256"]:
            raise RouteIsolatedCoreError("v17 model tokenizer hash mismatch")
        declared = _DeclaredTokenizer(
            model_tokenizer["tokenizers_json"], int(model_tokenizer["eos_token_id"])
        )
        router_tokenizer = Utf8ConcatenativeBpeTokenizer.from_document(
            architecture["router_tokenizer"]
        )
        if router_tokenizer.vocab_size != int(router_config["vocabulary"]):
            raise RouteIsolatedCoreError("v17 router tokenizer vocabulary mismatch")
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
        }

    def _require_active(self):
        values = (
            self.model,
            self.router,
            self.residual,
            self.model_tokenizer,
            self.router_tokenizer,
        )
        if any(value is None for value in values):
            raise RouteIsolatedCoreError("no v17 English core is active")
        return values

    def _features(self, text: str) -> tuple[list[int], list[int]]:
        tokenizer = self.router_tokenizer
        if tokenizer is None:
            raise RouteIsolatedCoreError("v17 router tokenizer is absent")
        try:
            bpe = [tokenizer.lexeme_to_id[item] for item in tokenizer.split(text)]
        except KeyError as exc:
            raise RouteIsolatedCoreError("prompt contains an unknown router lexeme") from exc
        if not bpe:
            raise RouteIsolatedCoreError("empty router segment")
        return bpe, _character_features(
            text,
            self.router_config["character_hash_buckets"],
            self.router_config["character_ngram_minimum"],
            self.router_config["character_ngram_maximum"],
            self.router_config["hash_seed"],
        )

    @torch.inference_mode()
    def route(self, prompt: str) -> str:
        if self.router is None:
            raise RouteIsolatedCoreError("v17 router is inactive")
        segments = _semantic_segments(prompt)
        values = [self._features(segment) for segment in segments]
        bpe_ids, bpe_offsets = _bag([value[0] for value in values], self.device)
        character_ids, character_offsets = _bag(
            [value[1] for value in values], self.device
        )
        probabilities = self.router(
            bpe_ids, bpe_offsets, character_ids, character_offsets
        ).softmax(dim=-1)
        labels = (*CAPABILITIES, METADATA_LABEL)
        details = []
        for row in probabilities:
            predicted = int(row.argmax())
            probability, index = row[: len(CAPABILITIES)].max(dim=0)
            details.append(
                {
                    "predicted": labels[predicted],
                    "best_capability": CAPABILITIES[int(index)],
                    "best_probability": float(probability),
                }
            )
        eligible = [value for value in details if value["predicted"] != METADATA_LABEL]
        selected = max(
            eligible if eligible else details,
            key=lambda value: value["best_probability"],
        )
        return str(selected["best_capability"])

    @torch.inference_mode()
    def prefill(self, prompt: bytes | str) -> dict[str, Any]:
        model, _, _, tokenizer, _ = self._require_active()
        if isinstance(prompt, bytes):
            text = prompt.decode("utf-8", errors="strict")
        else:
            prompt.encode("utf-8", errors="strict")
            text = prompt
        capability = self.route(text)
        weak_route = (
            WEAK_CAPABILITIES.index(capability)
            if capability in WEAK_CAPABILITIES
            else -1
        )
        self._set_residual_route(weak_route)
        prompt_ids = tokenizer.encode(text.rstrip() + "\n")
        if not prompt_ids:
            raise RouteIsolatedCoreError("v17 prompt encodes to no tokens")
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        forced = (
            torch.tensor(
                [CAPABILITY_TO_TASK_ROUTE[capability]],
                dtype=torch.long,
                device=self.device,
            )
            if weak_route >= 0
            else None
        )
        result = model(
            ids,
            prompt_lengths=torch.tensor(
                [len(prompt_ids)], dtype=torch.long, device=self.device
            ),
            task_routes=forced,
            use_cache=True,
        )
        return {
            "past_key_values": result["past_key_values"],
            "task_route": result["task_routes"].detach().clone(),
            "capability": capability,
            "weak_route": weak_route,
            "next_logits": result["logits"][:, -1],
            "generated_ids": [],
            "terminated_by_guard": False,
            "finished": False,
        }

    @torch.inference_mode()
    def decode_step(self, state: dict[str, Any]) -> int | None:
        model, _, _, tokenizer, _ = self._require_active()
        if state["finished"]:
            return None
        self._set_residual_route(int(state["weak_route"]))
        selected = state["next_logits"].argmax(dim=-1)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            state["finished"] = True
            return None
        candidate = [*state["generated_ids"], token]
        if int(state["weak_route"]) >= 0 and repetition_collapse(
            tokenizer.decode(candidate)
        ):
            state["terminated_by_guard"] = True
            state["finished"] = True
            return None
        state["generated_ids"].append(token)
        result = model(
            selected[:, None],
            task_routes=state["task_route"],
            past_key_values=state["past_key_values"],
            use_cache=True,
        )
        state["past_key_values"] = result["past_key_values"]
        state["next_logits"] = result["logits"][:, -1]
        return token

    def realize(self, state: Mapping[str, Any]) -> bytes:
        _, _, _, tokenizer, _ = self._require_active()
        value = tokenizer.decode(state["generated_ids"])
        if state["capability"] == "abstention" and not any(
            marker.casefold() in value.casefold()
            for marker in self.guard["abstention_markers"]
        ):
            clause = str(self.guard["abstention_clause"])
            value = clause + (" " + value if value else "")
        raw = value.encode("utf-8", errors="strict")
        raw.decode("utf-8", errors="strict")
        return raw

    def generate(
        self, prompt: bytes | str, *, maximum_tokens: int = 128
    ) -> bytes:
        if maximum_tokens < 1:
            raise RouteIsolatedCoreError("maximum_tokens must be positive")
        state = self.prefill(prompt)
        for _ in range(maximum_tokens):
            if self.decode_step(state) is None:
                break
        return self.realize(state)

    def verify(self) -> dict[str, Any]:
        if self.active_cake_id is None:
            raise RouteIsolatedCoreError("no v17 English core is active")
        result = self.installer.verify(self.active_cake_id)
        if (
            result["archive_hash"] != self.active_archive_hash
            or result["payload_hash"] != self.active_payload_hash
        ):
            raise RouteIsolatedCoreError("active v17 package identity changed")
        return {**result, "role": ROLE, "utf8": "STRICT"}

    def remove(self) -> dict[str, Any]:
        if self.active_cake_id is None:
            raise RouteIsolatedCoreError("no v17 English core is active")
        result = self.installer.remove(self.active_cake_id)
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.model = None
        self.router = None
        self.residual = None
        self.model_tokenizer = None
        self.router_tokenizer = None
        self.active_cake_id = None
        self.active_archive_hash = None
        self.active_payload_hash = None
        return result
