"""Context-sliced CPU decoder for high-throughput CountCake generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class _OrderIndex:
    order: int
    encoding: str
    keys: np.ndarray
    targets: np.ndarray
    counts: np.ndarray
    contexts: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    totals: np.ndarray
    hash_bits: int


class CountCakeCPUDecoder:
    """Exact CPU decoder that indexes observed continuations by context.

    The generic tensor reference searches all 256 possible joint keys.  This
    index performs one scalar search per n-gram order and updates only byte
    continuations that were actually observed for the matched context.
    """

    def __init__(self, model) -> None:
        if next(model.parameters()).device.type != "cpu":
            raise ValueError("CountCakeCPUDecoder requires a CPU model")
        self.model = model.eval()
        self.max_order = model.count_cake.max_order
        self.backoff_mode = model.count_cake.backoff_mode
        unigram = model.count_cake.unigram_counts.detach().numpy().astype(np.float32)
        self.base_probability = (unigram + 0.5) / (unigram.sum() + 128.0)
        self.strengths = tuple(
            model.count_cake.backoff_strengths[: self.max_order]
        )
        self._probability_cache: dict[tuple[int, ...], np.ndarray] = {}
        self._feature_cache: dict[tuple[int, ...], np.ndarray] = {}
        self._probability_cache_limit = 8192
        self._certificate_lower = np.empty(256, dtype=np.float64)
        self._certificate_upper = np.empty(256, dtype=np.float64)
        indices: list[_OrderIndex] = []
        for order in range(1, self.max_order + 1):
            keys = (
                getattr(model.count_cake, f"keys_{order}")
                .detach()
                .numpy()
            )
            counts = (
                getattr(model.count_cake, f"counts_{order}")
                .detach()
                .numpy()
                .astype(np.float32)
            )
            context_ids, starts = np.unique(keys >> 8, return_index=True)
            ends = np.concatenate((starts[1:], np.array([keys.size])))
            totals = np.add.reduceat(counts, starts)
            encoding = model.count_cake.order_encodings[order - 1]
            contexts = (
                context_ids
                if encoding == "packed"
                else getattr(model.count_cake, f"context_keys_{order}")
                .detach()
                .numpy()
            )
            if contexts.size != starts.size:
                raise ValueError("CountCake context index is not dense")
            indices.append(
                _OrderIndex(
                    order=order,
                    encoding=encoding,
                    keys=keys,
                    targets=(keys & 255).astype(np.uint8),
                    counts=counts,
                    contexts=contexts,
                    starts=starts,
                    ends=ends,
                    totals=totals,
                    hash_bits=model.count_cake.context_hash_bits[order - 1],
                )
            )
        self.indices = tuple(indices)

    def _certified_cache_byte(self, memory, history: bytearray) -> int | None:
        """Return an argmax proven independently of the uncomputed base model."""
        if memory is None or memory.recent is None or memory.normalized is None:
            return None
        lower = self._certificate_lower
        upper = self._certificate_upper
        lower.fill(0.0)
        upper.fill(1.0)
        recent = memory.recent
        for stage, (order, strength) in zip(recent._recent, recent.specs):
            if len(history) < order:
                continue
            match = stage.get(bytes(history[-order:]))
            if match is None or recent.position - match[1] > recent.window:
                continue
            scale = strength / (1.0 + strength)
            addition = 1.0 / (1.0 + strength)
            lower *= scale
            upper *= scale
            lower[int(match[0])] += addition
            upper[int(match[0])] += addition
        normalized = memory.normalized
        for stage, (order, strength) in zip(
            normalized._counts, normalized.specs
        ):
            if len(history) < order:
                continue
            continuations = stage.get(normalized._context(history, order))
            if not continuations:
                continue
            total = sum(continuations.values())
            denominator = total + strength
            lower *= strength / denominator
            upper *= strength / denominator
            for target, count in continuations.items():
                addition = count / denominator
                lower[int(target)] += addition
                upper[int(target)] += addition
        candidate = int(lower.argmax())
        competitor = max(
            float(upper[:candidate].max(initial=-np.inf)),
            float(upper[candidate + 1 :].max(initial=-np.inf)),
        )
        return candidate if float(lower[candidate]) > competitor else None

    def next_probabilities(
        self,
        history: list[int],
        *,
        return_features: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if len(history) < self.max_order:
            raise ValueError(
                f"optimized decoding requires {self.max_order} history bytes"
            )
        cache_key = tuple(history[-self.max_order :])
        cached = self._probability_cache.get(cache_key)
        if cached is not None:
            if return_features:
                return cached, self._feature_cache[cache_key]
            return cached
        probability = self.base_probability.copy()
        matched_order = 0.0
        matched_total = 0.0
        matched_density = 0.0
        matched_peak = 0.0
        for order_index, index in enumerate(self.indices):
            if index.encoding == "packed":
                context = sum(
                    int(history[-1 - lag]) << (8 * lag)
                    for lag in range(index.order)
                )
            else:
                context = 0
                for byte in history[-index.order :]:
                    context = (context * 257 + int(byte) + 1) & (
                        (1 << index.hash_bits) - 1
                    )
            location = int(np.searchsorted(index.contexts, context))
            if location >= index.contexts.size or index.contexts[location] != context:
                continue
            total = index.totals[location]
            distinct = float(index.ends[location] - index.starts[location])
            matched_order = index.order / max(self.max_order, 1)
            matched_total = float(np.log1p(total) / 16.0)
            matched_density = distinct / max(float(total), 1.0)
            strength = (
                distinct
                if self.backoff_mode == "distinct"
                else self.strengths[order_index]
            )
            probability *= strength / (total + strength)
            start = index.starts[location]
            end = index.ends[location]
            targets = index.targets[start:end]
            matched_peak = float(index.counts[start:end].max()) / max(
                float(total), 1.0
            )
            probability[targets] += index.counts[start:end] / (total + strength)
        if len(self._probability_cache) >= self._probability_cache_limit:
            self._probability_cache.clear()
            self._feature_cache.clear()
        features = np.asarray(
            [matched_order, matched_total, matched_density, matched_peak],
            dtype=np.float32,
        )
        self._probability_cache[cache_key] = probability
        self._feature_cache[cache_key] = features
        if return_features:
            return probability, features
        return probability

    def clear_cache(self) -> None:
        self._probability_cache.clear()
        self._feature_cache.clear()

    @torch.no_grad()
    def _neural_patch(self, context: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        model = self.model
        if model.local_recurrent:
            raise ValueError("recurrent local decoding is computed byte by byte")
        composed = context + model.from_abi(model.to_abi(context))
        positions = torch.arange(model.patch_size)
        local = model.local_norm(
            model.local_projection(composed).unsqueeze(-2)
            + model.local_positions(positions).unsqueeze(0)
        )[0]
        if model.byte_head == "direct":
            probability = torch.softmax(model.direct_head(local), dim=-1)
        else:
            high_log_probability = torch.log_softmax(
                model.high_head(local), dim=-1
            )
            high_values = torch.arange(16)
            low_hidden = model.low_norm(
                local.unsqueeze(-2)
                * (1.0 + model.high_scale(high_values))
                + model.high_embedding(high_values)
            )
            low_log_probability = torch.log_softmax(
                model.low_head(low_hidden), dim=-1
            )
            probability = (
                high_log_probability.unsqueeze(-1) + low_log_probability
            ).flatten(-2).exp()
        return probability.numpy(), local

    @torch.no_grad()
    def generate_cached(
        self,
        state: dict,
        *,
        patches: int = 1,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> torch.Tensor:
        if patches <= 0:
            raise ValueError("patches must be positive")
        if temperature > 0 and rng is None:
            rng = np.random.default_rng()
        history = [int(value) for value in state["history"].tolist()]
        online_cache = state.get("online_cache")
        online_history = state.get("online_history")
        outputs: list[torch.Tensor] = []
        certified_bytes = 0
        exact_bytes = 0
        for _ in range(patches):
            if self.model.local_recurrent:
                context = state["recurrent_state"].squeeze(0)
                composed = context + self.model.from_abi(self.model.to_abi(context))
                local_state = self.model.local_projection(composed).unsqueeze(0)
                local_input = self.model.local_bos.reshape(1, 1, -1)
                neural = local_patch = None
            else:
                neural, local_patch = self._neural_patch(
                    state["recurrent_state"].squeeze(0)
                )
            generated: list[int] = []
            offset = 0
            while offset < self.model.patch_size:
                if (
                    temperature <= 0
                    and online_cache is not None
                ):
                    certified_run: list[int] = []
                    while offset + len(certified_run) < self.model.patch_size:
                        byte = self._certified_cache_byte(
                            online_cache, online_history
                        )
                        if byte is None:
                            break
                        history.append(byte)
                        generated.append(byte)
                        certified_run.append(byte)
                        online_cache.update(online_history, byte)
                        online_history.append(byte)
                        if len(online_history) > online_cache.max_order:
                            del online_history[
                                : len(online_history) - online_cache.max_order
                            ]
                    if certified_run:
                        if self.model.local_recurrent:
                            emitted = torch.tensor(
                                certified_run, dtype=torch.long
                            ).reshape(1, -1)
                            teacher = torch.cat(
                                [
                                    local_input,
                                    self.model.byte_embedding(emitted[:, :-1]),
                                ],
                                dim=1,
                            )
                            _, local_state = self.model.local_core(
                                teacher, local_state
                            )
                            local_input = self.model.byte_embedding(
                                emitted[:, -1]
                            ).reshape(1, 1, -1)
                        offset += len(certified_run)
                        certified_bytes += len(certified_run)
                        continue
                if self.model.local_recurrent:
                    _, local_state = self.model.local_core(local_input, local_state)
                    local = self.model.local_norm(
                        local_state.squeeze(0)
                        + self.model.local_positions.weight[offset]
                    )[0]
                    if self.model.byte_head == "direct":
                        neural_tensor = torch.softmax(
                            self.model.direct_head(local), dim=-1
                        )
                    else:
                        high_log_probability = torch.log_softmax(
                            self.model.high_head(local), dim=-1
                        )
                        high_values = torch.arange(16)
                        low_hidden = self.model.low_norm(
                            local.unsqueeze(0)
                            * (1.0 + self.model.high_scale(high_values))
                            + self.model.high_embedding(high_values)
                        )
                        low_log_probability = torch.log_softmax(
                            self.model.low_head(low_hidden), dim=-1
                        )
                        neural_tensor = (
                            high_log_probability.unsqueeze(-1)
                            + low_log_probability
                        ).flatten().exp()
                    neural_step = neural_tensor.numpy()
                else:
                    neural_step = neural[offset]
                    local = local_patch[offset]
                    neural_tensor = torch.from_numpy(neural_step)
                count_probability, count_features = self.next_probabilities(
                    history, return_features=True
                )
                expert_confidence = (
                    self.model._expert_confidence_features(neural_tensor)
                    if self.model.expert_confidence_gate_enabled
                    else None
                )
                gate = float(
                    torch.sigmoid(
                        self.model._gate_logits(
                            local,
                            torch.from_numpy(count_features),
                            expert_confidence,
                        )
                    ).squeeze()
                )
                probability = (
                    (1.0 - gate) * count_probability
                    + gate * neural_step
                )
                if online_cache is not None:
                    probability = online_cache.probabilities_numpy(
                        probability, online_history
                    )
                if temperature <= 0:
                    byte = int(probability.argmax())
                else:
                    logits = np.log(np.maximum(probability, 1e-30)) / temperature
                    probability = np.exp(logits - logits.max())
                    probability /= probability.sum()
                    byte = int(rng.choice(256, p=probability))
                history.append(byte)
                generated.append(byte)
                exact_bytes += 1
                if online_cache is not None:
                    online_cache.update(online_history, byte)
                    online_history.append(byte)
                    if len(online_history) > online_cache.max_order:
                        del online_history[
                            : len(online_history) - online_cache.max_order
                        ]
                if self.model.local_recurrent:
                    local_input = self.model.byte_embedding(
                        torch.tensor(byte)
                    ).reshape(1, 1, -1)
                offset += 1
            patch = torch.tensor(generated, dtype=torch.long).unsqueeze(0)
            outputs.append(patch)
            state["history"] = torch.tensor(
                history[-self.max_order :], dtype=torch.long
            )
            feature = torch.tanh(
                self.model.patch_projection(
                    self.model.byte_embedding(patch).flatten(-2)
                )
            ).unsqueeze(1)
            _, state["recurrent_state"] = self.model.patch_core(
                feature,
                state["recurrent_state"],
            )
        state["cpu_certified_bytes"] = certified_bytes
        state["cpu_exact_bytes"] = exact_bytes
        return torch.cat(outputs, dim=-1)
