"""Exact count-draft speculative decoding for byte-recurrent CountCakes."""

from __future__ import annotations

import numpy as np
import torch

from .count_cake_cpu import CountCakeCPUDecoder


class CountCakeSpeculativeDecoder:
    """Verify blocks of cheap count proposals with one fused byte recurrence.

    The count cake proposes greedy bytes without evaluating the neural host.
    The byte-level GRU then scores the entire proposed block in one call.  The
    decoder accepts the common prefix and, at the first disagreement, emits
    the target model's byte.  This is exact greedy speculative decoding: every
    emitted byte is identical to serial target-model generation.

    A CPU count index is intentional even for a CUDA target.  Drafting touches
    sparse host-resident state and transfers only a small candidate block and
    its 4-value confidence features to the accelerator.
    """

    def __init__(
        self,
        model,
        draft_decoder: CountCakeCPUDecoder,
        *,
        block_size: int = 32,
    ) -> None:
        if block_size <= 0:
            raise ValueError("speculative block size must be positive")
        if model.patch_size != 1:
            raise ValueError("count speculation currently requires byte patches")
        if model.chunking_mode != "fixed" or model.patch_core_type != "gru":
            raise ValueError("count speculation requires a fixed-patch GRU host")
        if model.patch_layers != 1:
            raise ValueError("count speculation currently requires one GRU layer")
        if model.local_decoder != "position":
            raise ValueError("count speculation requires the position byte head")
        if model.cache_enabled:
            raise ValueError("online-cache speculation is not yet certified")
        if (
            draft_decoder.max_order != model.count_cake.max_order
            or draft_decoder.model.count_cake.state_entries
            != model.count_cake.state_entries
        ):
            raise ValueError("draft and target count cakes do not match")
        self.model = model.eval()
        self.draft_decoder = draft_decoder
        self.block_size = int(block_size)

    def clear_cache(self) -> None:
        self.draft_decoder.clear_cache()

    def _patch_features(self, values: torch.Tensor) -> torch.Tensor:
        embedded = self.model.byte_embedding(values).unsqueeze(-2)
        return torch.tanh(
            self.model.patch_projection(embedded.flatten(-2))
        )

    @torch.no_grad()
    def generate_cached(
        self,
        state: dict,
        *,
        patches: int = 1,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        if patches <= 0:
            raise ValueError("patches must be positive")
        if temperature > 0:
            raise ValueError("exact count speculation currently supports greedy decode")
        device = next(self.model.parameters()).device
        history = [int(value) for value in state["history"].detach().cpu().tolist()]
        outputs: list[torch.Tensor] = []
        remaining = int(patches)
        rounds = 0
        accepted_total = 0
        while remaining:
            width = min(self.block_size, remaining)
            proposed_history = list(history)
            proposal: list[int] = []
            count_probabilities: list[np.ndarray] = []
            count_features: list[np.ndarray] = []
            for _ in range(width):
                probability, features = self.draft_decoder.next_probabilities(
                    proposed_history,
                    return_features=True,
                )
                byte = int(probability.argmax())
                proposal.append(byte)
                count_probabilities.append(probability)
                count_features.append(features)
                proposed_history.append(byte)

            candidate = torch.tensor(
                proposal, device=device, dtype=torch.long
            ).reshape(1, -1)
            hidden, candidate_state = self.model.patch_core(
                self._patch_features(candidate), state["recurrent_state"]
            )
            contexts = torch.cat(
                [state["recurrent_state"][-1].unsqueeze(1), hidden[:, :-1]],
                dim=1,
            )
            composed = contexts + self.model.from_abi(self.model.to_abi(contexts))
            local = self.model.local_norm(
                self.model.local_projection(composed)
                + self.model.local_positions.weight[0]
            )
            neural = self.model._neural_probabilities(local)
            feature_tensor = torch.from_numpy(np.stack(count_features)).to(
                device=device, dtype=local.dtype
            ).unsqueeze(0)
            count_tensor = torch.from_numpy(np.stack(count_probabilities)).to(
                device=device, dtype=neural.dtype
            ).unsqueeze(0)
            expert_confidence = (
                self.model._expert_confidence_features(neural)
                if self.model.expert_confidence_gate_enabled
                else None
            )
            gate = torch.sigmoid(
                self.model._gate_logits(local, feature_tensor, expert_confidence)
            )
            greedy = ((1.0 - gate) * count_tensor + gate * neural).argmax(dim=-1)
            mismatch = torch.nonzero(greedy[0] != candidate[0])
            accepted = width if not mismatch.numel() else int(mismatch[0, 0].item())
            accepted_total += accepted
            rounds += 1

            if accepted == width:
                actual = candidate
                state["recurrent_state"] = candidate_state
            else:
                correct = greedy[:, accepted : accepted + 1]
                actual = torch.cat([candidate[:, :accepted], correct], dim=1)
                accepted_state = (
                    state["recurrent_state"]
                    if accepted == 0
                    else hidden[:, accepted - 1].unsqueeze(0)
                )
                _, state["recurrent_state"] = self.model.patch_core(
                    self._patch_features(correct), accepted_state
                )

            actual_values = actual[0].detach().cpu().tolist()
            history.extend(int(value) for value in actual_values)
            history = history[-self.model.count_cake.max_order :]
            outputs.append(actual)
            remaining -= actual.shape[1]

        state["history"] = torch.tensor(
            history, device=device, dtype=torch.long
        )
        state["speculative_rounds"] = rounds
        state["speculative_accepted_bytes"] = accepted_total
        state["speculative_emitted_bytes"] = int(patches)
        return torch.cat(outputs, dim=-1)
