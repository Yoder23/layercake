"""Progressive replacement core with a source-native causal prompt boundary."""

from __future__ import annotations

import torch

from .progressive_replacement_core import (
    ProgressiveReplacementCore,
    ProgressiveReplacementCoreState,
)


class SourceAlignedProgressiveReplacementCore(ProgressiveReplacementCore):
    """Predict the first response action from the final prompt action.

    Unlike older direct-core constructs, this boundary does not inject a host
    BOS action between the externally tokenized prompt and generated response.
    """

    def prefill_ids(
        self, source_ids: list[int], source_lexemes: list[bytes]
    ) -> ProgressiveReplacementCoreState:
        if not source_ids or len(source_ids) > self.maximum_source_actions:
            raise ValueError("source-aligned progressive replacement source is empty or exceeds bound")
        device = self.token_embedding.weight.device
        values = torch.tensor([source_ids], dtype=torch.long, device=device)
        positions = torch.arange(values.shape[1], device=device)
        hidden = self.token_embedding(values)
        keys: list[torch.Tensor] = []
        cached_values: list[torch.Tensor] = []
        for layer in self.layers:
            hidden, key, value = layer.forward_with_cache(hidden, positions)
            keys.append(key)
            cached_values.append(value)
        logits = self.lm_head(self.final_norm(hidden[:, -1]))
        return ProgressiveReplacementCoreState(
            values,
            source_lexemes,
            [],
            tuple(keys),
            tuple(cached_values),
            logits,
            values.shape[1],
        )
