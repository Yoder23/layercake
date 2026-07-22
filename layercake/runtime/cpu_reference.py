"""Correctness-first full-context CPU runtime used only as an oracle."""

from __future__ import annotations

import torch


class CPUReferenceRuntime:
    def __init__(self, model: torch.nn.Module, *, route: int = 0):
        self.model = model.cpu().eval()
        self.route = int(route)

    @torch.inference_mode()
    def generate(self, prompt: bytes | torch.Tensor, count: int) -> torch.Tensor:
        if isinstance(prompt, bytes):
            prompt = torch.tensor(list(prompt), dtype=torch.long)[None]
        generated = prompt.cpu()
        for _ in range(count):
            try:
                logits = self.model(generated, route=self.route)
            except TypeError:
                self.model.set_route(self.route)
                logits = self.model(generated)
            generated = torch.cat([generated, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return generated

