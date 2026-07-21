import importlib.util
from pathlib import Path
import sys

import torch


def load_module():
    path = Path("scripts/benchmark_micro_scale_curriculum_frontier_v2.py")
    spec = importlib.util.spec_from_file_location("micro_frontier_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_empirical_byte_priors_are_shaped_and_finite():
    module = load_module()
    stream = torch.tensor(
        list(b"the the cat sat. the cat ate."),
        dtype=torch.long,
    )
    priors = module._build_empirical_byte_priors(
        stream,
        context_specs={(128, 2)},
    )
    transition = priors["transition_logits"]
    context = priors["context_logits"][(128, 2)]
    assert transition.shape == (256, 256)
    assert context.shape == (128, 256)
    assert torch.isfinite(transition).all()
    assert torch.isfinite(context).all()
    assert int(transition[ord("t")].argmax()) == ord("h")


def test_repeat_unlikelihood_penalizes_recent_repeated_bytes():
    module = load_module()
    x = torch.tensor([[ord("a"), ord("b"), ord("a"), ord("c")]], dtype=torch.long)
    y = torch.tensor([[ord("b"), ord("a"), ord("c"), ord("d")]], dtype=torch.long)
    logits = torch.zeros((1, 4, 256), dtype=torch.float32)
    logits[..., ord("a")] = 8.0

    loss = module._repeat_unlikelihood_loss(logits, x, y, window=3)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_pick_next_blocks_repeated_ngram():
    module = load_module()
    history = [1, 2, 3, 1, 2]
    logits = torch.full((256,), -10.0)
    logits[3] = 10.0
    logits[4] = 9.0

    nxt = module._pick_next(logits, history, top_k=4, no_repeat_ngram=3)

    assert nxt == 4
