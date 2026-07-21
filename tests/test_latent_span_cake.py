import math

import torch

from layercake.latent_span_cake import LatentSpanCakeLM


def _model() -> LatentSpanCakeLM:
    torch.manual_seed(7)
    return LatentSpanCakeLM(
        span_bytes=4,
        d_byte=8,
        d_model=16,
        layers=1,
        latent_states=8,
        d_abi=4,
    )


def test_span_probability_is_causal() -> None:
    model = _model().eval()
    rows = torch.randint(0, 256, (2, 16))
    changed = rows.clone()
    changed[:, 12:] = torch.randint(0, 256, (2, 4))
    original = model.span_log_probs(rows)
    modified = model.span_log_probs(changed)
    torch.testing.assert_close(original[:, :2], modified[:, :2])


def test_emission_components_are_normalized() -> None:
    model = _model().eval()
    high = model.emission_high_logits.log_softmax(-1)
    low = model.emission_low_logits.log_softmax(-1)
    byte = (high.unsqueeze(-1) + low).flatten(-2).exp()
    torch.testing.assert_close(
        byte.sum(-1), torch.ones_like(byte.sum(-1)), atol=1e-6, rtol=1e-6
    )


def test_loss_is_finite_per_byte() -> None:
    model = _model()
    rows = torch.randint(0, 256, (3, 20))
    loss = model.loss(rows)
    assert loss.ndim == 0
    assert math.isfinite(float(loss))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_generation_appends_complete_raw_byte_spans() -> None:
    model = _model().eval()
    prompt = torch.randint(0, 256, (2, 8))
    generated = model.generate_spans(prompt, spans=3)
    assert generated.shape == (2, 20)
    assert int(generated.min()) >= 0
    assert int(generated.max()) <= 255


def test_autoregressive_latent_emission_is_causal_and_trainable() -> None:
    model = LatentSpanCakeLM(
        span_bytes=4,
        d_byte=8,
        d_model=16,
        layers=1,
        latent_states=4,
        d_abi=4,
        emission_mode="autoregressive",
        local_width=12,
    )
    rows = torch.randint(0, 256, (2, 16))
    changed = rows.clone()
    changed[:, 12:] = torch.randint(0, 256, (2, 4))
    original = model.span_log_probs(rows)
    modified = model.span_log_probs(changed)
    torch.testing.assert_close(original[:, :2], modified[:, :2])
    model.loss(rows).backward()
    assert model.local_core.weight_hh_l0.grad is not None
