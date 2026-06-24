import torch

from layercake.rolling.baselines import TinyByteTransformer, baseline_bpb, baseline_training_smoke_loop


def test_transformer_baseline_smoke_runs():
    batch = torch.randint(0, 255, (1, 16), dtype=torch.long)
    model = TinyByteTransformer(d_model=16, layers=1, heads=4, max_len=16)
    assert baseline_bpb(model, batch) > 0
    result = baseline_training_smoke_loop(batch, steps=1, target_params=1000)
    assert result["after_bpb"] > 0
