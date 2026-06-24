from layercake.rolling.early_stop import EarlyStopper


def test_early_stop_triggers_on_divergence_and_stall():
    diverge = EarlyStopper(divergence_factor=2.0)
    assert not diverge.update(1.0, step=0).should_stop
    assert diverge.update(3.0, step=1).reason == "loss_diverged"
    stall = EarlyStopper(patience=2, min_delta=0.0)
    stall.update(1.0, step=0)
    stall.update(1.0, step=1)
    assert stall.update(1.0, step=2).reason == "no_improvement"
