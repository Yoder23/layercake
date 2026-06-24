from layercake.rolling.efficiency import (
    ComputeWasteGate,
    PreviewBenefitGate,
    QualityPerStepGate,
    QualityPerTrainableParamGate,
    RollbackRecoveryGate,
    TimeToMetricGate,
    TrainingRegressionGate,
    TransformerBaselineGate,
)


def test_efficiency_gates_pass_and_fail():
    ctx = {"bpb": 1.0, "parent": {"bpb": 2.0}, "steps": 2, "training_seconds": 1.0, "trainable_params": 10}
    assert TimeToMetricGate(target=1.1, max_seconds=2).run(ctx).passed
    assert QualityPerStepGate(min_gain_per_step=0.4).run(ctx).passed
    assert QualityPerTrainableParamGate(min_gain_per_param=0.09).run(ctx).passed
    assert not TrainingRegressionGate(max_regression=-2.0).run(ctx).passed
    assert not ComputeWasteGate(patience=2).run({"loss_history": [1.0, 1.0, 1.0]}).passed
    assert RollbackRecoveryGate().run({"rollback": {"restored_commit": "abc"}}).passed
    assert PreviewBenefitGate().run({"preview_guided": {"bpb": 1.0}, "blind": {"bpb": 1.2}}).passed
    assert TransformerBaselineGate(max_delta=0.1).run({"layercake": {"bpb": 1.0}, "transformer": {"bpb": 1.05}}).passed
