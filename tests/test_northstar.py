from layercake.northstar import NorthStarMetrics


def complete_metrics(**changes):
    values = {
        "parameters": 9,
        "baseline_parameters": 10,
        "heldout_bpb": 1.9,
        "baseline_heldout_bpb": 2.0,
        "training_bytes": 100,
        "baseline_training_bytes": 100,
        "training_seconds": 9,
        "baseline_training_seconds": 10,
        "mobile_prefill_ratio": 1.1,
        "mobile_generation_ratio": 1.1,
        "desktop_prefill_ratio": 1.1,
        "desktop_generation_ratio": 1.1,
        "gpu_prefill_ratio": 1.1,
        "gpu_generation_ratio": 1.1,
        "migration_ppl_ratio": 1.0,
        "migration_max_logit_diff": 0.0,
        "migrated_domain_bpb": 1.5,
        "baseline_domain_bpb": 2.0,
    }
    values.update(changes)
    return NorthStarMetrics(**values)


def test_northstar_requires_every_dimension():
    certificate = complete_metrics().certificate()
    assert certificate["status"] == "PASS"
    assert certificate["failed_required"] == []


def test_northstar_rejects_gpu_only_regression():
    certificate = complete_metrics(gpu_generation_ratio=0.99).certificate()
    assert certificate["status"] == "FAIL"
    assert certificate["failed_required"] == ["faster_gpu_generation"]


def test_northstar_rejects_non_lossless_migration():
    certificate = complete_metrics(migration_ppl_ratio=1.000001).certificate()
    assert certificate["status"] == "FAIL"
    assert "lossless_migration_ppl" in certificate["failed_required"]
