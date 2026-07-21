import json
from pathlib import Path


RESULTS = Path("results/breakthrough_equal")


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_northstar_v23_certificate_is_fail_closed_pass():
    certificate = _load("northstar_v23_release_certificate.json")

    assert certificate["status"] == "PASS"
    assert certificate["failed_required"] == []
    assert certificate["required_gates"]
    assert all(certificate["required_gates"].values())
    assert certificate["limitations"]["training_scope"]
    assert certificate["limitations"]["legacy_decoder_scope"]


def test_northstar_v23_training_artifact_recomputes_above_five_x():
    artifact = _load("northstar_v23_domain_cake_training_speed.json")

    assert artifact["status"] == "PASS"
    assert artifact["protocol"]["layercake_training_mode"] == (
        "shared3_routed_tail_int8_foundation"
    )
    assert artifact["protocol"]["cpu_batch_size"] == 16
    assert artifact["protocol"]["gpu_batch_size"] == 128
    for device in ("cpu", "cuda"):
        details = artifact["devices"][device]["repeat_details"]
        ratios = [
            layercake["logical_bytes_per_second"]
            / transformer["logical_bytes_per_second"]
            for layercake, transformer in zip(
                details["layercake"], details["transformer"]
            )
        ]
        assert len(ratios) >= 3
        assert min(ratios) >= 5.0
        assert artifact["devices"][device]["status"] == "PASS"


def test_northstar_v23_migration_and_route_isolation_are_exact():
    migration = _load("northstar_v23_lossless_migration.json")
    isolation = _load("northstar_v23_route_isolation.json")

    assert migration["status"] == "PASS"
    assert migration["verification"]["next_byte_logits_max_abs_diff"] == 0.0
    assert migration["verification"]["patch_prediction_max_abs_diff"] == 0.0
    assert isolation["status"] == "PASS"
    assert isolation["route_zero_generation_path_bit_exact"] is True
    assert isolation["training"]["loss_ratio_final_over_first"] < 1.0
