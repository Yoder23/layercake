from scripts.write_mobile_local_northstar_report import build_report


def _fair(exact: float = 0.25) -> dict:
    return {
        "best_current_candidate": "candidate",
        "candidates": {
            "candidate": {
                "cpu_heldout": {
                    "layercake_exact": exact,
                    "speed_ratio_layercake_over_transformer": 6.0,
                },
                "gpu_heldout": {
                    "layercake_exact": exact,
                    "speed_ratio_layercake_over_transformer": 10.0,
                },
            }
        },
    }


def _game() -> dict:
    return {
        "gates": {
            "layercake_cpu_relevance_full": True,
            "layercake_gpu_relevance_full": True,
            "lossless_game_layer_transfer": True,
        },
        "metrics": {
            "layercake_cpu": {"generation_bytes_per_second": 5000.0},
            "bpe_cpu": {"generation_bytes_per_second": 500.0},
            "layercake_gpu": {"generation_bytes_per_second": 6000.0},
            "bpe_gpu": {"generation_bytes_per_second": 500.0},
        },
    }


def _resources(prefill_ratio: float = 0.8) -> dict:
    return {
        "layercake": {
            "artifact_bytes": 32 * 1024 * 1024,
            "peak_rss_bytes": 400 * 1024 * 1024,
        },
        "metrics": {
            "generation_speed_ratio": 2.0,
            "prefill_speed_ratio": prefill_ratio,
        },
    }


def _mobile_int8() -> dict:
    return {
        "payload_bytes": 512 * 1024,
        "greedy_generation": {"bytes_per_second": 900.0},
    }


def test_mobile_northstar_rejects_proxy_only_claim():
    report = build_report(
        fair_neural=_fair(exact=0.99),
        game_domain=_game(),
        cpu_resources=_resources(prefill_ratio=1.1),
        mobile_int8=_mobile_int8(),
        platform={"status": "PASS"},
    )

    assert report["status"] == "FAIL"
    assert report["gates"]["real_phone_hardware_evidence"] is False
    assert "real_phone_hardware_evidence" in report["failed"]


def test_mobile_northstar_rejects_current_fair_neural_quality_gap():
    report = build_report(
        fair_neural=_fair(exact=0.25),
        game_domain=_game(),
        cpu_resources=_resources(prefill_ratio=1.1),
        mobile_int8=_mobile_int8(),
        platform={"status": "PASS"},
    )

    assert report["gates"]["fair_neural_cpu_5x_speed"] is True
    assert report["gates"]["fair_neural_gpu_5x_speed"] is True
    assert report["gates"]["fair_neural_high_exact_quality"] is False


def test_mobile_northstar_rejects_prefill_regression():
    report = build_report(
        fair_neural=_fair(exact=0.99),
        game_domain=_game(),
        cpu_resources=_resources(prefill_ratio=0.8),
        mobile_int8=_mobile_int8(),
        platform={"status": "PASS"},
    )

    assert report["gates"]["desktop_cpu_generation_faster_than_bpe"] is True
    assert report["gates"]["desktop_cpu_prefill_faster_than_bpe"] is False
