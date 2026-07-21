from scripts.verify_domain_orchestrator import verify


def test_domain_orchestrator_routes_specialists_and_bounds_compute():
    result = verify()
    assert result["status"] == "PASS"
    assert result["required_gates"]["all_prompts_routed_to_expected_model"] is True
    assert result["required_gates"]["active_compute_bounded"] is True
    assert result["metrics"]["routing_accuracy"] == 1.0
    assert result["metrics"]["max_active_model_count"] == 1


def test_domain_orchestrator_fails_when_specialist_missing():
    result = verify(
        models=[
            {
                "id": "lc-general-cpu-large",
                "domains": ["general"],
                "keywords": ["explain", "summarize", "question"],
                "cost": 1.0,
                "capacity": 1.0,
                "abi_version": "lc-abi/2",
                "top_k": 1,
            }
        ]
    )
    assert result["status"] == "FAIL"
    assert result["required_gates"]["all_prompts_routed_to_expected_model"] is False
