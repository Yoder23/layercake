import json

import pytest

from layercake.orchestration import Claim, HandoffPacket, LayerCakeOrchestrator


def test_packet_serialization_and_hash_validation():
    packet = HandoffPacket(
        abi_version="lc-abi/2",
        input_mode="byte_patch",
        patching_mode="fixed:4",
        source_model_id="mobile-25m",
        active_domain_bricks=["python"],
        claims=[Claim("syntax is valid", 0.8)],
        uncertainty=0.2,
    )
    restored = HandoffPacket.from_json(packet.to_json())
    assert restored.compute_hash() == packet.compute_hash()
    envelope = json.loads(packet.to_json())
    envelope["payload"]["uncertainty"] = 0.9
    with pytest.raises(ValueError, match="hash mismatch"):
        HandoffPacket.from_json(json.dumps(envelope))


def test_orchestrator_escalates_and_limits_active_bricks():
    orchestrator = LayerCakeOrchestrator(escalation_threshold=0.5)
    models = [
        {"id": "small", "cost": 1, "abi_version": "lc-abi/2", "top_k": 1},
        {"id": "large", "cost": 10, "abi_version": "lc-abi/2", "top_k": 2},
    ]
    bricks = [
        {"id": "python-a", "domain": "python", "abi_version": "lc-abi/2"},
        {"id": "python-b", "domain": "python", "abi_version": "lc-abi/2"},
    ]
    result = orchestrator.route("debug python", models, bricks, uncertainty=0.9)
    assert result["model_id"] == "large"
    assert result["escalate"]
    assert result["active_bricks"] == ["python-a", "python-b"]
