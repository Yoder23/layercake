from types import SimpleNamespace

from layercake.cake.registry import CakeRegistry
from layercake.routing.orchestrator import LocalLayerCakeOrchestrator
from layercake.routing.policies import CakePermissionPolicy, RoutingPolicy


class _SemanticStub:
    def route(self, prompt, *, installed, top_k):
        if "crawler" in prompt and "python" in installed:
            return SimpleNamespace(
                selected=("python",), confidence=0.9, probabilities={"python": 0.9},
                reason="learned-semantic",
            )
        return SimpleNamespace(
            selected=(), confidence=0.05, probabilities={"python": 0.05},
            reason="uncertain-core-fallback",
        )


def test_learned_orchestrator_maps_domain_to_installed_cake(tmp_path):
    registry = CakeRegistry(tmp_path / "registry")
    registry.activate({
        "cake_id": "python-fusion-v2", "domains": ["python"], "keywords": [],
        "signed": True, "trusted_local": False, "permissions": ["local-inference"],
        "archive_hash": "0" * 64,
    })
    policy = RoutingPolicy(permissions=CakePermissionPolicy(
        allowed_permissions=frozenset({"local-inference"})
    ))
    orchestrator = LocalLayerCakeOrchestrator(
        registry, policy=policy, semantic_router=_SemanticStub(), loader=lambda record: (record, 1)
    )
    selected = orchestrator.route("Implement a bounded crawler")
    assert selected.selected == ("python-fusion-v2",)
    assert selected.policy_version == "layercake-learned-router/2"
    abstained = orchestrator.route("Tell me a story")
    assert abstained.core_fallback
