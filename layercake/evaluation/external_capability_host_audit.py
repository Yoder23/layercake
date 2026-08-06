"""Audit LayerCake's sealed interfaces for hosting an external English core artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class ExternalCapabilityHostAuditError(ValueError):
    """Raised when the audit protocol or sealed evidence changes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExternalCapabilityHostAuditError(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != "layercake-external-capability-host-interface-audit/1"
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_POSTRELEASE_AUDIT"
        or protocol.get("sealed_campaign_mutation_allowed") is not False
        or protocol.get("external_acquisition_code_allowed") is not False
        or protocol.get("training_allowed") is not False
        or protocol.get("final_test_access_allowed") is not False
    ):
        raise ExternalCapabilityHostAuditError("host-interface audit governance changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or _sha256(target) != expected:
            raise ExternalCapabilityHostAuditError(f"host-interface binding changed: {relative}")
    return protocol, _sha256(path)


def _semantic_branch(ledger: Mapping[str, Any]) -> Mapping[str, Any]:
    branches = ledger.get("branches")
    if not isinstance(branches, list):
        raise ExternalCapabilityHostAuditError("Phase 4 branch ledger is malformed")
    matches = [row for row in branches if row.get("branch_id") == "semantic-token-plan-residual-seed10240"]
    if len(matches) != 1:
        raise ExternalCapabilityHostAuditError("semantic token-plan branch is not unique")
    return matches[0]


def audit(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    semantic = _json(root / "moonshot/phase2_canonical_semantic_abi_r3.json")
    direct = _json(root / "moonshot/phase4_canonical_direct_decoder_abi_v1.json")
    direct_certificate = _json(root / "results/moonshot/phase4/direct_decoder_transfer_certificate.json")
    action_decision = _json(root / "moonshot/phase4_semantic_action_plan_lexical_repair_decision.json")
    branch = _semantic_branch(_json(root / "results/moonshot/phase4/branch_ledger.json"))
    portable_source = (root / "layercake/portable_token_plan.py").read_text(encoding="utf-8")

    facts = {
        "sealed_semantic_interface_is_residual_only": semantic.get("attachment", {}).get("cake_output")
        == "semantic residual with identical shape",
        "sealed_direct_interface_is_capability_cake_only": direct.get("scope")
        == "One explicitly selected neural capability cake executing without a router",
        "direct_interface_is_byte_facing": direct.get("external_boundary")
        == {"input": "UTF-8 bytes", "output": "UTF-8 bytes", "lossless_roundtrip_required": True},
        "direct_interface_requires_incremental_state": direct.get("neural_generation", {}).get("persistent_incremental_state_required") is True,
        "direct_package_transfer_certified": all(
            direct_certificate.get("gates", {}).get(name) is True
            for name in (
                "canonical_direct_decoder_abi_conformance",
                "core_immutable",
                "cpu_install_and_execution",
                "cuda_install_and_execution",
                "incremental_action_equivalence",
                "receiver_learning_zero",
                "three_receivers",
            )
        ),
        "semantic_teacher_forced_fit_was_near_perfect": branch.get("teacher_forced_final_report", {}).get("full_vocabulary_cross_entropy", 1.0) < 0.001,
        "semantic_autonomous_generation_failed": branch.get("autonomous_validation", {}).get("strict_functional_successes") == 0,
        "self_causal_semantic_realization_still_failed_gate": action_decision.get("status")
        == "FAILED_LEXICAL_SCREEN_BRANCH_CLOSED"
        and action_decision.get("lexical_validation", {}).get("exact_response_successes") == 189
        and action_decision.get("lexical_validation", {}).get("minimum_exact_response_successes") == 231,
        "direct_decoder_has_prefill_api": "def prefill_bytes(" in portable_source,
        "direct_decoder_has_incremental_step_api": "def step_bytes(" in portable_source,
        "signed_direct_english_core_interface_absent": not (root / "moonshot/canonical_direct_neural_core_abi_v1.json").exists(),
    }
    if not all(facts.values()):
        failed = [name for name, passed in facts.items() if not passed]
        raise ExternalCapabilityHostAuditError(f"host-interface audit fact failed: {failed}")

    result: dict[str, Any] = {
        "format": "layercake-external-capability-host-interface-audit-result/1",
        "status": "HOST_SCOPE_GAP_DIRECT_ENGLISH_CORE_ARTIFACT_INTERFACE_ABSENT",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "facts": facts,
        "interface_matrix": {
            "ad_hoc_hidden_state_hooks": {
                "canonical": False,
                "eligible_for_host_claims": False,
                "finding": "not a sealed LayerCake package or core interface",
            },
            "lc-semantic-gpt2-768/1": {
                "canonical": True,
                "persistent_external_state": True,
                "autonomous_external_english_core_certified": False,
                "finding": "teacher-forced fit did not survive autonomous host-state realization",
            },
            "lc-direct-neural-decoder/1": {
                "canonical": True,
                "byte_facing": True,
                "persistent_incremental_state": True,
                "signed_exact_transfer": True,
                "external_english_core_role_certified": False,
                "finding": "proven for selected capability cakes, not for replacing or supplying the English core",
            },
        },
        "ownership": {
            "sealed_layercake_regression": False,
            "general_layercake_quality_failure": False,
            "host_interface_scope_gap": True,
            "external_extraction_or_labeling_failure_evaluated": False,
        },
        "required_repair": {
            "design": "version a signed byte-facing self-causal direct neural core artifact interface using existing package safety and incremental execution primitives",
            "must_not": [
                "reuse ad hoc block hooks",
                "claim semantic-residual fusion",
                "inherit prior core or capability-package quality or speed",
                "mutate the sealed release lineage",
            ],
            "recertification": [
                "core-only English behavior",
                "artifact immutability",
                "teacher absence",
                "CPU and GPU execution",
                "persistent incremental state",
                "same-artifact CPU speed, TTFT, RSS, and memory",
                "future domain-package compatibility",
            ],
        },
        "training_performed": False,
        "runtime_changed": False,
        "sealed_campaign_changed": False,
        "external_final_data_accessed": False,
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result["evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def write_result(result: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise ExternalCapabilityHostAuditError("host-interface audit result is immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="moonshot/postrelease_external_capability_host_interface_audit_v1.json")
    parser.add_argument("--output", default="results/moonshot/postrelease/external_capability_host_interface_audit_v1.json")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = audit(root, (root / args.protocol).resolve())
    if args.write:
        write_result(result, (root / args.output).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
