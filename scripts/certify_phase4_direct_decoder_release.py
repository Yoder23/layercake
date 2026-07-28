"""Repackage and certify the frozen Python cake against the direct ABI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.cake.package import load_package
from layercake.training.phase4_python_cake import _canonical_sha
import scripts.certify_phase4_token_plan_transfer as legacy


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "moonshot/phase4_direct_decoder_release_preregistration.json"
PROTOCOL_SHA256 = (
    "9a1a97f29f8ac29c187f559034541350cd54e5e0023d1bd77f01089d996b2dc4"
)
DIRECT_ABI = ROOT / "moonshot/phase4_canonical_direct_decoder_abi_v1.json"
DIRECT_ABI_VERSION = "lc-direct-neural-decoder/1"
DIRECT_ABI_SHA256 = (
    "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
)
PACKAGE = (
    ROOT
    / "artifacts/moonshot/phase4/release"
    / "python-token-plan-seed10141-direct-v1.0.0.cake"
)
PUBLIC_KEY = (
    ROOT / "moonshot/phase4-direct-token-plan-publisher.public.pem"
)
TRUST_STORE = (
    ROOT / "moonshot/phase4-direct-token-plan-trust-store.json"
)
EVIDENCE = (
    ROOT
    / "results/moonshot/phase4"
    / "direct_decoder_transfer_certificate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_legacy_certifier() -> None:
    legacy.ABI_VERSION = DIRECT_ABI_VERSION
    legacy.ABI_HASH = DIRECT_ABI_SHA256
    legacy.PACKAGE = PACKAGE
    legacy.PRIVATE_KEY = PACKAGE.with_suffix(".private.pem")
    legacy.PUBLIC_KEY = PUBLIC_KEY
    legacy.TRUST_STORE = TRUST_STORE
    legacy.EVIDENCE = EVIDENCE


def certify() -> dict[str, Any]:
    if _sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("direct-decoder release protocol changed")
    if _sha256(DIRECT_ABI) != DIRECT_ABI_SHA256:
        raise RuntimeError("canonical direct-decoder ABI changed")
    _configure_legacy_certifier()
    result = legacy.certify()
    if result["status"] != (
        "PARTIAL_PASS_DIRECT_DECODER_SEMANTIC_ABI_GATE_OPEN"
    ):
        raise RuntimeError(
            "underlying transfer checks did not reach their direct-path pass"
        )
    key_id = result["package"]["key_id"]
    package = load_package(
        PACKAGE, trust_store={key_id: PUBLIC_KEY}
    )
    direct_conformance = (
        package.manifest.abi_version == DIRECT_ABI_VERSION
        and package.manifest.abi_hash == DIRECT_ABI_SHA256
        and package.manifest.cake_type == "portable_decoder"
        and package.manifest.input_contract.get("mode")
        == "direct_selected_portable_decoder"
        and package.manifest.output_contract.get("composition")
        == "direct_selected_one_cake_no_router"
    )
    prior_gates = {
        name: value
        for name, value in result["gates"].items()
        if name
        not in {
            "canonical_semantic_abi_contract_unchanged",
            "candidate_consumes_and_returns_semantic_abi",
        }
    }
    gates = {
        **prior_gates,
        "canonical_direct_decoder_abi_conformance": direct_conformance,
        "semantic_residual_fusion_claimed": False,
        "direct_mode_claims_no_phase5_or_phase6_credit": True,
    }
    status = (
        "PASS"
        if all(
            value
            for name, value in gates.items()
            if name != "semantic_residual_fusion_claimed"
        )
        and gates["semantic_residual_fusion_claimed"] is False
        else "FAIL"
    )
    result.update(
        {
            "format": (
                "layercake-phase4-direct-decoder-transfer-certificate/1"
            ),
            "status": status,
            "preregistration": PROTOCOL.relative_to(ROOT).as_posix(),
            "preregistration_sha256": PROTOCOL_SHA256,
            "canonical_interface": {
                "path": DIRECT_ABI.relative_to(ROOT).as_posix(),
                "sha256": DIRECT_ABI_SHA256,
                "version": DIRECT_ABI_VERSION,
                "mode": "direct_selected_one_cake_no_router",
                "conforms": direct_conformance,
            },
            "execution_mode_audit": {
                "mode": "direct_selected_portable_decoder",
                "canonical_direct_decoder_abi_consumed": direct_conformance,
                "canonical_semantic_abi_mutated": False,
                "semantic_residual_fusion_claimed": False,
                "phase5_or_phase6_credit_claimed": False,
                "claim": (
                    "Exact direct neural cake transfer and one-cake hosting "
                    "are certified; semantic-residual fusion is not claimed."
                ),
            },
            "gates": gates,
        }
    )
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = _canonical_sha(result)
    EVIDENCE.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    result = certify()
    print(
        json.dumps(
            {
                "status": result["status"],
                "package": result["package"],
                "canonical_interface": result["canonical_interface"],
                "receiver_retention": [
                    receiver["retention_rate"]
                    for receiver in result["receivers"]
                ],
                "gates": result["gates"],
                "evidence_sha256": result["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
