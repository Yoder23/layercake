"""Benchmark the certified direct-ABI Python cake on the declared CPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.cake.package import load_package
from layercake.training.phase4_python_cake import _canonical_sha
import scripts.benchmark_phase4_token_plan_cpu as legacy


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "moonshot/phase4_direct_decoder_runtime_protocol.json"
PROTOCOL_SHA256 = (
    "3e059c1a28b766e234ee58484dd8a2231ffc5ea19f791a1cb320c87ead17df6d"
)
TRANSFER = (
    ROOT
    / "results/moonshot/phase4"
    / "direct_decoder_transfer_certificate.json"
)
TRANSFER_FILE_SHA256 = (
    "c72d37ad95feeb763190ca6ffad3edbb5cc1bf6443bb957f5c75c48b416b131a"
)
TRANSFER_EVIDENCE_SHA256 = (
    "48ce31f720376c87819501048cc8cde06774bb6741ae4628522062f4b29e5d89"
)
PACKAGE = (
    ROOT
    / "artifacts/moonshot/phase4/release"
    / "python-token-plan-seed10141-direct-v1.0.0.cake"
)
PACKAGE_SHA256 = (
    "0585c79bfbea16b1c4165bf0030ba6985b8a8cdeab529cd7afe3f9c76c564ef7"
)
PUBLIC_KEY = (
    ROOT / "moonshot/phase4-direct-token-plan-publisher.public.pem"
)
OUTPUT = (
    ROOT
    / "results/moonshot/phase4"
    / "direct_decoder_cpu_product_benchmark.json"
)
DIRECT_ABI_VERSION = "lc-direct-neural-decoder/1"
DIRECT_ABI_SHA256 = (
    "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure() -> None:
    legacy.PACKAGE = PACKAGE
    legacy.PUBLIC_KEY = PUBLIC_KEY
    legacy.OUTPUT = OUTPUT


def benchmark() -> dict[str, Any]:
    if _sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("direct-decoder runtime protocol changed")
    if _sha256(TRANSFER) != TRANSFER_FILE_SHA256:
        raise RuntimeError("direct-decoder transfer certificate changed")
    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    if (
        transfer["status"] != "PASS"
        or transfer["evidence_sha256"] != TRANSFER_EVIDENCE_SHA256
    ):
        raise RuntimeError("direct-decoder transfer is not certified")
    if _sha256(PACKAGE) != PACKAGE_SHA256:
        raise RuntimeError("certified direct-decoder package changed")
    _configure()
    result = legacy.benchmark()
    package = load_package(
        PACKAGE,
        trust_store={
            result["package"].get(
                "key_id",
                transfer["package"]["key_id"],
            ): PUBLIC_KEY
        },
    )
    same_package = (
        result["package"]["archive_sha256"]
        == transfer["package"]["archive_sha256"]
        == PACKAGE_SHA256
        and result["package"]["tensor_payload_hash"]
        == transfer["package"]["tensor_payload_hash"]
    )
    direct_abi = (
        package.manifest.abi_version == DIRECT_ABI_VERSION
        and package.manifest.abi_hash == DIRECT_ABI_SHA256
    )
    gates = {
        name: value
        for name, value in result["gates"].items()
        if name
        not in {
            "one_payload_quality_and_speed",
            "candidate_consumes_and_returns_semantic_abi",
        }
    }
    gates.update(
        {
            "same_certified_package_quality_transfer_and_speed": (
                same_package
            ),
            "canonical_direct_decoder_abi_conformance": direct_abi,
            "semantic_residual_fusion_claimed": False,
        }
    )
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
                "layercake-phase4-direct-decoder-cpu-product-benchmark/1"
            ),
            "status": status,
            "protocol": {
                **result["protocol"],
                "preregistration": PROTOCOL.relative_to(ROOT).as_posix(),
                "preregistration_sha256": PROTOCOL_SHA256,
                "transfer_certificate": (
                    TRANSFER.relative_to(ROOT).as_posix()
                ),
                "transfer_certificate_sha256": (
                    TRANSFER_EVIDENCE_SHA256
                ),
                "canonical_interface_version": DIRECT_ABI_VERSION,
                "canonical_interface_sha256": DIRECT_ABI_SHA256,
            },
            "gates": gates,
        }
    )
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = _canonical_sha(result)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    result = benchmark()
    print(
        json.dumps(
            {
                "status": result["status"],
                "aggregates": result["aggregates"],
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
