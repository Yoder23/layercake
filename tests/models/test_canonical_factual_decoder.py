from __future__ import annotations

import pytest
import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, load_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.models.canonical_factual import (
    CanonicalFactualDecoder,
    canonical_factual_manifest_architecture,
)
from layercake.models.direct_cake_host import DirectCakeHost


FACTS = [
    {"relation": "capital", "entity": "Austria", "value": "Vienna"},
    {"relation": "capital", "entity": "India", "value": "New Delhi"},
]
ABI_VERSION = "lc-test/1"
ABI_HASH = "d024de52144a2d797d0501acb7deb55575ffca7e33f72900beff599cf0a97761"


def _package(tmp_path):
    private, public, signer = generate_keypair()
    model = CanonicalFactualDecoder("geography/capitals", FACTS)
    state = model.state_dict()
    manifest = CakeManifest(
        schema_version="1",
        cake_id="canonical-facts-test",
        name="Canonical facts test",
        description="Test-only canonical factual package",
        version="1.0.0",
        publisher={"id": "test", "name": "Test", "key_id": signer},
        abi_version=ABI_VERSION,
        abi_hash=ABI_HASH,
        cake_type="portable_decoder",
        input_contract={
            "external": "UTF-8 bytes",
            "mode": "direct_selected_portable_decoder",
        },
        output_contract={"external": "UTF-8 bytes", "composition": "selected"},
        architecture=canonical_factual_manifest_architecture(
            "geography/capitals", FACTS
        ),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={
            "features": ["byte_input", "safe_tensors", "incremental"]
        },
        tensor_payload_hash="",
        tensor_shapes=tensor_specs(state),
        package_hash="",
        training_data_provenance={"receiver_training_steps": 0},
        evaluation_evidence={"status": "TEST"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer},
        domains=("geography/capitals",),
    )
    path = build_package(
        tmp_path / "canonical-facts.cake", manifest, state, private_key=private
    )
    return path, public, signer


def test_canonical_factual_package_executes_and_abstains(tmp_path) -> None:
    path, public, signer = _package(tmp_path)
    package = load_package(path, trust_store={signer: public})
    assert package.manifest.architecture["facts"] == FACTS
    host = DirectCakeHost(
        tmp_path / "host",
        abi_version=ABI_VERSION,
        abi_hash=ABI_HASH,
        trust_store={signer: public},
    )
    host.install(path)
    assert host.generate("canonical-facts-test", "What is Austria's capital?").output == b"Vienna"
    assert host.generate("canonical-facts-test", b"Name the capital of India.").output == b"New Delhi"
    assert host.generate("canonical-facts-test", "What is Kenya's capital?").output == b""
    assert host.generate("canonical-facts-test", "Compare Austria and India.").output == b""


def test_canonical_factual_package_transfers_without_behavior_change(tmp_path) -> None:
    path, public, signer = _package(tmp_path)
    outputs = []
    for name in ("first", "second"):
        host = DirectCakeHost(
            tmp_path / name,
            abi_version=ABI_VERSION,
            abi_hash=ABI_HASH,
            trust_store={signer: public},
            device="cuda" if name == "second" and torch.cuda.is_available() else "cpu",
        )
        host.install(path)
        outputs.append(host.generate("canonical-facts-test", "Austria?").output)
    assert outputs == [b"Vienna", b"Vienna"]


def test_canonical_factual_schema_rejects_duplicates_and_oversize() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        canonical_factual_manifest_architecture(
            "domain", [FACTS[0], {**FACTS[0], "value": "Other"}]
        )
    with pytest.raises(ValueError, match="safety boundary"):
        canonical_factual_manifest_architecture(
            "domain", [{"relation": "r", "entity": "x", "value": "v" * 4097}]
        )
