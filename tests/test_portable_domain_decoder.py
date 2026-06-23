import torch

from layercake.causal_byte_models import CausalBytePatchLM
from layercake.portable_domain import (
    LayerCakeRuntime,
    PortableDomainDecoder,
    PortableDomainSpec,
    build_portable_artifact,
    load_portable_artifact,
    quantize_portable_artifact,
)


def test_portable_decoder_is_core_and_seed_independent():
    torch.manual_seed(11)
    source_core = CausalBytePatchLM(d_model=32, d_abi=16, layers=1, heads=4)
    torch.manual_seed(99)
    target_core = CausalBytePatchLM(d_model=48, d_abi=16, layers=1, heads=4)
    decoder = PortableDomainDecoder(d_abi=16, hidden=24)
    x = torch.randint(0, 256, (2, 16))

    source_logits, _ = source_core(x)
    target_logits, _ = target_core(x)
    portable_source = decoder(x)
    portable_target = decoder(x)

    assert not torch.equal(source_logits, target_logits)
    assert torch.equal(portable_source, portable_target)


def test_portable_decoder_copy_is_bit_exact():
    source = PortableDomainDecoder(d_abi=16, hidden=24)
    target = PortableDomainDecoder(d_abi=16, hidden=24)
    target.load_state_dict(source.state_dict())
    x = torch.randint(0, 256, (2, 16))

    assert torch.equal(source(x), target(x))


def test_artifact_hash_rejects_payload_mutation():
    model = PortableDomainDecoder(feature_width=16, hidden_width=24)
    spec = PortableDomainSpec("python", feature_width=16, hidden_width=24)
    artifact = build_portable_artifact(model, spec)
    _, loaded = load_portable_artifact(artifact)
    assert torch.equal(model.decoder[1].weight, loaded.decoder[1].weight)

    artifact["state_dict"]["decoder.1.weight"][0, 0] += 1
    try:
        load_portable_artifact(artifact)
    except ValueError as error:
        assert "payload hash mismatch" in str(error)
    else:
        raise AssertionError("mutated artifact was accepted")


def test_runtime_generation_is_identical_across_different_cores():
    torch.manual_seed(11)
    source_core = CausalBytePatchLM(d_model=32, d_abi=16, layers=1, heads=4)
    torch.manual_seed(99)
    target_core = CausalBytePatchLM(d_model=48, d_abi=32, layers=1, heads=4)
    decoder = PortableDomainDecoder(feature_width=16, hidden_width=24)
    artifact = build_portable_artifact(
        decoder,
        PortableDomainSpec("python", feature_width=16, hidden_width=24),
    )
    source = LayerCakeRuntime(source_core)
    target = LayerCakeRuntime(target_core)
    source.install_portable_domain(artifact)
    target.install_portable_domain(artifact)

    a = source.generate(b"def f(", max_new_bytes=8, domain_id="python")
    b = target.generate(b"def f(", max_new_bytes=8, domain_id="python")
    assert torch.equal(a, b)


def test_recurrent_portable_decoder_artifact_round_trip():
    decoder = PortableDomainDecoder(
        feature_width=16,
        hidden_width=24,
        architecture="byte_gru",
        embedding_width=8,
    )
    spec = PortableDomainSpec(
        "python",
        feature_width=16,
        hidden_width=24,
        architecture="byte_gru",
        embedding_width=8,
    )
    artifact = build_portable_artifact(decoder, spec)
    loaded_spec, loaded = load_portable_artifact(artifact)
    x = torch.randint(0, 256, (2, 16))
    assert loaded_spec == spec
    assert torch.equal(decoder(x), loaded(x))


def test_int8_artifact_is_portable_and_bounded():
    decoder = PortableDomainDecoder(
        feature_width=16,
        hidden_width=24,
        architecture="byte_gru",
        embedding_width=8,
    )
    artifact = build_portable_artifact(
        decoder,
        PortableDomainSpec(
            "python",
            feature_width=16,
            hidden_width=24,
            architecture="byte_gru",
            embedding_width=8,
        ),
    )
    quantized = quantize_portable_artifact(artifact)
    _, a = load_portable_artifact(quantized)
    _, b = load_portable_artifact(quantized)
    x = torch.randint(0, 256, (2, 16))
    assert torch.equal(a(x), b(x))
    assert (decoder(x) - a(x)).abs().max().item() < 0.1
