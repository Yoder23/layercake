import torch
import sys
from pathlib import Path

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


def test_persistent_byte_gru_matches_full_prefix_logits():
    torch.manual_seed(17)
    decoder = PortableDomainDecoder(
        feature_width=16,
        hidden_width=24,
        architecture="byte_gru",
        embedding_width=8,
    ).eval()
    prompt = torch.randint(0, 256, (2, 23))
    state = decoder.prefill_incremental(prompt)
    expected = decoder(prompt)[:, -1]
    assert torch.allclose(state["next_logits"], expected, atol=1e-6, rtol=1e-6)

    observed = torch.randint(0, 256, (2, 1))
    actual = decoder.decode_incremental(observed, state)
    expected = decoder(torch.cat([prompt, observed], dim=1))[:, -1]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_runtime_persistent_generation_is_identical_across_receivers():
    torch.manual_seed(23)
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
    first = LayerCakeRuntime()
    second = LayerCakeRuntime()
    first.install_portable_domain(artifact)
    second.install_portable_domain(artifact)
    a = first.generate_incremental(
        b"Return Python:\n", max_new_bytes=32, domain_id="python"
    )
    b = second.generate_incremental(
        b"Return Python:\n", max_new_bytes=32, domain_id="python"
    )
    assert torch.equal(a, b)


def test_functional_batch_identifier_mask_selects_exact_response_name():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from train_phase4_portable_functional_decoder import _batch
    finally:
        sys.path.remove(str(scripts))
    row = {
        "prompt": "Define exact_name(value).",
        "response": "def exact_name(value):\n    return value\n",
        "function_name": "exact_name",
    }
    inputs, targets, _, identifier_mask, pointer_labels = _batch(
        [row], [0], maximum_sequence_bytes=128
    )
    selected = bytes(targets[identifier_mask].tolist())
    assert selected == b"exact_name"
    source_positions = pointer_labels[identifier_mask]
    assert bytes(inputs[0, source_positions].tolist()) == b"exact_name"


def test_pointer_decoder_incremental_logits_match_full_prefix():
    torch.manual_seed(29)
    decoder = PortableDomainDecoder(
        feature_width=16,
        hidden_width=24,
        architecture="byte_gru_pointer",
        embedding_width=8,
        pointer_width=12,
    ).eval()
    prompt = torch.randint(0, 256, (2, 23))
    state = decoder.prefill_incremental(prompt)
    expected = decoder(prompt)[:, -1]
    assert torch.allclose(state["next_logits"], expected, atol=1e-6, rtol=1e-6)

    observed = torch.randint(0, 256, (2, 1))
    actual = decoder.decode_incremental(observed, state)
    expected = decoder(torch.cat([prompt, observed], dim=1))[:, -1]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_pointer_artifact_round_trip_preserves_logits():
    torch.manual_seed(31)
    decoder = PortableDomainDecoder(
        feature_width=16,
        hidden_width=24,
        architecture="byte_gru_pointer",
        embedding_width=8,
        pointer_width=12,
    ).eval()
    artifact = build_portable_artifact(
        decoder,
        PortableDomainSpec(
            "python",
            feature_width=16,
            hidden_width=24,
            architecture="byte_gru_pointer",
            embedding_width=8,
            pointer_width=12,
        ),
    )
    _, loaded = load_portable_artifact(artifact)
    prompt = torch.randint(0, 256, (2, 19))
    assert torch.equal(decoder(prompt), loaded(prompt))


def test_transition_pointer_incremental_logits_match_full_prefix():
    torch.manual_seed(37)
    decoder = PortableDomainDecoder(
        feature_width=16,
        hidden_width=24,
        architecture="byte_gru_pointer_transition",
        embedding_width=8,
        pointer_width=12,
    ).eval()
    with torch.no_grad():
        decoder.copy_transition_gate.bias.fill_(1.25)
        decoder.copy_transition_logits.copy_(
            torch.tensor([-2.0, -1.0, 0.0, 3.0, 0.5, -0.5, -1.5])
        )
    prompt = torch.randint(0, 256, (2, 23))
    state = decoder.prefill_incremental(prompt)
    expected = decoder(prompt)[:, -1]
    assert torch.allclose(state["next_logits"], expected, atol=1e-6, rtol=1e-6)

    observed = torch.randint(0, 256, (2, 1))
    actual = decoder.decode_incremental(observed, state)
    expected = decoder(torch.cat([prompt, observed], dim=1))[:, -1]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_transition_pointer_artifact_round_trip_preserves_logits():
    torch.manual_seed(41)
    decoder = PortableDomainDecoder(
        feature_width=16,
        hidden_width=24,
        architecture="byte_gru_pointer_transition",
        embedding_width=8,
        pointer_width=12,
    ).eval()
    artifact = build_portable_artifact(
        decoder,
        PortableDomainSpec(
            "python",
            feature_width=16,
            hidden_width=24,
            architecture="byte_gru_pointer_transition",
            embedding_width=8,
            pointer_width=12,
        ),
    )
    _, loaded = load_portable_artifact(artifact)
    prompt = torch.randint(0, 256, (2, 19))
    assert torch.equal(decoder(prompt), loaded(prompt))


def test_self_gated_transition_incremental_logits_match_full_prefix():
    torch.manual_seed(43)
    decoder = PortableDomainDecoder(
        feature_width=16,
        hidden_width=24,
        architecture="byte_gru_pointer_self_transition",
        embedding_width=8,
        pointer_width=12,
    ).eval()
    with torch.no_grad():
        decoder.copy_gate.bias.fill_(0.75)
        decoder.copy_transition_logits.copy_(
            torch.tensor([-2.0, -1.0, 0.0, 3.0, 0.5, -0.5, -1.5])
        )
    prompt = torch.randint(0, 256, (2, 23))
    state = decoder.prefill_incremental(prompt)
    expected = decoder(prompt)[:, -1]
    assert torch.allclose(state["next_logits"], expected, atol=1e-6, rtol=1e-6)

    observed = torch.randint(0, 256, (2, 1))
    actual = decoder.decode_incremental(observed, state)
    expected = decoder(torch.cat([prompt, observed], dim=1))[:, -1]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_markov_pointer_has_only_seven_transition_parameters():
    decoder = PortableDomainDecoder(
        feature_width=16,
        hidden_width=24,
        architecture="byte_gru_pointer_markov",
        embedding_width=8,
        pointer_width=12,
    )
    transition = {
        name: parameter.numel()
        for name, parameter in decoder.named_parameters()
        if name.startswith("copy_transition")
    }
    assert transition == {"copy_transition_logits": 7}


def test_markov_pointer_incremental_logits_match_full_prefix():
    torch.manual_seed(47)
    decoder = PortableDomainDecoder(
        feature_width=16,
        hidden_width=24,
        architecture="byte_gru_pointer_markov",
        embedding_width=8,
        pointer_width=12,
    ).eval()
    with torch.no_grad():
        decoder.copy_gate.bias.fill_(0.75)
        decoder.copy_transition_logits.copy_(
            torch.tensor([-2.0, -1.0, 0.0, 3.0, 0.5, -0.5, -1.5])
        )
    prompt = torch.randint(0, 256, (2, 23))
    state = decoder.prefill_incremental(prompt)
    expected = decoder(prompt)[:, -1]
    assert torch.allclose(state["next_logits"], expected, atol=1e-6, rtol=1e-6)

    observed = torch.randint(0, 256, (2, 1))
    actual = decoder.decode_incremental(observed, state)
    expected = decoder(torch.cat([prompt, observed], dim=1))[:, -1]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
