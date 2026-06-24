import json
from pathlib import Path


def test_transformer_dominance_matrix_schema():
    data = json.loads(
        Path("results/transformer_dominance_matrix.json").read_text(encoding="utf-8")
    )
    assert data["status"] == "PASS"
    promoted = data["promoted_tiers"]
    assert set(promoted) == {
        "local_methodology_ladder",
        "full_corpus_15m_source_and_transfer",
        "receiver_after_lossless_transfer",
    }
    transition = promoted["full_corpus_15m_source_and_transfer"]
    gates = transition["required_gates"]
    for gate in [
        "source_at_least_2pct_smaller_than_bpe",
        "source_at_least_0_5pct_better_general_bpb",
        "source_at_least_1pct_faster_training",
        "source_at_least_10pct_faster_cpu_generation",
        "source_generation_distinct_trigrams",
        "source_generation_no_repeated_4gram",
        "transfer_ppl_ratio_exact",
        "transfer_max_logit_diff_exact",
        "transfer_generation_exact",
        "transferred_domain_at_least_10pct_better_than_adapter",
    ]:
        assert gates[gate] is True
    assert data["unpromoted_tiers"]["full_corpus_20m_source"]["status"] == "OPEN"
    assert (
        "training_time_beats_bpe20"
        in data["unpromoted_tiers"]["full_corpus_20m_source"]["failed"]
    )


def test_scale15m_transition_certificate_uses_strict_generation_evidence():
    data = json.loads(
        Path("results/scale15m_transition_frontier_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["status"] == "PASS"
    assert data["artifacts"]["generation"].endswith("norepeat4.json")
    metrics = data["metrics"]
    assert metrics["generation_distinct_trigram_rate"] >= 0.80
    assert metrics["generation_has_repeated_8gram"] is False
    assert metrics["cpu_generation_speed_ratio"] >= 1.10
