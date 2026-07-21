from scripts.write_fair_neural_candidate_report import (
    NORTHSTAR_MIN_CHAR_SIMILARITY,
    NORTHSTAR_MIN_EXACT_JSON_ACCURACY,
    NORTHSTAR_MIN_PARSEABLE_JSON_RATE,
    NORTHSTAR_MIN_SAMPLES_PER_SPLIT,
    layercake_northstar_quality_gates,
)


def _doc(exact: float, parseable: float, similarity: float, samples: int) -> dict:
    return {
        "splits": {
            "heldout": {
                "summary": {
                    "layercake": {
                        "exact_json_accuracy": exact,
                        "parseable_json_rate": parseable,
                        "mean_char_similarity": similarity,
                    }
                },
                "samples": [{} for _ in range(samples)],
            }
        }
    }


def test_northstar_quality_gate_rejects_relative_only_win():
    gates = layercake_northstar_quality_gates(
        _doc(exact=0.1875, parseable=0.5, similarity=0.916, samples=16),
        "heldout",
        "cpu",
    )

    assert gates["cpu_heldout_generation_examples_present"] is True
    assert gates["cpu_heldout_layercake_exact_at_northstar_floor"] is False
    assert gates["cpu_heldout_layercake_parse_at_northstar_floor"] is False
    assert gates["cpu_heldout_layercake_similarity_at_northstar_floor"] is False


def test_northstar_quality_gate_accepts_high_absolute_generation_quality():
    gates = layercake_northstar_quality_gates(
        _doc(
            exact=NORTHSTAR_MIN_EXACT_JSON_ACCURACY,
            parseable=NORTHSTAR_MIN_PARSEABLE_JSON_RATE,
            similarity=NORTHSTAR_MIN_CHAR_SIMILARITY,
            samples=NORTHSTAR_MIN_SAMPLES_PER_SPLIT,
        ),
        "heldout",
        "cpu",
    )

    assert all(gates.values())
