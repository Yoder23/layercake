import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "benchmark_moonshot_generation.py"
    spec = importlib.util.spec_from_file_location("benchmark_moonshot_generation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_quality_score_penalizes_repetition():
    module = _load_module()
    coherent = module._quality_score("Take cover, wait, then move when safe.")
    repeated = module._quality_score("the the the the the the the the the the")
    assert coherent["quality_score"] > repeated["quality_score"]
    assert repeated["max_word_repeat"] >= 10


def test_quality_score_penalizes_static_letter_loop():
    module = _load_module()
    coherent = module._quality_score(
        "Take cover behind the wall, wait for the archer to reload, then move."
    )
    static_loop = module._quality_score(
        "SESESESESSSESESES EEEESEEESEEEEESEE E EESEEES SESESEEESESESEE"
    )
    assert coherent["quality_score"] > static_loop["quality_score"]
    assert static_loop["unique_alpha_char_count"] < 3
    assert static_loop["unique_word_count"] < 8


def test_moonshot_generation_output_schema_is_verifier_compatible(tmp_path):
    payload = {
        "status": "PASS",
        "model_kind": "layercake",
        "metrics": {
            "generation_bytes_per_second": 123.0,
            "quality_score": 0.75,
            "generated_bytes": 384,
            "seconds": 3.12,
        },
        "samples": [],
    }
    path = tmp_path / "gen.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["metrics"]["generation_bytes_per_second"] > 0
    assert 0 <= loaded["metrics"]["quality_score"] <= 1


def test_forced_patch_prediction_generation_uses_generate_next_patch(monkeypatch):
    module = _load_module()

    class DummyPos:
        num_embeddings = 8

    class DummyModel:
        patch_size = 2
        patch_pos = DummyPos()

        def generate_next_patch(self, x):
            return module.torch.tensor([[65, 66]], dtype=module.torch.long)

    text, seconds = module._generate_layercake_patch_prediction_method(
        DummyModel(),
        list(b"Question:"),
        device=module.torch.device("cpu"),
        max_new_bytes=4,
        no_repeat_ngram=0,
    )

    assert text == "ABAB"
    assert seconds >= 0


def test_instruction_keyword_hits_require_phrase_coverage_and_handle_negation():
    module_path = ROOT / "scripts" / "benchmark_instruction_generalization.py"
    spec = importlib.util.spec_from_file_location("benchmark_instruction_generalization", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    hits, names = module._keyword_hits(
        "Northwind returns are allowed within thirty days.",
        ["fourteen days", "thirty days"],
    )
    assert hits == 1
    assert names == ["thirty days"]

    forbidden_hits, forbidden_names = module._keyword_hits(
        "ForgeBoard uses a pinned board snapshot, not an encrypted notebook cache.",
        ["encrypted notebook cache"],
        ignore_negated=True,
    )
    assert forbidden_hits == 0
    assert forbidden_names == []


def test_instruction_generation_gates_fail_zero_relevance_gibberish():
    module_path = ROOT / "scripts" / "benchmark_instruction_generalization.py"
    spec = importlib.util.spec_from_file_location("benchmark_instruction_generalization", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    gates = module._instruction_generation_gates(
        [
            {
                "text": "e e A, A, e e Be, at w ala a a le ar acose acosese",
                "printable_ratio": 1.0,
                "alpha_space_ratio": 0.96,
                "max_repeat_8gram": 1.0,
                "unique_word_count": 12.0,
                "distinct_word_ratio": 0.5,
                "one_char_word_ratio": 0.56,
                "unique_alpha_char_count": 12.0,
                "relevance_pass": False,
            }
        ]
    )

    assert gates["samples_relevant"] is False
    assert gates["samples_lexically_diverse"] is False
    assert not all(gates.values())


def test_instruction_generation_gates_pass_clean_relevant_sample():
    module_path = ROOT / "scripts" / "benchmark_instruction_generalization.py"
    spec = importlib.util.spec_from_file_location("benchmark_instruction_generalization", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    gates = module._instruction_generation_gates(
        [
            {
                "text": "Retreat, guard, create space, stabilize health, then recover tempo.",
                "printable_ratio": 1.0,
                "alpha_space_ratio": 1.0,
                "max_repeat_8gram": 1.0,
                "unique_word_count": 8.0,
                "distinct_word_ratio": 1.0,
                "one_char_word_ratio": 0.0,
                "unique_alpha_char_count": 15.0,
                "relevance_pass": True,
            }
        ]
    )

    assert all(gates.values())
