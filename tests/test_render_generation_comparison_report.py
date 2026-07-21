from __future__ import annotations

from scripts.render_generation_comparison_report import build_report


def _generation(model_kind: str, text: str, *, relevance: bool, keyword_hits: int):
    return {
        "checkpoint": f"runs/{model_kind}.pt",
        "device": "cpu",
        "model_kind": model_kind,
        "metrics": {
            "generation_bytes_per_second": 1000.0 if model_kind == "layercake" else 100.0,
            "quality_score": 0.9 if model_kind == "layercake" else 0.8,
            "relevance_rate": 1.0 if relevance else 0.0,
        },
        "samples": [
            {
                "prompt": "Question: What should I do? Answer:",
                "category": "paraphrase",
                "text": text,
                "runtime_path": "semantic_instruction_alias" if model_kind == "layercake" else "neural_bpe_transformer",
                "relevance_pass": relevance,
                "keyword_hits": keyword_hits,
                "min_keyword_hits": 2,
                "hit_keywords": ["guard", "retreat"][:keyword_hits],
                "expected_keywords": ["guard", "retreat"],
                "forbidden_keyword_hits": 0,
                "quality_score": 0.9 if model_kind == "layercake" else 0.8,
                "bytes_per_second": 1000.0 if model_kind == "layercake" else 100.0,
                "max_repeat_8gram": 1.0,
            }
        ],
    }


def test_generation_comparison_report_contains_side_by_side_outputs_and_scores():
    markdown, result = build_report(
        layercake=_generation("layercake", "Guard, retreat, then stabilize.", relevance=True, keyword_hits=2),
        transformer=_generation("bpe", "How do when surrounded.", relevance=False, keyword_hits=0),
        title="Review",
        certificate={"status": "PASS"},
    )

    assert result["summary"]["speed_ratio"] == 10.0
    assert result["comparisons"][0]["layercake"]["text"] == "Guard, retreat, then stabilize."
    assert result["comparisons"][0]["transformer"]["text"] == "How do when surrounded."
    assert "LayerCake generation" in markdown
    assert "BPE transformer generation" in markdown
    assert "2/2" in markdown
    assert "0/2" in markdown
    assert "Certificate status" in markdown
