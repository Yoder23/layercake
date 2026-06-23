from pathlib import Path


def test_equivalence_evaluator_defines_strict_unchanged_payload_contract():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/eval_ppl_equivalence.py").read_text(encoding="utf-8")
    assert '"unchanged_brick_payload": True' in text
    assert '"same_eval_bytes": True' in text
    assert "symmetric_ppl_ratio" in text
    assert "ppl_tolerance" in text
