from pathlib import Path


def test_required_v2_docs_exist_and_rubric_is_complete():
    root = Path(__file__).resolve().parents[1]
    required = [
        "RUBRIC.md",
        "ARCHITECTURE.md",
        "BENCHMARKS.md",
        "ORCHESTRATION.md",
        "TOKENIZER_FREE.md",
        "BYTE_PATCH_LAYERCAKE.md",
        "ROADMAP.md",
        "NEXT_STEPS.md",
    ]
    assert all((root / name).exists() for name in required)
    rubric = (root / "RUBRIC.md").read_text(encoding="utf-8")
    for level in range(8):
        assert f"L{level}" in rubric
