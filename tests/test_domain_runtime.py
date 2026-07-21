from layercake.domain_runtime import (
    load_portable_domain_chunks,
    load_instruction_aliases,
    match_instruction_alias,
    match_portable_domain_chunk,
    normalize_instruction_tokens,
    parse_portable_domain_chunks,
    parse_instruction_aliases,
    render_portable_domain_answer,
    render_instruction_alias_answer,
)


def test_instruction_alias_matches_paraphrased_game_prompt(tmp_path):
    corpus = tmp_path / "domain.txt"
    corpus.write_text(
        "Question: How should I recover after a mistake in combat? "
        "Answer: Stop chasing damage, retreat or guard to stabilize, remove the nearest threat.\n",
        encoding="utf-8",
    )
    aliases = load_instruction_aliases([corpus])
    match = match_instruction_alias(
        "Question: I made a combat mistake; how do I get stable again? Answer:",
        aliases,
        threshold=0.34,
    )
    assert match is not None
    assert match.alias.question == "How should I recover after a mistake in combat?"
    assert match.overlap >= 2


def test_instruction_alias_rejects_unrelated_prompt(tmp_path):
    corpus = tmp_path / "domain.txt"
    corpus.write_text(
        "Question: How do I fight an archer? Answer: Flank the archer safely.\n",
        encoding="utf-8",
    )
    aliases = load_instruction_aliases([corpus])
    assert (
        match_instruction_alias(
            "Question: What is the recipe for bread dough? Answer:",
            aliases,
            threshold=0.34,
        )
        is None
    )


def test_render_instruction_alias_answer_truncates_to_generation_budget():
    aliases = parse_instruction_aliases(
        "Question: Two threats appear. Answer: Create space first and kite safely.\n"
    )
    rendered = render_instruction_alias_answer(
        "Question: Two enemies show up. What is the safest opening move? Answer:",
        aliases,
        max_new_bytes=14,
        threshold=0.34,
    )
    assert rendered is not None
    text, match = rendered
    assert text == " Create space "
    assert match.alias.answer.startswith("Create space")


def test_normalize_instruction_tokens_applies_domain_synonyms():
    tokens = normalize_instruction_tokens("Two enemies show up and I need a safe opening.")
    assert "threat" in tokens
    assert "safest" in tokens
    assert "step" in tokens


def test_portable_domain_memory_matches_paraphrased_app_prompt(tmp_path):
    corpus = tmp_path / "manual.txt"
    corpus.write_text(
        "Atlas Notes keeps a local encrypted cache of recent notebooks so writing "
        "stays available offline. When the device reconnects, sync resumes automatically.\n",
        encoding="utf-8",
    )
    chunks = load_portable_domain_chunks([corpus])
    match = match_portable_domain_chunk(
        "Question: Can I keep writing if my phone loses service? Answer:",
        chunks,
        threshold=0.20,
        min_overlap=1,
    )
    assert match is not None
    assert "offline" in match.chunk.text
    rendered = render_portable_domain_answer(
        "Question: Can I keep writing if my phone loses service? Answer:",
        chunks,
        max_new_bytes=160,
        threshold=0.20,
        min_overlap=1,
    )
    assert rendered is not None
    assert "local encrypted cache" in rendered[0]


def test_portable_domain_memory_chunks_arbitrary_corpus_text():
    chunks = parse_portable_domain_chunks(
        "# Website\n"
        "Returns are allowed within thirty days with a receipt. "
        "Use Start Return to print the prepaid label.\n\n"
        "# App\n"
        "Open Settings and choose Rebuild Index when search feels stale.\n",
        source="docs",
        max_chunk_chars=120,
    )
    assert len(chunks) >= 2
    assert all(chunk.tokens for chunk in chunks)
