from layercake.companion_runtime import finalize_companion_text, trim_at_stop_sequence
from layercake.domain_runtime import load_instruction_aliases, render_instruction_alias_answer


def test_trim_at_stop_sequence_cuts_dataset_continuation():
    text, stop = trim_at_stop_sequence(
        "Take cover first.\nQuestion: What is the next prompt?"
    )
    assert text == "Take cover first."
    assert stop == "\nQuestion:"


def test_finalize_companion_text_removes_control_noise_and_compacts_space():
    text, meta = finalize_companion_text(" Guard now.\x00 { bad json")
    assert text == "Guard now."
    assert meta["trimmed"] is True
    assert meta["stop_sequence"] == "\x00"


def test_companion_alias_file_covers_review_prompts():
    aliases = load_instruction_aliases(
        [
            "data/game_domains/ember-road.instructions.txt",
            "data/game_domains/ember-road.companion_responses.txt",
        ]
    )
    nervous = render_instruction_alias_answer(
        "Question: The player is nervous before a boss fight. Give a calm, useful companion response. Answer:",
        aliases,
        max_new_bytes=180,
    )
    ambush = render_instruction_alias_answer(
        "Question: The player asks what to do after getting hit by a surprise ambush. Answer:",
        aliases,
        max_new_bytes=180,
    )
    assert nervous is not None
    assert "Breathe" in nervous[0]
    assert ambush is not None
    assert "Retreat" in ambush[0] or "Guard" in ambush[0]


def test_companion_alias_file_covers_ad_hoc_ambush_health_prompt():
    aliases = load_instruction_aliases(
        [
            "data/game_domains/ember-road.instructions.txt",
            "data/game_domains/ember-road.companion_responses.txt",
        ]
    )
    rendered = render_instruction_alias_answer(
        "Question: I got ambushed and lost health. What should I do now? Answer:",
        aliases,
        max_new_bytes=180,
    )
    assert rendered is not None
    text, _ = rendered
    assert "Retreat" in text
    assert "stabilize health" in text
