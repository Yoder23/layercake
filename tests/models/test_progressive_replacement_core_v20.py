from layercake.progressive_replacement_core import ProgressiveReplacementCore


def test_corrected_progressive_replacement_target_uses_runtime_vocabulary() -> None:
    assert ProgressiveReplacementCore.parameter_count_for_config(
        fixed_vocab_size=32_015,
        full_width=3_072,
        bottleneck_width=192,
        replacement_layers=32,
        intermediate_size=768,
    ) == 253_535_232
