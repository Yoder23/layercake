import torch

from layercake.models.baseline_transformer import (
    BytePairTokenizer, ModernBPETransformer, matched_transformer_config,
)
from layercake.models.foundation import FoundationConfig, LayerCakeFoundation


def test_bpe_is_learned_only_from_bytes_and_round_trips():
    corpus = b"banana bandana banana bandana"
    tokenizer = BytePairTokenizer.train(corpus, merge_count=8)
    encoded = tokenizer.encode(corpus)
    assert len(encoded) < len(corpus)
    assert tokenizer.decode(encoded) == corpus


def test_modern_transformer_is_parameter_matched_with_stable_initialization():
    foundation = LayerCakeFoundation(FoundationConfig(d_byte=24, d_model=64, recurrent_layers=1))
    target = sum(parameter.numel() for parameter in foundation.parameters())
    config = matched_transformer_config(target, vocab_size=288, max_tokens=65)
    transformer = ModernBPETransformer(config)
    actual = transformer.parameter_count()
    assert abs(actual - target) / target <= 0.05
    assert transformer.embedding.weight.std().item() < 0.03
    ids = torch.randint(0, 288, (2, 12))
    assert transformer(ids).shape == (2, 12, 288)
