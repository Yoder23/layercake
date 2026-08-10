from __future__ import annotations

import pytest
import torch

from layercake.routed_sparse_rank768_progressive_core import RoutedSparseRank768ProgressiveCore
from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from tests.models.test_decoder_direct_neural_core import _doc


def _model(kind):
    tokenizer = DecoderAwareExternalTokenizer(_doc())
    model = kind(
        fixed_vocab_size=tokenizer.vocab_size,
        full_width=24,
        bottleneck_width=8,
        attention_heads=2,
        replacement_layers=2,
        intermediate_size=16,
        residual_rank=16,
        sparse_width=8,
        maximum_source_actions=16,
        maximum_target_actions=8,
        maximum_sequence_actions=24,
    ).bind_tokenizer(tokenizer)
    return model, tokenizer


def test_v15_state_dict_is_exactly_compatible_and_fp16_cpu_router_executes() -> None:
    torch.manual_seed(16)
    source, _ = _model(RoutedSparseRank768ProgressiveCore)
    target, tokenizer = _model(PrecisionConformantRoutedSparseRank768ProgressiveCore)
    fp16 = {name: value.half() for name, value in source.state_dict().items()}
    incompatible = target.load_state_dict(fp16, strict=True, assign=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    assert set(target.state_dict()) == set(source.state_dict())
    ids, _ = tokenizer.encode_source("hello world")
    value = target._select_route(torch.tensor([ids]))
    assert value in range(3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_true_fp16_cuda_prefill_and_incremental_execution() -> None:
    torch.manual_seed(16)
    model, tokenizer = _model(PrecisionConformantRoutedSparseRank768ProgressiveCore)
    fp16 = {name: value.half().cuda() for name, value in model.state_dict().items()}
    model.load_state_dict(fp16, strict=True, assign=True)
    model = model.cuda().eval()
    ids, lexemes = tokenizer.encode_source("hello world")
    state = model.prefill_ids(ids, lexemes)
    assert state.route_index in range(3)
    assert torch.isfinite(state.next_logits).all()
    state.next_logits.zero_(); state.next_logits[0, 4] = 1
    _, state = model.decode_step(state)
    assert torch.isfinite(state.next_logits).all()
    assert state.sequence_length == len(ids) + 1
