from __future__ import annotations

import hashlib

import pytest

from layercake.evaluation.phase2_evidence import (
    Phase2EvidenceError,
    validate_inference_records,
)


def _candidate_row() -> dict:
    output = b"word " * 96
    elapsed = 0.48
    generated_tokens = 240
    return {
        "format": "layercake-phase2-raw-inference/1",
        "run_id": "candidate-1",
        "system_id": "layercake_sparse_bpe_primary",
        "suite": "functional_headline",
        "checkpoint_sha256": "a" * 64,
        "prompt_id": "prompt-1",
        "prompt_sha256": "b" * 64,
        "prompt_tokens": 10,
        "trial": 1,
        "status": "PASS",
        "device": "cpu",
        "cache_state": "warm",
        "generation_mode": "deterministic_constrained_english",
        "english_planner": {
            "enabled": True,
            "checkpoint_buffer_sha256": "c" * 64,
            "neural_prefill_selects_lexical_rotation": True,
            "forced_plan_tokens": generated_tokens,
            "frozen_evaluation_content": False,
        },
        "working_set_management": {
            "status": "ACTIVE_SET_COMPACTED",
            "post_compaction_warmup": True,
        },
        "generated_bytes": len(output),
        "generated_tokens": generated_tokens,
        "output_hex": output.hex(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "total_latency_seconds": elapsed,
        "time_to_first_output_seconds": 0.01,
        "bytes_per_second": len(output) / elapsed,
        "characters_per_second": len(output) / elapsed,
        "process_resident_bytes": 1000,
        "resident_model_tensor_bytes": 800,
        "active_parameter_bytes": 600,
        "installed_parameter_bytes": 800,
        "token_accounting_method": "authoritative_runtime_selected_ids_and_posthoc_locked_tokenizer",
        "persistent_state": {
            "decode_input_tokens_per_step": 1,
            "cached_tokens": [250, 250, 250],
        },
        "sparse_execution": {
            "maximum_active_experts_per_token": 1,
            "expert_forward_calls": [30] * 8,
            "total_decode_expert_invocations": generated_tokens,
        },
    }


def test_phase2_inference_validator_accepts_authoritative_sparse_cached_row() -> None:
    row = _candidate_row()
    result = validate_inference_records(
        [row], system_id=row["system_id"], suite=row["suite"],
        checkpoint_sha256=row["checkpoint_sha256"], minimum_bytes=480,
    )
    assert result["records"] == 1


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("bytes_per_second", 2000.0, "does not recompute"),
        ("token_accounting_method", "estimated", "not authoritative"),
        ("persistent_state", {}, "incremental state"),
    ],
)
def test_phase2_inference_validator_rejects_adversarial_mutations(
    field: str, value, error: str
) -> None:
    row = _candidate_row()
    row[field] = value
    with pytest.raises(Phase2EvidenceError, match=error):
        validate_inference_records(
            [row], system_id=row["system_id"], suite=row["suite"],
            checkpoint_sha256=row["checkpoint_sha256"], minimum_bytes=480,
        )


def test_phase2_inference_validator_rejects_dense_decode_trace() -> None:
    row = _candidate_row()
    row["sparse_execution"]["total_decode_expert_invocations"] *= 8
    with pytest.raises(Phase2EvidenceError, match="physically skipped"):
        validate_inference_records(
            [row], system_id=row["system_id"], suite=row["suite"],
            checkpoint_sha256=row["checkpoint_sha256"], minimum_bytes=480,
        )


def test_phase2_inference_validator_accepts_completed_qwen_token_scope() -> None:
    output = b"word " * 96
    completed = output + b"done"
    elapsed = 1.0
    request_elapsed = 1.25
    tokens = 121
    row = {
        "format": "layercake-phase2-raw-inference/1",
        "run_id": "qwen-1",
        "system_id": "qwen25_05b_optimized_cpu",
        "suite": "functional_headline",
        "checkpoint_sha256": "a" * 64,
        "prompt_id": "prompt-1",
        "trial": 1,
        "status": "PASS",
        "device": "cpu",
        "cache_state": "warm",
        "generation_mode": "deterministic",
        "generated_bytes": len(output),
        "generated_tokens": tokens,
        "output_hex": output.hex(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "total_latency_seconds": elapsed,
        "request_total_latency_seconds": request_elapsed,
        "time_to_first_output_seconds": 0.01,
        "bytes_per_second": len(output) / elapsed,
        "characters_per_second": len(output) / elapsed,
        "tokens_per_second": tokens / request_elapsed,
        "process_resident_bytes": 1000,
        "resident_model_tensor_bytes": 800,
        "active_parameter_bytes": 800,
        "installed_parameter_bytes": 800,
        "token_accounting_method": "ollama_terminal_eval_count",
        "token_accounting_scope": "completed_response_secondary_metric",
        "completed_response_bytes": len(completed),
        "completed_response_tokens": tokens,
        "completed_response_hex": completed.hex(),
        "completed_response_sha256": hashlib.sha256(completed).hexdigest(),
    }
    result = validate_inference_records(
        [row], system_id=row["system_id"], suite=row["suite"],
        checkpoint_sha256=row["checkpoint_sha256"], minimum_bytes=480,
    )
    assert result["records"] == 1


def test_phase2_inference_validator_rejects_qwen_prefix_token_mixing() -> None:
    output = b"word " * 96
    row = {
        "format": "layercake-phase2-raw-inference/1", "run_id": "qwen-1",
        "system_id": "qwen25_05b_optimized_cpu", "suite": "functional_headline",
        "checkpoint_sha256": "a" * 64, "prompt_id": "prompt-1", "trial": 1,
        "status": "PASS", "device": "cpu", "cache_state": "warm",
        "generation_mode": "deterministic", "generated_bytes": len(output),
        "generated_tokens": 121, "output_hex": output.hex(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "total_latency_seconds": 1.0, "request_total_latency_seconds": 1.25,
        "time_to_first_output_seconds": 0.01, "bytes_per_second": len(output),
        "characters_per_second": len(output), "tokens_per_second": 121.0,
        "process_resident_bytes": 1000, "resident_model_tensor_bytes": 800,
        "active_parameter_bytes": 800, "installed_parameter_bytes": 800,
        "token_accounting_method": "ollama_terminal_eval_count",
        "token_accounting_scope": "completed_response_secondary_metric",
        "completed_response_bytes": len(output), "completed_response_tokens": 121,
        "completed_response_hex": output.hex(),
        "completed_response_sha256": hashlib.sha256(output).hexdigest(),
    }
    with pytest.raises(Phase2EvidenceError, match="mixes prefix"):
        validate_inference_records(
            [row], system_id=row["system_id"], suite=row["suite"],
            checkpoint_sha256=row["checkpoint_sha256"], minimum_bytes=480,
        )
