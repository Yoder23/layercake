"""Fail-closed typed validation for the Phase 2 integrated-core proof."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from layercake.training.data import sha256_file


class Phase2EvidenceError(ValueError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase2EvidenceError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2EvidenceError(f"evidence document is not an object: {path}")
    return value


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Phase2EvidenceError(f"{field} is not finite numeric evidence")
    result = float(value)
    if positive and result <= 0:
        raise Phase2EvidenceError(f"{field} must be positive")
    return result


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_depth(
    rows: Sequence[Mapping[str, Any]], *, distinct: int, repeated: int, observations: int
) -> None:
    if len(rows) != observations:
        raise Phase2EvidenceError(f"expected {observations} observations, found {len(rows)}")
    prompt_trials: dict[str, set[int]] = {}
    for row in rows:
        prompt_trials.setdefault(str(row.get("prompt_id")), set()).add(int(row.get("trial", 0)))
    if len(prompt_trials) != distinct:
        raise Phase2EvidenceError(f"expected {distinct} distinct prompts, found {len(prompt_trials)}")
    repeated_count = sum(len(trials) >= 2 for trials in prompt_trials.values())
    if repeated_count < repeated:
        raise Phase2EvidenceError(
            f"expected at least {repeated} repeated prompt observations, found {repeated_count}"
        )


def validate_inference_records(
    rows: Sequence[Mapping[str, Any]], *, system_id: str, suite: str,
    checkpoint_sha256: str, minimum_bytes: int, planner_sha256: str | None = None,
) -> dict[str, Any]:
    if not rows:
        raise Phase2EvidenceError("inference record set is empty")
    run_ids: set[str] = set()
    prompt_pairs: set[tuple[str, int]] = set()
    for row in rows:
        if row.get("format") != "layercake-phase2-raw-inference/1":
            raise Phase2EvidenceError("raw inference format is invalid")
        if row.get("system_id") != system_id or row.get("suite") != suite:
            raise Phase2EvidenceError("inference system or suite identity is mixed")
        if row.get("checkpoint_sha256") != checkpoint_sha256:
            raise Phase2EvidenceError("quality/speed checkpoint lineage is mixed")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or run_id in run_ids:
            raise Phase2EvidenceError("run identifiers are missing or duplicated")
        run_ids.add(run_id)
        pair = (str(row.get("prompt_id")), int(row.get("trial", 0)))
        if pair in prompt_pairs:
            raise Phase2EvidenceError("prompt/trial pairing is duplicated")
        prompt_pairs.add(pair)
        if row.get("status") != "PASS" or row.get("device") != "cpu":
            raise Phase2EvidenceError("raw inference row did not complete on CPU")
        expected_mode = (
            "deterministic_constrained_english"
            if system_id == "layercake_sparse_bpe_primary"
            else "deterministic"
        )
        if row.get("cache_state") != "warm" or row.get("generation_mode") != expected_mode:
            raise Phase2EvidenceError("promoted Phase 2 rows must use the locked warm deterministic path")
        if system_id == "layercake_sparse_bpe_primary":
            planner = row.get("english_planner")
            if (
                not isinstance(planner, dict)
                or planner.get("enabled") is not True
                or planner.get("neural_prefill_selects_lexical_rotation") is not True
                or planner.get("frozen_evaluation_content") is not False
                or not isinstance(planner.get("checkpoint_buffer_sha256"), str)
                or planner.get("forced_plan_tokens") != row.get("generated_tokens")
            ):
                raise Phase2EvidenceError("checkpoint-bound neural-guided English planner evidence is absent")
            if planner_sha256 is not None and planner["checkpoint_buffer_sha256"] != planner_sha256:
                raise Phase2EvidenceError("English planner evidence is not bound to the checkpoint buffer")
        generated_bytes = int(_finite(row.get("generated_bytes"), "generated_bytes", positive=True))
        generated_tokens = int(_finite(row.get("generated_tokens"), "generated_tokens", positive=True))
        if generated_bytes < minimum_bytes:
            raise Phase2EvidenceError("generation stopped before the locked output target")
        try:
            output = bytes.fromhex(str(row.get("output_hex")))
        except ValueError as error:
            raise Phase2EvidenceError("output payload is not valid hexadecimal") from error
        if len(output) != generated_bytes or _hash_bytes(output) != row.get("output_sha256"):
            raise Phase2EvidenceError("output payload size or hash is stale")
        if suite == "long_context":
            expected = row.get("expected_codeword")
            if not isinstance(expected, str) or not expected:
                raise Phase2EvidenceError("long-context codeword is absent")
            observed = output.decode("utf-8", errors="replace").lstrip().casefold().startswith(
                expected.casefold()
            )
            if row.get("long_context_success") is not observed:
                raise Phase2EvidenceError("long-context success flag is stale")
        elapsed = _finite(row.get("total_latency_seconds"), "total_latency_seconds", positive=True)
        ttfo = _finite(row.get("time_to_first_output_seconds"), "time_to_first_output_seconds", positive=True)
        if ttfo > elapsed:
            raise Phase2EvidenceError("time to first output exceeds total latency")
        measured_bps = _finite(row.get("bytes_per_second"), "bytes_per_second", positive=True)
        if not math.isclose(measured_bps, generated_bytes / elapsed, rel_tol=1e-9, abs_tol=1e-9):
            raise Phase2EvidenceError("bytes/second does not recompute from raw duration")
        _finite(row.get("characters_per_second"), "characters_per_second", positive=True)
        _finite(row.get("process_resident_bytes"), "process_resident_bytes", positive=True)
        _finite(row.get("resident_model_tensor_bytes"), "resident_model_tensor_bytes", positive=True)
        _finite(row.get("active_parameter_bytes"), "active_parameter_bytes", positive=True)
        _finite(row.get("installed_parameter_bytes"), "installed_parameter_bytes", positive=True)
        method = row.get("token_accounting_method")
        if system_id.startswith("layercake"):
            if method != "authoritative_runtime_selected_ids_and_posthoc_locked_tokenizer":
                raise Phase2EvidenceError("LayerCake token accounting is not authoritative")
            state = row.get("persistent_state")
            sparse = row.get("sparse_execution")
            if not isinstance(state, dict) or state.get("decode_input_tokens_per_step") != 1:
                raise Phase2EvidenceError("persistent incremental state is absent")
            cached = state.get("cached_tokens")
            prompt_tokens = row.get("prompt_tokens")
            if (
                not isinstance(prompt_tokens, int)
                or not isinstance(cached, list)
                or len(cached) != 3
                or any(value != prompt_tokens + generated_tokens for value in cached)
            ):
                raise Phase2EvidenceError("KV cache did not advance exactly once per generated token")
            if not isinstance(sparse, dict):
                raise Phase2EvidenceError("physical sparse-execution trace is absent")
            calls = sparse.get("expert_forward_calls")
            if (
                sparse.get("maximum_active_experts_per_token") != 1
                or not isinstance(calls, list)
                or len(calls) != 8
                or sum(int(value) for value in calls) != generated_tokens
                or sparse.get("total_decode_expert_invocations") != generated_tokens
            ):
                raise Phase2EvidenceError("inactive experts were not physically skipped")
        else:
            if method != "ollama_terminal_eval_count":
                raise Phase2EvidenceError("transformer token count is not runtime-authoritative")
            if row.get("token_accounting_scope") != "completed_response_secondary_metric":
                raise Phase2EvidenceError("transformer token count scope is ambiguous")
            try:
                completed_output = bytes.fromhex(str(row.get("completed_response_hex")))
            except ValueError as error:
                raise Phase2EvidenceError("completed transformer response is not valid hexadecimal") from error
            completed_bytes = int(_finite(
                row.get("completed_response_bytes"), "completed_response_bytes", positive=True,
            ))
            completed_tokens = int(_finite(
                row.get("completed_response_tokens"), "completed_response_tokens", positive=True,
            ))
            request_elapsed = _finite(
                row.get("request_total_latency_seconds"), "request_total_latency_seconds", positive=True,
            )
            if (
                len(completed_output) != completed_bytes
                or _hash_bytes(completed_output) != row.get("completed_response_sha256")
                or completed_tokens != generated_tokens
                or completed_bytes < generated_bytes
                or completed_output[:generated_bytes] != output
                or request_elapsed < elapsed
            ):
                raise Phase2EvidenceError("completed transformer response evidence is stale")
            tokens_per_second = _finite(row.get("tokens_per_second"), "tokens_per_second", positive=True)
            if not math.isclose(
                tokens_per_second, completed_tokens / request_elapsed,
                rel_tol=1e-9, abs_tol=1e-9,
            ):
                raise Phase2EvidenceError("transformer tokens/second mixes prefix and completed-response timing")
    return {"records": len(rows), "pairs": len(prompt_pairs), "run_ids": len(run_ids)}


def _functional_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    primary = [row for row in rows if row["trial"] == 1]
    names = tuple(primary[0]["quality"])
    return {
        name: statistics.fmean(float(row["quality"][name]) for row in primary)
        for name in names
    }


def _load_raw(phase: Path, name: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    path = phase / "raw_runs" / name
    document = _read(path)
    rows = document.get("records")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise Phase2EvidenceError(f"{name} has no object-valued records")
    return path, document, rows


def validate_phase2_bundle(root: Path, phase: Path) -> dict[str, Any]:
    from layercake.phase2_campaign import _functional_scores

    config_path = root / "configs/moonshot/phase2/final_benchmark.json"
    config = _read(config_path)
    protocol = _read(phase / "protocol_manifest.json")
    if config.get("format") != "layercake-phase2-benchmark-lock/1" or config.get("locked_before_final_evaluation") is not True:
        raise Phase2EvidenceError("Phase 2 benchmark was not frozen before final evaluation")
    if protocol.get("status") != "LOCKED" or protocol.get("benchmark_config", {}).get("sha256") != sha256_file(config_path):
        raise Phase2EvidenceError("Phase 2 protocol manifest is stale")
    suite_path = root / config["phase1_quality_suite"]["path"]
    if sha256_file(suite_path) != config["phase1_quality_suite"]["sha256"]:
        raise Phase2EvidenceError("corrected Phase 1 quality suite changed")

    quality_path, quality, quality_rows = _load_raw(phase, "quality_seeds.json")
    if quality.get("format") != "layercake-phase2-quality-seeds/1":
        raise Phase2EvidenceError("three-seed quality format is invalid")
    seeds = {row.get("seed") for row in quality_rows}
    if seeds != {9824, 9825, 9826}:
        raise Phase2EvidenceError(f"promoted quality evidence must preserve all three seeds, got {seeds}")
    if not all(row.get("test_accessed") is True for row in quality_rows):
        raise Phase2EvidenceError("final quality evidence did not evaluate the frozen test split")
    primary_hash = next(row["checkpoint_sha256"] for row in quality_rows if row["seed"] == 9824)
    primary_planner_sha = None
    for row in quality_rows:
        if int(row.get("raw_training_bytes", 0)) < 100_000_000:
            raise Phase2EvidenceError("a promoted seed used less than the 100M-byte tier")
        sealed_root = root / str(config["candidate_lineage"]["sealed_checkpoint_template"]).format(
            seed=row["seed"]
        )
        source_root = root / str(config["candidate_lineage"]["checkpoint_template"]).format(
            seed=row["seed"]
        )
        checkpoint = (sealed_root if sealed_root.is_dir() else source_root) / "model.safetensors"
        if sha256_file(checkpoint) != row.get("checkpoint_sha256"):
            raise Phase2EvidenceError("quality checkpoint artifact hash is stale")
        metadata = _read(checkpoint.parent / "metadata.json")
        distillation = metadata.get("instruction_distillation")
        if (
            metadata.get("format") != "layercake-sparse-bpe-instruction-core/1"
            or not isinstance(distillation, dict)
            or distillation.get("steps") != config["candidate_lineage"]["distillation_steps"]
        ):
            raise Phase2EvidenceError("promoted checkpoint is not the locked integrated instruction lineage")
        planner = metadata.get("english_planner")
        architecture = metadata.get("architecture", {})
        if (
            architecture.get("constrained_english_planner") is not True
            or architecture.get("prompt_conditioning") is not True
            or not isinstance(planner, dict)
            or planner.get("enabled") is not True
            or planner.get("frozen_evaluation_content") is not False
            or not isinstance(planner.get("checkpoint_buffer_sha256"), str)
        ):
            raise Phase2EvidenceError("promoted checkpoint is missing its bounded English realization state")
        if row["seed"] == 9824:
            primary_planner_sha = planner["checkpoint_buffer_sha256"]
        corpus = Path(str(distillation.get("corpus_path", "")))
        if not corpus.is_absolute():
            corpus = root / corpus
        if not corpus.is_file() or sha256_file(corpus) != distillation.get("corpus_sha256"):
            raise Phase2EvidenceError("promoted instruction corpus is missing or stale")
        _finite(row.get("validation_bpb"), "validation_bpb", positive=True)
        _finite(row.get("validation_calibration_error"), "validation_calibration_error")
        _finite(row.get("test_bpb"), "test_bpb", positive=True)

    layer_path, _, layer = _load_raw(phase, "layercake_functional.json")
    qwen_path, _, qwen = _load_raw(phase, "qwen_functional.json")
    layer_s_path, _, layer_s = _load_raw(phase, "layercake_sustained.json")
    qwen_s_path, _, qwen_s = _load_raw(phase, "qwen_sustained.json")
    layer_l_path, _, layer_l = _load_raw(phase, "layercake_long_context.json")
    qwen_l_path, _, qwen_l = _load_raw(phase, "qwen_long_context.json")
    qwen_hash = config["product_reference"]["checkpoint_sha256"]
    validate_inference_records(layer, system_id="layercake_sparse_bpe_primary", suite="functional_headline", checkpoint_sha256=primary_hash, minimum_bytes=480, planner_sha256=primary_planner_sha)
    validate_inference_records(qwen, system_id="qwen25_05b_optimized_cpu", suite="functional_headline", checkpoint_sha256=qwen_hash, minimum_bytes=480)
    validate_inference_records(layer_s, system_id="layercake_sparse_bpe_primary", suite="sustained_1024", checkpoint_sha256=primary_hash, minimum_bytes=1024, planner_sha256=primary_planner_sha)
    validate_inference_records(qwen_s, system_id="qwen25_05b_optimized_cpu", suite="sustained_1024", checkpoint_sha256=qwen_hash, minimum_bytes=1024)
    validate_inference_records(layer_l, system_id="layercake_sparse_bpe_primary", suite="long_context", checkpoint_sha256=primary_hash, minimum_bytes=64, planner_sha256=primary_planner_sha)
    validate_inference_records(qwen_l, system_id="qwen25_05b_optimized_cpu", suite="long_context", checkpoint_sha256=qwen_hash, minimum_bytes=64)
    _validate_depth(layer, distinct=100, repeated=20, observations=120)
    _validate_depth(qwen, distinct=100, repeated=20, observations=120)
    _validate_depth(layer_s, distinct=20, repeated=20, observations=40)
    _validate_depth(qwen_s, distinct=20, repeated=20, observations=40)
    _validate_depth(layer_l, distinct=20, repeated=0, observations=20)
    _validate_depth(qwen_l, distinct=20, repeated=0, observations=20)
    layer_pairs = {(row["prompt_id"], row["prompt_sha256"], row["trial"]) for row in layer}
    qwen_pairs = {(row["prompt_id"], row["prompt_sha256"], row["trial"]) for row in qwen}
    if layer_pairs != qwen_pairs:
        raise Phase2EvidenceError("product functional/speed comparison is not prompt paired")
    if {(row["prompt_id"], row["trial"]) for row in layer_s} != {(row["prompt_id"], row["trial"]) for row in qwen_s}:
        raise Phase2EvidenceError("sustained comparison is not prompt paired")
    if {
        (row["prompt_id"], row["prompt_sha256"], row["trial"], row["expected_codeword"])
        for row in layer_l
    } != {
        (row["prompt_id"], row["prompt_sha256"], row["trial"], row["expected_codeword"])
        for row in qwen_l
    }:
        raise Phase2EvidenceError("long-context comparison is not prompt paired")

    derived_path = phase / "derived_evidence.json"
    derived = _read(derived_path)
    payload = _read(phase / "certificate_payload.json")
    if derived.get("format") != "layercake-phase2-derived-evidence/1" or derived.get("status") != "PASS":
        raise Phase2EvidenceError(f"Phase 2 derived gates failed: {derived.get('failed_gates')}")
    if payload.get("format") != "layercake-phase2-certificate-payload/1" or payload.get("status") != "PASS":
        raise Phase2EvidenceError("Phase 2 certificate payload is not passing")
    expected_sources = {
        name: sha256_file(path)
        for name, path in {
            "layercake_functional.json": layer_path,
            "qwen_functional.json": qwen_path,
            "layercake_sustained.json": layer_s_path,
            "qwen_sustained.json": qwen_s_path,
            "quality_seeds.json": quality_path,
            "layercake_long_context.json": layer_l_path,
            "qwen_long_context.json": qwen_l_path,
        }.items()
    }
    if derived.get("raw_artifacts") != expected_sources:
        raise Phase2EvidenceError("derived evidence source hashes are stale")
    if payload.get("derived_evidence", {}).get("sha256") != sha256_file(derived_path):
        raise Phase2EvidenceError("certificate payload does not bind the derived evidence")
    if payload.get("lineage", {}).get("primary_checkpoint_sha256") != primary_hash:
        raise Phase2EvidenceError("certificate quality/speed checkpoint identity is mixed")

    candidate_bps = statistics.fmean(float(row["bytes_per_second"]) for row in layer)
    qwen_bps = statistics.fmean(float(row["bytes_per_second"]) for row in qwen)
    candidate_s_bps = statistics.fmean(float(row["bytes_per_second"]) for row in layer_s)
    qwen_s_bps = statistics.fmean(float(row["bytes_per_second"]) for row in qwen_s)
    validation_mean = statistics.fmean(float(row["validation_bpb"]) for row in quality_rows)
    calibration_mean = statistics.fmean(float(row["validation_calibration_error"]) for row in quality_rows)
    reference_calibration = _finite(
        quality.get("reference", {}).get("validation_calibration_error"),
        "reference_validation_calibration_error",
    )
    layer_long_accuracy = statistics.fmean(float(row["long_context_success"]) for row in layer_l)
    qwen_long_accuracy = statistics.fmean(float(row["long_context_success"]) for row in qwen_l)
    recomputed = {
        "heldout_bpb_delta": validation_mean - float(config["quality_reference"]["validation_bpb"]),
        "cpu_throughput_ratio": candidate_bps / qwen_bps,
        "cpu_median_latency_ratio": statistics.median(float(row["total_latency_seconds"]) for row in layer) / statistics.median(float(row["total_latency_seconds"]) for row in qwen),
        "time_to_first_output_ratio": statistics.median(float(row["time_to_first_output_seconds"]) for row in layer) / statistics.median(float(row["time_to_first_output_seconds"]) for row in qwen),
        "active_memory_ratio": max(float(row["active_parameter_bytes"]) for row in layer) / max(float(row["active_parameter_bytes"]) for row in qwen),
        "installed_model_memory_ratio": max(float(row["installed_parameter_bytes"]) for row in layer) / max(float(row["installed_parameter_bytes"]) for row in qwen),
        "process_resident_memory_ratio": max(float(row["process_resident_bytes"]) for row in layer) / max(float(row["process_resident_bytes"]) for row in qwen),
        "sustained_1024_byte_throughput_ratio": candidate_s_bps / qwen_s_bps,
        "physical_sparse_execution": 1.0,
        "long_context_accuracy_delta": layer_long_accuracy - qwen_long_accuracy,
    }
    aggregates = derived.get("aggregates", {})
    for name, value in recomputed.items():
        if not math.isclose(value, float(aggregates.get(name, math.nan)), rel_tol=0.0, abs_tol=1e-12):
            raise Phase2EvidenceError(f"derived gate {name} is stale or hand edited")
    functional = {
        "layercake": {**_functional_metrics(layer), **_functional_scores(layer)},
        "qwen": {**_functional_metrics(qwen), **_functional_scores(qwen)},
    }
    recorded_functional = derived.get("functional_quality", {})
    for system, metrics in functional.items():
        for name, value in metrics.items():
            if not math.isclose(
                value, float(recorded_functional.get(system, {}).get(name, math.nan)),
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise Phase2EvidenceError(f"functional quality metric {system}/{name} is stale")
    if not math.isclose(
        calibration_mean - reference_calibration,
        float(aggregates.get("calibration_error_delta", math.nan)),
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise Phase2EvidenceError("calibration non-inferiority evidence is stale")
    if not all(derived.get("gates", {}).values()):
        raise Phase2EvidenceError("one or more quality or performance gates failed")
    adversarial = _read(phase / "adversarial_checks.json")
    if adversarial.get("status") != "PASS" or adversarial.get("detected", 0) < 13:
        raise Phase2EvidenceError("adversarial verifier tests are incomplete")
    tests = _read(phase / "test_results.json")
    if tests.get("status") != "PASS" or tests.get("failures") != 0 or tests.get("errors") != 0:
        raise Phase2EvidenceError("complete Phase 2 regression evidence is not green")
    junit = root / tests.get("junit_path", "")
    if not junit.is_file() or sha256_file(junit) != tests.get("junit_sha256"):
        raise Phase2EvidenceError("Phase 2 JUnit artifact is missing or stale")
    manifest = _read(phase / "evidence_manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise Phase2EvidenceError("Phase 2 evidence manifest is empty")
    for artifact in artifacts:
        path = root / artifact.get("path", "")
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise Phase2EvidenceError(f"Phase 2 manifested artifact is stale: {artifact.get('path')}")

    return {
        "status": "PASS",
        "architecture_hash": quality["architecture_hash"],
        "primary_checkpoint_sha256": primary_hash,
        "transformer_checkpoint_sha256": qwen_hash,
        "seeds": sorted(seeds),
        "quality": {"validation_bpb_mean": validation_mean, "test_bpb_mean": statistics.fmean(float(row["test_bpb"]) for row in quality_rows)},
        "gates": recomputed,
        "functional_quality": functional,
        "raw_records": (
            len(layer) + len(qwen) + len(layer_s) + len(qwen_s)
            + len(layer_l) + len(qwen_l) + len(quality_rows)
        ),
    }
