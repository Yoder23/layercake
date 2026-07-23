"""Typed, fail-closed validation for Moonshot Phase 1 benchmark evidence."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .campaign_statistics import mean, median, p50, p95, p99, paired_bootstrap_difference


CORRECTION_PROTOCOL = "phase1-final-corrections/1"
TAIL_QUANTILE_MINIMUM = 20


PHASE1_FORMATS = {
    "hardware": "layercake-phase1-hardware/1",
    "runtime": "layercake-phase1-runtime-manifest/1",
    "model": "layercake-phase1-model-manifest/1",
    "matrix": "layercake-phase1-benchmark-matrix/1",
    "raw": "layercake-phase1-raw-timings/1",
    "quality": "layercake-phase1-quality-suite/1",
    "thresholds": "layercake-phase1-threshold-lock/1",
    "commands": "layercake-phase1-execution-commands/1",
    "evidence": "layercake-phase1-evidence-manifest/1",
}
REQUIRED_QUALITY_METRICS = {
    "heldout_bpb",
    "repetition_rate",
    "unique_ngram_rate",
    "entropy_collapse",
    "continuation_quality",
    "instruction_following",
    "long_context",
    "invalid_output_rate",
    "sample_inspection",
    "contamination",
}
REQUIRED_RAW_KEYS = {
    "run_id",
    "system_id",
    "runtime_id",
    "model_id",
    "model_sha256",
    "tokenizer_sha256",
    "configuration_sha256",
    "precision",
    "seed",
    "trial",
    "device",
    "threads",
    "prompt",
    "output",
    "generation",
    "cache_state",
    "order",
    "timing",
    "memory",
    "execution",
    "status",
}


class Phase1EvidenceError(RuntimeError):
    pass


def _document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase1EvidenceError(f"cannot read typed evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase1EvidenceError(f"typed evidence must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise Phase1EvidenceError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _require_format(document: Mapping[str, Any], kind: str) -> None:
    if document.get("format") != PHASE1_FORMATS[kind]:
        raise Phase1EvidenceError(f"invalid {kind} evidence format")


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase1EvidenceError(f"{label} must be non-empty text")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase1EvidenceError(f"{label} must be a positive integer")
    return value


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Phase1EvidenceError(f"{label} must be finite numeric evidence")
    converted = float(value)
    if minimum is not None and converted < minimum:
        raise Phase1EvidenceError(f"{label} must be >= {minimum}")
    return converted


def validate_hardware_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    _require_format(document, "hardware")
    capture = document.get("capture")
    if not isinstance(capture, dict):
        raise Phase1EvidenceError("hardware identity has no capture procedure")
    _nonempty_text(capture.get("command"), "hardware capture command")
    _nonempty_text(capture.get("stdout_sha256"), "hardware capture stdout hash")
    cpu = document.get("cpu")
    if not isinstance(cpu, dict):
        raise Phase1EvidenceError("CPU hardware record is missing")
    _nonempty_text(cpu.get("model"), "CPU model")
    physical = _positive_int(cpu.get("physical_cores"), "physical CPU cores")
    logical = _positive_int(cpu.get("logical_cores"), "logical CPU cores")
    if physical > logical:
        raise Phase1EvidenceError("physical CPU core count exceeds logical count")
    features = cpu.get("instruction_sets")
    if not isinstance(features, list) or not features or not all(isinstance(item, str) for item in features):
        raise Phase1EvidenceError("CPU instruction-set evidence is missing")
    memory = document.get("memory")
    if not isinstance(memory, dict):
        raise Phase1EvidenceError("memory hardware record is missing")
    _positive_int(memory.get("total_physical_bytes"), "physical memory")
    gpus = document.get("gpus")
    if not isinstance(gpus, list):
        raise Phase1EvidenceError("GPU inventory must be a list")
    for gpu in gpus:
        if not isinstance(gpu, dict):
            raise Phase1EvidenceError("GPU inventory row must be an object")
        _nonempty_text(gpu.get("name"), "GPU name")
        _nonempty_text(gpu.get("uuid"), "GPU UUID")
        _positive_int(gpu.get("memory_bytes"), "GPU memory")
        _nonempty_text(gpu.get("driver_version"), "GPU driver")
    return {"gpu_available": bool(gpus), "physical_cores": physical, "logical_cores": logical}


def validate_runtime_manifest(document: Mapping[str, Any], *, optimized: bool = False) -> str:
    _require_format(document, "runtime")
    runtime_id = _nonempty_text(document.get("id"), "runtime id")
    executable = document.get("executable")
    if not isinstance(executable, dict):
        raise Phase1EvidenceError(f"runtime {runtime_id} lacks executable evidence")
    _nonempty_text(executable.get("path"), "runtime executable path")
    _nonempty_text(executable.get("sha256"), "runtime executable hash")
    version = document.get("version")
    if not isinstance(version, dict):
        raise Phase1EvidenceError(f"runtime {runtime_id} lacks version evidence")
    _nonempty_text(version.get("command"), "runtime version command")
    _nonempty_text(version.get("stdout"), "runtime version output")
    _nonempty_text(version.get("stdout_sha256"), "runtime version output hash")
    _nonempty_text(document.get("backend"), "runtime backend")
    if document.get("target_device") not in {"cpu", "gpu", "cpu_and_gpu"}:
        raise Phase1EvidenceError(f"runtime {runtime_id} has no target-device contract")
    _nonempty_text(document.get("precision_contract"), "runtime precision contract")
    if optimized:
        optimization = document.get("optimization_evidence")
        if not isinstance(optimization, dict):
            raise Phase1EvidenceError(f"optimized runtime {runtime_id} has no structured evidence")
        kv = optimization.get("kv_cache")
        if not isinstance(kv, dict):
            raise Phase1EvidenceError(f"optimized runtime {runtime_id} has no KV-cache evidence")
        _nonempty_text(kv.get("mechanism"), "KV-cache mechanism")
        traces = kv.get("raw_trace_run_ids")
        if not isinstance(traces, list) or not traces or not all(isinstance(item, str) for item in traces):
            raise Phase1EvidenceError("KV-cache evidence must reference raw run traces")
        kernels = optimization.get("kernels")
        if not isinstance(kernels, dict):
            raise Phase1EvidenceError("optimized runtime has no kernel evidence")
        _nonempty_text(kernels.get("implementation"), "optimized kernel implementation")
        instruction_sets = kernels.get("instruction_sets")
        if not isinstance(instruction_sets, list) or not instruction_sets:
            raise Phase1EvidenceError("optimized runtime has no SIMD/kernel feature evidence")
        threading = optimization.get("threading")
        if not isinstance(threading, dict) or not threading.get("raw_trace_run_ids"):
            raise Phase1EvidenceError("optimized runtime has no thread-scaling evidence")
        batch = optimization.get("batch_one")
        if not isinstance(batch, dict) or batch.get("batch_size") != 1:
            raise Phase1EvidenceError("optimized runtime lacks batch-one execution evidence")
        probe = optimization.get("device_probe")
        if not isinstance(probe, dict):
            raise Phase1EvidenceError("optimized runtime lacks a measured device probe")
        _nonempty_text(probe.get("path"), "runtime device-probe path")
        _nonempty_text(probe.get("sha256"), "runtime device-probe hash")
        if probe.get("observed_target") != document.get("target_device"):
            raise Phase1EvidenceError("runtime device probe disagrees with target device")
        if "optimized" in document:
            raise Phase1EvidenceError("bare self-asserted optimized booleans are forbidden")
    return runtime_id


def validate_model_manifest(
    document: Mapping[str, Any], root: Path, *, verify_local_files: bool = True
) -> str:
    _require_format(document, "model")
    model_id = _nonempty_text(document.get("id"), "model id")
    _nonempty_text(document.get("architecture"), "model architecture")
    parameters = document.get("parameters")
    if not isinstance(parameters, dict):
        raise Phase1EvidenceError(f"model {model_id} has no parameter accounting")
    total = _positive_int(parameters.get("total"), "total parameters")
    active = _positive_int(parameters.get("active"), "active parameters")
    if active > total:
        raise Phase1EvidenceError("active model parameters exceed total parameters")
    checkpoint = document.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise Phase1EvidenceError(f"model {model_id} has no checkpoint identity")
    kind = checkpoint.get("kind")
    expected = _nonempty_text(checkpoint.get("sha256"), "checkpoint hash")
    if kind == "local_file":
        relative = _nonempty_text(checkpoint.get("path"), "checkpoint path")
        path = (root / relative).resolve()
        if verify_local_files and (not path.is_file() or _sha256(path) != expected):
            raise Phase1EvidenceError(f"model checkpoint is missing or stale: {relative}")
    elif kind == "external_content_addressed":
        _nonempty_text(checkpoint.get("provider"), "external checkpoint provider")
        _nonempty_text(checkpoint.get("manifest_sha256"), "external checkpoint manifest hash")
    else:
        raise Phase1EvidenceError("checkpoint kind must be local_file or external_content_addressed")
    tokenizer = document.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise Phase1EvidenceError(f"model {model_id} has no tokenizer contract")
    _nonempty_text(tokenizer.get("kind"), "tokenizer kind")
    _nonempty_text(tokenizer.get("sha256"), "tokenizer hash")
    config = document.get("configuration")
    if not isinstance(config, dict):
        raise Phase1EvidenceError(f"model {model_id} has no configuration identity")
    _nonempty_text(config.get("sha256"), "model configuration hash")
    _nonempty_text(document.get("runtime_id"), "model runtime binding")
    state = document.get("incremental_state")
    if not isinstance(state, dict):
        raise Phase1EvidenceError(f"model {model_id} has no incremental-state evidence")
    if state.get("status") == "MEASURED":
        _nonempty_text(state.get("mechanism"), "incremental-state mechanism")
        if not state.get("raw_trace_run_ids"):
            raise Phase1EvidenceError("incremental-state evidence must reference raw runs")
    elif state.get("status") == "NOT_AVAILABLE_IN_CURRENT_IMPLEMENTATION":
        _nonempty_text(state.get("reason"), "incremental-state limitation")
    elif state.get("status") == "IMPLEMENTED_NOT_BENCHMARKED":
        _nonempty_text(state.get("mechanism"), "incremental-state mechanism")
        _nonempty_text(state.get("reason"), "unbenchmarked-state reason")
    else:
        raise Phase1EvidenceError("incremental-state status is not typed")
    if "model_identity" in document and isinstance(document["model_identity"], bool):
        raise Phase1EvidenceError("bare model-identity booleans are forbidden")
    return model_id


def _validate_timing_row(row: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_RAW_KEYS - set(row))
    if missing:
        raise Phase1EvidenceError(f"raw timing row is missing fields: {missing}")
    for key in ("run_id", "system_id", "runtime_id", "model_id", "model_sha256", "tokenizer_sha256", "configuration_sha256", "precision"):
        _nonempty_text(row.get(key), f"raw {key}")
    _positive_int(row.get("trial"), "trial")
    if not isinstance(row.get("seed"), int):
        raise Phase1EvidenceError("raw seed must be an integer")
    device = row.get("device")
    if not isinstance(device, dict) or device.get("kind") not in {"cpu_one_thread", "cpu_all_core", "gpu"}:
        raise Phase1EvidenceError("raw device identity is invalid")
    _nonempty_text(device.get("hardware_id"), "raw hardware id")
    threads = row.get("threads")
    if not isinstance(threads, dict):
        raise Phase1EvidenceError("raw thread evidence is missing")
    requested = _positive_int(threads.get("requested"), "requested threads")
    observed = _positive_int(threads.get("observed_limit"), "observed thread limit")
    if requested != observed:
        raise Phase1EvidenceError("requested and observed thread limits differ")
    prompt = row.get("prompt")
    if not isinstance(prompt, dict):
        raise Phase1EvidenceError("raw prompt identity is missing")
    _nonempty_text(prompt.get("id"), "raw prompt id")
    _nonempty_text(prompt.get("sha256"), "raw prompt hash")
    _positive_int(prompt.get("bytes"), "raw prompt bytes")
    _nonempty_text(prompt.get("bucket"), "raw prompt bucket")
    output = row.get("output")
    if not isinstance(output, dict):
        raise Phase1EvidenceError("raw output evidence is missing")
    _positive_int(output.get("target_bytes"), "output target bytes")
    generated = _positive_int(output.get("generated_bytes"), "generated bytes")
    _positive_int(output.get("generated_tokens"), "generated tokens")
    _positive_int(output.get("generated_characters"), "generated characters")
    if row.get("status") == "PASS" and generated < output["target_bytes"]:
        raise Phase1EvidenceError("successful run did not reach its output-byte target")
    _nonempty_text(output.get("sha256"), "generated output hash")
    if row.get("evidence_protocol") == CORRECTION_PROTOCOL:
        encoded = _nonempty_text(output.get("hex"), "generated output bytes")
        try:
            output_bytes = bytes.fromhex(encoded)
        except ValueError as error:
            raise Phase1EvidenceError("generated output hex is invalid") from error
        if len(output_bytes) != generated:
            raise Phase1EvidenceError("generated byte count differs from the captured output")
        if hashlib.sha256(output_bytes).hexdigest() != output.get("sha256"):
            raise Phase1EvidenceError("generated output hash does not recompute")
        accounting = output.get("token_accounting")
        if not isinstance(accounting, dict):
            raise Phase1EvidenceError("corrected evidence requires structured token accounting")
        if accounting.get("method") not in {
            "runtime_final_eval_count", "posthoc_tokenizer", "raw_byte_vocabulary"
        }:
            raise Phase1EvidenceError("token accounting is not authoritative or post-hoc tokenized")
        if accounting.get("scope") != "completed_response":
            raise Phase1EvidenceError("token accounting must cover the completed response")
        counted = _positive_int(accounting.get("count"), "authoritative token count")
        if counted != output.get("generated_tokens"):
            raise Phase1EvidenceError("generated token count differs from token-accounting evidence")
        completed_bytes = _positive_int(
            accounting.get("completed_response_bytes"), "completed response bytes"
        )
        if completed_bytes < generated:
            raise Phase1EvidenceError("completed response is shorter than its benchmark prefix")
        _nonempty_text(accounting.get("completed_response_sha256"), "completed response hash")
    generation = row.get("generation")
    if not isinstance(generation, dict) or generation.get("mode") not in {"deterministic", "sampled"}:
        raise Phase1EvidenceError("raw generation mode is invalid")
    _finite_number(generation.get("temperature"), "generation temperature", minimum=0.0)
    if not isinstance(generation.get("seed"), int):
        raise Phase1EvidenceError("generation seed must be an integer")
    cache = row.get("cache_state")
    if not isinstance(cache, dict) or cache.get("kind") not in {"cold", "warm"}:
        raise Phase1EvidenceError("cache state requires a structured cold/warm record")
    _nonempty_text(cache.get("procedure"), "cache-state procedure")
    if row.get("evidence_protocol") == CORRECTION_PROTOCOL and cache.get("kind") == "cold":
        cold = cache.get("single_request_evidence")
        if not isinstance(cold, dict):
            raise Phase1EvidenceError("cold evidence must identify one measured streaming request")
        if cold.get("measured_streaming_requests") != 1 or cold.get("load_probe_requests") != 0:
            raise Phase1EvidenceError("cold evidence used an extra load probe or multiple requests")
        if cold.get("model_load_source") != "same_streaming_request":
            raise Phase1EvidenceError("cold model-load timing is not from the measured request")
    if "cold" in row:
        raise Phase1EvidenceError("bare cold/warm booleans are forbidden")
    order = row.get("order")
    if not isinstance(order, dict):
        raise Phase1EvidenceError("randomized order evidence is missing")
    if not isinstance(order.get("randomization_seed"), int):
        raise Phase1EvidenceError("randomization seed is missing")
    _positive_int(order.get("index"), "randomized order index")
    _nonempty_text(order.get("permutation_sha256"), "randomized order hash")
    timing = row.get("timing")
    if not isinstance(timing, dict):
        raise Phase1EvidenceError("direct timing evidence is missing")
    if timing.get("clock") != "perf_counter_ns":
        raise Phase1EvidenceError("timing must use perf_counter_ns")
    started = _positive_int(timing.get("request_started_ns"), "request start timestamp")
    first = _positive_int(timing.get("first_output_ns"), "first-output timestamp")
    completed = _positive_int(timing.get("target_completed_ns"), "completion timestamp")
    if not started <= first <= completed:
        raise Phase1EvidenceError("timing timestamps are not monotonic")
    ttfo = _finite_number(timing.get("time_to_first_output_seconds"), "TTFO", minimum=0.0)
    total = _finite_number(timing.get("total_latency_seconds"), "total latency", minimum=0.0)
    if not math.isclose(ttfo, (first - started) / 1e9, rel_tol=0.0, abs_tol=1e-6):
        raise Phase1EvidenceError("TTFO is not derived from raw timestamps")
    if not math.isclose(total, (completed - started) / 1e9, rel_tol=0.0, abs_tol=1e-6):
        raise Phase1EvidenceError("latency is not derived from raw timestamps")
    phases = timing.get("phase_timings")
    if not isinstance(phases, dict):
        raise Phase1EvidenceError("timing lacks structured load/prefill/decode phases")
    _finite_number(phases.get("model_load_seconds"), "model load time", minimum=0.0)
    if row.get("evidence_protocol") == CORRECTION_PROTOCOL and cache.get("kind") == "cold":
        if phases.get("model_load_source") != "same_streaming_request":
            raise Phase1EvidenceError("cold phase timing does not bind load to the same request")
    _nonempty_text(phases.get("measurement"), "phase timing measurement method")
    if "prefill_seconds" in phases:
        for name in ("prompt_preprocessing_seconds", "prefill_seconds", "decode_seconds"):
            _finite_number(phases.get(name), name, minimum=0.0)
    else:
        for name in ("prompt_preprocessing_seconds", "prefill_and_first_decode_seconds", "decode_after_first_seconds"):
            _finite_number(phases.get(name), name, minimum=0.0)
    memory = row.get("memory")
    if not isinstance(memory, dict):
        raise Phase1EvidenceError("memory evidence is missing")
    _nonempty_text(memory.get("method"), "memory measurement method")
    _finite_number(memory.get("resident_bytes"), "resident memory", minimum=0.0)
    _finite_number(memory.get("peak_resident_bytes"), "peak resident memory", minimum=0.0)
    accelerator = memory.get("accelerator_allocation")
    if not isinstance(accelerator, dict):
        raise Phase1EvidenceError("accelerator allocation status is missing")
    if accelerator.get("status") not in {
        "MEASURED", "NOT_APPLICABLE_CPU", "NOT_EXPOSED_BY_EXTERNAL_RUNTIME"
    }:
        raise Phase1EvidenceError("accelerator allocation status is invalid")
    _nonempty_text(accelerator.get("method"), "accelerator memory method")
    _finite_number(accelerator.get("peak_bytes"), "accelerator peak bytes", minimum=0.0)
    execution = row.get("execution")
    if not isinstance(execution, dict):
        raise Phase1EvidenceError("execution evidence is missing")
    _nonempty_text(execution.get("command_id"), "execution command id")
    if row.get("status") not in {"PASS", "FAIL"} or not isinstance(execution.get("exit_code"), int):
        raise Phase1EvidenceError("raw execution status is invalid")
    if row.get("status") == "PASS" and execution.get("exit_code") != 0:
        raise Phase1EvidenceError("successful raw execution has a nonzero exit code")


def validate_raw_timing_samples(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require_format(document, "raw")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise Phase1EvidenceError("raw timing evidence has no records")
    seen: set[str] = set()
    validated = []
    for row in records:
        if not isinstance(row, dict):
            raise Phase1EvidenceError("raw timing rows must be objects")
        _validate_timing_row(row)
        if row["run_id"] in seen:
            raise Phase1EvidenceError(f"duplicate raw run id: {row['run_id']}")
        seen.add(row["run_id"])
        validated.append(row)
    return validated


def validate_execution_commands(document: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    _require_format(document, "commands")
    commands = document.get("commands")
    if not isinstance(commands, list) or not commands:
        raise Phase1EvidenceError("execution command ledger is empty")
    by_id = {}
    for command in commands:
        if not isinstance(command, dict):
            raise Phase1EvidenceError("execution command rows must be objects")
        identifier = _nonempty_text(command.get("id"), "command id")
        if identifier in by_id:
            raise Phase1EvidenceError(f"duplicate command id: {identifier}")
        _nonempty_text(command.get("executable"), "command executable")
        arguments = command.get("arguments")
        if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
            raise Phase1EvidenceError("command arguments must be a string list")
        _nonempty_text(command.get("configuration_sha256"), "command configuration hash")
        by_id[identifier] = command
    missing = sorted({row["execution"]["command_id"] for row in rows} - set(by_id))
    if missing:
        raise Phase1EvidenceError(f"raw rows reference missing execution commands: {missing}")


def validate_seed_and_trial_counts(
    rows: Sequence[Mapping[str, Any]], *, minimum_trials_per_cell: int
) -> None:
    if minimum_trials_per_cell < 2:
        raise Phase1EvidenceError("repeated trials require a minimum of at least two")
    cells: dict[tuple[Any, ...], set[int]] = {}
    for row in rows:
        if row["status"] != "PASS":
            continue
        key = (
            row["system_id"],
            row["device"]["kind"],
            row["cache_state"]["kind"],
            row["generation"]["mode"],
            row["prompt"]["bucket"],
            row["output"]["target_bytes"],
        )
        cells.setdefault(key, set()).add(row["trial"])
    insufficient = {str(key): sorted(trials) for key, trials in cells.items() if len(trials) < minimum_trials_per_cell}
    if insufficient:
        raise Phase1EvidenceError(f"benchmark cells have insufficient repeated trials: {insufficient}")


def _expected_cells(system: Mapping[str, Any], matrix: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    devices = system.get("required_devices")
    if not isinstance(devices, list) or not devices:
        raise Phase1EvidenceError("matrix system has no required devices")
    axes = matrix.get("axes")
    if not isinstance(axes, dict):
        raise Phase1EvidenceError("benchmark matrix has no axes")
    return set(
        itertools.product(
            devices,
            axes["cache_states"],
            axes["generation_modes"],
            axes["prompt_buckets"],
            axes["output_target_bytes"],
        )
    )


def _headline_rows(
    rows: Sequence[Mapping[str, Any]], configuration: Mapping[str, Any], system_id: str,
    device: str,
) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if row["status"] == "PASS"
        and row["system_id"] == system_id
        and row["device"]["kind"] == device
        and row["cache_state"]["kind"] == configuration.get("cache_state")
        and row["generation"]["mode"] == configuration.get("generation_mode")
        and row["output"]["target_bytes"] == configuration.get("output_target_bytes")
        and row["prompt"]["bucket"] == "headline"
    ]


def validate_promoted_depth(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    minimum_prompts = _positive_int(
        protocol.get("minimum_distinct_prompts"), "headline distinct-prompt minimum"
    )
    minimum_repeats = _positive_int(
        protocol.get("minimum_repeated_prompt_observations"), "headline repeat minimum"
    )
    tail_minimum = _positive_int(
        protocol.get("tail_quantile_minimum_observations"), "tail-quantile minimum"
    )
    if minimum_prompts < 100 or minimum_repeats < 20 or tail_minimum < TAIL_QUANTILE_MINIMUM:
        raise Phase1EvidenceError("promoted benchmark depth is below the correction mandate")
    if protocol.get("pairing_key") != "prompt.id":
        raise Phase1EvidenceError("headline comparisons must be paired at prompt level")
    configurations = protocol.get("headline_configurations")
    if not isinstance(configurations, list) or len(configurations) != 2:
        raise Phase1EvidenceError("exactly two corrected headline comparisons are required")
    required_kinds = {"same_scale_architecture", "product"}
    observed_kinds = {item.get("comparison_kind") for item in configurations if isinstance(item, dict)}
    if observed_kinds != required_kinds:
        raise Phase1EvidenceError("same-scale and product comparisons must be certified separately")
    summaries: dict[str, Any] = {}
    for configuration in configurations:
        identifier = _nonempty_text(configuration.get("id"), "headline configuration id")
        left_id = _nonempty_text(configuration.get("layercake_system_id"), "headline LayerCake system")
        right_id = _nonempty_text(configuration.get("transformer_system_id"), "headline transformer system")
        left_rows = _headline_rows(
            rows, configuration, left_id,
            _nonempty_text(configuration.get("layercake_device"), "headline LayerCake device"),
        )
        right_rows = _headline_rows(
            rows, configuration, right_id,
            _nonempty_text(configuration.get("transformer_device"), "headline transformer device"),
        )
        system_summaries = {}
        prompt_rate_maps = []
        for system_id, system_rows in ((left_id, left_rows), (right_id, right_rows)):
            prompt_ids = {row["prompt"]["id"] for row in system_rows}
            by_prompt: dict[str, list[float]] = {}
            for row in system_rows:
                by_prompt.setdefault(row["prompt"]["id"], []).append(
                    row["output"]["generated_bytes"] / row["timing"]["total_latency_seconds"]
                )
            repeated = sum(1 for observations in by_prompt.values() if len(observations) >= 2)
            if len(prompt_ids) < minimum_prompts:
                raise Phase1EvidenceError(
                    f"headline {identifier}/{system_id} has only {len(prompt_ids)} distinct prompts"
                )
            if repeated < minimum_repeats:
                raise Phase1EvidenceError(
                    f"headline {identifier}/{system_id} has only {repeated} repeated prompt observations"
                )
            if len(system_rows) < tail_minimum:
                raise Phase1EvidenceError(f"headline {identifier}/{system_id} cannot promote tail latency")
            rate_map = {key: mean(values) for key, values in by_prompt.items()}
            prompt_rate_maps.append(rate_map)
            system_summaries[system_id] = {
                "observations": len(system_rows),
                "distinct_prompts": len(prompt_ids),
                "repeated_prompts": repeated,
            }
        if set(prompt_rate_maps[0]) != set(prompt_rate_maps[1]):
            raise Phase1EvidenceError(f"headline {identifier} is not paired on identical prompts")
        interval = paired_bootstrap_difference(
            prompt_rate_maps[0], prompt_rate_maps[1],
            confidence=float(protocol.get("bootstrap_confidence", 0.95)),
            resamples=_positive_int(protocol.get("bootstrap_resamples"), "headline bootstrap resamples"),
            seed=int(protocol.get("bootstrap_seed")),
        )
        summaries[identifier] = {
            "systems": system_summaries,
            "paired_prompt_count": len(prompt_rate_maps[0]),
            "bytes_per_second_paired_difference_ci": interval.__dict__,
        }
    return summaries


def validate_benchmark_matrix(
    document: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    _require_format(document, "matrix")
    hardware_summary = validate_hardware_manifest(hardware)
    axes = document.get("axes")
    required_axes = {
        "cache_states": ["cold", "warm"],
        "generation_modes": ["deterministic", "sampled"],
        "prompt_buckets": ["short", "medium", "long"],
        "output_target_bytes": [64, 256, 1024],
    }
    if not isinstance(axes, dict):
        raise Phase1EvidenceError("benchmark matrix axes are missing")
    for name, required in required_axes.items():
        if set(axes.get(name, [])) != set(required):
            raise Phase1EvidenceError(f"benchmark matrix axis {name} is incomplete")
    systems = document.get("systems")
    if not isinstance(systems, list) or not systems:
        raise Phase1EvidenceError("benchmark matrix has no systems")
    passed_rows = [row for row in rows if row["status"] == "PASS"]
    result: dict[str, Any] = {}
    for system in systems:
        identifier = _nonempty_text(system.get("id"), "matrix system id")
        required_devices = system.get("required_devices", [])
        if "gpu" in required_devices and not hardware_summary["gpu_available"]:
            if system.get("gpu_status") != "NOT_RUN_NO_HARDWARE":
                raise Phase1EvidenceError("missing GPU requires NOT_RUN_NO_HARDWARE")
            required_devices = [device for device in required_devices if device != "gpu"]
            system = {**system, "required_devices": required_devices}
        expected = _expected_cells(system, document)
        actual = {
            (
                row["device"]["kind"],
                row["cache_state"]["kind"],
                row["generation"]["mode"],
                row["prompt"]["bucket"],
                row["output"]["target_bytes"],
            )
            for row in passed_rows
            if row["system_id"] == identifier
        }
        missing = sorted(expected - actual, key=str)
        if missing:
            raise Phase1EvidenceError(f"benchmark matrix is incomplete for {identifier}: {missing[:10]}")
        result[identifier] = {"required_cells": len(expected), "observed_cells": len(actual)}
    minimum = document.get("minimum_trials_per_cell")
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise Phase1EvidenceError("matrix minimum trial count is not an integer")
    validate_seed_and_trial_counts(passed_rows, minimum_trials_per_cell=minimum)
    roles = {system.get("role") for system in systems}
    required_roles = {"optimized_transformer_baseline", "fastest_layercake_baseline"}
    if not required_roles.issubset(roles):
        raise Phase1EvidenceError("matrix lacks a required optimized-transformer or LayerCake role")
    if hardware_summary["gpu_available"]:
        gpu_roles = {
            system.get("role") for system in systems if "gpu" in system.get("required_devices", [])
        }
        if not {"optimized_transformer_gpu_baseline", "fastest_layercake_baseline"}.issubset(gpu_roles):
            raise Phase1EvidenceError("available GPU lacks matched transformer and LayerCake coverage")
    correction = document.get("correction_protocol")
    if correction is not None:
        if correction != CORRECTION_PROTOCOL:
            raise Phase1EvidenceError("unknown Phase 1 correction protocol")
        if any(row.get("evidence_protocol") != CORRECTION_PROTOCOL for row in passed_rows):
            raise Phase1EvidenceError("corrected Phase 1 contains uncorrected active raw rows")
        result["promoted_headlines"] = validate_promoted_depth(
            document.get("promoted_benchmark_protocol", {}), passed_rows
        )
    return result


def validate_quality_suite(
    document: Mapping[str, Any], root: Path, *, verify_dataset_files: bool = True
) -> dict[str, Any]:
    _require_format(document, "quality")
    metrics = document.get("metrics")
    if not isinstance(metrics, list):
        raise Phase1EvidenceError("quality suite metric definitions are missing")
    metric_ids = {row.get("id") for row in metrics if isinstance(row, dict)}
    missing = sorted(REQUIRED_QUALITY_METRICS - metric_ids)
    if missing:
        raise Phase1EvidenceError(f"quality suite metrics are incomplete: {missing}")
    for metric in metrics:
        _nonempty_text(metric.get("implementation"), "quality metric implementation")
        _nonempty_text(metric.get("direction"), "quality metric direction")
    prompts = document.get("prompts")
    if not isinstance(prompts, list) or len(prompts) < 3:
        raise Phase1EvidenceError("quality suite requires frozen prompts")
    if document.get("correction_protocol") is not None:
        if document.get("correction_protocol") != CORRECTION_PROTOCOL or len(prompts) < 100:
            raise Phase1EvidenceError("corrected quality suite requires at least 100 frozen prompts")
    prompt_ids = []
    prompt_hashes = []
    for prompt in prompts:
        prompt_ids.append(_nonempty_text(prompt.get("id"), "quality prompt id"))
        prompt_hashes.append(_nonempty_text(prompt.get("sha256"), "quality prompt hash"))
        _nonempty_text(prompt.get("category"), "quality prompt category")
    if len(prompt_ids) != len(set(prompt_ids)) or len(prompt_hashes) != len(set(prompt_hashes)):
        raise Phase1EvidenceError("quality suite contains duplicate prompts")
    data = document.get("datasets")
    if not isinstance(data, list) or not data:
        raise Phase1EvidenceError("quality suite has no immutable dataset manifest")
    for dataset in data:
        relative = _nonempty_text(dataset.get("path"), "quality dataset path")
        expected = _nonempty_text(dataset.get("sha256"), "quality dataset hash")
        path = (root / relative).resolve()
        if verify_dataset_files and (not path.is_file() or _sha256(path) != expected):
            raise Phase1EvidenceError(f"quality dataset is missing or stale: {relative}")
        if dataset.get("selection_access_allowed") not in {True, False}:
            raise Phase1EvidenceError("dataset selection-access policy is missing")
        if dataset.get("split") == "test" and dataset.get("selection_access_allowed") is not False:
            raise Phase1EvidenceError("test split may not influence selection")
    contamination = document.get("contamination_report")
    if not isinstance(contamination, dict):
        raise Phase1EvidenceError("quality suite has no contamination report")
    report_path = (root / _nonempty_text(contamination.get("path"), "contamination report path")).resolve()
    if not report_path.is_file() or _sha256(report_path) != contamination.get("sha256"):
        raise Phase1EvidenceError("contamination report is missing or stale")
    report = _document(report_path)
    if report.get("duplicate_prompt_ids") or report.get("cross_split_exact_overlaps"):
        raise Phase1EvidenceError("quality suite contamination or duplicate prompts detected")
    return {"prompt_count": len(prompts), "metric_count": len(metrics)}


def validate_threshold_lock(
    document: Mapping[str, Any], quality_manifest_path: Path
) -> dict[str, Any]:
    _require_format(document, "thresholds")
    if document.get("quality_suite_sha256") != _sha256(quality_manifest_path):
        raise Phase1EvidenceError("quality threshold lock is stale")
    methodology = document.get("statistical_methodology")
    if not isinstance(methodology, dict):
        raise Phase1EvidenceError("threshold lock has no statistical methodology")
    confidence = _finite_number(methodology.get("confidence"), "confidence")
    if not 0 < confidence < 1:
        raise Phase1EvidenceError("confidence must be in (0, 1)")
    _positive_int(methodology.get("bootstrap_seed"), "bootstrap seed")
    if _positive_int(methodology.get("resamples"), "bootstrap resamples") < 1000:
        raise Phase1EvidenceError("frozen bootstrap requires at least 1000 resamples")
    _nonempty_text(methodology.get("pairing_key"), "bootstrap pairing key")
    margins = document.get("non_inferiority_margins")
    if not isinstance(margins, dict) or not margins:
        raise Phase1EvidenceError("non-inferiority margins are not frozen")
    for name, value in margins.items():
        _finite_number(value, f"non-inferiority margin {name}", minimum=0.0)
    if document.get("locked_before_phase2") is not True:
        raise Phase1EvidenceError("quality thresholds were not locked before Phase 2")
    return {"confidence": confidence, "resamples": methodology["resamples"]}


def validate_baseline_optimization(
    runtime: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, runtime_id: str
) -> None:
    validate_runtime_manifest(runtime, optimized=True)
    runtime_rows = [row for row in rows if row["runtime_id"] == runtime_id and row["status"] == "PASS"]
    if not runtime_rows:
        raise Phase1EvidenceError(f"optimized runtime {runtime_id} has no successful raw runs")
    traces = runtime["optimization_evidence"]["kv_cache"]["raw_trace_run_ids"]
    row_ids = {row["run_id"] for row in runtime_rows}
    if not set(traces).issubset(row_ids):
        raise Phase1EvidenceError("KV-cache trace references do not exist in raw evidence")
    if {row["cache_state"]["kind"] for row in runtime_rows} != {"cold", "warm"}:
        raise Phase1EvidenceError("optimized runtime lacks measured cold/warm behavior")
    target = runtime.get("target_device")
    if target in {"cpu", "cpu_and_gpu"}:
        cpu_rows = [row for row in runtime_rows if row["device"]["kind"].startswith("cpu_")]
        if len({row["threads"]["requested"] for row in cpu_rows}) < 2:
            raise Phase1EvidenceError("optimized CPU runtime lacks measured thread scaling")
    if target in {"gpu", "cpu_and_gpu"} and not any(
        row["device"]["kind"] == "gpu" for row in runtime_rows
    ):
        raise Phase1EvidenceError("optimized GPU runtime lacks measured GPU runs")


def validate_evidence_manifest(document: Mapping[str, Any], root: Path) -> dict[str, str]:
    _require_format(document, "evidence")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise Phase1EvidenceError("Phase 1 evidence manifest is empty")
    result = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise Phase1EvidenceError("evidence manifest entries must be objects")
        relative = _nonempty_text(artifact.get("path"), "evidence path").replace("\\", "/")
        if relative in result:
            raise Phase1EvidenceError(f"duplicate evidence manifest path: {relative}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise Phase1EvidenceError(f"evidence path escapes repository: {relative}") from error
        expected = _nonempty_text(artifact.get("sha256"), "evidence hash")
        if not path.is_file() or _sha256(path) != expected:
            raise Phase1EvidenceError(f"evidence artifact is missing or stale: {relative}")
        result[relative] = expected
    raw_manifest_hash = document.get("raw_evidence_manifest_sha256")
    raw_rows = sorted(
        (path, digest) for path, digest in result.items()
        if "/raw_runs/" in path and "/history/" not in path
    )
    computed = hashlib.sha256(
        json.dumps(raw_rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if raw_manifest_hash != computed:
        raise Phase1EvidenceError("raw-evidence manifest hash does not recompute")
    return result


def derive_performance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] != "PASS":
            continue
        key = (
            row["system_id"],
            row["device"]["kind"],
            row["cache_state"]["kind"],
            row["generation"]["mode"],
            row["prompt"]["bucket"],
            row["output"]["target_bytes"],
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        latencies = [row["timing"]["total_latency_seconds"] for row in group]
        ttfo = [row["timing"]["time_to_first_output_seconds"] for row in group]
        throughputs = [row["output"]["generated_bytes"] / row["timing"]["total_latency_seconds"] for row in group]
        character_rates = [row["output"]["generated_characters"] / row["timing"]["total_latency_seconds"] for row in group]
        token_rates = [row["output"]["generated_tokens"] / row["timing"]["total_latency_seconds"] for row in group]
        tail_eligible = len(group) >= TAIL_QUANTILE_MINIMUM
        latency = {"mean": mean(latencies), "median": median(latencies), "p50": p50(latencies)}
        first_output = {"p50": p50(ttfo)}
        if tail_eligible:
            latency.update({"p95": p95(latencies), "p99": p99(latencies)})
            first_output.update({"p95": p95(ttfo), "p99": p99(ttfo)})
        summaries.append(
            {
                "system_id": key[0],
                "device": key[1],
                "cache_state": key[2],
                "generation_mode": key[3],
                "prompt_bucket": key[4],
                "output_target_bytes": key[5],
                "trials": len(group),
                "tail_quantiles": {
                    "status": "PROMOTED" if tail_eligible else "INSUFFICIENT_OBSERVATIONS",
                    "minimum_observations": TAIL_QUANTILE_MINIMUM,
                    "observations": len(group),
                },
                "latency_seconds": latency,
                "time_to_first_output_seconds": first_output,
                "bytes_per_second": {"mean": mean(throughputs), "p50": p50(throughputs)},
                "characters_per_second": {"mean": mean(character_rates), "p50": p50(character_rates)},
                "tokens_per_second_secondary_non_cross_model": {"mean": mean(token_rates), "p50": p50(token_rates)},
                "resident_memory_bytes": {"p50": p50([row["memory"]["resident_bytes"] for row in group]), "peak": max(row["memory"]["peak_resident_bytes"] for row in group)},
            }
        )
    return {
        "format": "layercake-phase1-derived-performance/1",
        "cross_model_primary_throughput_metrics": ["bytes_per_second", "characters_per_second"],
        "token_throughput_status": "SECONDARY_TOKENIZER_SPECIFIC",
        "summaries": summaries,
    }


def validate_comparison_certificate(
    document: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
    baseline_quality: Mapping[str, Any],
    functional_quality: Mapping[str, Any],
) -> dict[str, Any]:
    if document.get("format") != "layercake-phase1-comparison-certificate/1":
        raise Phase1EvidenceError("corrected comparison certificate format is invalid")
    if document.get("correction_protocol") != CORRECTION_PROTOCOL:
        raise Phase1EvidenceError("comparison certificate is not bound to the correction protocol")
    comparisons = document.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 2:
        raise Phase1EvidenceError("comparison certificate requires exactly two comparisons")
    by_kind = {item.get("kind"): item for item in comparisons if isinstance(item, dict)}
    if set(by_kind) != {"same_scale_architecture", "product"}:
        raise Phase1EvidenceError("same-scale and product certifications are not separate")
    baseline_records = {
        row.get("model_id"): row for row in baseline_quality.get("records", [])
        if isinstance(row, dict)
    }
    functional_records = functional_quality.get("systems")
    if not isinstance(functional_records, dict):
        raise Phase1EvidenceError("functional quality evidence is missing system records")
    summary = {}
    for kind, comparison in by_kind.items():
        layercake_model = _nonempty_text(comparison.get("layercake_model_id"), "comparison LayerCake model")
        transformer_model = _nonempty_text(comparison.get("transformer_model_id"), "comparison transformer model")
        if layercake_model not in models or transformer_model not in models:
            raise Phase1EvidenceError("comparison references an unknown model")
        lc_system = _nonempty_text(comparison.get("layercake_speed_system_id"), "comparison LayerCake speed system")
        tf_system = _nonempty_text(comparison.get("transformer_speed_system_id"), "comparison transformer speed system")
        for system, expected_model in ((lc_system, layercake_model), (tf_system, transformer_model)):
            system_rows = [row for row in rows if row["system_id"] == system and row["status"] == "PASS"]
            if not system_rows or {row["model_id"] for row in system_rows} != {expected_model}:
                raise Phase1EvidenceError(
                    f"comparison {kind} mixes speed from a different model lineage: {system}"
                )
        if kind == "same_scale_architecture":
            if comparison.get("quality_protocol") != "heldout_bpb_same_checkpoint":
                raise Phase1EvidenceError("same-scale comparison must use checkpoint-bound heldout BPB")
            if layercake_model not in baseline_records or transformer_model not in baseline_records:
                raise Phase1EvidenceError("same-scale speed models lack checkpoint-bound quality records")
            lc_parameters = int(models[layercake_model]["parameters"]["total"])
            tf_parameters = int(models[transformer_model]["parameters"]["total"])
            ratio = max(lc_parameters, tf_parameters) / min(lc_parameters, tf_parameters)
            if ratio > 1.10:
                raise Phase1EvidenceError("same-scale comparison models differ by more than 10% parameters")
        else:
            if comparison.get("quality_protocol") != "functional_output_suite":
                raise Phase1EvidenceError("product comparison must use the functional output suite")
            if set(functional_records) != {layercake_model, transformer_model}:
                raise Phase1EvidenceError("product functional quality and speed use different models")
            prompt_sets = {
                model_id: set(record.get("prompt_ids", []))
                for model_id, record in functional_records.items()
                if isinstance(record, dict)
            }
            if len(prompt_sets) != 2 or len(next(iter(prompt_sets.values()))) < 100:
                raise Phase1EvidenceError("product functional quality suite has fewer than 100 prompts")
            if len({frozenset(value) for value in prompt_sets.values()}) != 1:
                raise Phase1EvidenceError("product functional quality is not paired on identical prompts")
        summary[kind] = {
            "layercake_model_id": layercake_model,
            "transformer_model_id": transformer_model,
            "layercake_checkpoint_sha256": models[layercake_model]["checkpoint"]["sha256"],
            "transformer_checkpoint_sha256": models[transformer_model]["checkpoint"]["sha256"],
        }
    return summary


def validate_phase1_bundle(
    root: Path, phase_dir: Path, *, verify_external_files: bool = True
) -> dict[str, Any]:
    hardware_path = phase_dir / "hardware.json"
    hardware = _document(hardware_path)
    hardware_summary = validate_hardware_manifest(hardware)
    runtime_paths = sorted((phase_dir / "runtime_manifests").glob("*.json"))
    model_paths = sorted((phase_dir / "model_manifests").glob("*.json"))
    if not runtime_paths or not model_paths:
        raise Phase1EvidenceError("runtime or model manifests are missing")
    runtime_documents = [_document(path) for path in runtime_paths]
    model_documents = [_document(path) for path in model_paths]
    runtimes = {document["id"]: document for document in runtime_documents}
    models = {document["id"]: document for document in model_documents}
    if len(runtimes) != len(runtime_documents) or len(models) != len(model_documents):
        raise Phase1EvidenceError("duplicate runtime or model manifest id")
    for runtime in runtimes.values():
        validate_runtime_manifest(runtime)
    for model in models.values():
        validate_model_manifest(model, root, verify_local_files=verify_external_files)
        if model["runtime_id"] not in runtimes:
            raise Phase1EvidenceError(f"model {model['id']} references an unknown runtime")
    raw_paths = sorted((phase_dir / "raw_runs").glob("*.json"))
    if not raw_paths:
        raise Phase1EvidenceError("Phase 1 contains no raw timing files")
    rows = [row for path in raw_paths for row in validate_raw_timing_samples(_document(path))]
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise Phase1EvidenceError("raw run ids are duplicated across files")
    for row in rows:
        if row["runtime_id"] not in runtimes or row["model_id"] not in models:
            raise Phase1EvidenceError("raw row references an unknown runtime or model")
        model = models[row["model_id"]]
        if row["runtime_id"] != model["runtime_id"]:
            raise Phase1EvidenceError("raw runtime/model binding differs from the manifest")
        if row["model_sha256"] != model["checkpoint"]["sha256"]:
            raise Phase1EvidenceError("raw model hash differs from the model manifest")
        if row["tokenizer_sha256"] != model["tokenizer"]["sha256"]:
            raise Phase1EvidenceError("raw tokenizer hash differs from the model manifest")
        if row["configuration_sha256"] != model["configuration"]["sha256"]:
            raise Phase1EvidenceError("raw configuration hash differs from the model manifest")
    matrix = _document(phase_dir / "benchmark_matrix.json")
    matrix_summary = validate_benchmark_matrix(matrix, rows, hardware)
    commands = _document(phase_dir / "execution_commands.json")
    validate_execution_commands(commands, rows)
    quality_path = phase_dir / "quality_suite_manifest.json"
    quality_summary = validate_quality_suite(
        _document(quality_path), root, verify_dataset_files=verify_external_files
    )
    threshold_summary = validate_threshold_lock(
        _document(phase_dir / "quality_threshold_lock.json"), quality_path
    )
    comparison_summary = None
    if matrix.get("correction_protocol") == CORRECTION_PROTOCOL:
        comparison_summary = validate_comparison_certificate(
            _document(phase_dir / "comparison_certificate.json"), rows, models,
            _document(phase_dir / "baseline_quality.json"),
            _document(phase_dir / "functional_quality.json"),
        )
    tests = _document(phase_dir / "test_results.json")
    if tests.get("format") != "layercake-phase1-test-results/1":
        raise Phase1EvidenceError("Phase 1 test-result format is invalid")
    if tests.get("status") != "PASS" or tests.get("failures") != 0 or tests.get("errors") != 0:
        raise Phase1EvidenceError("Phase 1 complete regression suite is not green")
    junit_path = (root / _nonempty_text(tests.get("junit_path"), "Phase 1 JUnit path")).resolve()
    if not junit_path.is_file() or _sha256(junit_path) != tests.get("junit_sha256"):
        raise Phase1EvidenceError("Phase 1 JUnit evidence is missing or stale")
    for runtime_id in matrix.get("optimized_runtime_ids", []):
        if runtime_id not in runtimes:
            raise Phase1EvidenceError(f"unknown optimized runtime: {runtime_id}")
        validate_baseline_optimization(runtimes[runtime_id], rows, runtime_id=runtime_id)
    evidence = validate_evidence_manifest(_document(phase_dir / "evidence_manifest.json"), root)
    lifecycle_names = {
        "candidate.json", "candidate_verification.json", "release_certificate.json",
        "handoff.json", "seal.json", "evidence_manifest.json",
    }
    discovered = {
        path.relative_to(root).as_posix()
        for path in phase_dir.rglob("*")
        if path.is_file()
        and not (path.parent == phase_dir and path.name in lifecycle_names)
    }
    if set(evidence) != discovered:
        missing = sorted(discovered - set(evidence))
        extra = sorted(set(evidence) - discovered)
        raise Phase1EvidenceError(
            f"evidence manifest does not exactly cover Phase 1 artifacts; missing={missing}, extra={extra}"
        )
    return {
        "hardware": hardware_summary,
        "runtime_ids": sorted(runtimes),
        "model_ids": sorted(models),
        "raw_run_count": len(rows),
        "matrix": matrix_summary,
        "quality": quality_summary,
        "thresholds": threshold_summary,
        "comparisons": comparison_summary,
        "tests": {key: tests[key] for key in ("tests", "passed", "failures", "errors", "skipped")},
        "evidence_artifact_count": len(evidence),
        "derived_performance": derive_performance(rows),
    }
