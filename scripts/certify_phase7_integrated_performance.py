"""Freeze and certify the Phase 7 integrated CPU/GPU performance matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import tempfile
import time
from typing import Any, Mapping
import urllib.request

import psutil
import torch

import _common
from layercake.evaluation.phase2_r3_evidence import validate_phase2_r3_bundle
from layercake.evaluation.phase6_evidence import validate_phase6_bundle
from layercake.moonshot_campaign import component_hashes, governed_source_hash
from layercake.routing import DirectCakeOrchestrator, load_archive_bound_profiles
from layercake.training.generic_domain import evaluate_generated, load_dataset
from layercake.training.phase4_python_cake import (
    _execute_tests,
    _extract_function,
    _load_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "moonshot/phase7_integrated_performance_preregistration.json"
CONTRACT_SHA256 = "a70df7bbb62ed4869bcc9b9c62ed84635c80461b5837db40f93a3aa16588afa1"
PROFILES = ROOT / "moonshot/phase6_router_profiles.json"
PROFILES_SHA256 = "5397a0f28d145c15ee2aeaf13c021aaabb969b4226d16a2f66f776d2b48c3ec2"
PHASE6_CPU = ROOT / "results/moonshot/phase6/raw_runs/mixed_cpu_benchmark.json"
PHASE6_CPU_SHA256 = "5aac50d5259c4e8d43fddf57d9e8771ce890682b84c992d636c2f73b9efbb303"
PHASE6_FUNCTIONAL = ROOT / "results/moonshot/phase6/raw_runs/functional_execution.json"
PHASE2_RESULTS = ROOT / "results/moonshot/phase2_recertification"
PHASE6_RESULTS = ROOT / "results/moonshot/phase6"
DIRECT_ABI_VERSION = "lc-direct-neural-decoder/1"
DIRECT_ABI_SHA256 = "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
QWEN_MODEL = "qwen2.5:0.5b"
QWEN_DIGEST = "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67"
SEEDS = (10701, 10702, 10703)
PACKAGES = {
    "python": ROOT
    / "artifacts/moonshot/phase4/release/"
    "python-token-plan-seed10141-direct-v1.0.0.cake",
    "sql": ROOT / "artifacts/moonshot/phase5/release/sql-token-plan-v1.0.0.cake",
    "regex": ROOT / "artifacts/moonshot/phase5/release/regex-token-plan-v1.0.0.cake",
}
PACKAGE_HASHES = {
    "python": "0585c79bfbea16b1c4165bf0030ba6985b8a8cdeab529cd7afe3f9c76c564ef7",
    "sql": "24efdb68885581318bee3a0b7c3cac0b0fe0b75eefa90070c601384a8bf5e105",
    "regex": "c336a552415b2d6161eb6696338086f89c3c3b6e25949e6cf0a94373e9690f15",
}
CAKE_IDS = {
    "python": "python-token-plan",
    "sql": "sql-token-plan",
    "regex": "regex-token-plan",
}
DATASETS = {
    "python": ROOT / "data/moonshot/phase4/python_functional_v1.jsonl",
    "sql": ROOT / "data/moonshot/phase5/sql_v1.jsonl",
    "regex": ROOT / "data/moonshot/phase5/regex_v1.jsonl",
}
PUBLIC_KEYS = {
    "4d64fb4eb20e06035d287ced76b54be9": (
        ROOT / "moonshot/phase4-direct-token-plan-publisher.public.pem"
    ),
    "0047fe0d71576d696b37ee1eb29a56db": (
        ROOT / "moonshot/phase5-multidomain-publisher.public.pem"
    ),
}
RESULTS = ROOT / "results/moonshot/phase7"
RAW = RESULTS / "raw_runs"
FRAMEWORK = RESULTS / "framework_freeze.json"
SOURCE_AUDIT = RESULTS / "source_audit.json"
COLD = RAW / "cold_start.json"
CPU_SUPPLEMENT = RAW / "cpu_layercake_timing_supplement.json"
GPU_PERFORMANCE = RAW / "gpu_performance.json"
GPU_RETENTION = RAW / "gpu_domain_retention.json"
GENERAL_QUALITY = RAW / "general_quality_retention.json"
GATES = RAW / "gate_observations.json"
PAYLOAD = RESULTS / "certificate_payload.json"
CERTIFICATE = RESULTS / "integrated_performance_certificate.json"
OLLAMA_GENERATE = "http://localhost:11434/api/generate"
OLLAMA_TAGS = "http://localhost:11434/api/tags"
OLLAMA_PS = "http://localhost:11434/api/ps"
IMPLEMENTATION_FILES = (
    Path("layercake/evaluation/phase7_evidence.py"),
    Path("scripts/certify_phase7_integrated_performance.py"),
    Path("tests/evaluation/test_phase7_evidence.py"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "evidence_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"Phase 7 evidence is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    value["evidence_sha256"] = _canonical_sha(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _implementation_hashes() -> dict[str, str]:
    return {
        path.as_posix(): _sha256(ROOT / path)
        for path in IMPLEMENTATION_FILES
    }


def _trust_store() -> dict[str, Path]:
    return dict(PUBLIC_KEYS)


def _new_orchestrator(registry: Path, device: str) -> DirectCakeOrchestrator:
    orchestrator = DirectCakeOrchestrator(
        registry,
        abi_version=DIRECT_ABI_VERSION,
        abi_hash=DIRECT_ABI_SHA256,
        trust_store=_trust_store(),
        profiles=load_archive_bound_profiles(PROFILES),
        device=device,
    )
    for package in PACKAGES.values():
        orchestrator.install(package)
    return orchestrator


def _functional_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "python": [
            row
            for row in _load_rows(DATASETS["python"])
            if row["split"] == "test"
        ],
        "sql": [
            row
            for row in load_dataset(DATASETS["sql"])
            if row["split"] == "test"
        ],
        "regex": [
            row
            for row in load_dataset(DATASETS["regex"])
            if row["split"] == "test"
        ],
    }


def _functional_result(
    domain: str, output: bytes, row: Mapping[str, Any]
) -> tuple[bool, int]:
    if domain != "python":
        passed, checks = evaluate_generated(output, row)
        return passed, len(checks)
    text = output.decode("utf-8", errors="strict")
    source, _ = _extract_function(text, str(row["function_name"]))
    if source is None:
        return False, 1
    passed, checks = _execute_tests(
        source, str(row["function_name"]), row["tests"]
    )
    return passed, len(checks)


def _module_bytes(module: torch.nn.Module) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in list(module.parameters()) + list(module.buffers())
    )


def _timed_layercake(
    orchestrator: DirectCakeOrchestrator,
    prompt: str,
    *,
    domain: str,
    cold: bool = False,
) -> dict[str, Any]:
    if torch.cuda.is_available() and orchestrator.host.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter_ns()
    route = orchestrator.plan(prompt, mode="automatic_top1")
    route_done = time.perf_counter_ns()
    if route.selected != (CAKE_IDS[domain],):
        raise RuntimeError(f"LayerCake route mismatch for {domain}: {route.selected}")
    load_started = time.perf_counter_ns()
    model = orchestrator.host._load_selected(route.selected[0])
    if orchestrator.host.device.type == "cuda":
        torch.cuda.synchronize()
    loaded = time.perf_counter_ns()
    state = model.prefill_bytes(prompt if prompt.endswith("\n") else prompt + "\n")
    if orchestrator.host.device.type == "cuda":
        torch.cuda.synchronize()
    prefilled = time.perf_counter_ns()
    decode_calls = 0
    if not state.complete:
        model.decode_step(state)
        decode_calls += 1
    if orchestrator.host.device.type == "cuda":
        torch.cuda.synchronize()
    first = time.perf_counter_ns()
    while not state.complete:
        model.decode_step(state)
        decode_calls += 1
    if orchestrator.host.device.type == "cuda":
        torch.cuda.synchronize()
    completed = time.perf_counter_ns()
    output = model.tokenizer.decode_actions(
        state.generated_actions, state.source_lexemes
    )
    seconds = (completed - started) / 1e9
    passed, check_count = _functional_result(domain, output, _row_by_prompt(domain, prompt))
    return {
        "output_hex": output.hex(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "generated_bytes": len(output),
        "generated_characters": len(output.decode("utf-8")),
        "generated_actions": len(state.generated_actions),
        "functional_success": passed,
        "functional_check_count": check_count,
        "selected": list(route.selected),
        "prefill_calls": 1,
        "decode_step_calls": decode_calls,
        "inactive_cake_forward_calls": 0,
        "active_tensor_bytes": _module_bytes(model),
        "cold": cold,
        "timing": {
            "route_seconds": (route_done - started) / 1e9,
            "model_load_seconds": (loaded - load_started) / 1e9,
            "prefill_seconds": (prefilled - loaded) / 1e9,
            "time_to_first_output_seconds": (first - started) / 1e9,
            "total_latency_seconds": seconds,
            "bytes_per_second_total": len(output) / max(seconds, 1e-12),
            "characters_per_second_total": (
                len(output.decode("utf-8")) / max(seconds, 1e-12)
            ),
        },
    }


_ROWS_CACHE: dict[str, list[dict[str, Any]]] | None = None


def _row_by_prompt(domain: str, prompt: str) -> dict[str, Any]:
    global _ROWS_CACHE
    if _ROWS_CACHE is None:
        _ROWS_CACHE = _functional_rows()
    for row in _ROWS_CACHE[domain]:
        if row["prompt"] == prompt:
            return row
    raise KeyError(f"functional prompt is not frozen for {domain}")


def _ollama_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("Ollama response is not an object")
    return value


def _qwen_request(
    prompt: str,
    *,
    num_gpu: int,
    seed: int,
    cold: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": QWEN_MODEL,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_predict": 128,
            "num_thread": 14,
            "num_gpu": num_gpu,
        },
    }
    request = urllib.request.Request(
        OLLAMA_GENERATE,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter_ns()
    first: int | None = None
    pieces: list[str] = []
    final: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line)
            piece = str(event.get("response", ""))
            if piece:
                pieces.append(piece)
                if first is None:
                    first = time.perf_counter_ns()
            if event.get("done"):
                final = event
    completed = time.perf_counter_ns()
    text = "".join(pieces)
    raw = text.encode("utf-8")
    total = (completed - started) / 1e9
    return {
        "output_hex": raw.hex(),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "generated_bytes": len(raw),
        "generated_characters": len(text),
        "authoritative_generated_tokens": int(final.get("eval_count", 0)),
        "done_reason": final.get("done_reason"),
        "cold": cold,
        "timing": {
            "load_seconds": int(final.get("load_duration", 0)) / 1e9,
            "prompt_eval_seconds": int(final.get("prompt_eval_duration", 0)) / 1e9,
            "eval_seconds": int(final.get("eval_duration", 0)) / 1e9,
            "time_to_first_output_seconds": ((first or completed) - started) / 1e9,
            "total_latency_seconds": total,
            "bytes_per_second_total": len(raw) / max(total, 1e-12),
            "characters_per_second_total": len(text) / max(total, 1e-12),
            "tokens_per_second_total": int(final.get("eval_count", 0))
            / max(total, 1e-12),
        },
    }


def _unload_qwen(num_gpu: int) -> dict[str, Any]:
    payload = {
        "model": QWEN_MODEL,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
        "options": {"num_gpu": num_gpu, "num_thread": 14},
    }
    request = urllib.request.Request(
        OLLAMA_GENERATE,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        value = json.load(response)
    return {
        "done": bool(value.get("done")),
        "done_reason": value.get("done_reason"),
        "load_duration_seconds": int(value.get("load_duration", 0)) / 1e9,
    }


def _bootstrap_mean(
    values: list[float], *, seed: int, samples: int = 5000
) -> list[float]:
    generator = random.Random(seed)
    estimates = [
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    estimates.sort()
    return [
        estimates[math.floor(0.025 * (samples - 1))],
        estimates[math.ceil(0.975 * (samples - 1))],
    ]


def _nearest_rank(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _system_descriptives(
    records: list[Mapping[str, Any]], system: str
) -> dict[str, Any]:
    observations = [row[system] for row in records]
    timing = [row["timing"] for row in observations]

    def quantiles(field: str) -> dict[str, float]:
        values = [float(row[field]) for row in timing]
        return {
            "p50": statistics.median(values),
            "p95_nearest_rank": _nearest_rank(values, 0.95),
            "p99_nearest_rank": _nearest_rank(values, 0.99),
        }

    result: dict[str, Any] = {
        "observations": len(observations),
        "total_latency_seconds": quantiles("total_latency_seconds"),
        "time_to_first_output_seconds": quantiles(
            "time_to_first_output_seconds"
        ),
        "bytes_per_second_total": quantiles("bytes_per_second_total"),
        "characters_per_second_total": quantiles(
            "characters_per_second_total"
        ),
        "correct_task_completions_per_wall_second": (
            sum(bool(row["functional_success"]) for row in observations)
            / sum(float(row["timing"]["total_latency_seconds"]) for row in observations)
        ),
    }
    if all("authoritative_generated_tokens" in row for row in observations):
        token_rates = [
            float(row["authoritative_generated_tokens"])
            / float(row["timing"]["total_latency_seconds"])
            for row in observations
        ]
        result["authoritative_tokens_per_second_total"] = {
            "p50": statistics.median(token_rates),
            "p95_nearest_rank": _nearest_rank(token_rates, 0.95),
            "p99_nearest_rank": _nearest_rank(token_rates, 0.99),
        }
    return result


def _hardware() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    nvidia = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "system_ram_bytes": psutil.virtual_memory().total,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_compute_capability": [properties.major, properties.minor],
        "nvidia_smi": nvidia.stdout.strip(),
        "energy": "NOT_MEASURED_NO_CALIBRATED_ENERGY_METER",
    }


def freeze_framework() -> dict[str, Any]:
    if _git("status", "--porcelain=v1"):
        raise RuntimeError("Phase 7 framework freeze requires a clean committed worktree")
    if _sha256(CONTRACT) != CONTRACT_SHA256:
        raise RuntimeError("Phase 7 preregistration changed")
    if _sha256(PROFILES) != PROFILES_SHA256:
        raise RuntimeError("sealed router profiles changed")
    if _sha256(PHASE6_CPU) != PHASE6_CPU_SHA256:
        raise RuntimeError("sealed Phase 6 CPU evidence changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 7 requires the declared CUDA device")
    commit = _git("rev-parse", "HEAD")
    matrix = _read(ROOT / "moonshot/invalidation_matrix.yaml")
    document = {
        "format": "layercake-phase7-framework-freeze/1",
        "status": "FROZEN",
        "framework_commit": commit,
        "framework_tree": _git("show", "-s", "--format=%T", commit),
        "contract_sha256": CONTRACT_SHA256,
        "profiles_sha256": PROFILES_SHA256,
        "phase6_cpu_sha256": PHASE6_CPU_SHA256,
        "governed_source_sha256": governed_source_hash(ROOT),
        "implementation_hashes": _implementation_hashes(),
        "component_hashes": component_hashes(ROOT, matrix),
        "hardware": _hardware(),
        "raw_evidence_present_at_freeze": RAW.exists(),
    }
    if document["raw_evidence_present_at_freeze"]:
        raise RuntimeError("Phase 7 raw evidence predates the framework freeze")
    _write(FRAMEWORK, document)
    return document


def _source_audit() -> dict[str, Any]:
    framework = _read(FRAMEWORK)
    current = governed_source_hash(ROOT)
    current_impl = _implementation_hashes()
    passed = (
        current == framework["governed_source_sha256"]
        and current_impl == framework["implementation_hashes"]
    )
    document = {
        "format": "layercake-phase7-source-audit/1",
        "status": "PASS" if passed else "FAIL",
        "framework_commit": framework["framework_commit"],
        "governed_source_sha256": current,
        "implementation_hashes": current_impl,
        "source_changes_after_freeze": not passed,
    }
    _write(SOURCE_AUDIT, document)
    if not passed:
        raise RuntimeError("Phase 7 source changed after framework freeze")
    return document


def _cold_evidence(first_domain: str, first_row: Mapping[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for device in ("cpu", "cuda"):
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with tempfile.TemporaryDirectory(prefix=f"layercake-phase7-cold-{device}-") as temporary:
            install_started = time.perf_counter_ns()
            orchestrator = _new_orchestrator(Path(temporary) / "registry", device)
            if device == "cuda":
                torch.cuda.synchronize()
            install_seconds = (time.perf_counter_ns() - install_started) / 1e9
            result = _timed_layercake(
                orchestrator,
                str(first_row["prompt"]),
                domain=first_domain,
                cold=True,
            )
            records.append(
                {
                    "system": (
                        "layercake_gpu" if device == "cuda" else "layercake_cpu"
                    ),
                    "device": device,
                    "install_and_verify_seconds": install_seconds,
                    "single_real_request": result,
                }
            )
    for device, num_gpu in (("cpu", 0), ("cuda", 99)):
        unload = _unload_qwen(num_gpu)
        result = _qwen_request(
            str(first_row["prompt"]),
            num_gpu=num_gpu,
            seed=SEEDS[0],
            cold=True,
        )
        report = _ollama_json(OLLAMA_PS)
        records.append(
            {
                "system": (
                    "transformer_gpu"
                    if device == "cuda"
                    else "transformer_cpu"
                ),
                "device": device,
                "unload_control": unload,
                "single_real_streaming_request": result,
                "model_report_after": report,
            }
        )
    document = {
        "format": "layercake-phase7-cold-start/1",
        "status": "RAW",
        "protocol": {
            "single_real_request_per_system": True,
            "transformer_load_probe_request": False,
            "transformer_unload_control_before_request": True,
        },
        "records": records,
    }
    _write(COLD, document)
    return document


def _performance_prompt_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cpu = _read(PHASE6_CPU)
    datasets = _functional_rows()
    by_id = {
        row["id"]: {"domain": domain, "row": row}
        for domain, values in datasets.items()
        for row in values
    }
    rows = []
    for source in cpu["records"]:
        item = by_id[source["prompt_id"]]
        row = item["row"]
        if hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest() != source[
            "prompt_sha256"
        ]:
            raise RuntimeError("Phase 6 CPU prompt identity changed")
        rows.append(
            {
                "prompt_id": source["prompt_id"],
                "trial": source["trial"],
                "prompt_sha256": source["prompt_sha256"],
                "domain": item["domain"],
                "row": row,
            }
        )
    return rows, by_id


def _cpu_layercake_supplement() -> dict[str, Any]:
    prompt_rows, _ = _performance_prompt_rows()
    records: list[dict[str, Any]] = []
    rss_samples: list[int] = []
    loaded_tensor_bytes: list[int] = []
    torch.set_num_threads(14)
    for seed_index, seed in enumerate(SEEDS):
        seed_rows = prompt_rows[seed_index * 40 : (seed_index + 1) * 40]
        with tempfile.TemporaryDirectory(
            prefix=f"layercake-phase7-cpu-seed{seed}-"
        ) as temporary:
            orchestrator = _new_orchestrator(
                Path(temporary) / "registry", "cpu"
            )
            _timed_layercake(
                orchestrator,
                seed_rows[0]["row"]["prompt"],
                domain=seed_rows[0]["domain"],
            )
            for item in seed_rows:
                result = _timed_layercake(
                    orchestrator,
                    item["row"]["prompt"],
                    domain=item["domain"],
                )
                records.append(
                    {
                        "seed": seed,
                        "prompt_id": item["prompt_id"],
                        "prompt_sha256": item["prompt_sha256"],
                        "domain": item["domain"],
                        "trial": item["trial"],
                        "layercake_cpu": result,
                    }
                )
            loaded_tensor_bytes.append(
                sum(
                    _module_bytes(model)
                    for model in orchestrator.host._models.values()
                )
            )
            rss_samples.append(psutil.Process().memory_info().rss)
    document = {
        "format": "layercake-phase7-cpu-timing-supplement/1",
        "status": "RAW",
        "purpose": (
            "Measure warm LayerCake CPU TTFT and descriptive quantiles omitted "
            "from the sealed Phase 6 raw record; Phase 7 CPU promotion gates "
            "remain derived from the immutable Phase 6 paired evidence."
        ),
        "protocol": {
            "distinct_prompts": 100,
            "repeated_prompt_observations": 20,
            "observations": 120,
            "seeds": list(SEEDS),
            "observations_per_seed": 40,
            "threads": 14,
            "warmup_per_seed_excluded": 1,
        },
        "memory": {
            "loaded_tensor_bytes_by_seed": loaded_tensor_bytes,
            "process_rss_samples_bytes": rss_samples,
        },
        "aggregates": {
            "layercake_cpu": _system_descriptives(
                records, "layercake_cpu"
            )
        },
        "records": records,
    }
    _write(CPU_SUPPLEMENT, document)
    return document


def _gpu_performance() -> dict[str, Any]:
    tags = _ollama_json(OLLAMA_TAGS)
    models = {
        str(row.get("name") or row.get("model")): row
        for row in tags.get("models", [])
    }
    if models.get(QWEN_MODEL, {}).get("digest") != QWEN_DIGEST:
        raise RuntimeError("locked Qwen model digest is unavailable")
    prompt_rows, _ = _performance_prompt_rows()
    # The cold GPU request left the baseline resident. One additional warmup is
    # explicitly excluded from promoted observations.
    qwen_warmup = _qwen_request(
        prompt_rows[0]["row"]["prompt"],
        num_gpu=99,
        seed=SEEDS[0],
    )
    warm_report = _ollama_json(OLLAMA_PS)
    active = [
        row
        for row in warm_report.get("models", [])
        if row.get("digest") == QWEN_DIGEST
    ]
    if not active or int(active[0].get("size_vram", 0)) <= 0:
        raise RuntimeError("optimized Qwen baseline is not physically GPU resident")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    loaded_tensor_bytes: list[int] = []
    for seed_index, seed in enumerate(SEEDS):
        seed_rows = prompt_rows[seed_index * 40 : (seed_index + 1) * 40]
        with tempfile.TemporaryDirectory(
            prefix=f"layercake-phase7-gpu-seed{seed}-"
        ) as temporary:
            orchestrator = _new_orchestrator(
                Path(temporary) / "registry", "cuda"
            )
            # Warm the same lineage before promoted timing.
            _timed_layercake(
                orchestrator,
                seed_rows[0]["row"]["prompt"],
                domain=seed_rows[0]["domain"],
            )
            for local_index, item in enumerate(seed_rows):
                order = (
                    ("layercake_gpu", "transformer_gpu")
                    if (local_index + seed_index) % 2 == 0
                    else ("transformer_gpu", "layercake_gpu")
                )
                values: dict[str, Any] = {}
                for system in order:
                    if system == "layercake_gpu":
                        values[system] = _timed_layercake(
                            orchestrator,
                            item["row"]["prompt"],
                            domain=item["domain"],
                        )
                    else:
                        result = _qwen_request(
                            item["row"]["prompt"],
                            num_gpu=99,
                            seed=4242,
                        )
                        output = bytes.fromhex(result["output_hex"])
                        passed, check_count = _functional_result(
                            item["domain"], output, item["row"]
                        )
                        result["functional_success"] = passed
                        result["functional_check_count"] = check_count
                        values[system] = result
                records.append(
                    {
                        "seed": seed,
                        "prompt_id": item["prompt_id"],
                        "prompt_sha256": item["prompt_sha256"],
                        "domain": item["domain"],
                        "trial": item["trial"],
                        "order": list(order),
                        **values,
                    }
                )
            loaded_tensor_bytes.append(
                sum(
                    _module_bytes(model)
                    for model in orchestrator.host._models.values()
                )
            )
        gc.collect()
        torch.cuda.empty_cache()
    final_report = _ollama_json(OLLAMA_PS)
    document = {
        "format": "layercake-phase7-gpu-performance/1",
        "status": "RAW",
        "protocol": {
            "distinct_prompts": 100,
            "repeated_prompt_observations": 20,
            "observations_per_system": 120,
            "seeds": list(SEEDS),
            "layercake_device": "cuda:0",
            "layercake_precision": "fp32",
            "transformer_num_gpu": 99,
            "transformer_threads": 14,
            "transformer_model": QWEN_MODEL,
            "transformer_digest": QWEN_DIGEST,
            "primary_throughput": "UTF-8 output bytes per complete wall second",
            "paired_keys": ["prompt_id", "trial"],
        },
        "hardware": _hardware(),
        "qwen_warmup_excluded": qwen_warmup,
        "qwen_warm_model_report": warm_report,
        "qwen_final_model_report": final_report,
        "memory": {
            "layercake_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "layercake_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "layercake_loaded_tensor_bytes_by_seed": loaded_tensor_bytes,
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "qwen_size_vram_bytes": int(active[0]["size_vram"]),
        },
        "aggregates": {
            "layercake_gpu": _system_descriptives(
                records, "layercake_gpu"
            ),
            "transformer_gpu": _system_descriptives(
                records, "transformer_gpu"
            ),
        },
        "records": records,
    }
    _write(GPU_PERFORMANCE, document)
    return document


def _inactive_forward_calls(
    delta: Mapping[str, Mapping[str, int]], selected: str
) -> int:
    return sum(
        int(values.get("prefill_calls", 0))
        + int(values.get("decode_step_calls", 0))
        for cake_id, values in delta.items()
        if cake_id != selected
    )


def _gpu_retention() -> dict[str, Any]:
    parent = _read(PHASE6_FUNCTIONAL)
    cpu = {
        (row["domain"], row["id"]): row
        for row in parent["records"]
        if row["mode"] == "automatic_top1"
    }
    rows = _functional_rows()
    records: list[dict[str, Any]] = []
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix="layercake-phase7-gpu-retention-") as temporary:
        orchestrator = _new_orchestrator(Path(temporary) / "registry", "cuda")
        for domain in ("python", "sql", "regex"):
            cake_id = CAKE_IDS[domain]
            for index, row in enumerate(rows[domain]):
                torch.cuda.synchronize()
                result = orchestrator.execute(
                    row["prompt"], mode="automatic_top1"
                )
                torch.cuda.synchronize()
                output = result.output
                if not isinstance(output, bytes):
                    raise RuntimeError("GPU retention output is not bytes")
                passed, checks = _functional_result(domain, output, row)
                reference = cpu[(domain, row["id"])]
                records.append(
                    {
                        "seed": SEEDS[index % len(SEEDS)],
                        "domain": domain,
                        "id": row["id"],
                        "selected": list(result.selected),
                        "functional_success": passed,
                        "functional_check_count": checks,
                        "output_sha256": hashlib.sha256(output).hexdigest(),
                        "cpu_reference_output_sha256": reference["output_sha256"],
                        "cpu_output_equal": (
                            hashlib.sha256(output).hexdigest()
                            == reference["output_sha256"]
                        ),
                        "inactive_forward_calls": _inactive_forward_calls(
                            result.telemetry_delta, cake_id
                        ),
                        "telemetry_delta": result.telemetry_delta,
                    }
                )
    document = {
        "format": "layercake-phase7-gpu-domain-retention/1",
        "status": "RAW",
        "package_hashes": {
            domain: _sha256(path) for domain, path in PACKAGES.items()
        },
        "cpu_reference": {
            "path": _relative(PHASE6_FUNCTIONAL),
            "sha256": _sha256(PHASE6_FUNCTIONAL),
            "successful_automatic_rows": len(cpu),
        },
        "records": records,
    }
    _write(GPU_RETENTION, document)
    return document


def _general_quality_retention() -> dict[str, Any]:
    summary = validate_phase2_r3_bundle(ROOT, PHASE2_RESULTS)
    campaign = _read(ROOT / "moonshot/campaign.yaml")
    expected = campaign["lineage"]["core_checkpoint_hashes"]
    core_paths = {
        key: (
            ROOT
            / "artifacts/moonshot/phase2_shallow_sparse_pretrained"
            / f"student2400-seed-{key.split('-')[1]}"
            / "model.safetensors"
        )
        for key in expected
    }
    actual = {key: _sha256(path) for key, path in core_paths.items()}
    suite = [
        json.loads(line)
        for line in (
            ROOT / "results/moonshot/phase6/final_routing_suite.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    core_rows = [
        row
        for row in suite
        if row["seed"] == 10601 and row["category"] == "core"
    ][:100]
    abstentions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="layercake-phase7-core-abstention-") as temporary:
        orchestrator = _new_orchestrator(Path(temporary) / "registry", "cpu")
        for index, row in enumerate(core_rows):
            result = orchestrator.execute(
                row["prompt"],
                mode="automatic_top1",
                core_handler=lambda prompt: f"CORE:{prompt}",
            )
            abstentions.append(
                {
                    "seed": SEEDS[index % len(SEEDS)],
                    "id": row["id"],
                    "selected": list(result.selected),
                    "execution_path": result.execution_path,
                    "inactive_forward_calls": sum(
                        values["prefill_calls"] + values["decode_step_calls"]
                        for values in result.telemetry_delta.values()
                    ),
                }
            )
    passed = (
        summary["status"] == "PASS"
        and actual == expected
        and len(abstentions) == 100
        and all(
            not row["selected"]
            and row["execution_path"] == "core_only"
            and row["inactive_forward_calls"] == 0
            for row in abstentions
        )
    )
    document = {
        "format": "layercake-phase7-general-quality-retention/1",
        "status": "PASS" if passed else "FAIL",
        "phase2_typed_summary": summary,
        "expected_core_checkpoint_hashes": expected,
        "actual_core_checkpoint_hashes": actual,
        "core_only_abstentions": abstentions,
        "general_quality_noninferior": passed,
    }
    _write(GENERAL_QUALITY, document)
    if not passed:
        raise RuntimeError("sealed general-quality lineage did not retain")
    return document


def _derived_metrics(
    gpu: Mapping[str, Any],
    retention: Mapping[str, Any],
    general: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    cpu = _read(PHASE6_CPU)
    cpu_rows = {
        (row["prompt_id"], row["trial"]): row for row in cpu["records"]
    }
    gpu_rows = {
        (row["prompt_id"], row["trial"]): row for row in gpu["records"]
    }
    if set(cpu_rows) != set(gpu_rows):
        raise RuntimeError("CPU and GPU benchmark pairing differs")
    pairs = []
    for key in sorted(cpu_rows):
        cpu_row = cpu_rows[key]
        gpu_row = gpu_rows[key]
        pairs.append(
            {
                "prompt_id": key[0],
                "trial": key[1],
                "domain": cpu_row["domain"],
                "cpu_cpu_throughput_ratio": (
                    cpu_row["layercake"]["timing"]["bytes_per_second_total"]
                    / cpu_row["qwen"]["timing"]["bytes_per_second_total"]
                ),
                "cpu_cpu_latency_ratio": (
                    cpu_row["layercake"]["timing"]["total_latency_seconds"]
                    / cpu_row["qwen"]["timing"]["total_latency_seconds"]
                ),
                "gpu_gpu_throughput_ratio": (
                    gpu_row["layercake_gpu"]["timing"]["bytes_per_second_total"]
                    / gpu_row["transformer_gpu"]["timing"]["bytes_per_second_total"]
                ),
                "cpu_gpu_throughput_ratio": (
                    cpu_row["layercake"]["timing"]["bytes_per_second_total"]
                    / gpu_row["transformer_gpu"]["timing"]["bytes_per_second_total"]
                ),
                "cpu_gpu_latency_ratio": (
                    cpu_row["layercake"]["timing"]["total_latency_seconds"]
                    / gpu_row["transformer_gpu"]["timing"]["total_latency_seconds"]
                ),
            }
        )
    distinct_keys = {
        (row["prompt_id"], row["trial"])
        for row in gpu["records"]
        if row["trial"] == 1
    }
    cpu_quality_delta = [
        float(cpu_rows[key]["layercake"]["functional_success"])
        - float(cpu_rows[key]["qwen"]["functional_success"])
        for key in sorted(distinct_keys)
    ]
    gpu_quality_delta = [
        float(gpu_rows[key]["layercake_gpu"]["functional_success"])
        - float(gpu_rows[key]["transformer_gpu"]["functional_success"])
        for key in sorted(distinct_keys)
    ]
    cpu_ci = _bootstrap_mean(cpu_quality_delta, seed=SEEDS[0])
    gpu_ci = _bootstrap_mean(gpu_quality_delta, seed=SEEDS[1])
    mixed_superior = (
        statistics.fmean(cpu_quality_delta) > 0
        and statistics.fmean(gpu_quality_delta) > 0
        and cpu_ci[0] > 0
        and gpu_ci[0] > 0
    )
    retention_pass = (
        len(retention["records"]) == 384
        and all(
            row["functional_success"]
            and row["cpu_output_equal"]
            and row["inactive_forward_calls"] == 0
            for row in retention["records"]
        )
    )
    metrics = {
        "cpu_cpu_throughput_ratio": statistics.median(
            row["cpu_cpu_throughput_ratio"] for row in pairs
        ),
        "cpu_cpu_median_latency_ratio": statistics.median(
            row["cpu_cpu_latency_ratio"] for row in pairs
        ),
        "gpu_gpu_throughput_ratio": statistics.median(
            row["gpu_gpu_throughput_ratio"] for row in pairs
        ),
        "cpu_gpu_throughput_ratio": statistics.median(
            row["cpu_gpu_throughput_ratio"] for row in pairs
        ),
        "cpu_gpu_median_latency_ratio": statistics.median(
            row["cpu_gpu_latency_ratio"] for row in pairs
        ),
        "general_quality_noninferior": float(
            general["general_quality_noninferior"]
        ),
        "mixed_domain_quality_superior": float(mixed_superior),
        "promoted_domain_success_retention": float(retention_pass),
    }
    quality = {
        "cpu_layercake_successes": sum(
            cpu_rows[key]["layercake"]["functional_success"]
            for key in distinct_keys
        ),
        "cpu_transformer_successes": sum(
            cpu_rows[key]["qwen"]["functional_success"]
            for key in distinct_keys
        ),
        "gpu_layercake_successes": sum(
            gpu_rows[key]["layercake_gpu"]["functional_success"]
            for key in distinct_keys
        ),
        "gpu_transformer_successes": sum(
            gpu_rows[key]["transformer_gpu"]["functional_success"]
            for key in distinct_keys
        ),
        "cpu_mean_success_delta": statistics.fmean(cpu_quality_delta),
        "gpu_mean_success_delta": statistics.fmean(gpu_quality_delta),
        "cpu_paired_bootstrap_95ci": cpu_ci,
        "gpu_paired_bootstrap_95ci": gpu_ci,
        "paired_rows": len(distinct_keys),
    }
    return metrics, {"pairs": pairs, "quality": quality}


def _write_derived(
    metrics: Mapping[str, float],
    details: Mapping[str, Any],
    descriptive_performance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate_records = [
        {
            "seed": seed,
            "gate_id": gate_id,
            "value": float(value),
            "derivation_scope": "typed verifier recomputes from bound raw evidence",
        }
        for seed in SEEDS
        for gate_id, value in sorted(metrics.items())
    ]
    gates = {
        "format": "layercake-phase7-gate-observations/1",
        "status": "RAW_DERIVED",
        "source_commit": _read(FRAMEWORK)["framework_commit"],
        "source_evidence": {
            "phase6_cpu": _relative(PHASE6_CPU),
            "gpu_performance": _relative(GPU_PERFORMANCE),
            "gpu_retention": _relative(GPU_RETENTION),
            "general_quality": _relative(GENERAL_QUALITY),
            "cold_start": _relative(COLD),
        },
        "quality_details": details["quality"],
        "records": gate_records,
    }
    _write(GATES, gates)
    gate_hash = _sha256(GATES)
    claims = [
        {
            "gate_id": gate_id,
            "kind": (
                "functional_quality"
                if "quality" in gate_id or "retention" in gate_id
                else "integrated_performance"
            ),
            "promoted": True,
            "value": float(value),
            "raw_artifact": _relative(GATES),
            "raw_sha256": gate_hash,
            "derivation": {
                "operation": "mean",
                "field": "value",
                "where": {"gate_id": gate_id},
            },
            "absolute_tolerance": 1e-12,
        }
        for gate_id, value in sorted(metrics.items())
    ]
    campaign = _read(ROOT / "moonshot/campaign.yaml")
    payload = {
        "format": "layercake-phase7-certificate-payload/1",
        "status": "PASS",
        "claims": claims,
        "headline_claims": claims,
        "lineage": campaign["lineage"],
        "systems": {
            "layercake_cpu": "sealed Phase 6 routed product",
            "transformer_cpu": f"{QWEN_MODEL} {QWEN_DIGEST} Q4_K_M Ollama CPU",
            "layercake_gpu": "same sealed packages and router on CUDA FP32",
            "transformer_gpu": f"{QWEN_MODEL} {QWEN_DIGEST} Q4_K_M Ollama GPU",
        },
        "quality": details["quality"],
        "descriptive_performance": descriptive_performance,
        "claim_boundaries": {
            "physical_mobile_hardware_claimed": False,
            "gpu_training_dominance_claimed": False,
            "latent_neural_fusion_claimed": False,
            "performance_hardware": _hardware(),
        },
    }
    _write(PAYLOAD, payload)
    return gates, payload


def certify() -> dict[str, Any]:
    framework = _read(FRAMEWORK)
    _source_audit()
    if _sha256(CONTRACT) != CONTRACT_SHA256:
        raise RuntimeError("Phase 7 preregistration changed")
    if _sha256(PHASE6_CPU) != PHASE6_CPU_SHA256:
        raise RuntimeError("sealed Phase 6 CPU evidence changed")
    if any(_sha256(PACKAGES[key]) != value for key, value in PACKAGE_HASHES.items()):
        raise RuntimeError("sealed package lineage changed")
    validate_phase6_bundle(ROOT, PHASE6_RESULTS)
    validate_phase2_r3_bundle(ROOT, PHASE2_RESULTS)
    rows, _ = _performance_prompt_rows()
    cold = _cold_evidence(rows[0]["domain"], rows[0]["row"])
    cpu_supplement = _cpu_layercake_supplement()
    gpu = _gpu_performance()
    retention = _gpu_retention()
    general = _general_quality_retention()
    metrics, details = _derived_metrics(gpu, retention, general)
    cpu = _read(PHASE6_CPU)
    descriptive_performance = {
        "layercake_cpu": cpu_supplement["aggregates"]["layercake_cpu"],
        "transformer_cpu": _system_descriptives(cpu["records"], "qwen"),
        "layercake_gpu": gpu["aggregates"]["layercake_gpu"],
        "transformer_gpu": gpu["aggregates"]["transformer_gpu"],
        "source_policy": {
            "layercake_cpu": _relative(CPU_SUPPLEMENT),
            "transformer_cpu": _relative(PHASE6_CPU),
            "layercake_gpu": _relative(GPU_PERFORMANCE),
            "transformer_gpu": _relative(GPU_PERFORMANCE),
            "headline_cpu_gate_source": _relative(PHASE6_CPU),
        },
    }
    gates, payload = _write_derived(
        metrics, details, descriptive_performance
    )
    thresholds = _read(CONTRACT)["promotion_thresholds"]
    passed = (
        metrics["cpu_cpu_throughput_ratio"]
        >= thresholds["cpu_cpu_throughput_ratio"]
        and metrics["cpu_cpu_median_latency_ratio"]
        <= thresholds["cpu_cpu_median_latency_ratio"]
        and metrics["gpu_gpu_throughput_ratio"]
        > thresholds["gpu_gpu_throughput_ratio_strictly_greater_than"]
        and metrics["cpu_gpu_throughput_ratio"]
        >= thresholds["cpu_gpu_throughput_ratio"]
        and metrics["cpu_gpu_median_latency_ratio"]
        <= thresholds["cpu_gpu_median_latency_ratio"]
        and metrics["general_quality_noninferior"] == 1.0
        and metrics["mixed_domain_quality_superior"] == 1.0
        and metrics["promoted_domain_success_retention"] == 1.0
    )
    certificate = {
        "format": "layercake-phase7-integrated-performance-certificate/1",
        "status": "PASS" if passed else "FAIL",
        "framework_commit": framework["framework_commit"],
        "contract_sha256": CONTRACT_SHA256,
        "phase6_cpu_sha256": PHASE6_CPU_SHA256,
        "profiles_sha256": PROFILES_SHA256,
        "metrics": metrics,
        "quality": details["quality"],
        "descriptive_performance": descriptive_performance,
        "cold_start_evidence_sha256": cold["evidence_sha256"],
        "gate_observations_sha256": gates["evidence_sha256"],
        "payload_sha256": payload["evidence_sha256"],
        "hardware": _hardware(),
        "claim_boundaries": {
            "physical_mobile_hardware_claimed": False,
            "gpu_training_dominance_claimed": False,
            "latent_neural_fusion_claimed": False,
        },
    }
    _write(CERTIFICATE, certificate)
    if not passed:
        raise RuntimeError(f"Phase 7 gates failed: {metrics}")
    return certificate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "certify"))
    arguments = parser.parse_args(argv)
    value = freeze_framework() if arguments.command == "freeze" else certify()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
