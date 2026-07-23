"""Reproducible Phase 1 inventory, benchmark, and evidence builder.

This module does not select or redesign an architecture.  It captures the existing
LayerCake and transformer references, executes the frozen generation matrix, and
emits raw rows consumed by :mod:`layercake.evaluation.phase1_evidence`.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import urllib.request
import winreg
import xml.etree.ElementTree as ET

import psutil
import torch

from .evaluation.phase1_evidence import (
    CORRECTION_PROTOCOL,
    Phase1EvidenceError,
    derive_performance,
    validate_baseline_optimization,
    validate_benchmark_matrix,
    validate_comparison_certificate,
    validate_evidence_manifest,
    validate_phase1_bundle,
    validate_raw_timing_samples,
    validate_runtime_manifest,
)
from .training.baseline import load_transformer_checkpoint
from .training.foundation import load_core_checkpoint


ROOT = Path(__file__).resolve().parents[1]
PHASE = Path("results/moonshot/phase1")
FORMAT_RAW = "layercake-phase1-raw-timings/1"
RANDOMIZATION_SEED = 20260722
TRIALS = 2
HEADLINE_OUTPUT_BYTES = 128
HEADLINE_REPEAT_PROMPTS = 20
OUTPUT_TARGETS = (64, 256, 1024)
PROMPTS = {
    "short": (
        "short-continuation",
        "Continue with uninterrupted natural-language prose for at least 1600 bytes: "
        "The future of efficient language models is",
    ),
    "medium": (
        "medium-continuation",
        "Continue with uninterrupted natural-language prose for at least 1600 bytes. "
        "Do not conclude early. Efficient language systems should balance accuracy, "
        "latency, memory, transparency, and accessibility. A useful comparison must "
        "hold prompts and stopping rules constant while measuring each implementation "
        "directly. In that setting, the most important engineering lesson is",
    ),
    "long": (
        "long-continuation",
        (
            "Continue with uninterrupted natural-language prose for at least 1600 bytes. "
            "Do not conclude early. A research team is evaluating language-model systems "
            "under a locked protocol. The protocol records executable identity, model and "
            "tokenizer hashes, precision, thread count, device, cache state, prompt identity, "
            "output bytes, timestamps, memory, and failures. Cold requests include model "
            "loading; warm requests use an already resident model. Deterministic and sampled "
            "decoding are kept separate. Prompt and output buckets are randomized with a "
            "published seed. Every aggregate is derived from per-request rows. Quality is not "
            "inferred from speed, and speed is not promoted without a frozen quality suite. "
            "A held-out test split cannot influence architecture selection. Hardware absence "
            "is recorded instead of converted into a passing performance claim. The campaign "
            "uses candidates, verifier promotion, release commits, annotated tags, and clean "
            "sealed verification. Historical measurements remain visible but cannot be mixed "
            "with current checkpoints. Given these constraints, a trustworthy benchmark "
            "should help future work by"
        ),
    ),
}


def _headline_prompts() -> list[dict[str, Any]]:
    topics = (
        "efficient computing", "public libraries", "urban gardens", "coastal weather",
        "scientific replication", "music practice", "safe navigation", "local history",
        "water conservation", "collaborative design",
    )
    tasks = (
        ("continuation", "Continue in clear natural prose about {topic}. Avoid repeating a sentence."),
        ("explanation", "Explain {topic} to a curious reader using two concrete details."),
        ("planning", "Give a concise three-step plan for improving {topic}."),
        ("comparison", "Compare two reasonable approaches to {topic} and state one tradeoff."),
        ("instruction_following", "Write exactly two complete sentences about {topic}."),
        ("reasoning", "State a likely cause and a likely consequence involving {topic}."),
        ("summarization", "Summarize why {topic} matters without using a list."),
        ("question_answering", "Answer directly: what is one practical benefit of {topic}?"),
        ("coherence", "Write a short coherent paragraph that connects people, tools, and {topic}."),
        ("repetition_control", "Describe {topic} with varied vocabulary and no repeated clause."),
    )
    prompts = []
    for topic_index, topic in enumerate(topics):
        for task_index, (category, template) in enumerate(tasks):
            text = template.format(topic=topic) + " Your response must contain at least 80 words."
            identifier = f"functional-{topic_index:02d}-{task_index:02d}"
            prompts.append({
                "id": identifier,
                "category": category,
                "text": text,
                "bytes": len(text.encode("utf-8")),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
    return prompts


def _path(root: Path, relative: str | Path) -> Path:
    result = (root / relative).resolve()
    result.relative_to(root.resolve())
    return result


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object in {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _run(arguments: Sequence[str]) -> tuple[int, str, str]:
    process = subprocess.run(arguments, text=True, capture_output=True, check=False)
    return process.returncode, process.stdout.strip(), process.stderr.strip()


def _post_json(url: str, payload: Mapping[str, Any], *, timeout: float = 120.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"non-object response from {url}")
    return value


def _ollama_show(endpoint: str, model: str) -> dict[str, Any]:
    return _post_json(f"{endpoint}/api/show", {"model": model})


def _ollama_digest(endpoint: str, model: str) -> str:
    with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=10) as response:
        tags = json.load(response)
    for row in tags.get("models", []):
        if row.get("name") == model or row.get("model") == model:
            return str(row["digest"])
    raise RuntimeError(f"model {model} is absent from {endpoint}")


def _cpu_name() -> str:
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        return platform.processor() or platform.machine()


def _gpu_inventory() -> tuple[list[dict[str, Any]], str]:
    code, stdout, stderr = _run([
        "nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if code:
        return [], f"nvidia-smi exit={code}\nstdout={stdout}\nstderr={stderr}\n"
    rows = []
    for line in stdout.splitlines():
        name, uuid, memory_mib, driver = [part.strip() for part in line.split(",", 3)]
        rows.append({
            "name": name, "uuid": uuid, "memory_bytes": int(memory_mib) * 1024 * 1024,
            "driver_version": driver,
        })
    return rows, stdout + "\n"


def _file_identity(root: Path, relative: str) -> tuple[str, dict[str, Any]]:
    path = _path(root, relative)
    metadata = _read(path.parent / "metadata.json")
    return _sha(path), metadata


def _model_manifest(
    *, identifier: str, architecture: str, total: int, active: int,
    checkpoint: Mapping[str, Any], tokenizer_kind: str, tokenizer_sha: str,
    configuration_sha: str, runtime_id: str, incremental_state: Mapping[str, Any],
    role: str, training_data: Mapping[str, Any], limitations: Sequence[str],
) -> dict[str, Any]:
    return {
        "format": "layercake-phase1-model-manifest/1",
        "id": identifier,
        "role": role,
        "architecture": architecture,
        "parameters": {"total": total, "active": active},
        "checkpoint": dict(checkpoint),
        "tokenizer": {"kind": tokenizer_kind, "sha256": tokenizer_sha},
        "configuration": {"sha256": configuration_sha},
        "runtime_id": runtime_id,
        "incremental_state": dict(incremental_state),
        "training_data": dict(training_data),
        "known_limitations": list(limitations),
    }


def prepare(
    root: Path, endpoint: str, model: str, *, preserve_history: bool = False
) -> dict[str, Any]:
    phase = _path(root, PHASE)
    existing = list(phase.iterdir()) if phase.exists() else []
    if existing and not preserve_history:
        raise RuntimeError(f"refusing to overwrite non-empty Phase 1 directory: {phase}")
    if preserve_history:
        history = phase / "history"
        if not history.is_dir() or not any(history.rglob("archive_manifest.json")):
            raise RuntimeError("corrected preparation requires the verifier-created historical archive")
        for path in existing:
            if path.name == "history":
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    for directory in ("raw_runs", "runtime_manifests", "model_manifests"):
        (phase / directory).mkdir(parents=True, exist_ok=True)

    config = {
        "format": "layercake-phase1-benchmark-config/1",
        "correction_protocol": CORRECTION_PROTOCOL,
        "randomization_seed": RANDOMIZATION_SEED,
        "minimum_trials_per_cell": TRIALS,
        "output_target_bytes": list(OUTPUT_TARGETS),
        "generation_modes": {
            "deterministic": {"temperature": 0.0, "top_p": 1.0},
            "sampled": {"temperature": 0.8, "top_p": 0.95},
        },
        "prompts": [
            {
                "bucket": bucket, "id": identifier, "text": text,
                "bytes": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            for bucket, (identifier, text) in PROMPTS.items()
        ],
        "stopping_rule": (
            "capture first streaming prefix containing at least target UTF-8 bytes, continue the "
            "same request to its terminal runtime record, and hash both benchmark prefix and completed response"
        ),
        "cross_model_primary_throughput_metrics": ["bytes_per_second", "characters_per_second"],
        "headline": {
            "distinct_prompts": 100,
            "repeated_prompt_observations": HEADLINE_REPEAT_PROMPTS,
            "output_target_bytes": HEADLINE_OUTPUT_BYTES,
        },
        "batch_size": 1,
        "ollama_model": model,
        "ollama_gpu_endpoint": endpoint,
        "layercake_checkpoint": "artifacts/final/medium-cores/seed-9801",
    }
    config_path = phase / "benchmark_config.json"
    _write(config_path, config)
    config_sha = _sha(config_path)

    gpus, nvidia_output = _gpu_inventory()
    hardware_capture = (
        f"platform={platform.platform()}\npython={sys.version}\n"
        f"cpu={_cpu_name()}\nphysical_cores={psutil.cpu_count(logical=False)}\n"
        f"logical_cores={psutil.cpu_count(logical=True)}\n"
        f"memory={psutil.virtual_memory().total}\n"
        f"torch_cpu_capability={torch.backends.cpu.get_cpu_capability()}\n"
        f"nvidia-smi:\n{nvidia_output}"
    )
    capture_path = phase / "hardware_capture.txt"
    capture_path.write_text(hardware_capture, encoding="utf-8")
    hardware = {
        "format": "layercake-phase1-hardware/1",
        "capture": {
            "command": "phase1_campaign prepare; psutil + torch CPU capability + nvidia-smi query",
            "stdout_path": _relative(root, capture_path),
            "stdout_sha256": _sha(capture_path),
        },
        "cpu": {
            "model": _cpu_name(),
            "physical_cores": int(psutil.cpu_count(logical=False) or 1),
            "logical_cores": int(psutil.cpu_count(logical=True) or 1),
            "instruction_sets": [torch.backends.cpu.get_cpu_capability()],
        },
        "memory": {"total_physical_bytes": int(psutil.virtual_memory().total)},
        "gpus": gpus,
    }
    _write(phase / "hardware.json", hardware)

    ollama = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe"
    if not ollama.is_file():
        raise RuntimeError(f"Ollama executable is missing: {ollama}")
    code, version_out, version_err = _run([str(ollama), "--version"])
    if code:
        raise RuntimeError(f"Ollama version command failed: {version_err}")
    show = _ollama_show(endpoint, model)
    digest = _ollama_digest(endpoint, model)
    info = show["model_info"]
    tokenizer_subset = {key: value for key, value in info.items() if key.startswith("tokenizer.")}
    tokenizer_sha = _canonical_sha(tokenizer_subset)
    qwen_config_sha = _canonical_sha({"details": show["details"], "model_info": info})
    show_path = phase / "ollama_model_show.json"
    _write(show_path, {
        "details": show["details"], "model_info": info, "capabilities": show.get("capabilities"),
        "modified_at": show.get("modified_at"), "digest": digest,
    })
    runtime_base = {
        "format": "layercake-phase1-runtime-manifest/1",
        "executable": {"path": str(ollama.resolve()), "sha256": _sha(ollama)},
        "version": {
            "command": f'"{ollama}" --version', "stdout": version_out,
            "stdout_sha256": hashlib.sha256(version_out.encode()).hexdigest(),
        },
        "backend": "Ollama native runner using llama.cpp/ggml",
        "precision_contract": str(show["details"]["quantization_level"]),
    }
    for runtime_id, target in (("ollama-cpu", "cpu"), ("ollama-gpu", "gpu")):
        _write(phase / f"runtime_manifests/{runtime_id}.json", {
            **runtime_base, "id": runtime_id, "target_device": target,
            "optimization_evidence": {
                "kv_cache": {"mechanism": "native llama.cpp per-layer KV cache", "raw_trace_run_ids": ["PENDING"]},
                "kernels": {
                    "implementation": "ggml quantized Q4_K_M matrix kernels",
                    "instruction_sets": [torch.backends.cpu.get_cpu_capability()] if target == "cpu" else ["CUDA"],
                },
                "threading": {"raw_trace_run_ids": ["PENDING"]},
                "batch_one": {"batch_size": 1},
            },
        })
    python_version = f"Python {platform.python_version()}; torch {torch.__version__}"
    _write(phase / "runtime_manifests/pytorch-foundation-v2.json", {
        "format": "layercake-phase1-runtime-manifest/1", "id": "pytorch-foundation-v2",
        "executable": {"path": sys.executable, "sha256": _sha(Path(sys.executable))},
        "version": {
            "command": f'"{sys.executable}" -c "import torch; print(torch.__version__)"',
            "stdout": python_version, "stdout_sha256": hashlib.sha256(python_version.encode()).hexdigest(),
        },
        "backend": "PyTorch eager recurrent/patch kernels with exact persistent state",
        "target_device": "cpu_and_gpu", "precision_contract": "fp32",
    })
    _write(phase / "runtime_manifests/pytorch-reference-controls.json", {
        "format": "layercake-phase1-runtime-manifest/1", "id": "pytorch-reference-controls",
        "executable": {"path": sys.executable, "sha256": _sha(Path(sys.executable))},
        "version": {
            "command": f'"{sys.executable}" -c "import torch; print(torch.__version__)"',
            "stdout": python_version, "stdout_sha256": hashlib.sha256(python_version.encode()).hexdigest(),
        },
        "backend": "PyTorch reference-quality controls; not the optimized headline runtime",
        "target_device": "cpu_and_gpu", "precision_contract": "fp32/fp16 as checkpoint metadata records",
    })

    train_data = {
        "corpus": "WikiText-103 byte corpus",
        "train_path": "data/moonshot/v2/wikitext103/train_medium.bin",
        "train_sha256": "ec54bd8fa09c2cf1a6d442538a98c62ce8e62de14378a19556310836891d23b6",
        "train_bytes": 100_000_000,
    }
    qwen_checkpoint = {
        "kind": "external_content_addressed", "provider": f"Ollama {model}",
        "sha256": digest, "manifest_sha256": _sha(show_path),
    }
    qwen_parameters = int(info["general.parameter_count"])
    qwen_state = {"status": "MEASURED", "mechanism": "native llama.cpp KV cache", "raw_trace_run_ids": ["PENDING"]}
    for suffix, runtime in (("cpu", "ollama-cpu"), ("gpu", "ollama-gpu")):
        _write(phase / f"model_manifests/qwen25-05b-{suffix}.json", _model_manifest(
            identifier=f"qwen25-05b-{suffix}", architecture="Qwen2.5 0.5B Instruct, 24-layer GQA transformer",
            total=qwen_parameters, active=qwen_parameters, checkpoint=qwen_checkpoint,
            tokenizer_kind="Qwen2 GPT2-style BPE in GGUF", tokenizer_sha=tokenizer_sha,
            configuration_sha=qwen_config_sha, runtime_id=runtime, incremental_state=qwen_state,
            role=f"optimized_{suffix}_transformer", training_data={"provider": "Qwen upstream; not trained in this repository"},
            limitations=["Instruct-tuned external reference is larger than the in-repository 3.35M BPE quality control"],
        ))

    lc_checkpoint = root / "artifacts/final/medium-cores/seed-9801/model.safetensors"
    lc_metadata = _read(lc_checkpoint.parent / "metadata.json")
    lc_tokenizer_sha = _canonical_sha({"contract": "raw bytes", "vocabulary": 256})
    _write(phase / "model_manifests/layercake-foundation-v2.json", _model_manifest(
        identifier="layercake-foundation-v2", architecture=lc_metadata["architecture"]["architecture_version"],
        total=int(lc_metadata["parameters"]["total_parameters"]),
        active=int(lc_metadata["parameters"]["active_parameters"]),
        checkpoint={"kind": "local_file", "path": _relative(root, lc_checkpoint), "sha256": _sha(lc_checkpoint)},
        tokenizer_kind="raw-byte vocabulary 0..255", tokenizer_sha=lc_tokenizer_sha,
        configuration_sha=str(lc_metadata["config"]["sha256"]), runtime_id="pytorch-foundation-v2",
        incremental_state={"status": "MEASURED", "mechanism": "FoundationV2State recurrent and patch state", "raw_trace_run_ids": ["PENDING"]},
        role="fastest_credible_and_integrated_layercake", training_data=train_data,
        limitations=["Current validation BPB is weaker than the BPE reference", "PyTorch runtime is not yet a native packaged CPU runtime"],
    ))

    bpe_root = root / "artifacts/final/medium-transformers/seed-9801"
    bpe_meta = _read(bpe_root / "metadata.json")
    _write(phase / "model_manifests/bpe-reference.json", _model_manifest(
        identifier="bpe-reference", architecture=bpe_meta["architecture"]["architecture_version"],
        total=int(bpe_meta["parameters"]), active=int(bpe_meta["parameters"]),
        checkpoint={"kind": "local_file", "path": _relative(root, bpe_root / "model.safetensors"), "sha256": _sha(bpe_root / "model.safetensors")},
        tokenizer_kind="trained 384-token byte-pair tokenizer", tokenizer_sha=str(bpe_meta["tokenizer"]["sha256"]),
        configuration_sha=str(bpe_meta["config"]["sha256"]), runtime_id="pytorch-reference-controls",
        incremental_state={"status": "MEASURED", "mechanism": "per-layer PyTorch KV cache", "raw_trace_run_ids": ["PENDING"]},
        role="strongest_existing_bpe_quality_reference", training_data=train_data,
        limitations=["512-token maximum context/output envelope", "same-scale architecture comparator, not the optimized product runtime"],
    ))
    adaptive_root = root / "artifacts/final/adaptive-medium-pilot/routed_adaptive_5x5_top1_8e/seed-9811"
    adaptive_meta = _read(adaptive_root / "metadata.json")
    _write(phase / "model_manifests/layercake-adaptive-quality.json", _model_manifest(
        identifier="layercake-adaptive-quality", architecture="causal adaptive 2/4-byte patch transformer",
        total=int(adaptive_meta["parameters"]), active=int(adaptive_meta["active_parameters"]),
        checkpoint={"kind": "local_file", "path": _relative(root, adaptive_root / "model.safetensors"), "sha256": _sha(adaptive_root / "model.safetensors")},
        tokenizer_kind="causal adaptive raw-byte patches", tokenizer_sha=lc_tokenizer_sha,
        configuration_sha=str(adaptive_meta["run_fingerprint"]), runtime_id="pytorch-reference-controls",
        incremental_state={"status": "NOT_AVAILABLE_IN_CURRENT_IMPLEMENTATION", "reason": "current adaptive model exposes full forward but no persistent decode state"},
        role="highest_quality_current_layercake", training_data=train_data,
        limitations=["No persistent incremental decode API", "therefore excluded from speed headlines"],
    ))
    byte_checkpoint = root / "runs_experiment/scale5m_seed4242_continued.pt"
    byte_config = root / "results/scale5m_seed4242_continued.json"
    _write(phase / "model_manifests/byte-transformer-control.json", _model_manifest(
        identifier="byte-transformer-control", architecture="CausalByteLM absolute-position transformer",
        total=14_566_048, active=14_566_048,
        checkpoint={"kind": "local_file", "path": _relative(root, byte_checkpoint), "sha256": _sha(byte_checkpoint)},
        tokenizer_kind="raw-byte vocabulary 0..255", tokenizer_sha=lc_tokenizer_sha,
        configuration_sha=_sha(byte_config), runtime_id="pytorch-reference-controls",
        incremental_state={"status": "NOT_AVAILABLE_IN_CURRENT_IMPLEMENTATION", "reason": "historical control uses full-sequence forward and fixed absolute positions"},
        role="byte_transformer_control", training_data={"historical_run": _relative(root, byte_config)},
        limitations=["unsafe legacy pickle container", "no KV cache", "fixed context", "not valid as optimized speed baseline"],
    ))

    model_paths = {
        path.stem: path for path in (phase / "model_manifests").glob("*.json")
    }
    runtime_paths = {
        path.stem: path for path in (phase / "runtime_manifests").glob("*.json")
    }
    baseline_specs = [
        ("bpe_reference", "bpe-reference", "pytorch-reference-controls"),
        ("optimized_cpu_transformer", "qwen25-05b-cpu", "ollama-cpu"),
        ("optimized_gpu_transformer", "qwen25-05b-gpu", "ollama-gpu"),
        ("byte_transformer", "byte-transformer-control", "pytorch-reference-controls"),
        ("fastest_existing_layercake", "layercake-foundation-v2", "pytorch-foundation-v2"),
        ("highest_quality_existing_layercake", "layercake-adaptive-quality", "pytorch-reference-controls"),
        ("integrated_layercake", "layercake-foundation-v2", "pytorch-foundation-v2"),
    ]
    baselines = []
    for baseline_id, model_id, runtime_id in baseline_specs:
        model_path = next(path for path in model_paths.values() if _read(path)["id"] == model_id)
        runtime_path = next(path for path in runtime_paths.values() if _read(path)["id"] == runtime_id)
        entry = {
            "id": baseline_id, "model_id": model_id,
            "model_manifest": {"path": _relative(root, model_path), "sha256": _sha(model_path)},
            "runtime": {"name": _read(runtime_path)["backend"], "version": _read(runtime_path)["version"]["stdout"], "execution": "native" if runtime_id.startswith("ollama") else "pytorch_reference", "runtime_manifest": {"path": _relative(root, runtime_path), "sha256": _sha(runtime_path)}},
        }
        if baseline_id.startswith("optimized_"):
            entry["runtime"]["deployment_evidence"] = {"path": _relative(root, show_path), "sha256": _sha(show_path)}
            entry["runtime"]["kv_cache_evidence"] = {"path": _relative(root, runtime_path), "sha256": _sha(runtime_path)}
        baselines.append(entry)
    _write(phase / "baseline_inventory.json", {
        "format": "layercake-phase1-baseline-inventory/1", "baselines": baselines,
        "historical_invalid_or_unfair": [
            {"pattern": "results/moonshot/v2/cpu_vs_gpu_evidence.json", "reason": "historical runtime/checkpoints and locked matrix differ; never inherited"},
            {"pattern": "results/breakthrough_equal/*", "reason": "task-specific stored-answer/schema workloads do not establish general generation speed"},
            {"pattern": "scripts/benchmark_generation.py BPE path", "reason": "BPE comparator lacks KV caching and is not the optimized headline baseline"},
            {"pattern": "legacy certificates", "reason": "certificate summaries are not raw evidence"},
        ],
        "performance_matrix_scope": {
            "included": ["qwen25-05b cpu/gpu through Ollama", "layercake-foundation-v2 cpu/gpu"],
            "excluded_with_reason": {
                "bpe-reference": "512-token limit cannot satisfy locked 1024-byte output",
                "byte-transformer-control": "no incremental cache and fixed context",
                "layercake-adaptive-quality": "no persistent incremental decode implementation",
            },
        },
    })

    quality_prompts = _headline_prompts()
    for prompt in quality_prompts:
        prompt.setdefault("category", prompt.get("bucket", "continuation_quality"))
        prompt["sha256"] = hashlib.sha256(prompt["text"].encode()).hexdigest()
        prompt["bytes"] = len(prompt["text"].encode())
    data_paths = [
        ("architecture_selection", "data/moonshot/v2/wikitext103/architecture_selection.bin", True),
        ("validation", "data/moonshot/v2/wikitext103/validation.bin", True),
        ("test", "data/moonshot/v2/wikitext103/test.bin", False),
    ]
    datasets = [
        {"split": split, "path": path, "sha256": _sha(root / path), "selection_access_allowed": allowed}
        for split, path, allowed in data_paths
    ]
    duplicate_ids = sorted({item["id"] for item in quality_prompts if sum(row["id"] == item["id"] for row in quality_prompts) > 1})
    contamination_matches = []
    for split, relative, _ in data_paths:
        payload = (root / relative).read_bytes()
        for prompt in quality_prompts:
            if prompt["text"].encode() in payload:
                contamination_matches.append({"split": split, "prompt_id": prompt["id"]})
    corpus_manifest = root / "data/moonshot/v2/wikitext103/manifest.json"
    corpus_data = _read(corpus_manifest)
    overlaps = corpus_data.get("sampled_cross_split_64byte_overlaps", {})
    contamination = {
        "format": "layercake-phase1-contamination-report/1",
        "duplicate_prompt_ids": duplicate_ids,
        "exact_prompt_corpus_matches": contamination_matches,
        "cross_split_exact_overlaps": [key for key, value in overlaps.items() if value],
        "source_manifest": _relative(root, corpus_manifest), "source_manifest_sha256": _sha(corpus_manifest),
        "method": "exact byte search for every frozen prompt plus inherited sampled 64-byte cross-split scan",
    }
    contamination_path = phase / "contamination_report.json"
    _write(contamination_path, contamination)
    metrics = [
        ("heldout_bpb", "layercake.training baseline token/byte NLL converted to bits per raw byte", "lower"),
        ("repetition_rate", "layercake.evaluation quality repeated 4-gram fraction", "lower"),
        ("unique_ngram_rate", "layercake.evaluation quality unique 4-grams / total 4-grams", "higher"),
        ("entropy_collapse", "byte/token entropy and longest repeated-run diagnostic", "lower"),
        ("continuation_quality", "frozen blinded five-point continuation rubric", "higher"),
        ("instruction_following", "exact frozen task constraints and JSON validity", "higher"),
        ("long_context", "exact codeword-recall task accuracy", "higher"),
        ("invalid_output_rate", "invalid UTF-8 or task-schema failures / prompts", "lower"),
        ("sample_inspection", "hash-bound frozen prompt/output inspection ledger", "higher"),
        ("contamination", "exact prompt search and sampled cross-split overlap report", "lower"),
    ]
    quality = {
        "format": "layercake-phase1-quality-suite/1",
        "correction_protocol": CORRECTION_PROTOCOL,
        "metrics": [{"id": i, "implementation": impl, "direction": direction} for i, impl, direction in metrics],
        "prompts": quality_prompts,
        "datasets": datasets,
        "contamination_report": {"path": _relative(root, contamination_path), "sha256": _sha(contamination_path)},
        "sampling": config["generation_modes"], "stopping_rule": config["stopping_rule"],
        "test_policy": "test split is frozen and selection_access_allowed=false; Phase 2 selection uses architecture_selection and validation only",
    }
    quality_path = phase / "quality_suite_manifest.json"
    _write(quality_path, quality)
    _write(phase / "quality_threshold_lock.json", {
        "format": "layercake-phase1-threshold-lock/1", "quality_suite_sha256": _sha(quality_path),
        "locked_before_phase2": True,
        "statistical_methodology": {
            "confidence": 0.95, "bootstrap_seed": RANDOMIZATION_SEED,
            "resamples": 10_000, "pairing_key": "prompt_id/generation_mode/seed",
            "implementation": "layercake.evaluation.campaign_statistics",
        },
        "non_inferiority_margins": {
            "heldout_bpb": 0.03, "repetition_rate": 0.02, "unique_ngram_rate": 0.02,
            "entropy_collapse": 0.02, "continuation_quality": 0.25,
            "instruction_following": 0.02, "long_context": 0.02, "invalid_output_rate": 0.01,
        },
    })
    _write(phase / "benchmark_matrix.json", {
        "format": "layercake-phase1-benchmark-matrix/1", "minimum_trials_per_cell": TRIALS,
        "correction_protocol": CORRECTION_PROTOCOL,
        "axes": {
            "cache_states": ["cold", "warm"], "generation_modes": ["deterministic", "sampled"],
            "prompt_buckets": list(PROMPTS), "output_target_bytes": list(OUTPUT_TARGETS),
        },
        "systems": [
            {"id": "transformer_optimized_cpu", "role": "optimized_transformer_baseline", "required_devices": ["cpu_one_thread", "cpu_all_core"]},
            {"id": "transformer_optimized_gpu", "role": "optimized_transformer_gpu_baseline", "required_devices": ["gpu"]},
            {"id": "layercake_fastest_integrated", "role": "fastest_layercake_baseline", "required_devices": ["cpu_one_thread", "cpu_all_core"] + (["gpu"] if gpus else []), "gpu_status": "REQUIRED_AVAILABLE_HARDWARE" if gpus else "NOT_RUN_NO_HARDWARE"},
        ],
        "optimized_runtime_ids": ["ollama-cpu"] + (["ollama-gpu"] if gpus else []),
        "randomization_seed": RANDOMIZATION_SEED,
        "promoted_benchmark_protocol": {
            "minimum_distinct_prompts": 100,
            "minimum_repeated_prompt_observations": HEADLINE_REPEAT_PROMPTS,
            "tail_quantile_minimum_observations": 20,
            "pairing_key": "prompt.id",
            "bootstrap_confidence": 0.95,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": RANDOMIZATION_SEED,
            "headline_configurations": [
                {
                    "id": "same-scale-architecture-cpu",
                    "comparison_kind": "same_scale_architecture",
                    "layercake_system_id": "layercake_same_scale_headline",
                    "transformer_system_id": "transformer_same_scale_headline",
                    "layercake_device": "cpu_one_thread",
                    "transformer_device": "cpu_one_thread",
                    "cache_state": "warm",
                    "generation_mode": "deterministic",
                    "output_target_bytes": HEADLINE_OUTPUT_BYTES,
                },
                {
                    "id": "product-cpu",
                    "comparison_kind": "product",
                    "layercake_system_id": "layercake_product_headline",
                    "transformer_system_id": "transformer_product_headline",
                    "layercake_device": "cpu_one_thread",
                    "transformer_device": "cpu_all_core",
                    "cache_state": "warm",
                    "generation_mode": "deterministic",
                    "output_target_bytes": HEADLINE_OUTPUT_BYTES,
                },
            ],
        },
        "equivalence_contract": {
            "identical_prompts": _sha(config_path), "batch_size": 1,
            "output_semantics": "first exact target-byte streaming prefix",
            "retrieval": "none", "stored_answers": "none",
            "sampling_difference": "same temperature/top-p/seed request; runtime RNG algorithms may differ and are not treated as bit-identical",
        },
    })
    return {"phase": 1, "status": "PREPARED", "config_sha256": config_sha, "gpu_count": len(gpus)}


def _cells(devices: Sequence[str]) -> list[tuple[str, str, str, int, int]]:
    values = list(itertools.product(devices, ("cold", "warm"), ("deterministic", "sampled"), PROMPTS, OUTPUT_TARGETS, range(1, TRIALS + 1)))
    random.Random(RANDOMIZATION_SEED).shuffle(values)
    return values


def _select(logits: torch.Tensor, mode: str, seed: int) -> torch.Tensor:
    if mode == "deterministic":
        return logits.argmax(-1)
    generator = torch.Generator(device=logits.device).manual_seed(seed)
    probabilities = torch.softmax(logits / 0.8, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).flatten()


def _process_memory(processes: Iterable[psutil.Process]) -> tuple[int, int]:
    resident = 0
    peak = 0
    for process in processes:
        try:
            info = process.memory_info()
            resident += int(info.rss)
            peak += int(getattr(info, "peak_wset", info.rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return resident, peak


def _base_row(
    *, index: int, system: str, runtime: str, model: Mapping[str, Any],
    device: str, threads: int, cache: str, mode: str, bucket: str, target: int,
    trial: int, output: bytes, generated_tokens: int, started: int, first: int,
    completed: int, resident: int, peak: int, command_id: str, precision: str,
    phase_timings: Mapping[str, Any], accelerator: Mapping[str, Any], status: str = "PASS",
    exit_code: int = 0, completed_output: bytes | None = None,
    token_accounting_method: str = "raw_byte_vocabulary",
    prompt_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if prompt_record is None:
        prompt_id, prompt_text = PROMPTS[bucket]
        prompt_record = {
            "id": prompt_id, "text": prompt_text, "bucket": bucket,
            "sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
            "bytes": len(prompt_text.encode()),
        }
    completed_output = output if completed_output is None else completed_output
    phases = dict(phase_timings)
    if cache == "cold":
        phases["model_load_source"] = "same_streaming_request"
    return {
        "evidence_protocol": CORRECTION_PROTOCOL,
        "run_id": f"{system}-{index:04d}", "system_id": system, "runtime_id": runtime,
        "model_id": model["id"], "model_sha256": model["checkpoint"]["sha256"],
        "tokenizer_sha256": model["tokenizer"]["sha256"],
        "configuration_sha256": model["configuration"]["sha256"], "precision": precision,
        "seed": RANDOMIZATION_SEED + trial, "trial": trial,
        "device": {"kind": device, "hardware_id": "cpu-0" if device.startswith("cpu") else "GPU-e2863cae-6a92-5833-1d9e-cc308702a966"},
        "threads": {"requested": threads, "observed_limit": threads},
        "prompt": {
            "id": prompt_record["id"], "sha256": prompt_record["sha256"],
            "bytes": prompt_record["bytes"], "bucket": prompt_record.get("bucket", bucket),
        },
        "output": {
            "target_bytes": target, "generated_bytes": len(output),
            "generated_tokens": generated_tokens, "generated_characters": len(output.decode("utf-8", errors="replace")),
            "sha256": hashlib.sha256(output).hexdigest(), "hex": output.hex(),
            "token_accounting": {
                "method": token_accounting_method,
                "scope": "completed_response",
                "count": generated_tokens,
                "completed_response_bytes": len(completed_output),
                "completed_response_sha256": hashlib.sha256(completed_output).hexdigest(),
            },
        },
        "generation": {"mode": mode, "temperature": 0.0 if mode == "deterministic" else 0.8, "top_p": 1.0 if mode == "deterministic" else 0.95, "seed": RANDOMIZATION_SEED + trial},
        "cache_state": {
            "kind": cache,
            "procedure": "model/checkpoint reloaded inside measured interval" if cache == "cold" else "resident model received one unmeasured prefill/decode warm-up before measured request",
            **({"single_request_evidence": {
                "measured_streaming_requests": 1,
                "load_probe_requests": 0,
                "model_load_source": "same_streaming_request",
            }} if cache == "cold" else {}),
        },
        "order": {"randomization_seed": RANDOMIZATION_SEED, "index": index + 1, "permutation_sha256": "PENDING"},
        "timing": {
            "clock": "perf_counter_ns", "request_started_ns": started, "first_output_ns": first,
            "target_completed_ns": completed, "time_to_first_output_seconds": (first - started) / 1e9,
            "total_latency_seconds": (completed - started) / 1e9, "phase_timings": phases,
        },
        "memory": {
            "method": "Windows process RSS and peak_wset; accelerator allocation recorded separately",
            "resident_bytes": resident, "peak_resident_bytes": peak,
            "accelerator_allocation": dict(accelerator),
        },
        "execution": {"command_id": command_id, "exit_code": exit_code}, "status": status,
    }


def benchmark_layercake(root: Path) -> dict[str, Any]:
    phase = _path(root, PHASE)
    model_manifest = _read(phase / "model_manifests/layercake-foundation-v2.json")
    checkpoint = root / "artifacts/final/medium-cores/seed-9801"
    devices = ["cpu_one_thread", "cpu_all_core"]
    if _read(phase / "hardware.json")["gpus"] and torch.cuda.is_available():
        devices.append("gpu")
    cells = _cells(devices)
    permutation = _canonical_sha(cells)
    process = psutil.Process(os.getpid())
    rows = []
    resident_models: dict[str, Any] = {}
    physical = int(psutil.cpu_count(logical=False) or 1)
    for index, (device_kind, cache, mode, bucket, target, trial) in enumerate(cells):
        device = torch.device("cuda" if device_kind == "gpu" else "cpu")
        threads = 1 if device_kind in {"cpu_one_thread", "gpu"} else physical
        torch.set_num_threads(threads)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        load_started = time.perf_counter_ns()
        if cache == "cold" or device_kind not in resident_models:
            resident_models.pop(device_kind, None)
            if "model" in locals():
                del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model, _ = load_core_checkpoint(checkpoint, device=device)
            resident_models[device_kind] = model
        else:
            model = resident_models[device_kind]
        if device.type == "cuda":
            torch.cuda.synchronize()
        loaded = time.perf_counter_ns()
        if cache == "warm":
            warm = model.prefill(PROMPTS[bucket][1], sampler_seed=RANDOMIZATION_SEED)
            _, warm = model.decode_step(warm)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter_ns()
            load_started = started
            loaded = started
        preprocessing_started = time.perf_counter_ns()
        prompt_bytes = PROMPTS[bucket][1].encode("utf-8")
        preprocessing_done = time.perf_counter_ns()
        prefill_started = preprocessing_done
        state = model.prefill(prompt_bytes, sampler_seed=RANDOMIZATION_SEED + trial)
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_done = time.perf_counter_ns()
        generated = bytearray()
        generated_tokens = 0
        first = 0
        while len(generated) < target:
            selected = _select(state.next_logits, mode, RANDOMIZATION_SEED + trial + generated_tokens)
            _, state = model.decode_step(state, next_byte=selected)
            if device.type == "cuda":
                torch.cuda.synchronize()
            generated.extend(bytes([int(selected.item())]))
            generated_tokens += 1
            if not first:
                first = time.perf_counter_ns()
        completed = time.perf_counter_ns()
        resident, peak = _process_memory([process])
        accelerator = (
            {"status": "MEASURED", "method": "torch.cuda.max_memory_allocated", "peak_bytes": int(torch.cuda.max_memory_allocated())}
            if device.type == "cuda" else
            {"status": "NOT_APPLICABLE_CPU", "method": "none", "peak_bytes": 0}
        )
        row = _base_row(
            index=index, system="layercake_fastest_integrated", runtime="pytorch-foundation-v2",
            model=model_manifest, device=device_kind, threads=threads, cache=cache, mode=mode,
            bucket=bucket, target=target, trial=trial, output=bytes(generated[:target]),
            generated_tokens=generated_tokens, started=started, first=first, completed=completed,
            resident=resident, peak=peak, command_id="phase1-layercake-direct", precision="fp32",
            phase_timings={
                "model_load_seconds": (loaded - load_started) / 1e9,
                "prompt_preprocessing_seconds": (preprocessing_done - preprocessing_started) / 1e9,
                "prefill_seconds": (prefill_done - prefill_started) / 1e9,
                "decode_seconds": (completed - prefill_done) / 1e9,
                "measurement": "direct synchronized client wall clock",
            }, accelerator=accelerator,
        )
        row["order"]["permutation_sha256"] = permutation
        rows.append(row)
        if (index + 1) % 12 == 0:
            _write(phase / "raw_runs/layercake.json", {"format": FORMAT_RAW, "records": rows})
            print(f"layercake {index + 1}/{len(cells)}", flush=True)
    _write(phase / "raw_runs/layercake.json", {"format": FORMAT_RAW, "records": rows})
    return {"system": "layercake_fastest_integrated", "records": len(rows), "status": "PASS"}


def _ollama_processes() -> list[psutil.Process]:
    rows = []
    for process in psutil.process_iter(["name"]):
        try:
            if "ollama" in (process.info.get("name") or "").lower():
                rows.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rows


def _ollama_unload(endpoint: str, model: str) -> None:
    _post_json(f"{endpoint}/api/generate", {"model": model, "prompt": "", "stream": False, "keep_alive": 0})


def _ollama_warm(endpoint: str, model: str, threads: int) -> None:
    _post_json(f"{endpoint}/api/generate", {
        "model": model, "prompt": "warm up", "raw": True, "stream": False, "keep_alive": -1,
        "options": {"num_predict": 1, "num_thread": threads, "num_ctx": 4096, "temperature": 0},
    })


def _ollama_stream(
    endpoint: str, model: str, prompt: str, *, target: int, threads: int,
    mode: str, seed: int,
) -> tuple[bytes, int, int, int, int, dict[str, Any]]:
    payload = {
        "model": model, "prompt": prompt, "raw": True, "stream": True, "keep_alive": -1,
        "options": {
            "num_predict": max(64, (3 * target) // 4 + 16),
            "num_thread": threads, "num_ctx": 4096,
            "temperature": 0.0 if mode == "deterministic" else 0.8,
            "top_p": 1.0 if mode == "deterministic" else 0.95, "seed": seed,
        },
    }
    request = urllib.request.Request(
        f"{endpoint}/api/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    generated = bytearray()
    first = 0
    target_completed = 0
    request_completed = 0
    final: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            row = json.loads(line)
            piece = str(row.get("response", "")).encode("utf-8")
            if piece:
                generated.extend(piece)
                if not first:
                    first = time.perf_counter_ns()
                if len(generated) >= target and not target_completed:
                    target_completed = time.perf_counter_ns()
            if row.get("done"):
                final = row
                request_completed = time.perf_counter_ns()
    if len(generated) < target:
        raise RuntimeError(f"Ollama stopped at {len(generated)} bytes before target {target}")
    if not final or not request_completed:
        raise RuntimeError("Ollama streaming response did not include a terminal runtime record")
    eval_count = int(final.get("eval_count", 0))
    if eval_count <= 0:
        raise RuntimeError("Ollama terminal response lacks authoritative eval_count")
    return bytes(generated), eval_count, first, target_completed, request_completed, final


def benchmark_ollama(
    root: Path, endpoint: str, model: str, target_device: str,
) -> dict[str, Any]:
    if target_device not in {"cpu", "gpu"}:
        raise ValueError("target_device must be cpu or gpu")
    phase = _path(root, PHASE)
    hardware = _read(phase / "hardware.json")
    if target_device == "gpu" and not hardware["gpus"]:
        return {"system": "transformer_optimized_gpu", "status": "NOT_RUN_NO_HARDWARE"}
    devices = ["gpu"] if target_device == "gpu" else ["cpu_one_thread", "cpu_all_core"]
    runtime_id = f"ollama-{target_device}"
    model_manifest = _read(phase / f"model_manifests/qwen25-05b-{target_device}.json")
    system = f"transformer_optimized_{target_device}"
    cells = _cells(devices)
    permutation = _canonical_sha(cells)
    physical = int(psutil.cpu_count(logical=False) or 1)
    rows = []
    for index, (device_kind, cache, mode, bucket, target, trial) in enumerate(cells):
        threads = 1 if device_kind in {"cpu_one_thread", "gpu"} else physical
        _ollama_unload(endpoint, model) if cache == "cold" else _ollama_warm(endpoint, model, threads)
        started = time.perf_counter_ns()
        output, tokens, first, target_completed, completed, final = _ollama_stream(
            endpoint, model, PROMPTS[bucket][1], target=target, threads=threads,
            mode=mode, seed=RANDOMIZATION_SEED + trial,
        )
        load_seconds = float(final.get("load_duration", 0)) / 1e9
        resident, peak = _process_memory(_ollama_processes())
        row = _base_row(
            index=index, system=system, runtime=runtime_id, model=model_manifest,
            device=device_kind, threads=threads, cache=cache, mode=mode, bucket=bucket,
            target=target, trial=trial, output=output, generated_tokens=tokens,
            started=started, first=first, completed=completed, resident=resident, peak=peak,
            command_id=f"phase1-ollama-{target_device}", precision="Q4_K_M",
            phase_timings={
                "model_load_seconds": load_seconds,
                "prompt_preprocessing_seconds": 0.0,
                "prefill_and_first_decode_seconds": float(final.get("prompt_eval_duration", 0)) / 1e9,
                "decode_after_first_seconds": float(final.get("eval_duration", 0)) / 1e9,
                "target_prefix_latency_seconds": (target_completed - started) / 1e9,
                "runtime_total_duration_seconds": float(final.get("total_duration", 0)) / 1e9,
                "measurement": "one streaming request; client clocks TTFO/total and terminal runtime record supplies load/eval durations",
            },
            accelerator=(
                {"status": "NOT_EXPOSED_BY_EXTERNAL_RUNTIME", "method": "Ollama API", "peak_bytes": 0}
                if target_device == "gpu" else
                {"status": "NOT_APPLICABLE_CPU", "method": "none", "peak_bytes": 0}
            ), completed_output=output,
            token_accounting_method="runtime_final_eval_count",
        )
        row["order"]["permutation_sha256"] = permutation
        rows.append(row)
        if (index + 1) % 6 == 0:
            _write(phase / f"raw_runs/transformer_{target_device}.json", {"format": FORMAT_RAW, "records": rows})
            print(f"transformer-{target_device} {index + 1}/{len(cells)}", flush=True)
    _write(phase / f"raw_runs/transformer_{target_device}.json", {"format": FORMAT_RAW, "records": rows})
    return {"system": system, "records": len(rows), "status": "PASS"}


def _headline_observations() -> list[tuple[dict[str, Any], int]]:
    prompts = _headline_prompts()
    values = [(prompt, 1) for prompt in prompts]
    values.extend((prompt, 2) for prompt in prompts[:HEADLINE_REPEAT_PROMPTS])
    random.Random(RANDOMIZATION_SEED).shuffle(values)
    return values


def _headline_layercake_rows(
    root: Path, *, system: str, start_index: int = 0
) -> list[dict[str, Any]]:
    phase = _path(root, PHASE)
    manifest = _read(phase / "model_manifests/layercake-foundation-v2.json")
    checkpoint = root / "artifacts/final/medium-cores/seed-9801"
    torch.set_num_threads(1)
    model, _ = load_core_checkpoint(checkpoint, device="cpu")
    model.eval()
    with torch.inference_mode():
        warm = model.prefill(torch.tensor([[65]], dtype=torch.long))
        _, warm = model.decode_step(warm)
    process = psutil.Process()
    rows = []
    observations = _headline_observations()
    permutation = _canonical_sha([(item[0]["id"], item[1]) for item in observations])
    for offset, (prompt, trial) in enumerate(observations):
        started = time.perf_counter_ns()
        prompt_started = started
        with torch.inference_mode():
            state = model.prefill(torch.tensor([list(prompt["text"].encode())], dtype=torch.long))
        prefill_done = time.perf_counter_ns()
        generated = bytearray()
        first = 0
        with torch.inference_mode():
            while len(generated) < HEADLINE_OUTPUT_BYTES:
                selected = state.next_logits.argmax(-1)
                generated.append(int(selected.item()))
                if not first:
                    first = time.perf_counter_ns()
                _, state = model.decode_step(state, next_byte=selected)
        completed = time.perf_counter_ns()
        resident, peak = _process_memory([process])
        row = _base_row(
            index=start_index + offset, system=system, runtime="pytorch-foundation-v2",
            model=manifest, device="cpu_one_thread", threads=1, cache="warm",
            mode="deterministic", bucket="headline", target=HEADLINE_OUTPUT_BYTES,
            trial=trial, output=bytes(generated), generated_tokens=len(generated),
            started=started, first=first, completed=completed, resident=resident, peak=peak,
            command_id="phase1-headline-layercake", precision="fp32",
            phase_timings={
                "model_load_seconds": 0.0,
                "prompt_preprocessing_seconds": (prefill_done - prompt_started) / 1e9,
                "prefill_seconds": (prefill_done - prompt_started) / 1e9,
                "decode_seconds": (completed - prefill_done) / 1e9,
                "measurement": "direct persistent-state prefill/decode client clock",
            },
            accelerator={"status": "NOT_APPLICABLE_CPU", "method": "none", "peak_bytes": 0},
            prompt_record={**prompt, "bucket": "headline"},
        )
        row["order"]["permutation_sha256"] = permutation
        rows.append(row)
    return rows


def _headline_bpe_rows(root: Path, *, start_index: int = 0) -> list[dict[str, Any]]:
    phase = _path(root, PHASE)
    manifest = _read(phase / "model_manifests/bpe-reference.json")
    checkpoint = root / "artifacts/final/medium-transformers/seed-9801"
    torch.set_num_threads(1)
    model, tokenizer, _ = load_transformer_checkpoint(checkpoint, device="cpu")
    model.eval()
    with torch.inference_mode():
        warm = model.prefill(torch.tensor([[65]], dtype=torch.long))
        _, warm = model.decode_step(warm)
    process = psutil.Process()
    rows = []
    observations = _headline_observations()
    permutation = _canonical_sha([(item[0]["id"], item[1]) for item in observations])
    for offset, (prompt, trial) in enumerate(observations):
        prompt_ids = tokenizer.encode(prompt["text"])
        if len(prompt_ids) + HEADLINE_OUTPUT_BYTES >= model.config.max_tokens:
            raise RuntimeError(f"headline prompt exceeds BPE context: {prompt['id']}")
        started = time.perf_counter_ns()
        with torch.inference_mode():
            state = model.prefill(torch.tensor([prompt_ids], dtype=torch.long))
        prefill_done = time.perf_counter_ns()
        generated = bytearray()
        generated_ids = []
        first = 0
        with torch.inference_mode():
            while len(generated) < HEADLINE_OUTPUT_BYTES:
                selected = state.next_logits.argmax(-1)
                token_id = int(selected.item())
                generated_ids.append(token_id)
                generated.extend(tokenizer.decode([token_id]))
                if not first:
                    first = time.perf_counter_ns()
                _, state = model.decode_step(state, next_token=selected)
        completed = time.perf_counter_ns()
        resident, peak = _process_memory([process])
        row = _base_row(
            index=start_index + offset, system="transformer_same_scale_headline",
            runtime="pytorch-reference-controls", model=manifest,
            device="cpu_one_thread", threads=1, cache="warm", mode="deterministic",
            bucket="headline", target=HEADLINE_OUTPUT_BYTES, trial=trial,
            output=bytes(generated), generated_tokens=len(generated_ids),
            started=started, first=first, completed=completed, resident=resident, peak=peak,
            command_id="phase1-headline-bpe", precision="fp32",
            phase_timings={
                "model_load_seconds": 0.0,
                "prompt_preprocessing_seconds": 0.0,
                "prefill_seconds": (prefill_done - started) / 1e9,
                "decode_seconds": (completed - prefill_done) / 1e9,
                "measurement": "direct per-layer KV-cache prefill/decode client clock",
            },
            accelerator={"status": "NOT_APPLICABLE_CPU", "method": "none", "peak_bytes": 0},
            token_accounting_method="posthoc_tokenizer",
            prompt_record={**prompt, "bucket": "headline"},
        )
        row["order"]["permutation_sha256"] = permutation
        rows.append(row)
    return rows


def _headline_qwen_rows(
    root: Path, endpoint: str, model: str, *, start_index: int = 0
) -> list[dict[str, Any]]:
    phase = _path(root, PHASE)
    manifest = _read(phase / "model_manifests/qwen25-05b-cpu.json")
    physical = int(psutil.cpu_count(logical=False) or 1)
    _ollama_warm(endpoint, model, physical)
    rows = []
    observations = _headline_observations()
    permutation = _canonical_sha([(item[0]["id"], item[1]) for item in observations])
    for offset, (prompt, trial) in enumerate(observations):
        started = time.perf_counter_ns()
        output, token_count, first, target_completed, completed, final = _ollama_stream(
            endpoint, model, prompt["text"], target=HEADLINE_OUTPUT_BYTES,
            threads=physical, mode="deterministic", seed=RANDOMIZATION_SEED + trial,
        )
        resident, peak = _process_memory(_ollama_processes())
        row = _base_row(
            index=start_index + offset, system="transformer_product_headline",
            runtime="ollama-cpu", model=manifest, device="cpu_all_core", threads=physical,
            cache="warm", mode="deterministic", bucket="headline",
            target=HEADLINE_OUTPUT_BYTES, trial=trial, output=output,
            generated_tokens=token_count, started=started, first=first, completed=completed,
            resident=resident, peak=peak, command_id="phase1-headline-qwen", precision="Q4_K_M",
            phase_timings={
                "model_load_seconds": float(final.get("load_duration", 0)) / 1e9,
                "prompt_preprocessing_seconds": 0.0,
                "prefill_and_first_decode_seconds": float(final.get("prompt_eval_duration", 0)) / 1e9,
                "decode_after_first_seconds": float(final.get("eval_duration", 0)) / 1e9,
                "target_prefix_latency_seconds": (target_completed - started) / 1e9,
                "measurement": "one streaming request with terminal runtime token/duration record",
            },
            accelerator={"status": "NOT_APPLICABLE_CPU", "method": "none", "peak_bytes": 0},
            completed_output=output, token_accounting_method="runtime_final_eval_count",
            prompt_record={**prompt, "bucket": "headline"},
        )
        row["order"]["permutation_sha256"] = permutation
        rows.append(row)
        if (offset + 1) % 10 == 0:
            print(f"headline-qwen {offset + 1}/{len(observations)}", flush=True)
    return rows


def benchmark_headlines(root: Path, endpoint: str, model: str) -> dict[str, Any]:
    phase = _path(root, PHASE)
    same_layercake = _headline_layercake_rows(root, system="layercake_same_scale_headline")
    _write(phase / "raw_runs/headline_layercake_same_scale.json", {"format": FORMAT_RAW, "records": same_layercake})
    bpe = _headline_bpe_rows(root)
    _write(phase / "raw_runs/headline_bpe_same_scale.json", {"format": FORMAT_RAW, "records": bpe})
    product_layercake = _headline_layercake_rows(root, system="layercake_product_headline")
    _write(phase / "raw_runs/headline_layercake_product.json", {"format": FORMAT_RAW, "records": product_layercake})
    qwen = _headline_qwen_rows(root, endpoint, model)
    _write(phase / "raw_runs/headline_qwen_product.json", {"format": FORMAT_RAW, "records": qwen})
    return {
        "status": "PASS",
        "distinct_prompts": len(_headline_prompts()),
        "repeated_prompts_per_system": HEADLINE_REPEAT_PROMPTS,
        "records": len(same_layercake) + len(bpe) + len(product_layercake) + len(qwen),
    }


def capture_runtime(root: Path, endpoint: str, target_device: str) -> dict[str, Any]:
    if target_device not in {"cpu", "gpu"}:
        raise ValueError("target_device must be cpu or gpu")
    with urllib.request.urlopen(f"{endpoint}/api/ps", timeout=10) as response:
        process_state = json.load(response)
    models = process_state.get("models", [])
    if not models:
        raise RuntimeError(f"Ollama runtime at {endpoint} has no resident measured model")
    resident = models[0]
    size_vram = int(resident.get("size_vram", -1))
    observed = "cpu" if size_vram == 0 else "gpu"
    if observed != target_device:
        raise RuntimeError(
            f"runtime target mismatch at {endpoint}: requested {target_device}, observed {observed}"
        )
    nvidia = None
    if target_device == "gpu":
        code, stdout, stderr = _run([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        nvidia = {"exit_code": code, "stdout": stdout, "stderr": stderr}
    probe = {
        "format": "layercake-phase1-runtime-device-probe/1",
        "endpoint": endpoint, "requested_target": target_device,
        "observed_target": observed, "derivation": "Ollama /api/ps size_vram == 0 means CPU; size_vram > 0 means GPU",
        "process_state": process_state, "nvidia_compute_processes": nvidia,
        "clock": "time.time_ns", "captured_ns": time.time_ns(),
    }
    path = _path(root, PHASE / f"runtime_probe_{target_device}.json")
    _write(path, probe)
    return {"target": target_device, "observed": observed, "path": _relative(root, path), "sha256": _sha(path)}


def adversarial_checks(root: Path) -> dict[str, Any]:
    phase = _path(root, PHASE)
    raw_documents = [_read(path) for path in sorted((phase / "raw_runs").glob("*.json"))]
    rows = [row for document in raw_documents for row in document["records"]]
    matrix = _read(phase / "benchmark_matrix.json")
    hardware = _read(phase / "hardware.json")
    checks = []

    def detected(identifier: str, action) -> None:
        try:
            action()
        except Phase1EvidenceError as error:
            checks.append({"id": identifier, "status": "DETECTED", "error": str(error)})
            return
        raise RuntimeError(f"adversarial mutation escaped detection: {identifier}")

    first = rows[0]
    filtered = [
        row for row in rows
        if not (
            row["system_id"] == first["system_id"]
            and row["device"]["kind"] == first["device"]["kind"]
            and row["cache_state"]["kind"] == first["cache_state"]["kind"]
            and row["generation"]["mode"] == first["generation"]["mode"]
            and row["prompt"]["bucket"] == first["prompt"]["bucket"]
            and row["output"]["target_bytes"] == first["output"]["target_bytes"]
        )
    ]
    detected("missing_matrix_cell", lambda: validate_benchmark_matrix(matrix, filtered, hardware))

    timestamp_doc = copy.deepcopy(raw_documents[0])
    timestamp_doc["records"][0]["timing"]["total_latency_seconds"] += 1.0
    detected("tampered_direct_timestamp", lambda: validate_raw_timing_samples(timestamp_doc))

    boolean_doc = copy.deepcopy(raw_documents[0])
    boolean_doc["records"][0]["cold"] = True
    detected("self_asserted_cold_boolean", lambda: validate_raw_timing_samples(boolean_doc))

    evidence = copy.deepcopy(_read(phase / "evidence_manifest.json"))
    evidence["artifacts"][0]["sha256"] = "0" * 64
    detected("stale_artifact_hash", lambda: validate_evidence_manifest(evidence, root))

    runtime = _read(phase / "runtime_manifests/ollama-cpu.json")
    forged_runtime = copy.deepcopy(runtime)
    forged_runtime["optimization_evidence"]["kv_cache"]["raw_trace_run_ids"] = ["invented-run"]
    detected(
        "invented_kv_cache_trace",
        lambda: validate_baseline_optimization(forged_runtime, rows, runtime_id="ollama-cpu"),
    )
    boolean_runtime = copy.deepcopy(runtime)
    boolean_runtime["optimized"] = True
    detected("self_asserted_optimized_boolean", lambda: validate_runtime_manifest(boolean_runtime, optimized=True))

    corrected_doc = copy.deepcopy(raw_documents[0])
    corrected_doc["records"][0]["output"]["token_accounting"]["count"] += 1
    detected("forged_authoritative_token_count", lambda: validate_raw_timing_samples(corrected_doc))

    cold_document = copy.deepcopy(next(
        document for document in raw_documents
        if any(row["cache_state"]["kind"] == "cold" for row in document["records"])
    ))
    cold_row = next(row for row in cold_document["records"] if row["cache_state"]["kind"] == "cold")
    cold_row["cache_state"]["single_request_evidence"]["load_probe_requests"] = 1
    detected("extra_cold_load_probe", lambda: validate_raw_timing_samples(cold_document))

    shallow_rows = [
        row for row in rows
        if not (row["prompt"]["bucket"] == "headline" and row["prompt"]["id"] >= "functional-09-00")
    ]
    detected("shallow_promoted_headline", lambda: validate_benchmark_matrix(matrix, shallow_rows, hardware))

    comparison = copy.deepcopy(_read(phase / "comparison_certificate.json"))
    comparison["comparisons"][1]["transformer_model_id"] = "bpe-reference"
    models = {
        document["id"]: document
        for document in (_read(path) for path in (phase / "model_manifests").glob("*.json"))
    }
    detected(
        "mixed_product_quality_speed_lineage",
        lambda: validate_comparison_certificate(
            comparison, rows, models, _read(phase / "baseline_quality.json"),
            _read(phase / "functional_quality.json"),
        ),
    )

    result = {
        "format": "layercake-phase1-adversarial-checks/1",
        "status": "PASS", "checks": checks, "detected": len(checks),
    }
    _write(phase / "adversarial_checks.json", result)
    return result


def _output_quality(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        valid_utf8 = 1.0
    except UnicodeDecodeError:
        text = payload.decode("utf-8", errors="replace")
        valid_utf8 = 0.0
    characters = max(len(text), 1)
    printable = sum(character.isprintable() or character in "\n\r\t" for character in text) / characters
    raw_ngrams = [payload[index:index + 4] for index in range(max(0, len(payload) - 3))]
    unique_rate = len(set(raw_ngrams)) / max(len(raw_ngrams), 1)
    words = [word for word in text.lower().split() if word]
    word_diversity = len(set(words)) / max(len(words), 1)
    return {
        "valid_utf8": valid_utf8,
        "invalid_output": 1.0 - valid_utf8,
        "printable_character_rate": printable,
        "unique_4gram_rate": unique_rate,
        "repetition_rate": 1.0 - unique_rate,
        "word_diversity": word_diversity,
        "generated_characters": len(text),
    }


def _write_functional_quality(phase: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["system_id"] in {"layercake_product_headline", "transformer_product_headline"}
        and row["trial"] == 1 and row["status"] == "PASS"
    ]
    system_models = {
        "layercake_product_headline": "layercake-foundation-v2",
        "transformer_product_headline": "qwen25-05b-cpu",
    }
    systems = {}
    for system_id, model_id in system_models.items():
        system_rows = [row for row in selected if row["system_id"] == system_id]
        records = []
        for row in sorted(system_rows, key=lambda item: item["prompt"]["id"]):
            payload = bytes.fromhex(row["output"]["hex"])
            records.append({
                "prompt_id": row["prompt"]["id"],
                "prompt_sha256": row["prompt"]["sha256"],
                "run_id": row["run_id"],
                "output_sha256": row["output"]["sha256"],
                "metrics": _output_quality(payload),
            })
        if len(records) != 100:
            raise RuntimeError(f"functional quality for {system_id} has {len(records)} prompts")
        metric_names = tuple(records[0]["metrics"])
        aggregates = {
            name: sum(float(record["metrics"][name]) for record in records) / len(records)
            for name in metric_names
        }
        systems[model_id] = {
            "speed_system_id": system_id,
            "prompt_ids": [record["prompt_id"] for record in records],
            "records": records,
            "aggregates": aggregates,
        }
    document = {
        "format": "layercake-phase1-functional-quality/1",
        "correction_protocol": CORRECTION_PROTOCOL,
        "suite": "same 100 prompts and completed outputs used by the product speed headline",
        "systems": systems,
        "selection_use": False,
    }
    _write(phase / "functional_quality.json", document)
    return document


def _write_comparison_certificate(
    root: Path, phase: Path, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    quality_path = phase / "baseline_quality.json"
    functional_path = phase / "functional_quality.json"
    raw_by_system = {}
    for system_id in (
        "layercake_same_scale_headline", "transformer_same_scale_headline",
        "layercake_product_headline", "transformer_product_headline",
    ):
        run_ids = sorted(row["run_id"] for row in rows if row["system_id"] == system_id)
        raw_by_system[system_id] = {
            "run_ids": run_ids,
            "run_id_set_sha256": _canonical_sha(run_ids),
        }
    document = {
        "format": "layercake-phase1-comparison-certificate/1",
        "correction_protocol": CORRECTION_PROTOCOL,
        "cross_model_primary_throughput_metrics": ["bytes_per_second", "characters_per_second"],
        "comparisons": [
            {
                "kind": "same_scale_architecture",
                "layercake_model_id": "layercake-foundation-v2",
                "transformer_model_id": "bpe-reference",
                "layercake_speed_system_id": "layercake_same_scale_headline",
                "transformer_speed_system_id": "transformer_same_scale_headline",
                "quality_protocol": "heldout_bpb_same_checkpoint",
                "quality_artifact": {"path": _relative(root, quality_path), "sha256": _sha(quality_path)},
                "speed_evidence": {
                    "layercake": raw_by_system["layercake_same_scale_headline"],
                    "transformer": raw_by_system["transformer_same_scale_headline"],
                },
            },
            {
                "kind": "product",
                "layercake_model_id": "layercake-foundation-v2",
                "transformer_model_id": "qwen25-05b-cpu",
                "layercake_speed_system_id": "layercake_product_headline",
                "transformer_speed_system_id": "transformer_product_headline",
                "quality_protocol": "functional_output_suite",
                "quality_artifact": {"path": _relative(root, functional_path), "sha256": _sha(functional_path)},
                "speed_evidence": {
                    "layercake": raw_by_system["layercake_product_headline"],
                    "transformer": raw_by_system["transformer_product_headline"],
                },
            },
        ],
        "prohibited_combination": (
            "same-scale BPE quality is never attached to Qwen speed; each comparison binds quality "
            "and speed to the transformer_model_id named in that comparison"
        ),
    }
    _write(phase / "comparison_certificate.json", document)
    return document


def finalize(root: Path) -> dict[str, Any]:
    phase = _path(root, PHASE)
    raw_paths = sorted((phase / "raw_runs").glob("*.json"))
    rows = [row for path in raw_paths for row in _read(path)["records"]]
    if not rows:
        raise RuntimeError("no raw rows to finalize")
    by_runtime: dict[str, list[dict[str, Any]]] = {}
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_runtime.setdefault(row["runtime_id"], []).append(row)
        by_model.setdefault(row["model_id"], []).append(row)
    for path in (phase / "runtime_manifests").glob("*.json"):
        manifest = _read(path)
        runtime_rows = by_runtime.get(manifest["id"], [])
        if "optimization_evidence" in manifest and runtime_rows:
            ids = [row["run_id"] for row in runtime_rows]
            manifest["optimization_evidence"]["kv_cache"]["raw_trace_run_ids"] = ids[:2]
            manifest["optimization_evidence"]["threading"]["raw_trace_run_ids"] = ids
            target = manifest["target_device"]
            probe_path = phase / f"runtime_probe_{target}.json"
            probe = _read(probe_path)
            manifest["optimization_evidence"]["device_probe"] = {
                "path": _relative(root, probe_path), "sha256": _sha(probe_path),
                "observed_target": probe["observed_target"],
            }
            _write(path, manifest)
    for path in (phase / "model_manifests").glob("*.json"):
        manifest = _read(path)
        model_rows = by_model.get(manifest["id"], [])
        if model_rows and manifest["incremental_state"]["status"] == "MEASURED":
            manifest["incremental_state"]["raw_trace_run_ids"] = [row["run_id"] for row in model_rows[:2]]
            _write(path, manifest)
    inventory_path = phase / "baseline_inventory.json"
    inventory = _read(inventory_path)
    for baseline in inventory["baselines"]:
        model_path = root / baseline["model_manifest"]["path"]
        runtime_path = root / baseline["runtime"]["runtime_manifest"]["path"]
        baseline["model_manifest"]["sha256"] = _sha(model_path)
        baseline["runtime"]["runtime_manifest"]["sha256"] = _sha(runtime_path)
        if "kv_cache_evidence" in baseline["runtime"]:
            baseline["runtime"]["kv_cache_evidence"]["sha256"] = _sha(runtime_path)
    _write(inventory_path, inventory)
    commands = {
        "format": "layercake-phase1-execution-commands/1",
        "commands": [
            {
                "id": "phase1-layercake-direct", "executable": sys.executable,
                "arguments": ["-m", "layercake.phase1_campaign", "benchmark-layercake"],
                "configuration_sha256": _sha(phase / "benchmark_config.json"),
            },
            {
                "id": "phase1-ollama-cpu", "executable": sys.executable,
                "arguments": ["-m", "layercake.phase1_campaign", "benchmark-ollama", "--device", "cpu", "--endpoint", "http://127.0.0.1:11435"],
                "configuration_sha256": _sha(phase / "benchmark_config.json"),
            },
            {
                "id": "phase1-ollama-gpu", "executable": sys.executable,
                "arguments": ["-m", "layercake.phase1_campaign", "benchmark-ollama", "--device", "gpu", "--endpoint", "http://127.0.0.1:11434"],
                "configuration_sha256": _sha(phase / "benchmark_config.json"),
            },
            {
                "id": "phase1-headline-layercake", "executable": sys.executable,
                "arguments": ["-m", "layercake.phase1_campaign", "benchmark-headlines"],
                "configuration_sha256": _sha(phase / "benchmark_config.json"),
            },
            {
                "id": "phase1-headline-bpe", "executable": sys.executable,
                "arguments": ["-m", "layercake.phase1_campaign", "benchmark-headlines"],
                "configuration_sha256": _sha(phase / "benchmark_config.json"),
            },
            {
                "id": "phase1-headline-qwen", "executable": sys.executable,
                "arguments": ["-m", "layercake.phase1_campaign", "benchmark-headlines", "--endpoint", "http://127.0.0.1:11435"],
                "configuration_sha256": _sha(phase / "benchmark_config.json"),
            },
        ],
    }
    _write(phase / "execution_commands.json", commands)
    performance = derive_performance(rows)
    performance["source_raw_files"] = [
        {"path": _relative(root, path), "sha256": _sha(path)} for path in raw_paths
    ]
    performance["measurement_scope"] = {
        "energy": "NOT_MEASURED_NO_CALIBRATED_ENERGY_INTERFACE",
        "gpu_process_memory": "Ollama process RSS is separate from CUDA allocator; LayerCake reports both process and torch allocator",
    }
    _write(phase / "baseline_performance.json", performance)
    bpe = _read(root / "artifacts/final/medium-transformers/seed-9801/metadata.json")
    lc = _read(root / "artifacts/final/medium-cores/seed-9801/metadata.json")
    adaptive = _read(root / "artifacts/final/adaptive-medium-pilot/routed_adaptive_5x5_top1_8e/seed-9811/metadata.json")
    _write(phase / "baseline_quality.json", {
        "format": "layercake-phase1-baseline-quality/1",
        "scope": "historical validation metrics bound to exact checkpoint metadata; no Phase 1 test-set selection",
        "records": [
            {"model_id": "bpe-reference", "validation": bpe["quality"]["validation"], "metadata_sha256": _sha(root / "artifacts/final/medium-transformers/seed-9801/metadata.json")},
            {"model_id": "layercake-foundation-v2", "validation": lc["quality"]["validation"], "metadata_sha256": _sha(root / "artifacts/final/medium-cores/seed-9801/metadata.json")},
            {"model_id": "layercake-adaptive-quality", "validation": adaptive["quality"]["validation"], "metadata_sha256": _sha(root / "artifacts/final/adaptive-medium-pilot/routed_adaptive_5x5_top1_8e/seed-9811/metadata.json")},
            {"model_id": "qwen25-05b", "validation": "NOT_COMPARABLE_NO_TOKENIZER_NEUTRAL_LOGPROB_API_IN_OLLAMA", "metadata_sha256": _sha(phase / "ollama_model_show.json")},
        ],
        "test_accessed_for_selection": False,
    })
    _write_functional_quality(phase, rows)
    _write_comparison_certificate(root, phase, rows)
    junit_path = phase / "pytest.xml"
    if not junit_path.is_file():
        raise RuntimeError("Phase 1 complete-suite JUnit evidence is missing")
    junit_root = ET.parse(junit_path).getroot()
    suites = [junit_root] if junit_root.tag == "testsuite" else list(junit_root.findall("testsuite"))
    totals = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    _write(phase / "test_results.json", {
        "format": "layercake-phase1-test-results/1",
        "status": "PASS" if totals["tests"] > 0 and totals["failures"] == totals["errors"] == 0 else "FAIL",
        **totals,
        "passed": totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"],
        "command": f'"{sys.executable}" -m pytest -q --junitxml=results/moonshot/phase1/pytest.xml',
        "junit_path": _relative(root, junit_path), "junit_sha256": _sha(junit_path),
    })
    excluded = {"evidence_manifest.json", "candidate.json", "candidate_verification.json", "release_certificate.json", "handoff.json", "seal.json"}
    artifacts = sorted(
        path for path in phase.rglob("*")
        if path.is_file() and not (path.parent == phase and path.name in excluded)
    )
    raw_rows = sorted(
        (_relative(root, path), _sha(path)) for path in artifacts
        if "raw_runs" in path.parts and "history" not in path.parts
    )
    evidence = {
        "format": "layercake-phase1-evidence-manifest/1",
        "artifacts": [{"path": _relative(root, path), "sha256": _sha(path)} for path in artifacts],
        "raw_evidence_manifest_sha256": _canonical_sha(raw_rows),
    }
    _write(phase / "evidence_manifest.json", evidence)
    summary = validate_phase1_bundle(root, phase)
    concise = {key: value for key, value in summary.items() if key != "derived_performance"}
    return {"phase": 1, "status": "PASS", "raw_run_count": len(rows), "validation": concise}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m layercake.phase1_campaign")
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    prepare_parser.add_argument("--model", default="qwen2.5:0.5b")
    correction_parser = sub.add_parser("prepare-correction")
    correction_parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    correction_parser.add_argument("--model", default="qwen2.5:0.5b")
    sub.add_parser("benchmark-layercake")
    ollama_parser = sub.add_parser("benchmark-ollama")
    ollama_parser.add_argument("--endpoint", required=True)
    ollama_parser.add_argument("--model", default="qwen2.5:0.5b")
    ollama_parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    headline_parser = sub.add_parser("benchmark-headlines")
    headline_parser.add_argument("--endpoint", required=True)
    headline_parser.add_argument("--model", default="qwen2.5:0.5b")
    capture_parser = sub.add_parser("capture-runtime")
    capture_parser.add_argument("--endpoint", required=True)
    capture_parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    sub.add_parser("finalize")
    sub.add_parser("adversarial-checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare":
        result = prepare(root, args.endpoint, args.model)
    elif args.command == "prepare-correction":
        result = prepare(root, args.endpoint, args.model, preserve_history=True)
    elif args.command == "benchmark-layercake":
        result = benchmark_layercake(root)
    elif args.command == "benchmark-ollama":
        result = benchmark_ollama(root, args.endpoint, args.model, args.device)
    elif args.command == "benchmark-headlines":
        result = benchmark_headlines(root, args.endpoint, args.model)
    elif args.command == "capture-runtime":
        result = capture_runtime(root, args.endpoint, args.device)
    elif args.command == "finalize":
        result = finalize(root)
    elif args.command == "adversarial-checks":
        result = adversarial_checks(root)
    else:  # pragma: no cover
        raise RuntimeError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
