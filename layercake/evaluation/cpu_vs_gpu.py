"""Direct CPU LayerCake versus cached GPU transformer benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
import time

import psutil
import torch
from safetensors.torch import load_file

from layercake.cake.package import load_package
from layercake.models.portable_decoder import load_cake_module
from layercake.routing.learned_router import CompactSemanticRouter
from layercake.runtime.cpu import configure_cpu, parameter_bytes
from layercake.training.baseline import load_transformer_checkpoint
from layercake.training.foundation import _config, load_core_checkpoint


def _prompts(count: int) -> list[dict]:
    general = [
        "Explain why patient observation matters when resolving a disagreement.",
        "Write a concise history of navigation and its effect on trade.",
        "Compare two ways to organize a community project fairly.",
        "Describe a rainy evening from the perspective of a train conductor.",
        "Summarize the causes and consequences of urban migration.",
    ]
    python = [
        "Implement an asynchronous bounded-concurrency crawler in Python.",
        "Repair a Python generator that accidentally retains object references.",
        "Write a typed context manager with reliable cleanup behavior.",
        "Explain and fix a race condition between two asyncio tasks.",
        "Implement a streaming parser that handles incomplete UTF-8 input.",
    ]
    mixed = [
        "Explain the probability argument, then implement its simulation in Python.",
        "Describe the user workflow and emit Python for its structured action.",
        "Analyze a clinical cohort and write Python to compute the endpoint.",
        "Plan the game encounter and implement a Python state machine for it.",
        "Derive the recurrence and write Python that evaluates it safely.",
    ]
    rows = []
    groups = (("general", general, 40), ("python", python, 40), ("mixed", mixed, 20))
    for group, values, group_count in groups:
        for index in range(group_count):
            length_bucket = (32, 64, 128)[index % 3]
            base = values[index % len(values)]
            filler = " Provide assumptions and check the result." * 20
            prompt = (base + filler)[:length_bucket]
            rows.append({
                "id": f"{group}-{index:03d}", "group": group,
                "prompt": prompt, "prompt_bytes": len(prompt.encode("utf-8")),
                "generated_bytes": (256, 384)[index % 2],
                "sampling": "greedy" if index % 2 == 0 else "sampled",
            })
    if len(rows) != count:
        raise ValueError("locked prompt generator count mismatch")
    return rows


def _stats(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "p50": statistics.median(values),
        "p95": ordered[round(0.95 * (len(ordered) - 1))],
        "p99": ordered[round(0.99 * (len(ordered) - 1))],
        "mean": statistics.fmean(values),
    }


def _sample(logits: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(seed)
    return torch.multinomial(torch.softmax(logits.float(), dim=-1), 1, generator=generator).flatten()


@torch.inference_mode()
def benchmark_cpu_vs_gpu(
    config_path: str | Path,
    *,
    core_dir: str | Path,
    cake_path: str | Path,
    public_key_path: str | Path,
    transformer_dir: str | Path,
    router_path: str | Path,
    output_path: str | Path,
) -> dict:
    config = _config(config_path)
    prompts = _prompts(int(config["prompts"]))
    repeats = int(config["repetitions"])
    configure_cpu(int(config["cpu_threads"]))

    started = time.perf_counter_ns()
    core, core_metadata = load_core_checkpoint(core_dir, device="cpu")
    core_load_ms = (time.perf_counter_ns() - started) / 1_000_000
    import zipfile
    from layercake.cake.manifest import CakeManifest
    with zipfile.ZipFile(cake_path) as archive:
        key_id = CakeManifest.from_json(archive.read("manifest.json")).signature["key_id"]
    started = time.perf_counter_ns()
    package = load_package(cake_path, trust_store={key_id: public_key_path})
    package_verify_ms = (time.perf_counter_ns() - started) / 1_000_000
    started = time.perf_counter_ns()
    cake = load_cake_module(package).cpu().eval()
    cake_load_ms = (time.perf_counter_ns() - started) / 1_000_000
    router = CompactSemanticRouter()
    router.load_state_dict(load_file(str(router_path)), strict=True)
    router.eval()

    if not torch.cuda.is_available():
        raise RuntimeError("the locked CPU-versus-GPU benchmark requires CUDA")
    torch.cuda.empty_cache()
    started = time.perf_counter_ns()
    transformer, tokenizer, transformer_metadata = load_transformer_checkpoint(
        transformer_dir, device="cuda"
    )
    transformer.half().eval()
    torch.cuda.synchronize()
    transformer_load_ms = (time.perf_counter_ns() - started) / 1_000_000
    torch.cuda.reset_peak_memory_stats()

    schedule = [(system, row, repeat) for row in prompts for repeat in range(repeats) for system in ("layercake_cpu", "transformer_gpu")]
    random.Random(int(config["execution_order_seed"])).shuffle(schedule)
    rows = []
    route_times = []
    for system, item, repeat in schedule:
        prompt = item["prompt"]
        target_bytes = int(item["generated_bytes"])
        sampled = item["sampling"] == "sampled"
        if system == "layercake_cpu":
            end_to_end = time.perf_counter_ns()
            route_started = time.perf_counter_ns()
            route = router.route(prompt, installed={"python"}, top_k=1)
            route_ms = (time.perf_counter_ns() - route_started) / 1_000_000
            route_times.append(route_ms)
            fusion = cake if "python" in route.selected else None
            prefill_started = time.perf_counter_ns()
            state = core.prefill(prompt, route=int(core_metadata["route"]), fusion_cake=fusion)
            prefill_ms = (time.perf_counter_ns() - prefill_started) / 1_000_000
            decode_started = time.perf_counter_ns()
            for index in range(target_bytes):
                next_byte = _sample(state.next_logits, 9800 + repeat * 1000 + index) if sampled else None
                _, state = core.decode_step(state, next_byte=next_byte, fusion_cake=fusion)
            decode_ms = (time.perf_counter_ns() - decode_started) / 1_000_000
            total_ms = (time.perf_counter_ns() - end_to_end) / 1_000_000
            useful_bytes = target_bytes
            generated_units = target_bytes
            route_selected = list(route.selected)
        else:
            tokenization_started = time.perf_counter_ns()
            prompt_ids = tokenizer.encode(prompt)
            tokenization_ms = (time.perf_counter_ns() - tokenization_started) / 1_000_000
            ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
            torch.cuda.synchronize()
            end_to_end = time.perf_counter_ns()
            prefill_started = time.perf_counter_ns()
            state = transformer.prefill(ids)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter_ns() - prefill_started) / 1_000_000
            decode_started = time.perf_counter_ns()
            useful_bytes = 0
            generated_units = 0
            while useful_bytes < target_bytes:
                next_token = _sample(state.next_logits, 9900 + repeat * 1000 + generated_units) if sampled else None
                selected = state.next_logits.argmax(-1) if next_token is None else next_token
                _, state = transformer.decode_step(state, selected)
                useful_bytes += len(tokenizer.pieces[int(selected.item())])
                generated_units += 1
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter_ns() - decode_started) / 1_000_000
            total_ms = tokenization_ms + (time.perf_counter_ns() - end_to_end) / 1_000_000
            route_ms = 0.0
            route_selected = []
        rows.append({
            **{key: item[key] for key in ("id", "group", "prompt_bytes", "generated_bytes", "sampling")},
            "system": system, "repeat": repeat, "route_milliseconds": route_ms,
            "selected": route_selected, "prefill_milliseconds": prefill_ms,
            "decode_milliseconds": decode_ms, "end_to_end_milliseconds": total_ms,
            "useful_generated_bytes": useful_bytes, "generated_model_units": generated_units,
            "useful_bytes_per_second": useful_bytes / (decode_ms / 1000),
            "time_to_first_output_milliseconds": total_ms - decode_ms + decode_ms / generated_units,
        })
    systems = {}
    for system in ("layercake_cpu", "transformer_gpu"):
        selected = [row for row in rows if row["system"] == system]
        systems[system] = {
            metric: _stats([row[metric] for row in selected])
            for metric in (
                "prefill_milliseconds", "decode_milliseconds", "end_to_end_milliseconds",
                "useful_bytes_per_second", "time_to_first_output_milliseconds",
            )
        }
        systems[system]["groups"] = {
            group: {
                "useful_bytes_per_second": _stats([
                    row["useful_bytes_per_second"] for row in selected if row["group"] == group
                ]),
                "end_to_end_milliseconds": _stats([
                    row["end_to_end_milliseconds"] for row in selected if row["group"] == group
                ]),
            }
            for group in ("general", "python", "mixed")
        }
    cpu_rate = systems["layercake_cpu"]["useful_bytes_per_second"]["p50"]
    gpu_rate = systems["transformer_gpu"]["useful_bytes_per_second"]["p50"]
    cpu_latency = systems["layercake_cpu"]["end_to_end_milliseconds"]["p50"]
    gpu_latency = systems["transformer_gpu"]["end_to_end_milliseconds"]["p50"]
    layercake_quality = {
        "general_bpb": core_metadata["quality"]["test"]["bits_per_byte"],
        "python_bpb": json.loads(Path(cake_path).with_suffix(".evidence.json").read_text())["evaluation"]["heldout_domain"]["cake_bits_per_byte"],
        "python_task_parse_success": json.loads(Path(cake_path).with_suffix(".evidence.json").read_text())["evaluation"]["syntax_tasks"]["cake_parse_success_rate"],
    }
    transformer_quality = {
        "general_bpb": transformer_metadata["quality"]["general_test"]["bits_per_byte"],
        "python_bpb": transformer_metadata["quality"]["python_test"]["bits_per_byte"],
        "python_task_parse_success": None,
    }
    bpb_comparable = all((
        layercake_quality["general_bpb"] <= config["quality_thresholds"]["general_bpb"],
        transformer_quality["general_bpb"] <= config["quality_thresholds"]["general_bpb"],
        layercake_quality["python_bpb"] <= config["quality_thresholds"]["python_bpb"],
        transformer_quality["python_bpb"] <= config["quality_thresholds"]["python_bpb"],
    ))
    coverage_valid = int(config["required_distinct_specialist_domains"]) <= 1
    ordinary_task_valid = layercake_quality["python_task_parse_success"] > 0
    performance_pass = cpu_rate >= gpu_rate and cpu_latency <= gpu_latency
    status = "PASS" if bpb_comparable and coverage_valid and ordinary_task_valid and performance_pass else "INVALID_EVIDENCE"
    evidence = {
        "format": "layercake-cpu-vs-gpu/2",
        "status": status,
        "performance_condition_passed": performance_pass,
        "bpb_thresholds_passed": bpb_comparable,
        "ordinary_task_quality_valid": ordinary_task_valid,
        "domain_coverage_valid": coverage_valid,
        "locked_specification": config,
        "quality": {"layercake": layercake_quality, "transformer": transformer_quality},
        "headline": {
            "cpu_over_gpu_useful_byte_throughput": cpu_rate / gpu_rate,
            "cpu_over_gpu_end_to_end_latency": cpu_latency / gpu_latency,
        },
        "systems": systems,
        "cold_start": {
            "core_load_milliseconds": core_load_ms,
            "package_verification_milliseconds": package_verify_ms,
            "cake_load_milliseconds": cake_load_ms,
            "transformer_load_milliseconds": transformer_load_ms,
        },
        "routing": {
            "milliseconds": _stats(route_times),
            "overhead_fraction_of_warm_layercake_latency": statistics.median(route_times) / cpu_latency,
        },
        "parameters": {
            "layercake_total_installed": core_metadata["parameters"]["total_parameters"] + sum(p.numel() for p in cake.parameters()),
            "layercake_active": core_metadata["parameters"]["active_parameters"] + sum(p.numel() for p in cake.parameters()),
            "transformer_total_active": transformer_metadata["parameters"],
            "same_total_scale_relative_delta": abs(core_metadata["parameters"]["total_parameters"] - transformer_metadata["parameters"]) / transformer_metadata["parameters"],
        },
        "memory": {
            "layercake_parameter_bytes": parameter_bytes(core) + parameter_bytes(cake),
            "transformer_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "process_rss_bytes": psutil.Process().memory_info().rss,
        },
        "hardware": {
            "cpu_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_precision": "fp16",
            "energy": "NOT_MEASURED_NO_POWER_METER",
            "temperature": "NOT_MEASURED",
        },
        "raw_rows": rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence
