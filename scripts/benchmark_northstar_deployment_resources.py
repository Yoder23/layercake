from __future__ import annotations

import argparse
import gc
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_cpu_deployment_resources import memory_info
from eval_schema_action_generation import (
    _generate_bpe,
    _generate_layercake,
    _load_bpe,
    _load_layercake,
)
from layercake.deployment import PatchGenerationDeployment


PROMPT = (
    'Question: Convert XML node <item id="42">ok</item> to canonical JSON. '
    "Answer: "
)


def _serialized_deployment_bytes(
    *,
    model_name: str,
    checkpoint: dict,
    model: torch.nn.Module,
    dynamic_int8: bool,
) -> int:
    payload: dict = {
        "model_config": checkpoint["model_config"],
        "model": model.state_dict(),
        "dynamic_int8": dynamic_int8,
    }
    if model_name == "transformer":
        payload["training_config"] = {
            "seq_len": checkpoint["training_config"].get("seq_len", 256)
        }
        payload["tokenizer_model"] = checkpoint["tokenizer_model"]
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.tell()


def _quantize_in_place(model_name: str, model: torch.nn.Module) -> torch.nn.Module:
    module_types: set[type[torch.nn.Module]] = {torch.nn.Linear}
    if model_name == "transformer":
        torch.backends.mha.set_fastpath_enabled(False)
    converted = torch.ao.quantization.quantize_dynamic(
        model,
        module_types,
        dtype=torch.qint8,
        inplace=True,
    )
    return model if converted is None else converted


@torch.inference_mode()
def run_child(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    baseline = memory_info()
    artifact_path = Path(args.artifact)
    if args.model == "layercake":
        checkpoint, model = _load_layercake(artifact_path, device)
        tokenizer = None
    else:
        checkpoint, model, tokenizer = _load_bpe(artifact_path, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    after_fp32_load = memory_info()

    if args.dynamic_int8:
        if args.model == "layercake":
            model = PatchGenerationDeployment(model)
        model = _quantize_in_place(args.model, model)
    model.eval()
    gc.collect()
    after_deployment_conversion = memory_info()
    deployment_bytes = _serialized_deployment_bytes(
        model_name=args.model,
        checkpoint=checkpoint,
        model=model,
        dynamic_int8=bool(args.dynamic_int8),
    )
    del checkpoint
    gc.collect()
    after_checkpoint_release = memory_info()

    timings: list[float] = []
    emitted_bytes: list[int] = []
    for _ in range(max(args.repeats, 1) + 1):
        if args.model == "layercake":
            text, seconds = _generate_layercake(
                model,
                PROMPT,
                max_new_bytes=args.generated_bytes,
                no_repeat_ngram=0,
                device=device,
                neural_mode="patch",
                structured_schema_head=False,
                direct_domain_cache=False,
                stop_after_json=False,
            )
        else:
            assert tokenizer is not None
            text, seconds = _generate_bpe(
                model,
                tokenizer,
                PROMPT,
                max_new_bytes=args.generated_bytes,
                no_repeat_ngram=0,
                device=device,
                stop_after_json=False,
            )
        if timings:
            timings.append(seconds)
            emitted_bytes.append(len(text.encode("utf-8", errors="replace")))
        else:
            # First invocation is an unreported warm-up.
            timings.append(seconds)
            emitted_bytes.append(len(text.encode("utf-8", errors="replace")))
    timings = timings[1:]
    emitted_bytes = emitted_bytes[1:]
    after_generation = memory_info()
    median_seconds = statistics.median(timings)
    median_emitted = int(statistics.median(emitted_bytes))
    return {
        "model": args.model,
        "dynamic_int8": bool(args.dynamic_int8),
        "dynamic_int8_module_types": ["Linear"] if args.dynamic_int8 else [],
        "dynamic_int8_engine": (
            torch.backends.quantized.engine if args.dynamic_int8 else None
        ),
        "deployment_scope": (
            "global autoregressive patch generation only"
            if args.model == "layercake"
            else "autoregressive transformer generation"
        ),
        "parameter_count_fp32_source": parameter_count,
        "source_checkpoint_bytes": artifact_path.stat().st_size,
        "deployment_artifact_bytes": deployment_bytes,
        "memory": {
            "baseline": baseline,
            "after_fp32_load": after_fp32_load,
            "after_deployment_conversion": after_deployment_conversion,
            "after_checkpoint_release": after_checkpoint_release,
            "after_generation": after_generation,
            "deployment_rss_increment_bytes": max(
                after_generation["rss_bytes"] - baseline["rss_bytes"],
                0,
            ),
            "deployment_private_increment_bytes": max(
                after_generation["private_bytes"] - baseline["private_bytes"],
                0,
            ),
            "process_peak_rss_bytes": after_generation["peak_rss_bytes"],
            "process_peak_private_bytes": after_generation[
                "peak_private_bytes"
            ],
        },
        "generation": {
            "repeats": args.repeats,
            "generated_bytes": median_emitted,
            "median_seconds": median_seconds,
            "bytes_per_second": median_emitted / max(median_seconds, 1e-12),
            "all_seconds": timings,
        },
    }


def _invoke_child(
    *,
    model: str,
    artifact: Path,
    generated_bytes: int,
    repeats: int,
    dynamic_int8: bool,
) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--model",
        model,
        "--artifact",
        str(artifact),
        "--generated-bytes",
        str(generated_bytes),
        "--repeats",
        str(repeats),
    ]
    if dynamic_int8:
        command.append("--dynamic-int8")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isolated-process Layercake/transformer deployment resource gate"
    )
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--model", choices=["layercake", "transformer"])
    parser.add_argument("--artifact", type=Path)
    parser.add_argument(
        "--layercake",
        type=Path,
        default=Path("runs_experiment/northstar_v21_semantic_pointer/latest.pt"),
    )
    parser.add_argument(
        "--transformer",
        type=Path,
        default=Path("runs_experiment/northstar_v22_fair_corrected_bpe/latest.pt"),
    )
    parser.add_argument("--generated-bytes", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dynamic-int8", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/breakthrough_equal/northstar_v22_deployment_resources.json"
        ),
    )
    args = parser.parse_args()

    if args.child:
        if args.model is None or args.artifact is None:
            raise SystemExit("--child requires --model and --artifact")
        print(json.dumps(run_child(args), sort_keys=True))
        return 0

    layercake = _invoke_child(
        model="layercake",
        artifact=args.layercake,
        generated_bytes=args.generated_bytes,
        repeats=args.repeats,
        dynamic_int8=args.dynamic_int8,
    )
    transformer = _invoke_child(
        model="transformer",
        artifact=args.transformer,
        generated_bytes=args.generated_bytes,
        repeats=args.repeats,
        dynamic_int8=args.dynamic_int8,
    )
    artifact_ratio = layercake["deployment_artifact_bytes"] / max(
        transformer["deployment_artifact_bytes"], 1
    )
    rss_ratio = layercake["memory"]["deployment_rss_increment_bytes"] / max(
        transformer["memory"]["deployment_rss_increment_bytes"], 1
    )
    private_ratio = layercake["memory"][
        "deployment_private_increment_bytes"
    ] / max(transformer["memory"]["deployment_private_increment_bytes"], 1)
    peak_rss_ratio = layercake["memory"]["process_peak_rss_bytes"] / max(
        transformer["memory"]["process_peak_rss_bytes"], 1
    )
    peak_private_ratio = layercake["memory"][
        "process_peak_private_bytes"
    ] / max(transformer["memory"]["process_peak_private_bytes"], 1)
    speed_ratio = layercake["generation"]["bytes_per_second"] / max(
        transformer["generation"]["bytes_per_second"], 1e-12
    )
    gates = {
        "measured_in_isolated_processes": True,
        "same_quantization_policy": bool(args.dynamic_int8),
        "layercake_deployment_artifact_smaller": artifact_ratio < 1.0,
        "layercake_peak_rss_lower": peak_rss_ratio < 1.0,
        "layercake_peak_private_memory_lower": peak_private_ratio < 1.0,
        "layercake_generation_at_least_5x": speed_ratio >= 5.0,
    }
    failed = [name for name, passed in gates.items() if not passed]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "scope": (
            "One-thread x86 CPU deployment proxy measured in separate fresh "
            "processes. It is not an ARM phone, NPU, battery, or thermal test."
        ),
        "required_gates": gates,
        "failed_required": failed,
        "metrics": {
            "deployment_artifact_ratio_layercake_over_transformer": artifact_ratio,
            "deployment_rss_increment_ratio_layercake_over_transformer": rss_ratio,
            "deployment_private_increment_ratio_layercake_over_transformer": (
                private_ratio
            ),
            "peak_rss_ratio_layercake_over_transformer": peak_rss_ratio,
            "peak_private_memory_ratio_layercake_over_transformer": (
                peak_private_ratio
            ),
            "generation_speed_ratio_layercake_over_transformer": speed_ratio,
        },
        "layercake": layercake,
        "transformer": transformer,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
