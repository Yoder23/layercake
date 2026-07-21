from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_bpe_baseline import BPETokenLM
from layercake.causal_byte_models import CausalBytePatchLM


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def memory_info() -> dict[str, int]:
    if sys.platform != "win32":
        # ru_maxrss is kilobytes on Linux, bytes on macOS. This branch is only
        # a fallback; the current benchmark machine is Windows.
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        return {
            "rss_bytes": peak,
            "peak_rss_bytes": peak,
            "private_bytes": peak,
            "peak_private_bytes": peak,
        }
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise ctypes.WinError()
    return {
        "rss_bytes": int(counters.WorkingSetSize),
        "peak_rss_bytes": int(counters.PeakWorkingSetSize),
        "private_bytes": int(counters.PagefileUsage),
        "peak_private_bytes": int(counters.PeakPagefileUsage),
    }


def parameter_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def artifact_bytes(path: Path) -> int:
    return path.stat().st_size


def build_patch_model_only(artifact: dict, device: torch.device) -> CausalBytePatchLM:
    args = artifact["args"]
    model = CausalBytePatchLM(
        patch_size=args.get("patch_size", 4),
        d_byte=args.get("d_byte", 48),
        d_model=args.get("patch_d_model") or args.get("d_model", 128),
        d_abi=args.get("d_abi", 64),
        layers=args.get("patch_layers") or args.get("layers", 3),
        heads=args.get("patch_heads") or args.get("heads", 4),
        max_patches=args.get("seq", 128) // args.get("patch_size", 4),
        continuous_local=args.get("continuous_local", False),
        direct_global_context=args.get("direct_global_context", False),
        ngram_buckets=args.get("ngram_buckets", 0),
        local_decoder=args.get("local_decoder", "gru"),
        conv_layers=args.get("conv_layers", 4),
        mtp_depth=args.get("mtp_depth", 0),
        patch_unit_buckets=args.get("patch_unit_buckets", 0),
        local_layers=args.get("local_layers", 2),
        local_width=args.get("local_width", 0),
        dropout=args.get("dropout", 0.0),
        qk_norm=args.get("qk_norm", False),
        patch_encoder_layers=args.get("patch_encoder_layers", 0),
        patch_encoder_window=args.get("patch_encoder_window", 16),
        mod_layers=args.get("mod_layers", 0),
        mod_capacity=args.get("mod_capacity", 0.5),
        mod_group_size=args.get("mod_group_size", 8),
        mod_share_weights=args.get("mod_share_weights", False),
        patch_prediction=args.get("patch_prediction", False),
        patch_prediction_stride=args.get("patch_prediction_stride", 1),
        patch_prediction_mode=args.get("patch_prediction_mode", "factorized"),
        patch_generation_width=args.get("patch_generation_width", 96),
        patch_generation_context=args.get("patch_generation_context", 0),
        patch_prediction_detach_context=args.get(
            "patch_prediction_detach_context", False
        ),
        patch_prediction_context=args.get("patch_prediction_context", "global"),
        tie_byte_embeddings=args.get("tie_byte_embeddings", False),
        context_buckets=args.get("context_buckets", 0),
        context_order=args.get("context_order", 3),
        local_position_embeddings=args.get("local_position_embeddings", False),
        modern_blocks=args.get("modern_blocks", False),
        fused_attention=args.get("fused_attention", False),
        local_window=args.get("local_window", 16),
        coarse_patch_size=args.get("coarse_patch_size", 0),
        coarse_layers=args.get("coarse_layers", 0),
        global_conv_layers=args.get("global_conv_layers", 0),
        global_gru_layers=args.get("global_gru_layers", 0),
        global_block=args.get("global_block", "attention"),
        sparse_state_local_window=args.get("sparse_state_local_window", 32),
        sparse_state_dilated_offsets=tuple(
            args.get("sparse_state_dilated_offsets", (32, 48, 64, 96))
        ),
        sparse_state_chunk_size=args.get("sparse_state_chunk_size", 16),
    ).to(device)
    model.load_state_dict(artifact["patch_model"], strict=False)
    return model


@torch.inference_mode()
def run_layercake(
    artifact_path: Path,
    prompt_len: int,
    generated_bytes: int,
    repeats: int,
    *,
    fast_prefill: bool = False,
) -> dict:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    loaded_at = memory_info()
    artifact = torch.load(artifact_path, map_location="cpu")
    model = build_patch_model_only(artifact, device)
    model.eval()
    after_load = memory_info()
    max_prompt_len = max(
        model.patch_size,
        (model.patch_pos.num_embeddings - 1) * model.patch_size,
    )
    prompt_len = min(prompt_len, max_prompt_len)
    prompt = torch.arange(prompt_len, dtype=torch.long, device=device).remainder(251)
    prompt = prompt.unsqueeze(0)
    _ = model.begin_cached_generation(prompt, fast_prefill_if_aligned=fast_prefill)
    prefill_times = []
    generation_times = []
    generated_counts = []
    for _ in range(repeats):
        started = time.perf_counter()
        state = model.begin_cached_generation(
            prompt,
            fast_prefill_if_aligned=fast_prefill,
        )
        prefill_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        emitted = 0
        while emitted < generated_bytes:
            patch = model.cached_generation_step(state, no_repeat_ngram=4)
            emitted += patch.shape[1]
        generation_times.append(time.perf_counter() - started)
        generated_counts.append(emitted)
    prefill_seconds = statistics.median(prefill_times)
    after_prefill = memory_info()
    generation_seconds = statistics.median(generation_times)
    after_generation = memory_info()
    emitted = int(statistics.median(generated_counts))
    profiled_state = model.begin_cached_generation(
        prompt,
        profile=True,
        fast_prefill_if_aligned=fast_prefill,
    )
    return {
        "model": "layercake",
        "fast_prefill_if_aligned": fast_prefill,
        "fast_prefill_active": bool(profiled_state.get("fast_prefill_active", False)),
        "artifact_bytes": artifact_bytes(artifact_path),
        "parameter_bytes": parameter_bytes(model),
        "prompt_len": prompt_len,
        "loaded_at": loaded_at,
        "after_load": after_load,
        "after_prefill": after_prefill,
        "after_generation": after_generation,
        "peak_rss_bytes": max(
            after_load["peak_rss_bytes"],
            after_prefill["peak_rss_bytes"],
            after_generation["peak_rss_bytes"],
        ),
        "prefill_seconds": prefill_seconds,
        "prefill_seconds_all": prefill_times,
        "generation_seconds": generation_seconds,
        "generation_seconds_all": generation_times,
        "prefill_profile_seconds": profiled_state["profile_seconds"],
        "generated_bytes": emitted,
        "generation_bytes_per_second": emitted / generation_seconds,
    }


@torch.inference_mode()
def run_bpe(
    artifact_path: Path, prompt_len: int, generated_bytes: int, repeats: int
) -> dict:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    loaded_at = memory_info()
    artifact = torch.load(artifact_path, map_location="cpu")
    config = artifact["args"]
    model = BPETokenLM(
        artifact["vocab_size"],
        d_model=config["d_model"],
        layers=config["layers"],
        heads=config["heads"],
        max_len=config["seq"],
    ).to(device)
    model.load_state_dict(artifact["model"])
    model.eval()
    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as handle:
        handle.write(artifact["tokenizer_model"])
        tokenizer_path = Path(handle.name)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    tokenizer_path.unlink(missing_ok=True)
    after_load = memory_info()
    prompt_text = (
        "LayerCake deployment resource benchmark for game dialogue latency. "
        * 16
    )[:prompt_len]
    tokens = tokenizer.encode(prompt_text, out_type=int)
    prompt = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    original = tokenizer.decode(tokens)
    _ = model(prompt[:, -model.pos.num_embeddings :])
    prefill_times = []
    generation_times = []
    generated_counts = []
    for _ in range(repeats):
        started = time.perf_counter()
        _ = model(prompt[:, -model.pos.num_embeddings :])
        prefill_times.append(time.perf_counter() - started)
        generated = prompt
        continuation = ""
        started = time.perf_counter()
        while len(continuation.encode("utf-8")) < generated_bytes:
            context = generated[:, -model.pos.num_embeddings :]
            next_token = model(context)[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            decoded = tokenizer.decode(generated[0].tolist())
            continuation = decoded[len(original) :]
        generation_times.append(time.perf_counter() - started)
        generated_counts.append(len(continuation.encode("utf-8")[:generated_bytes]))
    prefill_seconds = statistics.median(prefill_times)
    after_prefill = memory_info()
    after_generation = memory_info()
    generation_seconds = statistics.median(generation_times)
    emitted = int(statistics.median(generated_counts))
    return {
        "model": "bpe",
        "artifact_bytes": artifact_bytes(artifact_path),
        "parameter_bytes": parameter_bytes(model),
        "loaded_at": loaded_at,
        "after_load": after_load,
        "after_prefill": after_prefill,
        "after_generation": after_generation,
        "peak_rss_bytes": max(
            after_load["peak_rss_bytes"],
            after_prefill["peak_rss_bytes"],
            after_generation["peak_rss_bytes"],
        ),
        "prefill_seconds": prefill_seconds,
        "prefill_seconds_all": prefill_times,
        "generation_seconds": generation_seconds,
        "generation_seconds_all": generation_times,
        "generated_bytes": emitted,
        "generation_bytes_per_second": emitted / generation_seconds,
    }


def run_child(args: argparse.Namespace) -> int:
    path = Path(args.artifact)
    if args.model == "layercake":
        result = run_layercake(
            path,
            args.prompt_len,
            args.generated_bytes,
            args.repeats,
            fast_prefill=args.fast_layercake_prefill,
        )
    else:
        result = run_bpe(path, args.prompt_len, args.generated_bytes, args.repeats)
    print(json.dumps(result, sort_keys=True))
    return 0


def invoke_child(
    model: str,
    artifact: Path,
    prompt_len: int,
    generated_bytes: int,
    repeats: int,
    *,
    fast_layercake_prefill: bool = False,
) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--model",
        model,
        "--artifact",
        str(artifact),
        "--prompt-len",
        str(prompt_len),
        "--generated-bytes",
        str(generated_bytes),
        "--repeats",
        str(repeats),
    ]
    if model == "layercake" and fast_layercake_prefill:
        command.append("--fast-layercake-prefill")
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def export_patch_only(source: Path, output: Path) -> Path:
    if output.exists() and output.stat().st_mtime >= source.stat().st_mtime:
        return output
    artifact = torch.load(source, map_location="cpu")
    payload = {
        "args": artifact["args"],
        "patch_model": artifact["patch_model"],
        "deployment_format": "layercake_patch_runtime_only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--model", choices=["layercake", "bpe"])
    parser.add_argument("--artifact")
    parser.add_argument(
        "--layercake",
        default="runs_experiment/scale15m_transition_lw280_2300_noprofile.pt",
    )
    parser.add_argument("--bpe", default="runs_experiment/scale15m_bpe_matched.pt")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--generated-bytes", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fast-layercake-prefill", action="store_true")
    parser.add_argument(
        "--output", default="results/cpu_deployment_resources_certificate.json"
    )
    args = parser.parse_args()

    if args.child:
        if not args.model or not args.artifact:
            raise SystemExit("--child requires --model and --artifact")
        return run_child(args)

    layercake_source = ROOT / args.layercake
    layercake_artifact = export_patch_only(
        layercake_source,
        ROOT / "runs_experiment" / f"{layercake_source.stem}_patch_only.pt",
    )
    layercake = invoke_child(
        "layercake",
        layercake_artifact,
        args.prompt_len,
        args.generated_bytes,
        args.repeats,
        fast_layercake_prefill=args.fast_layercake_prefill,
    )
    bpe = invoke_child(
        "bpe", ROOT / args.bpe, args.prompt_len, args.generated_bytes, args.repeats
    )
    gates = {
        "measured_in_isolated_processes": True,
        "layercake_parameter_memory_lower_than_bpe": (
            layercake["parameter_bytes"] < bpe["parameter_bytes"]
        ),
        "layercake_artifact_smaller_than_bpe": (
            layercake["artifact_bytes"] < bpe["artifact_bytes"]
        ),
        "layercake_peak_rss_no_more_than_bpe": (
            layercake["peak_rss_bytes"] <= bpe["peak_rss_bytes"]
        ),
        "layercake_prefill_faster_than_bpe": (
            layercake["prefill_seconds"] < bpe["prefill_seconds"]
        ),
        "layercake_generation_faster_than_bpe": (
            layercake["generation_bytes_per_second"]
            > bpe["generation_bytes_per_second"]
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    result = {
        "status": "PASS" if not failed else "OPEN",
        "scope": (
            "CPU deployment-resource benchmark from separate fresh Python "
            "processes. This is a local desktop CPU proxy, not phone, NPU, "
            "battery, or thermal evidence."
        ),
        "required_gates": gates,
        "failed_required": failed,
        "artifacts": {
            "layercake_source": args.layercake,
            "layercake_deployment": str(layercake_artifact.relative_to(ROOT)),
            "bpe": args.bpe,
        },
        "layercake": layercake,
        "bpe": bpe,
        "layercake_deployment_mode": {
            "fast_prefill_if_aligned": bool(args.fast_layercake_prefill),
            "fast_prefill_active": bool(layercake.get("fast_prefill_active", False)),
        },
        "metrics": {
            "repeats": args.repeats,
            "layercake_peak_rss_bytes": layercake["peak_rss_bytes"],
            "bpe_peak_rss_bytes": bpe["peak_rss_bytes"],
            "peak_rss_ratio": (
                layercake["peak_rss_bytes"] / max(bpe["peak_rss_bytes"], 1)
            ),
            "layercake_parameter_bytes": layercake["parameter_bytes"],
            "bpe_parameter_bytes": bpe["parameter_bytes"],
            "parameter_memory_ratio": (
                layercake["parameter_bytes"] / max(bpe["parameter_bytes"], 1)
            ),
            "layercake_artifact_bytes": layercake["artifact_bytes"],
            "bpe_artifact_bytes": bpe["artifact_bytes"],
            "artifact_ratio": layercake["artifact_bytes"]
            / max(bpe["artifact_bytes"], 1),
            "prefill_speed_ratio": bpe["prefill_seconds"]
            / max(layercake["prefill_seconds"], 1e-12),
            "generation_speed_ratio": layercake["generation_bytes_per_second"]
            / max(bpe["generation_bytes_per_second"], 1e-12),
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
