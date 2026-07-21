from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import tempfile
import time

import sentencepiece as spm
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifact_utils import build_models
from benchmark_bpe_baseline import BPETokenLM
from train_bpe_transformer_from_config import BPETokenTransformerLM


QUESTIONS = [
    (
        "xml_json_schema",
        "Question: Given an XML node <item id=\"42\">ok</item>, produce the matching JSON object. Answer:",
    ),
    (
        "screen_edit_action",
        "Question: A user says move the Save button to the top right of the app. What edit action should be taken? Answer:",
    ),
]


def _byte_quality(raw: bytes) -> dict[str, float | int | str]:
    text = raw.decode("utf-8", errors="replace")
    printable = sum(1 for byte in raw if byte in (9, 10, 13) or 32 <= byte <= 126)
    words = [word for word in text.replace("\n", " ").split(" ") if word]
    trigrams = list(zip(words, words[1:], words[2:]))
    counts = Counter(raw[index : index + 8] for index in range(max(len(raw) - 7, 0)))
    return {
        "text": text,
        "printable_ratio": printable / max(len(raw), 1),
        "word_count": len(words),
        "distinct_word_trigram": len(set(trigrams)) / max(len(trigrams), 1),
        "max_repeat_8gram": max(counts.values(), default=0),
    }


@torch.no_grad()
def _generate_layercake(
    model,
    prompt_text: str,
    new_bytes: int,
    device: torch.device,
    mode: str,
):
    prompt_bytes = list(prompt_text.encode("utf-8", errors="replace"))
    alignment = max(int(getattr(model, "patch_size", 1)), 1)
    if mode == "stateful_cached" and getattr(model, "local_decoder", None) == "window_transformer":
        alignment = max(alignment, int(getattr(model, "local_window", alignment)))
    left_pad = (-len(prompt_bytes)) % alignment
    prompt = torch.tensor(
        [([ord(" ")] * left_pad) + prompt_bytes],
        dtype=torch.long,
        device=device,
    )
    if mode == "stateful_cached":
        state = model.begin_cached_generation(prompt)
        continuation = torch.empty(1, new_bytes, dtype=torch.long, device=device)
        emitted = 0
    else:
        generated = prompt.clone()
    started = time.perf_counter()
    while True:
        if mode == "stateful_cached" and emitted >= new_bytes:
            break
        if mode != "stateful_cached" and generated.shape[1] - prompt.shape[1] >= new_bytes:
            break
        if mode == "stateful_cached":
            next_patch = model.cached_generation_step(state, no_repeat_ngram=8)
            take = min(next_patch.shape[1], new_bytes - emitted)
            continuation[:, emitted : emitted + take] = next_patch[:, :take]
            emitted += take
        else:
            context = generated[:, -model.patch_pos.num_embeddings * model.patch_size :]
            if mode == "cached":
                next_patch = model.generate_cached_patch(context)
            elif mode == "verified":
                next_patch = model.generate_verified_patch(context)
            else:
                next_patch = model.generate_next_patch(context)
            generated = torch.cat([generated, next_patch], dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if mode != "stateful_cached":
        continuation = generated[:, prompt.shape[1] : prompt.shape[1] + new_bytes]
    return bytes(continuation[0].cpu().tolist()), elapsed


@torch.no_grad()
def _generate_bpe(model, tokenizer, prompt_text: str, new_bytes: int, device: torch.device):
    token_ids = tokenizer.encode(prompt_text, out_type=int)
    generated = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    original_text = tokenizer.decode(token_ids)
    continuation = ""
    started = time.perf_counter()
    while len(continuation.encode("utf-8")) < new_bytes:
        context = generated[:, -model.pos.num_embeddings :]
        next_token = model(context)[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        decoded = tokenizer.decode(generated[0].tolist())
        continuation = decoded[len(original_text) :]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return continuation.encode("utf-8", errors="replace")[:new_bytes], elapsed


def _load_bpe(path: Path, device: torch.device):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as handle:
        handle.write(artifact["tokenizer_model"])
        tokenizer_path = Path(handle.name)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    tokenizer_path.unlink(missing_ok=True)
    if "model_config" in artifact and "training_config" in artifact:
        model_config = artifact["model_config"]
        training_config = artifact["training_config"]
        model = BPETokenTransformerLM(
            vocab_size=tokenizer.vocab_size(),
            d_model=int(model_config["d_model"]),
            layers=int(model_config["layers"]),
            heads=int(model_config["heads"]),
            max_len=int(training_config.get("seq_len", 256)),
            ff_mult=int(model_config.get("ff_mult", 4)),
            dropout=float(model_config.get("dropout", 0.0)),
        ).to(device)
    else:
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
    return artifact, model, tokenizer


def _training_summary(path: Path) -> dict:
    row = json.loads(path.read_text(encoding="utf-8-sig"))
    general = row.get("general", {})
    latest = row.get("latest", {})
    parameter_filter = latest.get("parameter_filter", row.get("parameter_filter", {}))
    return {
        "artifact": str(path),
        "parameters": row.get(
            "parameters",
            parameter_filter.get("total_params", latest.get("trainable_params")),
        ),
        "eval_bpb": general.get(
            "bpb",
            row.get("eval_bpb", latest.get("eval_bpb")),
        ),
        "eval_bytes": row.get("eval_bytes", latest.get("eval_bytes")),
        "train_seconds": row.get("elapsed_seconds", latest.get("elapsed_seconds")),
        "train_bytes": row.get("estimated_total_training_bytes", latest.get("train_bytes")),
    }


def _timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layercake", required=True, type=Path)
    parser.add_argument("--layercake-training", required=True, type=Path)
    parser.add_argument("--bpe", required=True, type=Path)
    parser.add_argument("--bpe-training", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--layercake-mode",
        choices=["draft", "verified", "cached", "stateful_cached"],
        default="stateful_cached",
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--new-bytes", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)

    layercake_artifact = torch.load(args.layercake, map_location="cpu", weights_only=True)
    _, layercake = build_models(layercake_artifact, device)
    layercake.eval()
    _, bpe, tokenizer = _load_bpe(args.bpe, device)
    cuda_graph_runtime = None
    if (
        device.type == "cuda"
        and args.layercake_mode == "draft"
        and hasattr(layercake, "prepare_patch_generator_cuda_graph")
    ):
        cuda_graph_runtime = layercake.prepare_patch_generator_cuda_graph()

    samples = []
    lc_speeds = []
    bpe_speeds = []
    for name, prompt in QUESTIONS:
        lc_trials = []
        bpe_trials = []
        lc_raw = b""
        bpe_raw = b""
        for _ in range(max(args.repeats, 1)):
            lc_raw, lc_seconds = _generate_layercake(
                layercake, prompt, args.new_bytes, device, args.layercake_mode
            )
            bpe_raw, bpe_seconds = _generate_bpe(
                bpe, tokenizer, prompt, args.new_bytes, device
            )
            lc_trials.append(
                {
                    "seconds": lc_seconds,
                    "bytes_per_second": args.new_bytes / lc_seconds,
                }
            )
            bpe_trials.append(
                {
                    "seconds": bpe_seconds,
                    "bytes_per_second": args.new_bytes / bpe_seconds,
                }
            )
        lc_bps_values = [float(item["bytes_per_second"]) for item in lc_trials]
        bpe_bps_values = [float(item["bytes_per_second"]) for item in bpe_trials]
        lc_bps = statistics.fmean(lc_bps_values)
        bpe_bps = statistics.fmean(bpe_bps_values)
        lc_speeds.extend(lc_bps_values)
        bpe_speeds.extend(bpe_bps_values)
        samples.append(
            {
                "name": name,
                "prompt": prompt,
                "layercake": {
                    "seconds": statistics.fmean(
                        [float(item["seconds"]) for item in lc_trials]
                    ),
                    "bytes_per_second": lc_bps,
                    "timing": {
                        "seconds": _timing_summary(
                            [float(item["seconds"]) for item in lc_trials]
                        ),
                        "bytes_per_second": _timing_summary(lc_bps_values),
                        "trials": lc_trials,
                    },
                    **_byte_quality(lc_raw),
                },
                "transformer": {
                    "seconds": statistics.fmean(
                        [float(item["seconds"]) for item in bpe_trials]
                    ),
                    "bytes_per_second": bpe_bps,
                    "timing": {
                        "seconds": _timing_summary(
                            [float(item["seconds"]) for item in bpe_trials]
                        ),
                        "bytes_per_second": _timing_summary(bpe_bps_values),
                        "trials": bpe_trials,
                    },
                    **_byte_quality(bpe_raw),
                },
                "speed_ratio_layercake_over_transformer": lc_bps / max(bpe_bps, 1e-12),
            }
        )

    result = {
        "scope": (
            "Same trained LayerCake and same trained BPE-token transformer, "
            "asked the same two prompts on the requested device. Quality is "
            "reported as raw text plus simple repetition/printability metrics; "
            "these are not instruction-tuned task-accuracy scores."
        ),
        "device": str(device),
        "layercake_mode": args.layercake_mode,
        "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
        "new_bytes": args.new_bytes,
        "repeats": max(args.repeats, 1),
        "layercake_cuda_graph_runtime": cuda_graph_runtime,
        "training": {
            "layercake": _training_summary(args.layercake_training),
            "transformer": _training_summary(args.bpe_training),
        },
        "samples": samples,
        "summary": {
            "mean_layercake_bytes_per_second": sum(lc_speeds) / len(lc_speeds),
            "mean_transformer_bytes_per_second": sum(bpe_speeds) / len(bpe_speeds),
            "mean_speed_ratio_layercake_over_transformer": (
                (sum(lc_speeds) / len(lc_speeds))
                / max(sum(bpe_speeds) / len(bpe_speeds), 1e-12)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
