"""Measure fresh-process RSS components for a Phase 2 prompt-memory checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _child(checkpoint: Path, threads: int, decode_steps: int) -> dict:
    import gc
    import os
    import psutil

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    process = psutil.Process(os.getpid())

    def rss() -> int:
        gc.collect()
        return int(process.memory_info().rss)

    stages = {"python_process_baseline": rss()}
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    stages["torch_runtime_loaded"] = rss()
    from layercake.training.phase2_sparse_bpe import load_sparse_bpe_checkpoint

    model, tokenizer, metadata = load_sparse_bpe_checkpoint(
        checkpoint, device="cpu"
    )
    model.eval()
    stages["checkpoint_loaded"] = rss()
    prompt = (
        "Explain efficient computing to a curious reader using two concrete "
        "details. Your response must contain at least 80 words."
    )
    ids = tokenizer.encode(prompt)
    with torch.inference_mode():
        state = model.prefill(torch.tensor([ids], dtype=torch.long))
        stages["prefill_state_ready"] = rss()
        peak = stages["prefill_state_ready"]
        for _ in range(decode_steps):
            _, state = model.decode_step(state)
            peak = max(peak, int(process.memory_info().rss))
    stages["decode_complete"] = rss()
    stages["decode_peak"] = peak
    components = {
        "torch_and_native_runtime": (
            stages["torch_runtime_loaded"] - stages["python_process_baseline"]
        ),
        "model_and_tokenizer": (
            stages["checkpoint_loaded"] - stages["torch_runtime_loaded"]
        ),
        "prefill_increment": (
            stages["prefill_state_ready"] - stages["checkpoint_loaded"]
        ),
        "decode_peak_increment": peak - stages["prefill_state_ready"],
    }
    prompt_memory = getattr(state, "hierarchical_prompt_memory", None)
    return {
        "format": "layercake-phase2-fresh-process-memory-profile/1",
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": hashlib.sha256(
            (checkpoint / "model.safetensors").read_bytes()
        ).hexdigest(),
        "threads": threads,
        "decode_steps": decode_steps,
        "stages_rss_bytes": stages,
        "components_rss_bytes": components,
        "active_state": {
            "kv_layers": len(state.keys_values),
            "generated_tokens": int(state.generated_ids.shape[1]),
            "prompt_memory_kind": (
                "hierarchical"
                if prompt_memory is not None
                else (
                    "fixed_fused_context"
                    if getattr(state, "prompt_context", None) is not None
                    else "none"
                )
            ),
            "prompt_memory_shapes": (
                [list(value.shape) for value in prompt_memory]
                if prompt_memory is not None
                else (
                    [list(state.prompt_context.shape)]
                    if getattr(state, "prompt_context", None) is not None
                    else []
                )
            ),
        },
        "status": "PASS",
        "metadata_checkpoint_sha256": metadata["checkpoint"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
    if args.child:
        print(json.dumps(_child(checkpoint, args.threads, args.decode_steps)))
        return 0
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--checkpoint",
        str(checkpoint),
        "--threads",
        str(args.threads),
        "--decode-steps",
        str(args.decode_steps),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=True
    )
    result = json.loads(completed.stdout)
    result["exact_command"] = command
    result["profile_sha256"] = hashlib.sha256(
        json.dumps(
            result, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if args.output is not None:
        output = args.output
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
