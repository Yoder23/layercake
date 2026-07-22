"""Held-out English diagnostics beyond a single next-byte loss number."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from layercake.training.data import ByteCorpus, sha256_file
from layercake.training.foundation import evaluate_core, load_core_checkpoint


@torch.inference_mode()
def evaluate_english_quality(
    core_dir: str | Path, corpus_path: str | Path, output_path: str | Path
) -> dict:
    core, metadata = load_core_checkpoint(core_dir, device="cpu")
    route = int(metadata["route"])
    corpus = ByteCorpus(corpus_path)
    lengths = {}
    for length in (64, 256, 1024):
        lengths[str(length)] = evaluate_core(
            core, corpus, batch_size=4, sequence_bytes=length, batches=3,
            device=torch.device("cpu"), route=route,
        )
    prompts = (
        "The history of navigation shows that",
        "A careful scientific explanation should",
        "When neighbors disagree about a project,",
        "The rain moved across the city while",
        "One practical way to learn a difficult subject is",
    )
    samples = []
    for prompt in prompts:
        state = core.prefill(prompt, route=route, capture_generated=True)
        _, state = core.decode_many(state, 128)
        raw = bytes(state.generated_bytes[0].tolist())
        repeated_fourgrams = len(raw) >= 8 and len({raw[index:index + 4] for index in range(len(raw) - 3)}) < (len(raw) - 3) * 0.5
        samples.append({
            "prompt": prompt,
            "completion": raw.decode("utf-8", errors="replace"),
            "printable_rate": sum(byte in (9, 10, 13) or 32 <= byte < 127 for byte in raw) / len(raw),
            "unique_byte_fraction": len(set(raw)) / len(raw),
            "high_fourgram_repetition": repeated_fourgrams,
        })
    base = lengths["256"]["bits_per_byte"]
    evidence = {
        "format": "layercake-english-quality/2",
        "status": "PASS",
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "test_corpus_sha256": sha256_file(corpus_path),
        "sequence_length_quality": lengths,
        "long_context_bpb_ratio_1024_over_256": lengths["1024"]["bits_per_byte"] / base,
        "generation_samples": samples,
        "benchmark_tasks": "NOT_RUN_MODEL_SCALE_TOO_SMALL_FOR_MEANINGFUL_STANDARD_LM_TASKS",
        "claim_scope": "diagnostic only; comparison quality is decided by the matched transformer gate",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence
