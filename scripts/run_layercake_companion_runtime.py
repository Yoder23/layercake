from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_instruction_generalization import _keyword_hits, _load_prompt_spec  # noqa: E402
from benchmark_moonshot_generation import _generate_layercake, _quality_score  # noqa: E402
from layercake.companion_runtime import finalize_companion_text  # noqa: E402
from layercake.domain_runtime import (  # noqa: E402
    load_instruction_aliases,
    load_portable_domain_chunks,
    render_instruction_alias_answer,
    render_portable_domain_answer,
)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _prompt_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.prompt_spec:
        rows.extend(_load_prompt_spec(args.prompt_spec))
    for prompt in args.prompt:
        rows.append(
            {
                "category": "ad_hoc",
                "prompt": prompt,
                "expected_keywords": [],
                "forbidden_keywords": [],
                "min_keyword_hits": 0,
            }
        )
    if not rows:
        raise ValueError("provide --prompt-spec or at least one --prompt")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the production LayerCake companion runtime with bounded answers."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt-spec", type=Path)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-new-bytes", type=int, default=180)
    parser.add_argument("--no-repeat-ngram", type=int, default=8)
    parser.add_argument("--instruction-alias-cache", nargs="*", type=Path, default=[])
    parser.add_argument("--alias-threshold", type=float, default=0.34)
    parser.add_argument("--alias-min-overlap", type=int, default=1)
    parser.add_argument("--portable-corpus-memory", nargs="*", type=Path, default=[])
    parser.add_argument("--portable-memory-threshold", type=float, default=0.30)
    parser.add_argument("--portable-memory-min-overlap", type=int, default=2)
    parser.add_argument("--portable-memory-max-chunk-chars", type=int, default=180)
    parser.add_argument(
        "--no-neural-fallback",
        action="store_true",
        help="Return a miss instead of unconstrained neural generation when no domain route matches.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)

    checkpoint = _load_checkpoint(args.checkpoint)
    prompt_rows = _prompt_rows(args)

    setup_started = time.perf_counter()
    aliases = load_instruction_aliases(args.instruction_alias_cache)
    portable_chunks = load_portable_domain_chunks(
        args.portable_corpus_memory,
        max_chunk_chars=args.portable_memory_max_chunk_chars,
    )
    setup_seconds = time.perf_counter() - setup_started

    samples: list[dict[str, Any]] = []
    total_seconds = 0.0
    total_bytes = 0
    relevant_count = 0
    quality_scores: list[float] = []

    for row in prompt_rows:
        prompt = str(row["prompt"])
        expected = [str(item) for item in row.get("expected_keywords", [])]
        forbidden = [str(item) for item in row.get("forbidden_keywords", [])]
        min_hits = int(row.get("min_keyword_hits", 0))
        metadata: dict[str, Any] = {}
        runtime_path = "neural_layercake"
        raw_text = ""
        seconds = 0.0

        started = time.perf_counter()
        alias_result = render_instruction_alias_answer(
            prompt,
            aliases,
            max_new_bytes=args.max_new_bytes,
            threshold=args.alias_threshold,
            min_overlap=args.alias_min_overlap,
        )
        if alias_result is not None:
            raw_text, match = alias_result
            seconds = time.perf_counter() - started
            runtime_path = "semantic_instruction_alias"
            metadata = {
                "alias_question": match.alias.question,
                "alias_score": match.score,
                "alias_overlap": match.overlap,
                "alias_query_tokens": sorted(match.query_tokens),
            }
        else:
            started = time.perf_counter()
            portable_result = render_portable_domain_answer(
                prompt,
                portable_chunks,
                max_new_bytes=args.max_new_bytes,
                threshold=args.portable_memory_threshold,
                min_overlap=args.portable_memory_min_overlap,
            )
            if portable_result is not None:
                raw_text, match = portable_result
                seconds = time.perf_counter() - started
                runtime_path = "portable_corpus_memory"
                metadata = {
                    "portable_memory_source": match.chunk.source,
                    "portable_memory_chunk_id": match.chunk.chunk_id,
                    "portable_memory_score": match.score,
                    "portable_memory_overlap": match.overlap,
                    "portable_memory_query_tokens": sorted(match.query_tokens),
                }
            elif args.no_neural_fallback:
                raw_text = "I do not have that information in the attached companion domain."
                seconds = time.perf_counter() - started
                runtime_path = "domain_miss"
            else:
                raw_text, seconds = _generate_layercake(
                    checkpoint,
                    prompt,
                    device=device,
                    max_new_bytes=args.max_new_bytes,
                    no_repeat_ngram=args.no_repeat_ngram,
                )

        text, finalization = finalize_companion_text(raw_text)
        emitted = len(text.encode("utf-8", errors="replace"))
        total_bytes += emitted
        total_seconds += seconds
        quality = _quality_score(text)
        quality_scores.append(float(quality["quality_score"]))
        hit_count, hit_keywords = _keyword_hits(text, expected)
        forbidden_count, forbidden_keywords = _keyword_hits(
            text,
            forbidden,
            ignore_negated=True,
        )
        relevance_pass = hit_count >= min_hits and forbidden_count == 0
        if relevance_pass:
            relevant_count += 1
        samples.append(
            {
                "prompt": prompt,
                "category": str(row.get("category", "unspecified")),
                "runtime_path": runtime_path,
                "text": text,
                "seconds": seconds,
                "generated_bytes": emitted,
                "bytes_per_second": emitted / max(seconds, 1e-12),
                "expected_keywords": expected,
                "forbidden_keywords": forbidden,
                "min_keyword_hits": min_hits,
                "keyword_hits": hit_count,
                "hit_keywords": hit_keywords,
                "forbidden_keyword_hits": forbidden_count,
                "forbidden_hit_keywords": forbidden_keywords,
                "relevance_pass": relevance_pass,
                **quality,
                **finalization,
                **metadata,
            }
        )

    result = {
        "status": "PASS" if relevant_count == len(samples) else "FAIL",
        "model_kind": "layercake",
        "runtime": "layercake_companion_runtime",
        "checkpoint": str(args.checkpoint),
        "prompt_spec": str(args.prompt_spec) if args.prompt_spec else None,
        "instruction_alias_cache": [str(path) for path in args.instruction_alias_cache],
        "portable_corpus_memory": [str(path) for path in args.portable_corpus_memory],
        "device": str(device),
        "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
        "max_new_bytes": args.max_new_bytes,
        "no_repeat_ngram": args.no_repeat_ngram,
        "metrics": {
            "generation_bytes_per_second": total_bytes / max(total_seconds, 1e-12),
            "quality_score": sum(quality_scores) / max(len(quality_scores), 1),
            "generated_bytes": total_bytes,
            "seconds": total_seconds,
            "domain_setup_seconds": setup_seconds,
            "relevance_rate": relevant_count / max(len(samples), 1),
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
