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

from benchmark_moonshot_generation import (  # noqa: E402
    _generate_bpe,
    _generate_layercake,
    _quality_score,
)
from layercake.companion_runtime import finalize_companion_text  # noqa: E402
from layercake.domain_runtime import (  # noqa: E402
    load_instruction_aliases,
    load_portable_domain_chunks,
    normalize_instruction_tokens,
    render_instruction_alias_answer,
    render_portable_domain_answer,
)


def _load_prompt_spec(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"prompt spec must be a non-empty list: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"prompt row {index} is not an object")
        if not str(row.get("prompt", "")).strip():
            raise ValueError(f"prompt row {index} is missing prompt")
        row.setdefault("category", "unspecified")
        row.setdefault("expected_keywords", [])
        row.setdefault("forbidden_keywords", [])
        row.setdefault("min_keyword_hits", 0)
        row.setdefault("expect_abstain", False)
    return rows


def _keyword_hits(
    text: str,
    expected_keywords: list[str],
    *,
    ignore_negated: bool = False,
) -> tuple[int, list[str]]:
    answer_tokens = normalize_instruction_tokens(text)
    lower_text = text.casefold()
    hits: list[str] = []
    for keyword in expected_keywords:
        keyword_tokens = normalize_instruction_tokens(keyword)
        if not keyword_tokens:
            continue
        if len(keyword_tokens) == 1:
            hit = bool(answer_tokens.intersection(keyword_tokens))
        else:
            hit = keyword_tokens.issubset(answer_tokens)
        if hit and ignore_negated:
            raw_keyword = keyword.casefold()
            raw_index = lower_text.find(raw_keyword)
            if raw_index >= 0:
                prefix = lower_text[max(0, raw_index - 32) : raw_index]
                if any(negation in prefix.split()[-5:] for negation in ("not", "never", "without")):
                    hit = False
        if hit:
            hits.append(keyword)
    return len(hits), hits


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _generate_one(
    *,
    checkpoint: dict[str, Any],
    model_kind: str,
    prompt: str,
    device: torch.device,
    max_new_bytes: int,
    no_repeat_ngram: int,
    layercake_generation_mode: str = "auto",
) -> tuple[str, float, str, dict[str, Any]]:
    if model_kind == "bpe":
        text, seconds = _generate_bpe(
            checkpoint,
            prompt,
            device=device,
            max_new_bytes=max_new_bytes,
            no_repeat_ngram=no_repeat_ngram,
        )
        return text, seconds, "neural_bpe_transformer", {}
    text, seconds = _generate_layercake(
        checkpoint,
        prompt,
        device=device,
        max_new_bytes=max_new_bytes,
        no_repeat_ngram=no_repeat_ngram,
        generation_mode=layercake_generation_mode,
    )
    runtime = (
        "neural_layercake_patch_prediction"
        if layercake_generation_mode == "patch_prediction"
        else "neural_layercake"
    )
    return text, seconds, runtime, {}


def _instruction_generation_gates(samples: list[dict[str, Any]]) -> dict[str, bool]:
    """Return strict gates for a standalone instruction-generation artifact.

    The old artifact status was unconditional, which allowed printable but
    semantically useless neural output to be labeled PASS. These gates are
    intentionally simple and auditable: a reviewed generation artifact only
    passes when every sample is non-empty, printable, non-degenerate, clean
    against no-repeat-8, and relevant to its prompt keywords/abstention rule.
    """

    has_samples = bool(samples)
    return {
        "samples_present": has_samples,
        "samples_nonempty": has_samples
        and all(bool(str(sample.get("text", "")).strip()) for sample in samples),
        "samples_printable": has_samples
        and all(float(sample.get("printable_ratio", 0.0)) >= 0.95 for sample in samples),
        "samples_alpha": has_samples
        and all(float(sample.get("alpha_space_ratio", 0.0)) >= 0.75 for sample in samples),
        "samples_no_repeat_8": has_samples
        and all(float(sample.get("max_repeat_8gram", 999.0)) <= 4.0 for sample in samples),
        "samples_lexically_diverse": has_samples
        and all(
            float(sample.get("unique_word_count", 0.0)) >= 6.0
            and float(sample.get("distinct_word_ratio", 0.0)) >= 0.25
            and float(sample.get("one_char_word_ratio", 1.0)) <= 0.45
            and float(sample.get("unique_alpha_char_count", 0.0)) >= 10.0
            for sample in samples
        ),
        "samples_relevant": has_samples
        and all(bool(sample.get("relevance_pass", False)) for sample in samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact and paraphrased instruction prompts with relevance gates. "
            "LayerCake can optionally use semantic instruction aliases or a portable "
            "corpus-memory domain runtime."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-kind", required=True, choices=["layercake", "bpe"])
    parser.add_argument("--prompt-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-new-bytes", type=int, default=160)
    parser.add_argument("--no-repeat-ngram", type=int, default=8)
    parser.add_argument(
        "--layercake-generation-mode",
        choices=["auto", "patch_prediction"],
        default="auto",
    )
    parser.add_argument("--instruction-alias-cache", nargs="*", type=Path, default=[])
    parser.add_argument("--alias-threshold", type=float, default=0.34)
    parser.add_argument("--alias-min-overlap", type=int, default=1)
    parser.add_argument("--portable-corpus-memory", nargs="*", type=Path, default=[])
    parser.add_argument("--portable-memory-threshold", type=float, default=0.30)
    parser.add_argument("--portable-memory-min-overlap", type=int, default=2)
    parser.add_argument("--portable-memory-max-chunk-chars", type=int, default=180)
    parser.add_argument("--portable-abstain-on-miss", action="store_true")
    parser.add_argument(
        "--portable-abstain-text",
        default=" I do not have that information in the attached domain layer.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)

    checkpoint = _load_checkpoint(args.checkpoint)
    prompt_rows = _load_prompt_spec(args.prompt_spec)
    domain_setup_started = time.perf_counter()
    aliases = (
        load_instruction_aliases(args.instruction_alias_cache)
        if args.model_kind == "layercake" and args.instruction_alias_cache
        else []
    )
    portable_chunks = (
        load_portable_domain_chunks(
            args.portable_corpus_memory,
            max_chunk_chars=args.portable_memory_max_chunk_chars,
        )
        if args.model_kind == "layercake" and args.portable_corpus_memory
        else []
    )
    domain_setup_seconds = time.perf_counter() - domain_setup_started

    samples: list[dict[str, Any]] = []
    total_bytes = 0
    total_seconds = 0.0
    quality_scores: list[float] = []
    relevant_count = 0
    alias_count = 0
    portable_memory_count = 0
    abstention_count = 0
    category_counts: dict[str, int] = {}
    category_relevant: dict[str, int] = {}
    category_alias: dict[str, int] = {}
    category_portable_memory: dict[str, int] = {}
    category_abstention: dict[str, int] = {}

    for row in prompt_rows:
        prompt = str(row["prompt"])
        category = str(row.get("category", "unspecified"))
        expected = [str(item) for item in row.get("expected_keywords", [])]
        forbidden = [str(item) for item in row.get("forbidden_keywords", [])]
        min_hits = int(row.get("min_keyword_hits", 0))
        expect_abstain = bool(row.get("expect_abstain", False))
        runtime_path = "neural_layercake" if args.model_kind == "layercake" else "neural_bpe_transformer"
        metadata: dict[str, Any] = {}
        portable_result = None
        portable_checked = False
        if portable_chunks:
            started = time.perf_counter()
            portable_result = render_portable_domain_answer(
                prompt,
                portable_chunks,
                max_new_bytes=args.max_new_bytes,
                threshold=args.portable_memory_threshold,
                min_overlap=args.portable_memory_min_overlap,
            )
            seconds = time.perf_counter() - started
            portable_checked = True
            if portable_result is not None:
                text, match = portable_result
                runtime_path = "portable_corpus_memory"
                metadata = {
                    "portable_memory_source": match.chunk.source,
                    "portable_memory_chunk_id": match.chunk.chunk_id,
                    "portable_memory_score": match.score,
                    "portable_memory_overlap": match.overlap,
                    "portable_memory_query_tokens": sorted(match.query_tokens),
                }
                portable_memory_count += 1
                category_portable_memory[category] = (
                    category_portable_memory.get(category, 0) + 1
                )
            elif args.portable_abstain_on_miss and args.model_kind == "layercake":
                text = args.portable_abstain_text.encode(
                    "utf-8",
                    errors="replace",
                )[: args.max_new_bytes].decode("utf-8", errors="replace")
                runtime_path = "portable_corpus_abstain"
                metadata = {"portable_memory_miss": True}
                abstention_count += 1
                category_abstention[category] = category_abstention.get(category, 0) + 1
        alias_result = None
        if (
            portable_result is None
            and runtime_path != "portable_corpus_abstain"
            and aliases
        ):
            started = time.perf_counter()
            alias_result = render_instruction_alias_answer(
                prompt,
                aliases,
                max_new_bytes=args.max_new_bytes,
                threshold=args.alias_threshold,
                min_overlap=args.alias_min_overlap,
            )
            seconds = time.perf_counter() - started
            if alias_result is not None:
                text, match = alias_result
                runtime_path = "semantic_instruction_alias"
                metadata = {
                    "alias_question": match.alias.question,
                    "alias_score": match.score,
                    "alias_overlap": match.overlap,
                    "alias_query_tokens": sorted(match.query_tokens),
                }
                alias_count += 1
                category_alias[category] = category_alias.get(category, 0) + 1
        if (
            portable_result is None
            and alias_result is None
            and runtime_path != "portable_corpus_abstain"
        ):
            text, seconds, runtime_path, metadata = _generate_one(
                checkpoint=checkpoint,
                model_kind=args.model_kind,
                prompt=prompt,
                device=device,
                max_new_bytes=args.max_new_bytes,
                no_repeat_ngram=args.no_repeat_ngram,
                layercake_generation_mode=args.layercake_generation_mode,
            )
        raw_text = text
        text, finalization = finalize_companion_text(raw_text)
        emitted = len(text.encode("utf-8", errors="replace"))
        quality = _quality_score(text)
        hit_count, hit_keywords = _keyword_hits(text, expected)
        forbidden_hit_count, forbidden_hit_keywords = _keyword_hits(
            text,
            forbidden,
            ignore_negated=True,
        )
        abstention_pass = (
            runtime_path == "portable_corpus_abstain"
            and forbidden_hit_count == 0
            if expect_abstain
            else True
        )
        relevant = (
            abstention_pass
            if expect_abstain
            else hit_count >= min_hits and forbidden_hit_count == 0
        )
        if relevant:
            relevant_count += 1
            category_relevant[category] = category_relevant.get(category, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        quality_scores.append(float(quality["quality_score"]))
        total_bytes += emitted
        total_seconds += seconds
        samples.append(
            {
                "prompt": prompt,
                "category": category,
                "text": text,
                **finalization,
                "runtime_path": runtime_path,
                "portable_memory_checked": portable_checked,
                "expect_abstain": expect_abstain,
                "abstention_pass": abstention_pass,
                "seconds": seconds,
                "generated_bytes": emitted,
                "bytes_per_second": emitted / max(seconds, 1e-12),
                "expected_keywords": expected,
                "forbidden_keywords": forbidden,
                "min_keyword_hits": min_hits,
                "keyword_hits": hit_count,
                "hit_keywords": hit_keywords,
                "forbidden_keyword_hits": forbidden_hit_count,
                "forbidden_hit_keywords": forbidden_hit_keywords,
                "relevance_pass": relevant,
                **quality,
                **metadata,
            }
        )

    category_metrics = {}
    for category, count in sorted(category_counts.items()):
        category_metrics[category] = {
            "count": count,
            "relevance_rate": category_relevant.get(category, 0) / max(count, 1),
            "alias_match_rate": category_alias.get(category, 0) / max(count, 1),
            "portable_memory_match_rate": category_portable_memory.get(category, 0)
            / max(count, 1),
            "abstention_rate": category_abstention.get(category, 0) / max(count, 1),
        }
    gates = _instruction_generation_gates(samples)
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "model_kind": args.model_kind,
        "checkpoint": str(args.checkpoint),
        "prompt_spec": str(args.prompt_spec),
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
            "domain_setup_seconds": domain_setup_seconds,
            "relevance_rate": relevant_count / max(len(samples), 1),
            "alias_match_rate": alias_count / max(len(samples), 1),
            "portable_memory_match_rate": portable_memory_count / max(len(samples), 1),
            "abstention_rate": abstention_count / max(len(samples), 1),
            "category_metrics": category_metrics,
        },
        "gates": gates,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
