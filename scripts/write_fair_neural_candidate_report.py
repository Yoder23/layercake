from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

NORTHSTAR_MIN_EXACT_JSON_ACCURACY = 0.95
NORTHSTAR_MIN_PARSEABLE_JSON_RATE = 0.95
NORTHSTAR_MIN_CHAR_SIMILARITY = 0.98
NORTHSTAR_MIN_SAMPLES_PER_SPLIT = 8


CANDIDATES = {
    "patch2_window_nocache": {
        "cpu": "results/breakthrough_equal/schema_action_patch2_nocache_stateful_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_patch2_nocache_stateful_gpu.json",
        "architecture": "2-byte window_transformer local decoder, stateful cached generation, no domain cache",
    },
    "patch2_window_nocache_answerft": {
        "cpu": "results/breakthrough_equal/schema_action_patch2_nocache_answerft_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_patch2_nocache_answerft_gpu.json",
        "architecture": "2-byte window_transformer local decoder, stateful cached generation, no domain cache, answer-span weighted fine-tune",
    },
    "patch2_window_nocache_answerft_uncached": {
        "cpu": "results/breakthrough_equal/schema_action_patch2_nocache_answerft_uncached_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_patch2_nocache_answerft_uncached_gpu.json",
        "architecture": "2-byte window_transformer local decoder, uncached full-forward generation, no domain cache, answer-span weighted fine-tune",
    },
    "patch2_copy_nocache": {
        "cpu": "results/breakthrough_equal/schema_action_patch2_copy_nocache_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_patch2_copy_nocache_gpu.json",
        "architecture": "2-byte window_transformer local decoder, causal byte-copy attention prior, stateful cached generation, no domain cache",
    },
    "span4_copy_nocache": {
        "cpu": "results/breakthrough_equal/schema_action_span4_copy_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span4_copy_gpu.json",
        "architecture": "4-byte span_patch_decoder with copy transducer, span_cached fair-neural generation, no domain cache",
    },
    "span8_copy_nocache": {
        "cpu": "results/breakthrough_equal/schema_action_span8_copy_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span8_copy_gpu.json",
        "architecture": "8-byte span_patch_decoder with copy transducer, span_cached fair-neural generation, no domain cache",
    },
    "span8_copy_2m_nocache": {
        "cpu": "results/breakthrough_equal/schema_action_span8_copy_2m_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span8_copy_2m_gpu.json",
        "architecture": "larger 8-byte span_patch_decoder with copy transducer, span_cached fair-neural generation, no domain cache",
    },
    "span4_copy_2m_nocache": {
        "cpu": "results/breakthrough_equal/schema_action_span4_copy_2m_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span4_copy_2m_gpu.json",
        "architecture": "larger 4-byte span_patch_decoder with autoregressive in-span prefix and copy transducer, answer-only span training, span_cached fair-neural generation, no domain cache",
    },
    "span64_copy_2m_oneshot": {
        "cpu": "results/breakthrough_equal/schema_action_span64_copy_2m_oneshot_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span64_copy_2m_oneshot_gpu.json",
        "architecture": "64-byte span_patch_decoder with autoregressive in-span prefix and copy transducer, one-shot fair-neural generation, no domain cache",
    },
    "span64_parallel_copy_2m_oneshot": {
        "cpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_2m_parallel_oneshot_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_2m_parallel_oneshot_gpu.json",
        "architecture": "64-byte no-prefix parallel span_patch_decoder with copy transducer and answer-aligned span training, one-shot fair-neural generation, no domain cache",
    },
    "span64_parallel_copy_3m_syntaxft2_oneshot": {
        "cpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_syntaxft2_parallel_oneshot_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_syntaxft2_parallel_oneshot_gpu.json",
        "architecture": "64-byte no-prefix refined parallel span_patch_decoder with copy transducer, row-preserved answer-aligned training, syntax fine-tuning, one-shot fair-neural generation, no domain cache",
    },
    "span64_parallel_copy_3m_copyft_oneshot": {
        "cpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_copyft_parallel_oneshot_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_copyft_parallel_oneshot_gpu.json",
        "architecture": "64-byte no-prefix parallel span_patch_decoder with stronger copy-transducer fine-tuning, one-shot fair-neural generation, no domain cache",
    },
    "span64_parallel_copy_3m_promptcopyft_oneshot": {
        "cpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_promptcopyft_parallel_oneshot_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_promptcopyft_parallel_oneshot_gpu.json",
        "architecture": "64-byte no-prefix parallel span_patch_decoder with prompt-bounded copy-transducer labels, one-shot fair-neural generation, no domain cache",
    },
    "span64_parallel_copy_3m_logcopyft_oneshot": {
        "cpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_logcopyft_parallel_oneshot_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_logcopyft_parallel_oneshot_gpu.json",
        "architecture": "64-byte no-prefix parallel span_patch_decoder with centered-log copy logits and prompt-bounded copy labels, one-shot fair-neural generation, no domain cache",
    },
    "span64_parallel_copy_3m_centerprobcopyft_oneshot": {
        "cpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_centerprobcopyft_parallel_oneshot_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_centerprobcopyft_parallel_oneshot_gpu.json",
        "architecture": "64-byte no-prefix parallel span_patch_decoder with centered-probability copy logits and prompt-bounded copy labels, one-shot fair-neural generation, no domain cache",
    },
    "span64_prefix_copy_3m_oneshot": {
        "cpu": "results/breakthrough_equal/schema_action_span64_prefix_copy_3m_oneshot_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span64_prefix_copy_3m_oneshot_gpu.json",
        "architecture": "64-byte prefix-conditioned span_patch_decoder with copy transducer, one-shot fair-neural generation, no domain cache",
    },
    "span64_parallel_copy_3m_argmaxcopyft_oneshot": {
        "cpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_argmaxcopyft_parallel_oneshot_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_argmaxcopyft_parallel_oneshot_gpu.json",
        "architecture": "64-byte no-prefix parallel span_patch_decoder with argmax pointer copy projection and prompt-bounded copy labels, one-shot fair-neural generation, no domain cache",
    },
    "span64_parallel_copy_3m_argmaxcopyft2_oneshot": {
        "cpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_argmaxcopyft2_parallel_oneshot_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_argmaxcopyft2_parallel_oneshot_gpu.json",
        "architecture": "64-byte no-prefix parallel span_patch_decoder with stronger second-stage argmax pointer copy projection, one-shot fair-neural generation, no domain cache",
    },
    "span64_parallel_copy_3m_argmaxpromptcopyft_oneshot": {
        "cpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_argmaxpromptcopyft_parallel_oneshot_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_span64_parallel_copy_3m_argmaxpromptcopyft_parallel_oneshot_gpu.json",
        "architecture": "64-byte no-prefix parallel span_patch_decoder with argmax pointer copy projection and fully prompt-bounded copy labels, one-shot fair-neural generation, no domain cache",
    },
    "stateful_window_cache64_audit_only": {
        "cpu": "results/breakthrough_equal/schema_action_neural_only_cpu_balanced.json",
        "gpu": "results/breakthrough_equal/schema_action_neural_only_gpu_balanced.json",
        "architecture": "window_transformer local decoder, stateful cached generation, cache-contaminated audit baseline",
        "audit_only_reason": "training config initializes an in-model domain cache, so this candidate is excluded from fair neural dominance gates",
    },
    "parallelpatch8": {
        "cpu": "results/breakthrough_equal/schema_action_parallelpatch_neural_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_parallelpatch_neural_gpu.json",
        "architecture": "8-byte parallel_patch neural decoder",
    },
    "patch4_prediction": {
        "cpu": "results/breakthrough_equal/schema_action_patch4_pred_neural_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_patch4_pred_neural_gpu.json",
        "architecture": "4-byte factorized patch prediction neural decoder",
    },
    "patch4_autoregressive": {
        "cpu": "results/breakthrough_equal/schema_action_patch4_ar_neural_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_patch4_ar_neural_gpu.json",
        "architecture": "4-byte autoregressive GRU patch prediction neural decoder",
    },
    "abi_patchcell": {
        "cpu": "results/breakthrough_equal/schema_action_abi_patchcell_neural_cpu.json",
        "gpu": "results/breakthrough_equal/schema_action_abi_patchcell_neural_gpu.json",
        "architecture": "2-byte ABI patch-cell cached neural decoder",
    },
}


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def split_summary(doc: dict, split: str) -> dict:
    summary = doc["splits"][split]["summary"]
    return {
        "layercake_exact": summary["layercake"]["exact_json_accuracy"],
        "transformer_exact": summary["transformer"]["exact_json_accuracy"],
        "layercake_parseable": summary["layercake"]["parseable_json_rate"],
        "transformer_parseable": summary["transformer"]["parseable_json_rate"],
        "layercake_bps": summary["layercake"]["mean_bytes_per_second"],
        "transformer_bps": summary["transformer"]["mean_bytes_per_second"],
        "speed_ratio_layercake_over_transformer": summary[
            "mean_speed_ratio_layercake_over_transformer"
        ],
    }


def layercake_summary(doc: dict, split: str) -> dict:
    return doc["splits"][split]["summary"]["layercake"]


def split_sample_count(doc: dict, split: str) -> int:
    return len(doc["splits"][split].get("samples", []))


def layercake_northstar_quality_gates(doc: dict, split: str, prefix: str) -> dict:
    summary = layercake_summary(doc, split)
    return {
        f"{prefix}_{split}_layercake_exact_at_northstar_floor": summary[
            "exact_json_accuracy"
        ]
        >= NORTHSTAR_MIN_EXACT_JSON_ACCURACY,
        f"{prefix}_{split}_layercake_parse_at_northstar_floor": summary[
            "parseable_json_rate"
        ]
        >= NORTHSTAR_MIN_PARSEABLE_JSON_RATE,
        f"{prefix}_{split}_layercake_similarity_at_northstar_floor": summary[
            "mean_char_similarity"
        ]
        >= NORTHSTAR_MIN_CHAR_SIMILARITY,
        f"{prefix}_{split}_generation_examples_present": split_sample_count(
            doc, split
        )
        >= NORTHSTAR_MIN_SAMPLES_PER_SPLIT,
    }


def candidate_score(candidate: dict) -> tuple[float, float]:
    exact = (
        candidate["cpu_seen"]["layercake_exact"]
        + candidate["cpu_heldout"]["layercake_exact"]
        + candidate["gpu_seen"]["layercake_exact"]
        + candidate["gpu_heldout"]["layercake_exact"]
    ) / 4.0
    speed = (
        candidate["cpu_seen"]["speed_ratio_layercake_over_transformer"]
        + candidate["cpu_heldout"]["speed_ratio_layercake_over_transformer"]
        + candidate["gpu_seen"]["speed_ratio_layercake_over_transformer"]
        + candidate["gpu_heldout"]["speed_ratio_layercake_over_transformer"]
    ) / 4.0
    return exact, speed


def main() -> None:
    candidates = {}
    for name, spec in CANDIDATES.items():
        cpu = load(spec["cpu"])
        gpu = load(spec["gpu"])
        fair_flags = {
            "structured_schema_head": bool(cpu.get("layercake_structured_schema_head"))
            or bool(gpu.get("layercake_structured_schema_head")),
            "direct_domain_cache": bool(cpu.get("layercake_direct_domain_cache"))
            or bool(gpu.get("layercake_direct_domain_cache")),
            "audit_only": bool(spec.get("audit_only_reason")),
            "not_fair_neural_mode": cpu.get("benchmark_mode", "fair_neural") != "fair_neural"
            or gpu.get("benchmark_mode", "fair_neural") != "fair_neural",
        }
        gates = {
            "fair_neural_path": not fair_flags["structured_schema_head"]
            and not fair_flags["direct_domain_cache"],
            "not_audit_only": not fair_flags["audit_only"],
            "benchmark_mode_fair_neural": not fair_flags["not_fair_neural_mode"],
            "cpu_seen_5x_speed": cpu["splits"]["seen"]["summary"][
                "mean_speed_ratio_layercake_over_transformer"
            ]
            >= 5.0,
            "cpu_heldout_5x_speed": cpu["splits"]["heldout"]["summary"][
                "mean_speed_ratio_layercake_over_transformer"
            ]
            >= 5.0,
            "gpu_seen_5x_speed": gpu["splits"]["seen"]["summary"][
                "mean_speed_ratio_layercake_over_transformer"
            ]
            >= 5.0,
            "gpu_heldout_5x_speed": gpu["splits"]["heldout"]["summary"][
                "mean_speed_ratio_layercake_over_transformer"
            ]
            >= 5.0,
            "cpu_seen_quality_noninferior": cpu["splits"]["seen"]["summary"][
                "layercake"
            ]["exact_json_accuracy"]
            >= cpu["splits"]["seen"]["summary"]["transformer"]["exact_json_accuracy"],
            "cpu_heldout_quality_noninferior": cpu["splits"]["heldout"]["summary"][
                "layercake"
            ]["exact_json_accuracy"]
            >= cpu["splits"]["heldout"]["summary"]["transformer"][
                "exact_json_accuracy"
            ],
            "gpu_seen_quality_noninferior": gpu["splits"]["seen"]["summary"][
                "layercake"
            ]["exact_json_accuracy"]
            >= gpu["splits"]["seen"]["summary"]["transformer"]["exact_json_accuracy"],
            "gpu_heldout_quality_noninferior": gpu["splits"]["heldout"]["summary"][
                "layercake"
            ]["exact_json_accuracy"]
            >= gpu["splits"]["heldout"]["summary"]["transformer"][
                "exact_json_accuracy"
            ],
            "cpu_seen_parse_noninferior": cpu["splits"]["seen"]["summary"][
                "layercake"
            ]["parseable_json_rate"]
            >= cpu["splits"]["seen"]["summary"]["transformer"]["parseable_json_rate"],
            "cpu_heldout_parse_noninferior": cpu["splits"]["heldout"]["summary"][
                "layercake"
            ]["parseable_json_rate"]
            >= cpu["splits"]["heldout"]["summary"]["transformer"][
                "parseable_json_rate"
            ],
            "gpu_seen_parse_noninferior": gpu["splits"]["seen"]["summary"][
                "layercake"
            ]["parseable_json_rate"]
            >= gpu["splits"]["seen"]["summary"]["transformer"]["parseable_json_rate"],
            "gpu_heldout_parse_noninferior": gpu["splits"]["heldout"]["summary"][
                "layercake"
            ]["parseable_json_rate"]
            >= gpu["splits"]["heldout"]["summary"]["transformer"][
                "parseable_json_rate"
            ],
        }
        gates.update(layercake_northstar_quality_gates(cpu, "seen", "cpu"))
        gates.update(layercake_northstar_quality_gates(cpu, "heldout", "cpu"))
        gates.update(layercake_northstar_quality_gates(gpu, "seen", "gpu"))
        gates.update(layercake_northstar_quality_gates(gpu, "heldout", "gpu"))
        candidates[name] = {
            "architecture": spec["architecture"],
            "artifacts": {"cpu": spec["cpu"], "gpu": spec["gpu"]},
            "parameters": cpu["checkpoint_parameters"],
            "training": {
                "layercake_seconds": cpu["training"]["layercake"]["train_seconds"],
                "transformer_seconds": cpu["training"]["transformer"]["train_seconds"],
                "layercake_eval_bpb": cpu["training"]["layercake"]["eval_bpb"],
                "transformer_eval_bpb": cpu["training"]["transformer"]["eval_bpb"],
            },
            "flags": fair_flags,
            "audit_only_reason": spec.get("audit_only_reason"),
            "gates": gates,
            "status": "PASS" if all(gates.values()) else "FAIL",
            "cpu_seen": split_summary(cpu, "seen"),
            "cpu_heldout": split_summary(cpu, "heldout"),
            "gpu_seen": split_summary(gpu, "seen"),
            "gpu_heldout": split_summary(gpu, "heldout"),
        }

    passing = [
        name for name, candidate in candidates.items() if candidate["status"] == "PASS"
    ]
    best_name = max(candidates, key=lambda name: candidate_score(candidates[name]))
    best = candidates[best_name]
    best_blockers = [
        name for name, passed in best["gates"].items() if not passed
    ]
    if passing:
        bottom_line = (
            f"{passing[0]} clears the strict fair-neural north-star gates: 5x "
            "CPU/GPU speed, noninferior transformer comparison, high absolute "
            "exact JSON accuracy, high parseability, generation examples, and no "
            "structured/domain/cache shortcut flags."
        )
    else:
        bottom_line = (
            "No current fair-neural LayerCake candidate proves the north-star "
            f"dominance claim. Best current candidate by exactness/speed is "
            f"{best_name}, but it fails {len(best_blockers)} gate(s), including: "
            + ", ".join(best_blockers[:8])
            + "."
        )

    result = {
        "status": "PASS" if passing else "FAIL",
        "scope": (
            "Fair trained-neural LayerCake candidate comparison against the same "
            "trained tokenizer transformer on the balanced schema-action benchmark. "
            "Structured parser heads and direct answer caches are excluded from "
            "passing neural dominance gates."
        ),
        "northstar_quality_floor": {
            "min_exact_json_accuracy": NORTHSTAR_MIN_EXACT_JSON_ACCURACY,
            "min_parseable_json_rate": NORTHSTAR_MIN_PARSEABLE_JSON_RATE,
            "min_char_similarity": NORTHSTAR_MIN_CHAR_SIMILARITY,
            "min_generation_examples_per_split": NORTHSTAR_MIN_SAMPLES_PER_SPLIT,
        },
        "generation_examples_artifact": (
            "results/breakthrough_equal/schema_action_fair_neural_generation_examples.md"
        ),
        "best_current_candidate": best_name,
        "best_current_candidate_blockers": best_blockers,
        "candidates": candidates,
        "bottom_line": bottom_line,
    }
    out = ROOT / "results/breakthrough_equal/schema_action_fair_neural_candidate_report.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
