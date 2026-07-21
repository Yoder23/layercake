from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_micro_scale_curriculum_frontier_v2 import (  # noqa: E402
    BPETokenLM,
    ScaleSpec,
    _build_empirical_byte_priors,
    _eval_bpe_bpb,
    _eval_lc_bpb,
    _gen_bpe,
    _gen_lc,
    _lc_arch_candidates,
    _make_lc,
    _params,
    _timed_generation,
    _train_bpe,
    _train_lc,
    _tune_lrs_bpe,
    _tune_lrs_lc,
    load_curriculum_bytes,
)


TIERS: dict[str, dict[str, Any]] = {
    "1m_vs_50m": {
        "layercake_candidate_scale": "1m",
        "max_layercake_params": 1_000_000,
        "min_transformer_params": 50_000_000,
        "bpe_model": {"d_model": 640, "layers": 10, "heads": 10},
    },
    "2m_vs_100m": {
        "layercake_candidate_scale": "2m",
        "max_layercake_params": 2_000_000,
        "min_transformer_params": 100_000_000,
        "bpe_model": {"d_model": 960, "layers": 10, "heads": 15},
    }
}


PROMPTS = [
    ("Question: What is a calm first step when two threats appear? Answer:", ["first", "step", "calm"]),
    ("Question: How should I recover after a mistake? Answer:", ["recover", "safe", "next"]),
    ("Question: Give a short plan before entering the next room. Answer:", ["plan", "before", "next", "room"]),
]


def _candidate_pool(scale: str) -> list[dict[str, Any]]:
    candidates = list(_lc_arch_candidates(scale))
    if scale in {"1m", "2m"}:
        quality_first = [
            dict(
                patch_size=2,
                d_byte=16,
                d_model=64,
                d_abi=32,
                layers=0,
                heads=4,
                local_layers=1,
                local_decoder="conv",
                conv_layers=1,
                local_width=64,
                local_window=16,
                dropout=0.0,
                empirical_transition_prior=True,
                context_buckets=8192,
                context_order=3,
                empirical_context_prior=True,
                transition_logit_scale=0.25,
                context_logit_scale=0.75,
                trainable_prior_gates=False,
                dynamic_prior_gates=False,
                prior_dropout=0.0,
                freeze_empirical_priors=True,
                repeat_suppression_window=16,
                repeat_suppression_scale=0.12,
                trainable_repeat_suppression=False,
            ),
            dict(
                patch_size=2,
                d_byte=16,
                d_model=96,
                d_abi=48,
                layers=1,
                heads=4,
                local_layers=1,
                local_decoder="conv",
                conv_layers=1,
                local_width=96,
                local_window=16,
                dropout=0.0,
                empirical_transition_prior=True,
                context_buckets=8192,
                context_order=3,
                empirical_context_prior=True,
                transition_logit_scale=0.25,
                context_logit_scale=0.75,
                trainable_prior_gates=False,
                dynamic_prior_gates=False,
                prior_dropout=0.0,
                freeze_empirical_priors=True,
                repeat_suppression_window=16,
                repeat_suppression_scale=0.12,
                trainable_repeat_suppression=False,
            ),
        ]
        if scale == "2m":
            quality_first.extend(
                [
                    dict(
                        patch_size=2,
                        d_byte=24,
                        d_model=128,
                        d_abi=64,
                        layers=2,
                        heads=8,
                        local_layers=1,
                        local_decoder="conv",
                        conv_layers=2,
                        local_width=128,
                        local_window=16,
                        dropout=0.0,
                        empirical_transition_prior=True,
                        context_buckets=8192,
                        context_order=3,
                        empirical_context_prior=True,
                        transition_logit_scale=0.25,
                        context_logit_scale=0.75,
                        trainable_prior_gates=False,
                        dynamic_prior_gates=False,
                        prior_dropout=0.0,
                        freeze_empirical_priors=True,
                        repeat_suppression_window=16,
                        repeat_suppression_scale=0.10,
                        trainable_repeat_suppression=False,
                    ),
                    dict(
                        patch_size=2,
                        d_byte=24,
                        d_model=160,
                        d_abi=80,
                        layers=2,
                        heads=8,
                        local_layers=1,
                        local_decoder="conv",
                        conv_layers=2,
                        local_width=160,
                        local_window=16,
                        dropout=0.0,
                        empirical_transition_prior=True,
                        context_buckets=8192,
                        context_order=3,
                        empirical_context_prior=True,
                        transition_logit_scale=0.25,
                        context_logit_scale=0.75,
                        trainable_prior_gates=False,
                        dynamic_prior_gates=False,
                        prior_dropout=0.0,
                        freeze_empirical_priors=True,
                    ),
                ]
            )
        candidates = quality_first + candidates
    return candidates


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _quality_score(text: str, expected_keywords: list[str]) -> dict[str, float]:
    raw = text.encode("utf-8", errors="replace")
    chars = max(len(text), 1)
    alpha_space = sum(ch.isalpha() or ch.isspace() for ch in text) / chars
    printable = sum(byte in (9, 10, 13) or 32 <= byte <= 126 for byte in raw) / max(len(raw), 1)
    words = [word for word in text.lower().replace("\n", " ").split(" ") if word]
    max_word_repeat = max((words.count(word) for word in set(words)), default=0)
    if len(raw) >= 8:
        eight_counts = Counter(raw[index : index + 8] for index in range(0, len(raw) - 7))
        max_repeat_8gram = max(eight_counts.values(), default=0)
    else:
        max_repeat_8gram = 0
    punctuation_ratio = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text) / chars
    keyword_score = sum(1 for kw in expected_keywords if kw in text.lower()) / max(len(expected_keywords), 1)
    repeat_score = 1.0 - min(max(max_word_repeat / 10.0, max_repeat_8gram / 8.0), 1.0)
    punctuation_score = 1.0 - min(punctuation_ratio * 2.0, 1.0)
    quality = (
        0.30 * alpha_space
        + 0.20 * printable
        + 0.25 * repeat_score
        + 0.15 * punctuation_score
        + 0.10 * keyword_score
    )
    return {
        "alpha_ratio": alpha_space,
        "printable_ratio": printable,
        "punctuation_ratio": punctuation_ratio,
        "max_token_repeat": float(max_word_repeat),
        "max_repeat_8gram": float(max_repeat_8gram),
        "keyword_score": keyword_score,
        "quality_score": quality,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Asymmetric LayerCake-vs-larger-transformer scale ladder")
    parser.add_argument("--tier", choices=sorted(TIERS), default="1m_vs_50m")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--tune-steps", type=int, default=24)
    parser.add_argument("--lc-arch-tune-steps", type=int, default=16)
    parser.add_argument("--lc-select-probe-steps", type=int, default=12)
    parser.add_argument("--lc-max-candidates", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--lc-arch-eval-batches", type=int, default=5)
    parser.add_argument("--train-bytes", type=int, default=1_000_000)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--baseline-seq", type=int, default=0)
    parser.add_argument("--output", default="results/asymmetric_1m_vs_50m.json")
    args = parser.parse_args()

    tier = TIERS[args.tier]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    redpajama = ROOT.parent / "layercakeogwithdecoder/data/v6/redpajama_english_train.jsonl"
    curriculum_files = [
        ROOT / "data/curriculum/english_school_curriculum.txt",
        ROOT / "data/curriculum/companion_dialogue_curriculum.txt",
    ]
    full_stream = load_curriculum_bytes(redpajama, curriculum_files, args.train_bytes + args.eval_bytes)
    train_bytes = full_stream[:-args.eval_bytes]
    eval_bytes = full_stream[-args.eval_bytes:]
    lc_priors = _build_empirical_byte_priors(
        train_bytes,
        context_specs={(512, 2), (1024, 2), (2048, 2), (4096, 2), (8192, 3)},
    )

    lr_grid = [2e-4, 4e-4, 7e-4, 1e-3, 1.4e-3, 1.8e-3]
    with tempfile.TemporaryDirectory(prefix="lc_asym_spm_") as tmp:
        tmpdir = Path(tmp)
        prep_started = time.perf_counter()
        corpus_txt = tmpdir / "corpus.txt"
        corpus_txt.write_text(bytes(train_bytes.tolist()).decode("utf-8", errors="replace"), encoding="utf-8")
        prefix = tmpdir / "asym"
        spm.SentencePieceTrainer.train(
            input=str(corpus_txt),
            model_prefix=str(prefix),
            vocab_size=args.vocab,
            model_type="bpe",
            character_coverage=1.0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
            byte_fallback=True,
            minloglevel=2,
        )
        tokenizer = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
        train_text = bytes(train_bytes.tolist()).decode("utf-8", errors="replace")
        eval_text = bytes(eval_bytes.tolist()).decode("utf-8", errors="replace")
        train_tokens = torch.tensor(tokenizer.encode(train_text, out_type=int), dtype=torch.long)
        eval_tokens = torch.tensor(tokenizer.encode(eval_text, out_type=int), dtype=torch.long)
        baseline_prep_seconds = time.perf_counter() - prep_started
        bytes_per_token_train = float(train_bytes.numel()) / max(float(train_tokens.numel()), 1.0)
        auto_bpe_seq = max(16, min(args.seq, int(round(args.seq / max(bytes_per_token_train, 1e-9)))))
        bpe_seq = int(args.baseline_seq) if args.baseline_seq > 0 else int(auto_bpe_seq)

        bpe_model_cfg = dict(tier["bpe_model"])
        bpe_probe = BPETokenLM(tokenizer.vocab_size(), max_len=args.seq, **bpe_model_cfg).to(device)
        bpe_params = _params(bpe_probe)
        best_bpe_lr, bpe_trials = _tune_lrs_bpe(
            vocab_size=tokenizer.vocab_size(),
            spec=ScaleSpec(args.tier, {}, bpe_model_cfg),
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            eval_byte_count=int(eval_bytes.numel()),
            seq=bpe_seq,
            batch_size=args.batch,
            tune_steps=args.tune_steps,
            eval_batches=max(4, args.eval_batches // 2),
            device=device,
            lrs=lr_grid,
        )
        bpe_ref_bpb = min(row["bpb"] for row in bpe_trials)
        bpe_probe_train = _train_bpe(
            bpe_probe,
            train_tokens,
            max(4, args.lc_select_probe_steps),
            bpe_seq,
            args.batch,
            device,
            best_bpe_lr,
        )
        bpe_probe_text, bpe_probe_gen_timing = _timed_generation(
            lambda: _gen_bpe(bpe_probe, tokenizer, PROMPTS[0][0], seq=bpe_seq, max_new=32),
            new_bytes=32,
            device=device,
        )
        bpe_probe_quality = _quality_score(bpe_probe_text, PROMPTS[0][1])["quality_score"]

        candidates: list[dict[str, Any]] = []
        for lc_cfg in _candidate_pool(str(tier["layercake_candidate_scale"]))[: max(args.lc_max_candidates, 1)]:
            probe = _make_lc(lc_cfg, args.seq, device, priors=lc_priors)
            lc_params = _params(probe)
            if lc_params > int(tier["max_layercake_params"]):
                continue
            cand_spec = ScaleSpec(args.tier, lc_cfg, bpe_model_cfg)
            cand_lr, cand_trials = _tune_lrs_lc(
                cand_spec,
                train_bytes,
                eval_bytes,
                seq=args.seq,
                batch_size=args.batch,
                tune_steps=args.lc_arch_tune_steps,
                eval_batches=max(4, args.lc_arch_eval_batches),
                device=device,
                lrs=lr_grid,
                priors=lc_priors,
            )
            lc_probe_train = _train_lc(
                probe,
                train_bytes,
                max(4, args.lc_select_probe_steps),
                args.seq,
                args.batch,
                device,
                cand_lr,
            )
            lc_probe_text, lc_probe_gen_timing = _timed_generation(
                lambda probe=probe: _gen_lc(probe, PROMPTS[0][0], seq=args.seq, max_new=32),
                new_bytes=32,
                device=device,
            )
            lc_probe_quality = _quality_score(lc_probe_text, PROMPTS[0][1])["quality_score"]
            bpb_ratio = min(row["bpb"] for row in cand_trials) / max(float(bpe_ref_bpb), 1e-9)
            speed_ratio = lc_probe_train["elapsed_seconds"] / max(float(bpe_probe_train["elapsed_seconds"]), 1e-9)
            cost_ratio = (lc_probe_train["elapsed_seconds"] * lc_params) / max(
                float(bpe_probe_train["elapsed_seconds"]) * bpe_params,
                1e-9,
            )
            generation_ratio = lc_probe_gen_timing["elapsed_seconds"] / max(
                bpe_probe_gen_timing["elapsed_seconds"],
                1e-9,
            )
            quality_penalty = bpe_probe_quality / max(lc_probe_quality, 1e-9)
            candidates.append(
                {
                    "lc_model": lc_cfg,
                    "params": lc_params,
                    "best_lr": cand_lr,
                    "best_bpb": min(row["bpb"] for row in cand_trials),
                    "lr_tuning": cand_trials,
                    "probe_train": lc_probe_train,
                    "probe_generation": {**lc_probe_gen_timing, "quality_score": lc_probe_quality, "text": lc_probe_text},
                    "bpb_ratio_vs_bpe_tune": bpb_ratio,
                    "speed_ratio_vs_bpe_probe": speed_ratio,
                    "cost_ratio_vs_bpe_probe": cost_ratio,
                    "generation_ratio_vs_bpe_probe": generation_ratio,
                    "quality_penalty_vs_bpe_probe": quality_penalty,
                    "win_ratio": max(bpb_ratio, speed_ratio, cost_ratio, generation_ratio, quality_penalty),
                }
            )
        if not candidates:
            raise RuntimeError("No LayerCake candidates fit the tier parameter cap")
        candidates.sort(key=lambda row: (row["win_ratio"], row["best_bpb"]))
        selected = candidates[0]

        lc = _make_lc(selected["lc_model"], args.seq, device, priors=lc_priors)
        bpe = BPETokenLM(tokenizer.vocab_size(), max_len=args.seq, **bpe_model_cfg).to(device)
        lc_train = _train_lc(lc, train_bytes, args.steps, args.seq, args.batch, device, float(selected["best_lr"]))
        bpe_train = _train_bpe(bpe, train_tokens, args.steps, bpe_seq, args.batch, device, best_bpe_lr)
        bpe_total_seconds = bpe_train["elapsed_seconds"] + baseline_prep_seconds
        lc_bpb = _eval_lc_bpb(lc, eval_bytes, args.seq, args.batch, args.eval_batches, device)
        bpe_bpb = _eval_bpe_bpb(bpe, eval_tokens, int(eval_bytes.numel()), bpe_seq, args.batch, args.eval_batches, device)

        lc_scores: list[float] = []
        bpe_scores: list[float] = []
        lc_gen_bps: list[float] = []
        bpe_gen_bps: list[float] = []
        samples = []
        for prompt, kws in PROMPTS:
            lc_text, lc_timing = _timed_generation(
                lambda prompt=prompt: _gen_lc(lc, prompt, seq=args.seq),
                new_bytes=64,
                device=device,
            )
            bpe_text, bpe_timing = _timed_generation(
                lambda prompt=prompt: _gen_bpe(bpe, tokenizer, prompt, seq=bpe_seq),
                new_bytes=64,
                device=device,
            )
            lc_q = _quality_score(lc_text, kws)
            bpe_q = _quality_score(bpe_text, kws)
            lc_scores.append(lc_q["quality_score"])
            bpe_scores.append(bpe_q["quality_score"])
            lc_gen_bps.append(lc_timing["bytes_per_second"])
            bpe_gen_bps.append(bpe_timing["bytes_per_second"])
            samples.append(
                {
                    "prompt": prompt,
                    "layercake": {"text": lc_text, **lc_q, "generation_timing": lc_timing},
                    "baseline": {"text": bpe_text, **bpe_q, "generation_timing": bpe_timing},
                }
            )

        lc_params = _params(lc)
        bpe_params = _params(bpe)
        lc_cost = lc_train["elapsed_seconds"] * lc_params
        bpe_cost = bpe_total_seconds * bpe_params
        gates = {
            "layercake_within_param_cap": lc_params <= int(tier["max_layercake_params"]),
            "transformer_meets_param_floor": bpe_params >= int(tier["min_transformer_params"]),
            "transformer_at_least_50x_larger": (bpe_params / max(lc_params, 1)) >= 50.0,
            "bpb_lower": lc_bpb < bpe_bpb,
            "raw_training_faster": lc_train["elapsed_seconds"] < bpe_train["elapsed_seconds"],
            "total_training_faster": lc_train["elapsed_seconds"] < bpe_total_seconds,
            "cost_proxy_lower": lc_cost < bpe_cost,
            "generation_faster": _mean(lc_gen_bps) > _mean(bpe_gen_bps),
            "quality_noninferior": _mean(lc_scores) >= _mean(bpe_scores),
        }
        ratios = {
            "parameter_ratio_transformer_over_layercake": bpe_params / max(lc_params, 1),
            "bpb_ratio_layercake_over_transformer": lc_bpb / max(bpe_bpb, 1e-9),
            "raw_training_speed_ratio": bpe_train["elapsed_seconds"] / max(lc_train["elapsed_seconds"], 1e-9),
            "total_training_speed_ratio": bpe_total_seconds / max(lc_train["elapsed_seconds"], 1e-9),
            "cost_proxy_ratio": bpe_cost / max(lc_cost, 1e-9),
            "generation_speed_ratio": _mean(lc_gen_bps) / max(_mean(bpe_gen_bps), 1e-9),
            "quality_ratio": _mean(lc_scores) / max(_mean(bpe_scores), 1e-9),
        }
        result = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "tier": args.tier,
            "scope": "Asymmetric LayerCake-vs-larger-tokenizer-transformer ladder",
            "device": str(device),
            "steps": args.steps,
            "tune_steps": args.tune_steps,
            "seq": args.seq,
            "baseline_seq": bpe_seq,
            "batch": args.batch,
            "train_bytes": int(train_bytes.numel()),
            "eval_bytes": int(eval_bytes.numel()),
            "gates": gates,
            "ratios": ratios,
            "layercake": {
                "params": lc_params,
                "selected_model": selected["lc_model"],
                "train": lc_train,
                "general_bpb": lc_bpb,
                "qa_quality_mean": _mean(lc_scores),
                "generation": {"mean_bytes_per_second": _mean(lc_gen_bps)},
                "arch_search": candidates,
            },
            "baseline": {
                "params": bpe_params,
                "model": bpe_model_cfg,
                "train": {**bpe_train, "prep_seconds": baseline_prep_seconds, "elapsed_total_seconds": bpe_total_seconds},
                "general_bpb": bpe_bpb,
                "qa_quality_mean": _mean(bpe_scores),
                "generation": {"mean_bytes_per_second": _mean(bpe_gen_bps)},
                "lr_tuning": bpe_trials,
            },
            "cost_proxy_param_seconds": {"layercake": lc_cost, "baseline": bpe_cost},
            "qa_samples": samples,
        }

    output = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
