# Breakthrough Dominance Protocol

This protocol is the hard target for proving a byte-level LayerCake breakthrough
against an equal-size tokenizer transformer. It is intentionally stricter than the
existing smoke, platform-slice, and 15M frontier gates.

## Claim Boundary

A breakthrough certificate may be promoted only when all of these are true:

- LayerCake and transformer trainable parameters are within 5%.
- Training and evaluation corpora are byte-identical, with frozen disjoint splits.
- Held-out evaluation uses at least 1,000,000 bytes.
- LayerCake held-out BPB is at least 5x better than transformer BPB.
- LayerCake training wall time is at least 5x faster.
- LayerCake parameter-time cost proxy is at least 5x lower.
- LayerCake uses no more training bytes.
- LayerCake CPU generation is at least 5x faster.
- LayerCake GPU generation is at least 5x faster.
- LayerCake generation quality or task score is at least 5x higher on CPU and GPU.
- Relevance is non-inferior, and the LayerCake relevance rate is complete.

The verifier is:

```powershell
python scripts/verify_breakthrough_equal_size_dominance.py `
  --layercake-training runs_experiment/breakthrough_equal_layercake/training_metrics.json `
  --transformer-training runs_experiment/breakthrough_equal_bpe/training_metrics.json `
  --layercake-cpu-generation results/breakthrough_equal/layercake_cpu_generation.json `
  --transformer-cpu-generation results/breakthrough_equal/transformer_cpu_generation.json `
  --layercake-gpu-generation results/breakthrough_equal/layercake_gpu_generation.json `
  --transformer-gpu-generation results/breakthrough_equal/transformer_gpu_generation.json `
  --output results/breakthrough_equal/breakthrough_equal_size_certificate.json
```

## First Serious Run

Use the strongest known architecture family as the first candidate:

- fixed two-byte ABI positions;
- empirical transition/context priors;
- transition-head or AR patch path from the promoted 15M frontier;
- no domain cache override;
- exact raw neural generation path;
- equal-size BPE transformer comparator, not smaller.

The current strict v3 certificate is not a breakthrough candidate. It loses held-out
BPB, training time, training cost, equal-size fairness, relevance, and GPU margin. Use
it as a negative control.

## What To Optimize First

The current evidence says CPU generation can win, transfer exactness can win, and
portable domain behavior can win. The breakthrough blocker is general held-out BPB
and training efficiency at equal size.

Optimization order:

1. Close held-out BPB on the frozen split.
2. Preserve or improve training time while closing BPB.
3. Add real task scoring for generation quality, not only printable text heuristics.
4. Re-run CPU and GPU generation on the same checkpoints.
5. Only then run the breakthrough verifier.

## Promotion Rule

Do not promote a partial certificate as breakthrough. If any gate fails, record the
certificate as a failed experiment and use its blockers to choose the next architecture
or training change.
