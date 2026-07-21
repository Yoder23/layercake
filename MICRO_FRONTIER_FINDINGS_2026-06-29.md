# Micro Frontier Findings (2026-06-29)

Source artifact:
- results/micro_scale_curriculum_frontier_1m_10m.json

## Verdict

- Overall: FAIL (quality gates not yet met at 1M/2M/5M/10M)
- LayerCake already wins speed and cost proxy when tokenizer prep is included.
- LayerCake does not yet beat baseline on bpb or QA quality at these step budgets.

## Scale summary

- 1M:
  - speed ratio (LayerCake / baseline total): 0.728 (win)
  - bpb gap (LayerCake - baseline): +0.670 (loss)
  - qa quality gap: -0.024 (loss)
  - param ratio: 1.071 (loss)

- 2M:
  - speed ratio: 0.706 (win)
  - bpb gap: +0.723 (loss)
  - qa quality gap: -0.027 (loss)
  - param ratio: 0.993 (pass)

- 5M:
  - speed ratio: 0.548 (win)
  - bpb gap: +0.770 (loss)
  - qa quality gap: -0.017 (loss)
  - param ratio: 0.747 (pass)

- 10M:
  - speed ratio: 0.541 (win)
  - bpb gap: +0.665 (loss)
  - qa quality gap: -0.088 (loss)
  - param ratio: 1.038 (loss)

## Interpretation

- The byte LayerCake stack is computationally efficient in this setup.
- Quality is the current blocker, not throughput.
- A strict "go to 100M" is not justified until quality gates pass in this micro frontier.

## Immediate optimization priorities

1. Close bpb gap first (primary objective).
2. Reduce repetition in LayerCake generation at low scale.
3. Rebalance 1M and 10M architecture points to ensure params <= baseline.
4. Increase curriculum quality weighting and move to staged curriculum -> redpajama schedule.
5. Re-run micro frontier with tuned LayerCake hyperparameters before promoting scale.

## 2026-06-30 byte-prior probe

Implemented an empirical byte transition/context prior path in
`benchmark_micro_scale_curriculum_frontier_v2.py`. This gives LayerCake a tokenizer-free
corpus preview comparable in spirit to the BPE tokenizer's corpus preparation step.

Probe artifact:
- `results/micro_prior_bpb_probe_1m10m.json`
- strict certificate: `results/micro_prior_bpb_probe_1m10m_strict_certificate.json`

Result:
- Overall strict dominance: FAIL.
- BPB-first selection now beats BPE BPB at 1M, 2M, and 5M in the bounded 200-step /
  1M-byte probe.
- 10M still misses BPB.
- BPB-first candidates lose too much raw training speed/cost at 1M/2M/5M.
- Low-scale generation repetition remains the major quality blocker.

Interpretation:
- Empirical byte priors are a useful direction and should stay in the search ladder.
- They are not sufficient by themselves. The next architectural target is a
  repetition-resistant small byte decoder / objective that preserves the BPB gains while
  restoring raw training speed and generation quality.
