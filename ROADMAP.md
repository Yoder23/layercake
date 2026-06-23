# LayerCake Research Roadmap

## Completed local tiers

- 0.35M compact patch core: selected small-scale BPB parity point estimate and bounded
  cross-seed/cross-size/int8 transfer.
- 5.40M patch core: 62.9% smaller than the byte baseline, approximately 2x faster,
  bounded sparse-brick adaptation and transfer; BPE general BPB still leads.
- 15.45M patch core: 5,000-step paired checkpoint; general BPB 2.430 versus 2.227
  for the 25.75M byte baseline, while running 2.45x faster in the selected CUDA benchmark.
  Exact portable-domain PPL/logit/generation transfer
  passes between 15.45M and 5.40M hosts.
- Portable recurrent Python domain: 148,736 parameters, 594,944-byte fp32 payload,
  148,808-byte int8 artifact, held-out PPL 2.71-2.86, and exact cross-host transfer.
- Mobile CPU domain contest: LayerCake beats a matched rank-16 BPE transformer adapter on
  Python BPB, training wall time, artifact size, and one-thread CPU throughput. The BPE
  adapter remains faster on GPU.

## Current and next scale work

The current configuration is:

```text
patch core: 15.45M parameters
byte baseline: 25.75M parameters
d_model patch/byte: 384 / 512
layers patch/byte: 8 / 8
d_abi: 128
context: 256 bytes
general corpus: >= 20 MB locally, then >= 1B bytes externally
```

The matched 5,000-step first-seed run is complete. Independent seeds remain before any
scaling-law or quality-parity announcement.

1. Train tokenized and byte-patch cores with paired canonical-ABI losses.
2. Require domain/general PPL preservation gates for every paste experiment.
3. Run cross-seed and cross-size matrices with at least three seeds.
4. Train a causal next-byte difficulty estimator and compare fixed/whitespace/dynamic patches.
5. Benchmark sparse brick training and installed-vs-active latency on CPU and GPU.
6. Add int8/fp8 artifact contracts and bounded-transfer gates.
7. Integrate authenticated HandoffPacket transport with CorticalSwarm.

## Required replication before broad marketing

1. Repeat the adapter comparison for at least three seeds.
2. Add at least two non-Python domains.
3. Compare against LoRA implementations that adapt attention projections.
4. Run on ARM/mobile hardware with memory, power, and thermal measurements.
5. Preserve the exact-transfer gate at 60M and 150M host tiers.
6. Close the general BPB and GPU throughput gaps before claiming general transformer
   superiority.

The near-term objective is bounded semantic transfer, not a claim of universal losslessness.
