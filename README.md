# LayerCake

**Tokenizer-free byte-patch language models with a canonical knowledge ABI and portable sparse domain bricks.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status: Research Preview](https://img.shields.io/badge/status-research%20preview-yellow.svg)]()

LayerCake is investigating a different way to build and extend language models:

```text
UTF-8 bytes
  -> causal byte patches
  -> compact global core
  -> deterministic canonical ABI
  -> portable top-k sparse domain brick
  -> shared canonical output contract
  -> byte predictions
```

The central hypothesis is that domain knowledge can live in a fixed ABI space rather than
inside tokenizer-specific or `d_model`-specific weights. A brick should be trainable once,
copied exactly, activated sparsely, quantized, and used by independently trained LayerCake
cores of different sizes.

This repository now contains both:

- the original tokenized fixed-ABI LayerCake prototype; and
- the v2 strictly causal tokenizer-free byte-patch research system.

## Current measured result

The current north-star experiment is a fixed-budget comparison between a 14.79M-parameter
two-byte LayerCake and a 14.84M-parameter 4,096-token BPE transformer. Both train on
approximately 10.3M sampled bytes from the same local general-text stream. LayerCake uses
four global and four window-local fused transformer blocks plus exact stateful cached
byte generation.

| Gate | LayerCake result | Comparator / threshold | Status |
|---|---:|---:|---|
| General held-out BPB, seed 6250 | **2.0446** | BPE: 2.0492 | PASS |
| General held-out BPB, seed 6263 | **2.0457** | BPE: 2.0492 | PASS |
| Parameters | **14.792M** | BPE: 14.844M | PASS |
| Mean training time, two seeds | **121.4 s** | BPE: 131.5 s | PASS |
| Batch-1 prefill latency | **2.96 ms** | BPE: 5.63 ms | PASS |
| Exact cached-generation BPB | **1.9953 / 1.9836** | BPE: 2.0492 | PASS |
| One-thread CPU generation | **2.91x / 2.96x BPE** | ratio > 1 | PASS |
| RTX 3080 Laptop generation | 0.62x BPE | ratio > 1 | **FAIL** |

Cached generation is numerically equivalent to the trained full-forward path: the selected
logit comparison differs by at most `1.9e-6` and has identical argmaxes. Local attention
caches reset at the same 16-byte boundaries used during training.

This is a replicated local-corpus result, not evidence of universal tokenizer-free
dominance. The CPU result is a one-thread x86 mobile proxy, not a phone, NPU, battery, or
thermal measurement. GPU generation remains an open optimization target.

Raw evidence: [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)

Verify the combined core and migration certificate:

```powershell
python scripts/verify_northstar_mobile.py
```

## Strict same-PPL transfer

The original additive sparse brick does **not** preserve absolute PPL across independent
cores:

| Additive transfer | Source PPL | Target PPL | Ratio | Strict gate |
|---|---:|---:|---:|---|
| Small cross-seed | 56.82 | 98.91 | 1.74 | FAIL |
| 5.40M -> 2.19M | 40.63 | 84.57 | 2.08 | FAIL |

The payload copies exactly, but different ABI states select different experts and the
shared correction is added to different base logits.

LayerCake now has a separate **core-independent lossless domain mode**. A
148,736-parameter recurrent byte decoder consumes deterministic causal anchors and owns
the domain prediction path instead of modifying target-core logits.

| Lossless decoder transfer | PPL on both | Top-1 byte accuracy | Logit max diff | Ratio |
|---|---:|---:|---:|---:|
| Small cross-seed, context 128 | 2.8553 | 72.60% | 0.0 | **1.0000** |
| 5.40M -> 2.19M, context 256 | 2.7143 | 73.76% | 0.0 | **1.0000** |
| 15.45M -> 5.40M, context 256 | 2.7143 | 73.76% | 0.0 | **1.0000** |
| int8 artifact, 15.45M -> 5.40M | 2.7165 | 73.77% | 0.0 | **1.0000** |

This proves exact domain-PPL portability for the explicit lossless mode. Additive mode
uses the host model and is bounded but not exact; lossless mode is exact because its
predictions do not depend on the host core.

The fp32 payload is 594,944 bytes. Symmetric per-tensor int8 storage is 148,808 bytes and
increases PPL by 0.083%. The current loader dequantizes to fp32; this is compact artifact
transport/storage evidence, not a native int8-kernel speed claim.

One-thread x86 CPU proxy results for 128-byte forward inference are 3.81 ms median and
33.6K bytes/s. This is not yet an Android, iOS, NPU, battery, or thermal benchmark.

### Mobile deployment thesis

The evidence now supports this precise positioning:

> Train a byte-level domain capsule once, verify its content hash, and install the same
> 149 KB int8 artifact on compatible LayerCake runtimes without retraining each host.

The tested artifact preserves its logits, PPL, byte accuracy, and deterministic output
across 2.19M, 5.40M, and 15.45M LayerCake hosts. This is the mechanism LayerCake is
developing for mobile domain deployment: a smaller general core plus installable,
domain-specific prediction payloads.

It is not evidence that a mobile core has the same general intelligence as a larger core.
PX transfers the domain capsule's behavior exactly because that capsule owns the selected
domain prediction path. Routing, task-level code quality, native mobile kernels, memory,
battery, and thermal behavior remain separate gates.

### Measured mobile domain-deployment win

LayerCake was compared with a matched 14.84M-parameter BPE transformer using a rank-16
residual adapter. Both systems adapted to the same local Python domain.

| Metric | LayerCake PX | BPE transformer adapter | Winner |
|---|---:|---:|---|
| Domain BPB | **1.4418** | 2.1101 | LayerCake |
| Domain training time | **51.3 s** | 183.1 s | LayerCake, 3.57x faster |
| Deployment artifact | **148,808 B** | 383,008 B | LayerCake, 2.57x smaller |
| One-thread CPU throughput | **31.9K B/s** | 7.1K B/s | LayerCake, 4.50x faster |
| RTX 3080 Laptop throughput | 153.6K B/s | **214.8K B/s** | Transformer |
| Exact cross-host transfer | **PASS** | model-specific adapter | LayerCake |

The transformer adapter has fewer trainable parameters (95,752 versus 148,736), but trains
slower, produces a larger artifact, reaches worse domain BPB, and requires the full
14.84M-parameter tokenizer transformer at inference. With the adapter active, its general
BPB changes from 2.041 to 2.420; it must be disabled outside the domain.

The domain-quality ordering replicated across two independent adaptation seeds:
LayerCake BPB 1.4418/1.4436 versus adapter BPB 2.1101/2.0951. The unchanged payload was
also reinstalled from the new 14.79M winning core into an independent 5.40M host:
max logit difference 0, PPL ratio 1.0, and identical generated bytes. The matched BPE
transformer no longer leads the selected general BPB or mobile CPU gates; it still wins
the selected GPU generation benchmark.

Verify:

```powershell
python scripts/verify_mobile_domain_win.py
```

## Historical scaling checkpoints

The checkpoints below document the quality gap that existed before the current fused,
window-local two-byte architecture. They remain useful negative controls but are no longer
the repository frontier.

Naive scaling beyond the selected frontier has also been tested and rejected. A 23.69M
5+5-block LayerCake reached 2.0299 BPB in 214.1 seconds, and a 25.24M width-scaled 4+4
model reached 2.0376 BPB in 204.6 seconds. The matched 24.09M BPE transformer reached
2.0035 BPB in 158.0 seconds. The next scale step therefore requires better patch
compression and fused training, not simply more dense blocks.

A subsequent additive multi-scale experiment also failed its early rejection gate:
four-byte coarse summaries combined with a two-byte fine stream reached 2.4216/2.4188 BPB
at 750 steps versus 2.3180 for the fixed two-byte reference, with no training-speed gain.
The implementation remains available as an experimental path, but the next full run will
require content-dependent patch boundaries rather than additive fixed-scale summaries.

The next tier increases the patch core from 0.35M to 5.40M parameters and the ABI from
64 to 96 dimensions. It uses 20 MB of general text and 256-byte contexts.

| Gate | 5.40M result | Status |
|---|---:|---|
| Patch vs byte parameters | 5.40M vs 14.57M | PASS |
| Patch vs BPE parameters | 5.40M vs 6.90M | PASS |
| Patch base inference | 243.6K vs 122.1K bytes/s | PASS |
| Patch + brick inference | 232.0K vs 122.1K bytes/s | PASS |
| Source Python PPL | 157.03 -> 40.94 | PASS |
| Source general ratio | 1.0105 | PASS |
| 5.40M -> independent 2.19M transfer | domain ratio 0.533; general 1.021 | PASS |
| Int8 transfer | domain ratio 0.532; general 1.021 | PASS |
| General BPB vs matched BPE | 2.261 vs 2.075 | **OPEN / BPE leads** |

This larger checkpoint preserves the architecture's size, speed, adaptation, transfer,
and quantization advantages. It does **not** yet reproduce the small-scale BPB parity
result. That negative result is part of the public evidence, not hidden.

The 15.45M patch checkpoint has completed 5,000 paired steps:

| Gate | 15.45M checkpoint |
|---|---:|
| Parameters | 15.45M vs 25.75M byte core |
| General BPB | 2.430 |
| 25.75M byte baseline BPB | 2.227 |
| Patch inference | 227.6K vs 93.0K bytes/s |
| Exact int8 portable-domain PPL | 2.7165 on both 15.45M and 5.40M hosts |
| Filesystem-disjoint stdlib PPL | 5.8296 on both hosts |
| Exact generated-byte identity | PASS |

The patch model is 40.0% smaller and 2.45x faster in this CUDA benchmark. The byte model
still leads quality by 0.203 BPB. This is a single-seed, 20 MB local-corpus result.

Verify it with:

```powershell
python scripts/verify_scale5m_results.py
```

## Verify the selected evidence

```powershell
pip install -e .[dev]
pytest -q
python scripts/verify_research_gates.py
python scripts/verify_scale5m_results.py
python scripts/verify_scale15m_results.py
python scripts/verify_lossless_domain_results.py
python scripts/verify_mobile_domain_win.py
python scripts/verify_northstar_mobile.py
python scripts/eval_lossless_domain_decoder.py `
  --decoder runs_experiment/portable_python_gru148k_v1.pt `
  --source-core runs_experiment/scale5m_seed4242_continued.pt `
  --target-core runs_experiment/scale2m_seed5151.pt `
  --output results/lossless_domain_scale5m_to_2m.json
```

Expected:

```text
all tests passed
"status": "PASS"
```

The verifier reads the committed result artifacts and checks every selected gate. It fails
non-zero if a required metric is missing or outside its threshold.

The original structural paste proof remains available:

```powershell
python verify_paste.py
```

## What changed about cross-seed generalization

The original LayerCake prototype copied brick weights exactly but failed functionally across
independent cores. A chess brick could move bit-for-bit while target PPL exploded because:

1. each seed learned a different ABI coordinate system; and
2. each seed decoded ABI deltas through a different output projection.

V2 addresses both causes:

- **Deterministic causal anchors:** every core aligns ABI states to the same byte-prefix
  target basis during training.
- **Canonical brick head:** brick deltas use a fixed ABI-to-byte-logit contract shared
  across interfaces, seeds, and model widths.
- **Correct temporal alignment:** the byte state after completed patch `n` aligns with the
  context used to predict patch `n+1`.
- **General-preservation loss:** brick training is constrained against the frozen base
  distribution and must pass an external non-regression gate.

Cross-seed failure without alignment remains an important negative control. It is no longer
an unresolved explanation for the selected v2 architecture.

## Why byte patches

Byte models avoid vocabulary lock-in but make global attention expensive. LayerCake uses:

- a local causal decoder at byte resolution;
- a smaller global transformer over patch states;
- continuous local hidden state across patch boundaries;
- a canonical ABI above the patch perception layer.

In the selected experiment, global sequence length is reduced 4x. The resulting patch core
is smaller and faster than both the byte transformer and the trained BPE baseline while
matching BPE general BPB by point estimate.

The current implementation uses fixed patches. Learned entropy/difficulty boundaries remain
a scale-up target.

## Portable sparse domain bricks

The selected brick has:

- 8 installed low-rank experts;
- rank 16 per expert;
- top-2 active experts;
- 16,897 parameters;
- a residual no-op initialization;
- exact state-dict portability;
- optional int8 fake-quantized transfer.

Installed knowledge does not require evaluating every expert. The router scores installed
experts, but only selected expert matrices execute.

This is different from:

| Method | Key distinction |
|---|---|
| LoRA | LoRA matrices are shaped by each target layer and `d_model`; LayerCake bricks bind to `d_abi`. |
| Adapter | Ordinary adapters remain model-specific; LayerCake uses a versioned canonical coordinate/output contract. |
| MoE | MoE experts usually belong to one core; LayerCake bricks are portable artifacts. |
| RAG | RAG retrieves external context; bricks modify model behavior in ABI space. |
| Fine-tuning | Full tuning changes the core; brick training freezes it. |
| BLT-style models | BLT targets tokenizer-free dynamic byte compute; LayerCake adds portable ABI-space knowledge. |

## Claim ladder

| Level | Meaning | Current evidence |
|---|---|---|
| L0 | Exact weight copy | Proven |
| L1 | Equal ABI input, equal brick function | Proven |
| L2 | Same-core generation identity | Proven on legacy tokenized path |
| L3 | Cross-size structural/function portability | Proven; bounded end-to-end v2 transfer passes locally |
| L4 | Bounded additive cross-seed semantic transfer | Small-scale PASS |
| L5 | Bounded quantized transfer | Small-scale int8 PASS |
| L6 | Bounded tokenizer-independent byte/patch transfer | Small-scale PASS |
| PX | Exact core-independent portable-domain transfer | PASS through 15.45M tier |
| L7 | Orchestrated swarm transfer | Interface implemented; task-level evidence pending |

See [RUBRIC.md](RUBRIC.md) for exact definitions.

## Reproduce or extend the experiments

Core paired training:

```powershell
python scripts/run_paired_byte_experiment.py `
  --seed 2028 `
  --d-model 128 --layers 3 --heads 4 `
  --patch-size 4 --continuous-local `
  --patch-d-model 96 --patch-layers 2 --patch-heads 4 `
  --d-byte 32 --d-abi 64 `
  --steps 4000 --brick-steps 1000 `
  --artifact runs_experiment/my_core.pt `
  --output results/my_core.json
```

Train a sparse portable brick:

```powershell
python scripts/train_sparse_brick_artifact.py `
  --core runs_experiment/my_core.pt `
  --steps 6000 --rank 16 --experts 8 --top-k 2 `
  --preserve-weight 2 `
  --artifact runs_experiment/my_brick.pt `
  --output results/my_brick.json
```

Transfer and quantize it:

```powershell
python scripts/eval_portable_brick.py `
  --brick runs_experiment/my_brick.pt `
  --target runs_experiment/another_core.pt `
  --quantize-int8 `
  --output results/my_transfer.json
```

Benchmark inference:

```powershell
python scripts/benchmark_canonical_artifact.py `
  --core runs_experiment/my_core.pt `
  --brick runs_experiment/my_brick.pt `
  --iterations 300 --rounds 9 `
  --output results/my_inference.json
```

## Repository map

```text
layercake/
  abi.py                  versioned compatibility contract
  abi_alignment.py        anchor, whitening, and alignment losses
  canonical_anchors.py    deterministic seed-independent prefix basis
  causal_byte_models.py   strictly causal byte and byte-patch models
  byte_patch.py           codecs, patchers, and metadata
  domain_bricks.py        low-rank and top-k sparse portable operators
  portable_domain.py      exact core-independent domain prediction payload
  orchestration.py        CorticalSwarm-style handoff packet and router
  transfer.py             copy, PPL, and degradation contracts

scripts/
  run_paired_byte_experiment.py
  train_sparse_brick_artifact.py
  eval_portable_brick.py
  eval_lossless_domain_decoder.py
  benchmark_bpe_baseline.py
  benchmark_canonical_artifact.py
  verify_research_gates.py

results/
  research_gate_certificate.json
  selected raw benchmark and transfer artifacts
```

The original flat `model.py`, training scripts, and paste proof remain intact for legacy
reproduction.

## What is not yet established

- Results have not yet been replicated at 25M, 60M, 150M, or 1B scale.
- The current parity result uses one selected byte-patch seed and one BPE seed.
- Confidence intervals, energy-to-quality, and matched wall-clock scaling curves remain.
- Dynamic learned patching is not implemented.
- Native int8 kernels were not benchmarked; current int8 evidence uses quantize/dequantize.
- L7 orchestration has serialization and routing tests but no end-to-end task benchmark.
- Production serving, security hardening, and distributed training are not complete.

The next milestone is a 25M-class, three-seed, matched-byte experiment with frozen hashes
and the same transfer matrix.

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Claims and evidence](CLAIMS.md)
- [Experiment results](EXPERIMENT_RESULTS.md)
- [Transfer rubric](RUBRIC.md)
- [Tokenizer-free design](TOKENIZER_FREE.md)
- [Benchmarks](BENCHMARKS.md)
- [Roadmap](ROADMAP.md)
- [Known blockers](BLOCKERS.md)
- [GitHub release checklist](GITHUB_RELEASE_CHECKLIST.md)

## Citation

```bibtex
@software{layercake2026,
  author  = {Yoder, Sam},
  title   = {LayerCake: Tokenizer-Free Byte-Patch Models with a Canonical Knowledge ABI},
  year    = {2026},
  url     = {https://github.com/Yoder23/layercake},
  license = {Apache-2.0}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
