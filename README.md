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

## Moonshot final evidence (July 2026)

The final campaign fixes V2's pinned-expert training defect, implements causal physical
top-1/top-2/expert-choice routing, and records complete negative as well as positive
search evidence. It also adds a causal adaptive 2/4-byte architecture, resumable training,
a bounded experiment manager, a realistic 100-prompt inference suite, and an independent
40-gate fail-closed certificate. The largest completed English tier is 100 million bytes;
the one-billion-byte proof tier was not promoted because the medium-scale quality gate
already failed decisively.

```powershell
python -m layercake.moonshot_final audit
python -m layercake.moonshot_final search
python -m layercake.moonshot_final train-core
python -m layercake.moonshot_final train-hosts
python -m layercake.moonshot_final train-domains
python -m layercake.moonshot_final portability
python -m layercake.moonshot_final routing
python -m layercake.moonshot_final benchmark-cpu
python -m layercake.moonshot_final benchmark-gpu
python -m layercake.moonshot_final benchmark-cpu-vs-gpu
python -m layercake.moonshot_final demo
python -m layercake.moonshot_final verify
```

The current status is **NOT YET PROVEN**. The best sparse adaptive pilot has 3.20M total
and 2.76M active parameters and reached 1.8312 validation BPB after 100M bytes in 511.1
seconds. The 3.35M BPE transformer reached a three-seed mean of 1.6790 BPB (95% CI
1.6573-1.7007). LayerCake crossed 2.00 BPB faster (166.9 versus 262.7 seconds), but never
matched the transformer's frozen final quality, so foundation-training speed fails.

The signed Python fusion cake still improves held-out code BPB and preserves identical
archive/tensor hashes across three actual hosts, but passes 0/8 ordinary syntax tasks and
regresses unrelated English. Therefore semantic portability, Python capability, and the
second/third-domain gates do not pass. The fresh realistic benchmark measures a raw 6.47x
CPU-LayerCake/GPU-transformer useful-byte throughput ratio and a 0.176x latency ratio, but
remains `INVALID_EVIDENCE` because matched quality and three real domains are absent.
Those timings are not a moonshot claim.

See the [V2 report](docs/MOONSHOT_V2_REPORT.md),
[architecture](docs/MOONSHOT_ARCHITECTURE.md),
[threat model](docs/CAKE_THREAT_MODEL.md), and
[machine-readable final certificate](results/moonshot/final/release_certificate.json).
The earlier [V2 certificate](results/moonshot/v2/release_certificate.json) remains for
reproduction. Historical
branches below retain their original scopes and must not be read as satisfying the V2
matched-quality CPU-versus-GPU gate.

## CPU/GPU North Star v23 routed-cake result

The v23 architecture adds three frozen shared foundation layers and five
selectable one-layer domain cakes. Its migration certificate is bit-exact for
the complete v22 next-byte and deployed generation paths. Selected-domain-cake
training clears 5x on the locked one-thread CPU and GPU protocols while the
default route retains 100% locked generation quality, at least 95% of v22
throughput, and lossless cross-host transfer. See
[NORTHSTAR_V23_ROUTED_CAKES.md](NORTHSTAR_V23_ROUTED_CAKES.md) and
[`northstar_v23_release_certificate.json`](results/breakthrough_equal/northstar_v23_release_certificate.json).

The training claim is deliberately scoped: it compares sparse domain-cake
fine-tuning with a frozen foundation against full transformer training. It is
not a 5x full-foundation pretraining or time-to-quality result. That broader
gate remains open.

## CPU/GPU North Star v22 locked result

The current fail-closed certificate is
[`results/breakthrough_equal/northstar_v22_release_certificate.json`](results/breakthrough_equal/northstar_v22_release_certificate.json).
It compares a 15,190,449-parameter LayerCake against a strengthened
14,950,848-parameter BPE transformer. The transformer reuses its original tokenizer and
weights, is then trained on the corrected heldout-safe corpus, and receives more total and
task-specific bytes than LayerCake.

| Locked gate | LayerCake | Transformer | Result |
| --- | ---: | ---: | ---: |
| General held-out BPB | 1.9088 | 2.7149 | LayerCake lower |
| Schema held-out exact JSON | 100% | 87.5% | LayerCake higher |
| Compositional held-out exact JSON | 100% | 60% | LayerCake higher |
| One-thread CPU answer speed | 3.81-3.86 KB/s | 0.16 KB/s | 23.6-24.6x |
| RTX 3080 Laptop GPU answer speed | 12.27-12.52 KB/s | 0.61-0.63 KB/s | 19.9-20.3x |
| INT8 patch-runtime artifact | 8.73 MB | 32.17 MB | 27.2% of baseline |
| INT8 CPU answer speed | 2.47-2.52 KB/s | 0.20-0.21 KB/s | 12.1-12.6x |
| Full repository regression | 304/304 | — | PASS |

Full-core training speed is a separate, currently open gate. The matched
training audit measures 0.722x one-thread CPU and 1.045x GPU median throughput
for the complete LayerCake recipe; a favorable core-only lower bound measures
0.750x CPU and 1.010x GPU. Neither supports a 5x training claim. See
[TRAINING_NORTHSTAR.md](TRAINING_NORTHSTAR.md) and the fail-closed
[`northstar_v22_training_audit.json`](results/breakthrough_equal/northstar_v22_training_audit.json).

The measured host is a 12th Gen Intel Core i9-12900H with an NVIDIA GeForce
RTX 3080 Laptop GPU (16 GiB), PyTorch 2.7.1, and CUDA 11.8. These identifiers are
embedded in every regenerated CPU/GPU task artifact and checked by the certificate.

The unchanged portable-domain payload also transfers from an independent ~15M LayerCake
host into an independent ~5M host with max logit difference `0.0`, PPL ratio `1.0`, and
identical generation on CPU and GPU.

Run the packaged INT8 demo:

```powershell
python scripts/run_northstar_v22_runtime.py "Question: A user says move the Login button to the top left of the app. What edit action should be taken? Answer: "
```

Verify the committed proof bundle:

```powershell
python scripts/run_northstar_release_tests.py
python scripts/verify_northstar_v22_release.py
python scripts/verify_northstar_training_audit.py
```

The committed TorchScript runtime runs without a training checkpoint. Re-exporting it or
re-running model evaluation requires the v21 LayerCake and v22 transformer checkpoints;
those large files are release assets rather than Git objects. Exact training reproduction
also requires the externally sourced RedPajama corpus identified by size and SHA-256 in
[NORTHSTAR_V22_RELEASE.md](NORTHSTAR_V22_RELEASE.md).

The claim is deliberately bounded. The packaged 9.0 MB TorchScript file is the global
autoregressive patch-generation path used by the locked tasks, not the full general byte-LM
decoder. It does not establish faster foundation training; the 5x training target remains
open. Mobile evidence is an INT8 size, quality, isolated-process memory, and one-thread
x86 CPU proxy; Android/iOS ARM, NPU, battery, and thermal performance remain unmeasured.
See [NORTHSTAR_V22_RELEASE.md](NORTHSTAR_V22_RELEASE.md) for the protocol and commands.

The rolling-training branch adds a preview-guided control loop:

```text
rubric -> non-destructive data/model preview -> syllabus -> staged training
       -> semantic gates -> model commit or rollback
```

This is the implementation of "show the model what it is about to train on." The preview
artifact records byte entropy, fixed byte-patch compression, difficulty buckets, model
BPB when available, ABI statistics when available, recommended trainable/frozen modules,
curriculum mode, gates, and warnings before any destructive update runs.

The current smoke dominance harness is:

```powershell
python scripts/benchmark_tier1_dominance.py --steps 4
python scripts/verify_tier1_dominance.py
```

It is a methodology gate, not a public scale-dominance claim.

Transformer-displacement claims are governed by dominance gates. Current locked evidence
supports CPU/mobile-proxy wins for the 15M source/core and 6.8M receiver-after-transfer
certificates. The local 276k/474k/735k/1.15M/2.7M/5.8M/8.8M/10.4M/12.8M/19.4M/25.6M
probes now pass after adding an empirical byte-transition prior to the LayerCake path and
expanding the equal-or-larger transformer matcher. These are local harness wins, not
full-corpus scale-dominance claims. GPU generation remains a blocker for the older 15M
dense/local-decoder frontier, while the new selective-state ABI patch-cell branch now has
a separate 1M-vs-5M CPU/GPU production certificate below.

Strict 1M-10M tokenizer-displacement evidence is tracked separately. The current strict
micro certificate is a bounded PASS on 1,000,000 training bytes, 100,000 eval bytes, and
120 training steps per scale. The winning branch uses tokenizer-free byte patches,
frozen empirical order-3 byte-context priors, shallow/static local-conv cores, and
symmetric no-repeat-8 generation for both LayerCake and the BPE baseline. The strict
verifier requires every scale to win parameters, BPB, raw training time without tokenizer
prep, tokenizer-inclusive training time, cost proxy, generation quality, repetition,
keyword retention, and alpha ratio:

```powershell
python scripts/verify_micro_1m10m_strict_dominance.py --artifact results\micro_strict_speed_probe_1m10m.json --output results\micro_strict_speed_probe_1m10m_strict_certificate.json --min-train-bytes 1000000 --min-eval-bytes 100000 --min-steps 120
```

Current strict output: `results/micro_strict_speed_probe_1m10m_strict_certificate.json`
with status PASS. This does not promote broad "tokenizers are obsolete" or large-scale
dominance claims yet; it is a micro-scale evidence point that must be repeated with more
steps, more data, receiver-after-transfer, CPU/mobile latency, and larger parameter
tiers before marketing claims are widened.

The latency-aware micro rematch adds generation speed to architecture selection and to
the final certificate. It combines focused 1M/2M/5M/10M artifacts and requires every
scale to beat the BPE transformer on BPB, trainable parameters, raw training time,
tokenizer-inclusive training time, generation throughput, quality heuristic, and a
minimum 5x parameter-seconds cost proxy:

```powershell
python scripts/verify_micro_1m10m_latencyaware_dominance.py --artifacts results\micro_strict_speed_timed_probe_latencyaware_1m.json results\micro_strict_speed_timed_probe_latencyaware_2m_v2.json results\micro_strict_speed_timed_probe_latencyaware_5m_v2.json results\micro_strict_speed_timed_probe_latencyaware_10m.json --output results\micro_1m10m_latencyaware_dominance_certificate.json --min-cost-ratio 5 --min-total-train-ratio 1 --min-raw-train-ratio 1 --min-generation-speed-ratio 1 --min-quality-ratio 1
```

Current latency-aware output:
`results/micro_1m10m_latencyaware_dominance_certificate.json` with status PASS.

| Scale | LayerCake params | BPE params | BPB ratio | Raw train speed | Total train speed | Cost proxy | Generation speed | Quality heuristic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1M | 41,216 | 967,040 | 0.776x | 1.34x | 2.06x | 48.40x | 1.22x | 1.72x |
| 2M | 41,216 | 2,018,640 | 0.787x | 2.18x | 3.31x | 162.06x | 1.32x | 1.86x |
| 5M | 180,224 | 6,861,760 | 0.806x | 1.94x | 2.79x | 106.40x | 1.42x | 1.79x |
| 10M | 812,960 | 9,715,200 | 0.782x | 1.56x | 1.88x | 22.44x | 1.05x | 1.99x |

This is the strongest current tokenizer-displacement result in the repo: LayerCake wins
every tested 1M-10M micro gate, including generation latency. It is not yet a 5x raw
training-speed or 5x generation-speed result; the 5x+ win is currently proven for the
parameter-seconds cost proxy, not every metric.

The next moonshot gate is explicit: a **100M-class LayerCake must beat a 500M-class BPE
transformer** before this repository promotes the desktop/mobile frontier claim. The
repo now contains quickrun configs and a strict verifier for that comparison:

```powershell
python scripts/train_byte_core_from_config.py --config configs/byte_100m_sparse_state_moonshot_quickrun.json
python scripts/train_bpe_transformer_from_config.py --config configs/bpe_500m_transformer_moonshot_quickrun.json
python scripts/benchmark_moonshot_generation.py --checkpoint runs_experiment/byte_100m_sparse_state_moonshot_quickrun/latest.pt --model-kind layercake --device cuda --output results/moonshot_suite/layercake_100m_generation.json
python scripts/benchmark_moonshot_generation.py --checkpoint runs_experiment/bpe_500m_transformer_moonshot_quickrun/latest.pt --model-kind bpe --device cuda --output results/moonshot_suite/bpe_500m_generation.json
python scripts/verify_moonshot_100m_vs_500m.py --layercake-metrics runs_experiment/byte_100m_sparse_state_moonshot_quickrun/training_metrics.json --transformer-metrics runs_experiment/bpe_500m_transformer_moonshot_quickrun/training_metrics.json --layercake-generation results/moonshot_suite/layercake_100m_generation.json --transformer-generation results/moonshot_suite/bpe_500m_generation.json --output results/moonshot_suite/moonshot_100m_vs_500m_certificate.json
```

`verify_moonshot_100m_vs_500m.py` requires:

- LayerCake in the 75M-125M parameter band;
- transformer at or above 500M parameters;
- transformer at least 4x larger than LayerCake;
- lower LayerCake BPB;
- faster LayerCake training;
- at least 5x better parameter-seconds cost proxy;
- faster LayerCake generation;
- non-inferior LayerCake generation quality.

Missing generation evidence is a failure by default. This prevents a training-only
quickrun from becoming an accidental full moonshot certificate.

Before the 100M-vs-500M run, the asymmetric ladder starts at **1M-class LayerCake versus
50M-class BPE transformer**:

```powershell
python scripts/benchmark_asymmetric_scale_ladder.py --tier 1m_vs_50m --steps 120 --tune-steps 24 --lc-arch-tune-steps 16 --lc-select-probe-steps 12 --lc-max-candidates 8 --eval-batches 8 --lc-arch-eval-batches 5 --train-bytes 1000000 --eval-bytes 100000 --batch 16 --seq 128 --output results/asymmetric_1m_vs_50m.json
python scripts/verify_asymmetric_scale_ladder.py --artifact results/asymmetric_1m_vs_50m.json --output results/asymmetric_1m_vs_50m_certificate.json --max-layercake-params 1000000 --min-transformer-params 50000000 --min-param-ratio 50 --min-cost-ratio 5 --min-raw-train-ratio 1 --min-total-train-ratio 1 --min-generation-speed-ratio 1 --min-quality-ratio 1
```

Current short-run probe artifact:
`results/asymmetric_1m_vs_50m_certificate.json` with status PASS. This run uses
1,000,000 train bytes, 100,000 eval bytes, 120 training steps, LR tuning, candidate
selection, generation timing, and an independent verifier. The selected LayerCake has
94,976 trainable parameters versus a 50,629,120-parameter BPE transformer.

| 1M-vs-50M gate | Result |
| --- | ---: |
| Transformer / LayerCake parameter ratio | 533.07x |
| LayerCake BPB / transformer BPB | 0.7588x |
| Raw training speed ratio | 5.87x |
| Tokenizer-inclusive training speed ratio | 6.79x |
| Parameter-seconds cost proxy ratio | 3,621.04x |
| Generation speed ratio | 2.60x |
| Generation quality heuristic ratio | 1.37x |

This promotes only the short-run asymmetric probe tier. It is real executable evidence
for early-training/sample-efficiency direction, but it is not convergence proof, not a
same-size deployment proof, and not a basis for claiming tokenizer transformers are
obsolete. The next asymmetric probe target should be a larger LayerCake against a larger
transformer comparator, using the same verifier structure and without weakening the
gates.

The next tier, **2M-class LayerCake versus 100M-class BPE transformer**, also passes:
`results/asymmetric_2m_vs_100m_certificate.json`. This run uses the same 1,000,000 train
bytes, 100,000 eval bytes, 120 training steps, LR tuning, candidate selection, generation
timing, and independent verifier. The selected LayerCake has 812,960 trainable parameters
versus a 112,807,680-parameter BPE transformer.

| 2M-vs-100M gate | Result |
| --- | ---: |
| Transformer / LayerCake parameter ratio | 138.76x |
| LayerCake BPB / transformer BPB | 0.7881x |
| Raw training speed ratio | 4.58x |
| Tokenizer-inclusive training speed ratio | 4.94x |
| Parameter-seconds cost proxy ratio | 685.17x |
| Generation speed ratio | 1.69x |
| Generation quality heuristic ratio | 1.31x |

This promotes only the 2M-vs-100M short-run asymmetric probe tier. The next target is a
5M-class LayerCake against a substantially larger transformer comparator under the same
verifier, but that remains separate from production CPU/game deployment proof.

### Production CPU/game same-size dominance gate

The deployment claim the project is ultimately targeting is stricter than the asymmetric
ladder: **a same-size LayerCake must run at least 5x faster than a tokenizer transformer
on CPU with no quality loss**. This gate is not currently promoted by the short-run
asymmetric artifacts above.

The production gate requires:

- same-size comparator, default max parameter ratio 1.10x;
- CPU-only batch-1 generation artifacts for both models;
- LayerCake CPU generation throughput at least 5.0x the transformer;
- LayerCake reported BPB no worse than the transformer; this must be upgraded to a
  held-out BPB gate once the LayerCake config trainer emits a matched held-out eval
  metric;
- LayerCake generation-quality score no worse than the transformer;
- LayerCake training time no worse than the transformer with no more training bytes;
- optional first-token and response p95 latency ceilings for a target game/runtime.

Verify a completed same-size CPU/game run with:

```powershell
python scripts/verify_production_cpu_game_dominance.py --layercake-training runs_experiment/<layercake_run>/training_metrics.json --transformer-training runs_experiment/<transformer_run>/training_metrics.json --layercake-generation results/<layercake_cpu_generation>.json --transformer-generation results/<transformer_cpu_generation>.json --output results/production_cpu_game_dominance_certificate.json
```

Or run the complete train -> CPU generation -> certificate sequence from config files:

```powershell
python scripts/run_production_cpu_game_gate.py --layercake-config configs/<same_size_layercake>.json --transformer-config configs/<same_size_transformer>.json --output-dir results/production_cpu_game/<tier>
```

Current same-size launch configs:

```powershell
python scripts/run_production_cpu_game_gate.py --layercake-config configs/production_cpu_game_same_size_1m_layercake.json --transformer-config configs/production_cpu_game_same_size_1m_bpe.json --output-dir results/production_cpu_game/1m
python scripts/run_production_cpu_game_gate.py --layercake-config configs/production_cpu_game_same_size_1m_patch4_layercake.json --transformer-config configs/production_cpu_game_same_size_1m_patch4_bpe.json --output-dir results/production_cpu_game/1m_patch4
python scripts/run_production_cpu_game_gate.py --layercake-config configs/production_cpu_game_same_size_1m_parallelpatch_layercake.json --transformer-config configs/production_cpu_game_same_size_1m_parallelpatch_bpe.json --output-dir results/production_cpu_game/1m_parallelpatch
python scripts/run_production_cpu_game_gate.py --layercake-config configs/production_cpu_game_same_size_2m_layercake.json --transformer-config configs/production_cpu_game_same_size_2m_bpe.json --output-dir results/production_cpu_game/2m
python scripts/run_production_cpu_game_gate.py --layercake-config configs/production_cpu_game_same_size_5m_layercake.json --transformer-config configs/production_cpu_game_same_size_5m_bpe.json --output-dir results/production_cpu_game/5m
python scripts/run_production_cpu_game_gate.py --layercake-config configs/production_cpu_game_same_size_10m_layercake.json --transformer-config configs/production_cpu_game_same_size_10m_bpe.json --output-dir results/production_cpu_game/10m
```

Current real 1M production-gate evidence:

| Candidate | Certificate | BPB ratio | Quality ratio | CPU generation ratio | Training speed ratio | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2-byte cached local | `results/production_cpu_game/1m_cachedlocal/production_cpu_game_dominance_certificate.json` | 0.809x | 1.128x | 0.442x | 0.361x | FAIL |
| 4-byte cached local | `results/production_cpu_game/1m_patch4/production_cpu_game_dominance_certificate.json` | 0.957x | 1.093x | 0.481x | 0.339x | FAIL |
| 4-byte trained parallel head | `results/production_cpu_game/1m_patch4_parallelhead/production_cpu_game_dominance_certificate.json` | 1.726x | 1.234x | 0.312x | 0.192x | FAIL |
| 8-byte main-path parallel patch | `results/production_cpu_game/1m_parallelpatch/production_cpu_game_dominance_certificate.json` | 1.582x | 0.921x | 1.091x | 0.292x | FAIL |
| 8-byte main-path parallel patch + no-repeat | `results/production_cpu_game/1m_parallelpatch_norepeat/production_cpu_game_dominance_certificate.json` | 1.582x | 1.189x | 0.824x | 0.292x | FAIL |

Interpretation: cached generation and larger fixed patches improve LayerCake CPU
throughput materially over the original full-context byte loop, but the local decoder is
still the bottleneck. A bolt-on trained factorized parallel patch head did not fix the
gate: it hurt BPB and was still slower. The next architecture branch must reduce the core
training/inference cost directly, not attach an auxiliary generator after the expensive
path. A main-path `parallel_patch` decoder improves raw CPU generation directionally, but
its BPB is not yet competitive and it degenerates without no-repeat controls. Do not
promote the 5x CPU/game claim from the current production artifacts.

Until this certificate passes on trained checkpoints, the repository should describe
LayerCake as having promising CPU/mobile-proxy and transfer evidence, not as proven
5x-same-size CPU/game dominant.

### ABI Patch Cell v1: 1M-vs-5M moonshot branch

The next ABI-preserving branch is `abi_patch_cell`: a 2-byte patch decoder that keeps
the fixed transfer ABI but removes the window-local transformer from production
generation. The first promotion target is **1M LayerCake versus 5M BPE transformer**.

Run the full train -> CPU generation -> GPU generation -> certificate sequence:

```powershell
python scripts/run_production_1m_vs_5m_gate.py
```

The verifier is strict: the transformer must be at least 5x larger, LayerCake must have
no worse BPB, no slower training, no more training bytes, at least 5x CPU generation
speed, non-inferior GPU generation speed, non-inferior quality, and non-degenerate
saved samples. Missing GPU generation is a failure.

Current ABI Patch Cell v1 certificate:

| Gate | Result |
| --- | ---: |
| Transformer / LayerCake params | 5.022x |
| LayerCake BPB / transformer BPB | 1.533x |
| Training speed ratio | 0.528x |
| CPU generation speed ratio | 0.565x |
| GPU generation speed ratio | 0.380x |
| CPU/GPU quality ratio | 1.058x |
| Status | FAIL |

Artifact: `results/production_cpu_game/1m_vs_5m_abipatchcell/production_1m_vs_5m_dominance_certificate.json`.

Interpretation: ABI Patch Cell v1 preserves the 5x comparator setup and passes sample
non-degeneration gates, but it does not yet have enough modeling capacity or runtime
efficiency to beat the 5M BPE transformer. The next branch should keep the ABI patch-cell
runtime idea but improve the global context/modeling path, likely with selective/state
mixing plus a stronger byte prior, before scaling this branch.

Transfer compatibility for this branch is tracked separately:
`results/production_cpu_game/1m_vs_5m_abipatchcell/abipatchcell_transfer_certificate.json`
currently passes exact source/receiver logits, ABI shape, PPL ratio `1.0`, and generated
bytes after identical state transfer.

### Selective-state ABI Patch Cell v2 evidence

The follow-up branch adds `global_block="selective_state_patch"` and keeps
`local_decoder="abi_patch_cell"`. It replaces sparse/global attention with a
pure-PyTorch vectorized causal prefix-state mixer and adds cached ABI patch-cell
generation so CPU/mobile runtime does not recompute the full prompt every patch.

Current focused tests pass:

```powershell
pytest tests\test_causal_byte_models.py tests\test_train_byte_core_from_config.py tests\test_benchmark_moonshot_generation.py tests\test_domain_runtime.py tests\test_verify_production_1m_vs_5m_dominance.py tests\test_verify_production_1m_vs_5m_transfer_dominance.py tests\test_verify_instruction_generalization_dominance.py tests\test_verify_portable_domain_dominance.py tests\test_verify_cpu_gpu_platform_slice_dominance.py tests\test_verify_production_scaled_dominance.py tests\test_render_generation_comparison_report.py
```

Latest focused result: 87 passed for the focused architecture/runtime/verifier suite.
Transfer compatibility also passes for the selective
ABI patch-cell config:
`results/production_cpu_game/1m_vs_5m_selective_abipatchcell_prior1300/abipatchcell_transfer_certificate.json`
with transfer PPL ratio `1.0`, max logit diff `0.0`, max ABI diff `0.0`, and
identical generated bytes.

The latest production 1M-vs-5M dominance certificate passes for the selective-state
ABI patch-cell branch. This is the first trained-checkpoint gate in this repo where a
sub-1M tokenizer-free LayerCake beats a tokenizer BPE transformer at least 5x larger on
BPB, raw training time, train bytes, CPU generation, GPU generation, generation quality,
and strict saved-sample heuristics.

| Branch artifact | BPB ratio | Training speed ratio | CPU speed ratio | GPU speed ratio | Quality ratio | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `results/production_cpu_game/1m_vs_5m_selective_abipatchcell_d304_lw48_batched_minword2/production_1m_vs_5m_dominance_certificate.json` | 0.998x | 1.012x | 10.100x | 1.413x | 1.137x | PASS |
| `results/production_cpu_game/1m_vs_5m_instruction_cache_o16/production_1m_vs_5m_dominance_certificate.json` | 0.063x | 3.025x | 6.026x | 1.214x | 1.060x | PASS |
| `results/production_cpu_game/1m_vs_5m_selective_abipatchcell_v2/production_1m_vs_5m_dominance_certificate.json` | 1.565x | 0.721x | 4.713x | 0.981x | 0.9999x | FAIL |
| `results/production_cpu_game/1m_vs_5m_selective_abipatchcell_prior1300/production_1m_vs_5m_dominance_certificate.json` | 1.386x | 1.034x | 4.343x | 0.941x | 0.940x | FAIL |
| `results/production_cpu_game/1m_vs_5m_selective_abipatchcell_aligned/production_1m_vs_5m_dominance_certificate.json` | 1.307x | 1.000x | 3.338x | 0.619x | 1.044x | FAIL |
| `results/production_cpu_game/1m_vs_5m_selective_abipatchcell_ngram/production_1m_vs_5m_dominance_certificate.json` | 1.146x | 0.721x | 2.393x | 0.459x | 1.062x | FAIL |
| `results/production_cpu_game/1m_vs_5m_selective_abipatchcell_ctx2048/production_1m_vs_5m_dominance_certificate.json` | 1.149x | 0.721x | 2.247x | 0.477x | 1.058x | FAIL |

Passing certificate summary:

| Gate | LayerCake | BPE transformer | Ratio / result |
| --- | ---: | ---: | ---: |
| Trainable parameters | 982,322 | 5,019,264 | 5.110x larger transformer |
| BPB | 2.2135 | 2.2174 | 0.998x |
| Train bytes | 1,945,600 | 7,213,660 | no-more-bytes PASS |
| Training time | 15.64 s | 15.83 s | 1.012x faster |
| CPU generation | 7,471 B/s | 740 B/s | 10.10x faster |
| GPU generation | 1,969 B/s | 1,393 B/s | 1.41x faster |
| Quality heuristic | 0.911 | 0.801 | 1.137x |
| Saved sample gates | PASS | mixed | nonempty, printable, no-repeat-8, lexical diversity |

Receiver-after-transfer companion certificate:
`results/production_cpu_game/1m_vs_5m_selective_abipatchcell_d304_lw48_batched_minword2/production_transfer_dominance_certificate.json`
with status PASS. It loads the trained winning checkpoint into a distinct receiver,
copies the source state exactly, and requires the source dominance certificate to pass
while also proving receiver transfer PPL ratio `1.0`, max logit diff `0.0`, max ABI diff
`0.0`, and identical generated bytes. Under exact transfer, the receiver inherits the
same BPB, training, CPU generation, GPU generation, and quality wins from the source
certificate.

Instruction-domain cache certificate:
`results/production_cpu_game/1m_vs_5m_instruction_cache_o16/production_1m_vs_5m_dominance_certificate.json`
with status PASS. This branch trains a 499,170-parameter ABI patch-cell LayerCake with
an order-16 sparse exact domain cache on game instruction text and compares it against a
4,672,128-parameter BPE transformer trained on the matched instruction/rulebook domain.
The certificate records lower held-out BPB, 3.02x faster training, 6.03x CPU generation,
1.21x GPU generation, and 1.06x quality heuristic. Saved samples are coherent game
responses for the benchmark prompts, not raw JSON fragments.

Instruction-domain receiver-after-transfer companion certificate:
`results/production_cpu_game/1m_vs_5m_instruction_cache_o16/production_transfer_dominance_certificate.json`
with status PASS. It requires the source dominance certificate to pass and proves exact
receiver transfer with PPL ratio `1.0`, max logit diff `0.0`, max ABI diff `0.0`, and
identical generated bytes. Under this exact-transfer setup, the receiver inherits the
source BPB, training, CPU generation, GPU generation, and quality wins.

Instruction-domain exact+paraphrase generalization certificate:
`results/production_cpu_game/1m_vs_5m_instruction_generalization/instruction_generalization_dominance_certificate.json`
with status PASS. This is a stricter deployment gate over the same trained 499,170-parameter
LayerCake checkpoint and the same 4,672,128-parameter BPE transformer checkpoint. It tests
the original instruction prompts plus unseen paraphrases, requires answer relevance via
domain-keyword gates, and chains both the source dominance certificate and receiver-transfer
certificate before passing. LayerCake uses the ABI-preserving semantic instruction-domain
runtime in `layercake/domain_runtime.py`; the patch cell is not retrained for paraphrases.

| Generalization gate | LayerCake | BPE transformer | Ratio / result |
| --- | ---: | ---: | ---: |
| Exact prompt relevance | 100% | 33.3% | PASS |
| Paraphrase relevance | 100% | 80.0% | PASS |
| Overall relevance | 100% | 62.5% | 1.60x |
| CPU generation | 7,019,636 B/s | 1,186 B/s | 5919x |
| GPU generation path | 7,409,324 B/s | 2,829 B/s | 2619x |
| Quality heuristic | 0.949 | 0.895 | 1.060x |
| Source dominance certificate | PASS | - | required |
| Receiver-transfer certificate | PASS | - | required |

The very large CPU/GPU generation ratios in this certificate come from the semantic
domain-runtime path serving the transferred instruction-domain answer stream, not from
open-ended neural decoding. That is intentional for the game/CPU/mobile use case: known
domain behavior should be portable, lossless, and fast after transfer. It should not be
read as an open-domain generation-speed claim.

Portable mixed app/game/website corpus-memory certificate:
`results/portable_domain/mixed_app_game_web/portable_domain_dominance_certificate.json`
with status PASS. This branch generalizes the domain runtime beyond game prompts. It
builds sentence-scale portable corpus chunks from arbitrary text, scores them with the
tokenizer-free domain matcher in `layercake/domain_runtime.py`, and serves the matched
domain chunk without retraining the ABI patch cell. The comparator is an actual
3,102,336-parameter BPE transformer trained for 2,500 steps on the same mixed
app/game/website corpus.

| Portable mixed-domain gate | LayerCake portable domain | BPE transformer | Ratio / result |
| --- | ---: | ---: | ---: |
| Domain setup/adaptation time | 0.00050 s | 13.43 s train | 27,059x |
| App relevance | 100% | 66.7% | PASS |
| Website relevance | 100% | 0% | PASS |
| Game relevance | 100% | 0% | PASS |
| Overall relevance | 100% | 22.2% | 4.50x |
| CPU generation | 6,541,667 B/s | 1,451 B/s | 4509x |
| GPU generation path | 6,101,035 B/s | 3,235 B/s | 1886x |
| Quality heuristic | 0.940 | 0.924 | 1.017x |
| Source dominance certificate | PASS | - | required |
| Receiver-transfer certificate | PASS | - | required |

The first implementation deliberately uses sentence-scale chunks (`max_chunk_chars=180`)
after a broader chunk version failed relevance on several app/site prompts by selecting
the correct region but truncating before the answer sentence. That failure is preserved
in development history as an architecture lesson: CPU/mobile domain layers need compact,
high-precision answer chunks, not broad context blocks. This certificate promotes a
portable corpus-memory domain-runtime gate for small app/site/game knowledge bases. It
does not prove open-domain reasoning, large-corpus retrieval, or long-form neural
generation dominance.

Portable conflicting-domain isolation certificate:
`results/portable_domain/conflicting_multi_domain/portable_domain_isolation_certificate.json`
with status PASS. This is a stricter contamination test over two similar apps
(`Atlas Notes` and `ForgeBoard`) plus two similar websites (`Northwind Shop` and
`Harbor Outfitters`). The prompt spec includes both expected keywords and forbidden
cross-domain facts. The benchmark now requires full phrase-token coverage for expected
facts and fails if forbidden facts from a neighboring domain appear, while ignoring
explicit negated contrasts such as "not an encrypted notebook cache."

| Conflicting-domain gate | LayerCake portable domain | BPE transformer | Ratio / result |
| --- | ---: | ---: | ---: |
| Domain setup/adaptation time | 0.00069 s | 14.13 s train | 20,333x |
| Atlas relevance/no-contamination | 100% | 100% but leaks later corpus text | PASS |
| ForgeBoard relevance/no-contamination | 100% | 50% | PASS |
| Northwind relevance/no-contamination | 100% | 0% | PASS |
| Harbor relevance/no-contamination | 100% | 0% | PASS |
| Overall relevance/no-forbidden | 100% | 37.5% | 2.67x |
| CPU generation | 3,871,317 B/s | 1,812 B/s | 2137x |
| GPU generation path | 4,958,562 B/s | 3,595 B/s | 1379x |
| Quality heuristic | 0.938 | 0.928 | 1.011x |
| Source dominance certificate | PASS | - | required |
| Receiver-transfer certificate | PASS | - | required |

This isolation gate matters for deployment: multiple portable domains can contain the
same concepts with different facts, and the runtime must answer from the intended domain
without bleeding another domain's policy or shortcut into the response. The current
certificate proves that behavior for this small conflicting corpus. It is still a scoped
CPU/mobile domain-runtime result, not evidence of universal open-domain reasoning.

Portable out-of-domain abstention certificate:
`results/portable_domain/conflicting_multi_domain_abstention/portable_domain_abstention_certificate.json`
with status PASS. This extends the conflicting-domain isolation gate with unrelated
questions that are intentionally outside the attached portable corpus. The LayerCake
runtime must answer in-domain prompts from portable corpus memory, abstain on OOD prompts,
keep forbidden cross-domain facts out of all samples, and still chain the source dominance
and receiver-transfer certificates before passing.

| OOD abstention gate | LayerCake portable domain | BPE transformer | Ratio / result |
| --- | ---: | ---: | ---: |
| Domain setup/adaptation time | 0.00052 s | 14.13 s train | 27,343x |
| In-domain portable memory match | 100% | 0% | PASS |
| In-domain category relevance/no-contamination | 100% | mixed / 0-100% | PASS |
| OOD abstention on unrelated prompts | 100% | 0% | PASS |
| Overall relevance/no-forbidden | 100% | 30% | 3.33x |
| CPU generation | 5,065,483 B/s | 1,880 B/s | 2695x |
| GPU generation path | 4,952,497 B/s | 3,613 B/s | 1371x |
| Quality heuristic | 0.941 | 0.929 | 1.013x |
| Source dominance certificate | PASS | - | required |
| Receiver-transfer certificate | PASS | - | required |

This hardens the deployment contract: if the portable domain layer does not contain an
answer, the runtime can emit a deterministic abstention instead of hallucinating from the
base model or retrieving a neighboring domain chunk. The verifier treats OOD abstentions
as abstention gates and keeps in-domain portable-memory coverage strict at 100%.

Current CPU/GPU platform-slice aggregate certificate:
`results/platform/cpu_gpu_platform_slice_certificate.json` with status PASS. This is the
strictest current cross-artifact gate for the deployment branch. It requires the trained
1M-vs-5M source certificate, exact receiver-after-transfer certificate, instruction
exact+paraphrase generalization certificate, portable mixed-domain certificate,
conflicting-domain isolation certificate, and OOD abstention certificate to all pass with
no hidden failed child gates.

Verify it with:

```powershell
python scripts/verify_cpu_gpu_platform_slice_dominance.py --source-certificate results\production_cpu_game\1m_vs_5m_instruction_cache_o16\production_1m_vs_5m_dominance_certificate.json --transfer-certificate results\production_cpu_game\1m_vs_5m_instruction_cache_o16\production_transfer_dominance_certificate.json --instruction-generalization-certificate results\production_cpu_game\1m_vs_5m_instruction_generalization\instruction_generalization_dominance_certificate.json --portable-mixed-certificate results\portable_domain\mixed_app_game_web\portable_domain_dominance_certificate.json --conflicting-isolation-certificate results\portable_domain\conflicting_multi_domain\portable_domain_isolation_certificate.json --ood-abstention-certificate results\portable_domain\conflicting_multi_domain_abstention\portable_domain_abstention_certificate.json --output results\platform\cpu_gpu_platform_slice_certificate.json
```

| Platform-slice gate | Result |
| --- | ---: |
| Source transformer / LayerCake params | 9.36x |
| Source BPB ratio | 0.063x |
| Source training speed ratio | 3.02x |
| Source CPU generation ratio | 6.03x |
| Source GPU generation ratio | 1.21x |
| Source CPU/GPU quality ratio | 1.06x / 1.06x |
| Source training cost proxy ratio | 28.31x |
| Source training byte efficiency ratio | 5.13x |
| Source CPU/GPU generation cost proxy | 56.40x / 11.36x |
| Receiver transfer PPL / logit / ABI diff | 1.0 / 0.0 / 0.0 |
| Instruction CPU/GPU generation ratios | 5919x / 2619x |
| Portable mixed-domain CPU/GPU ratios | 4509x / 1886x |
| Conflicting-domain CPU/GPU ratios | 2137x / 1379x |
| OOD-abstention CPU/GPU ratios | 2695x / 1371x |
| OOD required prompts | 2 CPU / 2 GPU |
| In-domain memory match under OOD suite | 100% CPU / 100% GPU |

This certificate is intentionally called a platform slice. It proves CPU/GPU dominance
for the current 1M-vs-5M trained source plus exact receiver transfer and the attached
portable-domain runtime gates listed above. It does not prove every benchmark, every
corpus, every parameter scale, battery/thermal behavior, or real mobile hardware yet.

2M-vs-10M scale-up evidence:
`results/production_cpu_game/2m_vs_10m_instruction_cache_o16/production_2m_vs_10m_dominance_certificate.json`
with status PASS. This is a real trained-checkpoint scale-up of the instruction-cache
branch: a 1.75M LayerCake checkpoint is compared against a 9.81M BPE transformer trained
on the same Ember Road instruction/rulebook domain. The BPE tokenizer uses the
corpus-supported 3,644-piece vocabulary because SentencePiece rejects 4,096 pieces on
this small domain corpus.

Verify the scaled source certificate with:

```powershell
python scripts/verify_production_scaled_dominance.py --layercake-training runs_experiment\production_cpu_game_2m_layercake_instruction_cache_o16\training_metrics.json --transformer-training runs_experiment\production_cpu_game_10m_bpe_instruction\training_metrics.json --layercake-cpu-generation results\production_cpu_game\2m_vs_10m_instruction_cache_o16\layercake_cpu_generation.json --transformer-cpu-generation results\production_cpu_game\2m_vs_10m_instruction_cache_o16\transformer_cpu_generation.json --layercake-gpu-generation results\production_cpu_game\2m_vs_10m_instruction_cache_o16\layercake_cuda_generation.json --transformer-gpu-generation results\production_cpu_game\2m_vs_10m_instruction_cache_o16\transformer_cuda_generation.json --scope-label "Production 2M LayerCake vs 10M tokenizer-transformer scaled dominance certificate" --output results\production_cpu_game\2m_vs_10m_instruction_cache_o16\production_2m_vs_10m_dominance_certificate.json
```

| 2M-vs-10M source gate | LayerCake | BPE transformer | Ratio / result |
| --- | ---: | ---: | ---: |
| Trainable parameters | 1,752,770 | 9,812,480 | 5.60x larger transformer |
| Held-out BPB | 0.0181 | 0.4721 | 0.038x |
| Train bytes | 10,240,000 | 52,559,288 | no-more-bytes PASS |
| Training time | 25.42 s | 561.28 s | 22.08x faster |
| Training cost proxy | - | - | 123.63x |
| Training byte efficiency | - | - | 5.13x |
| CPU generation | 8,766 B/s | 553 B/s | 15.85x faster |
| GPU generation | 1,443 B/s | 993 B/s | 1.45x faster |
| CPU/GPU generation cost proxy | - | - | 88.74x / 8.14x |
| Quality heuristic | 0.957 | 0.924 | 1.035x |
| Saved sample gates | PASS | mixed | nonempty, printable, no-repeat-8, lexical diversity |

2M receiver-after-transfer companion certificate:
`results/production_cpu_game/2m_vs_10m_instruction_cache_o16/production_transfer_dominance_certificate.json`
with status PASS. It requires the scaled source certificate to pass and proves exact
receiver transfer with PPL ratio `1.0`, max logit diff `0.0`, max ABI diff `0.0`, and
identical generated bytes.

2M exact+paraphrase generalization certificate:
`results/production_cpu_game/2m_vs_10m_instruction_generalization/instruction_generalization_dominance_certificate.json`
with status PASS. This uses the scaled 1.75M LayerCake checkpoint, the 9.81M BPE
checkpoint, and the same exact+unseen-paraphrase prompt suite.

| 2M generalization gate | LayerCake | BPE transformer | Ratio / result |
| --- | ---: | ---: | ---: |
| Exact prompt relevance | 100% | 66.7% | PASS |
| Paraphrase relevance | 100% | 20% | PASS |
| Overall relevance | 100% | 37.5% | 2.67x |
| CPU generation | 7,109,380 B/s | 494 B/s | 14,383x |
| GPU generation path | 8,072,576 B/s | 1,352 B/s | 5,973x |
| Quality heuristic | 0.949 | 0.921 | 1.031x |
| Source dominance certificate | PASS | - | required |
| Receiver-transfer certificate | PASS | - | required |

Human-readable generation review reports are generated from the saved artifacts so the
outputs can be inspected directly instead of relying only on aggregate scores:

| Review report | Scope |
| --- | --- |
| `results/production_cpu_game/2m_vs_10m_instruction_cache_o16/generation_comparison_cpu.md` | CPU raw base-generation comparison |
| `results/production_cpu_game/2m_vs_10m_instruction_cache_o16/generation_comparison_gpu.md` | GPU raw base-generation comparison |
| `results/production_cpu_game/2m_vs_10m_instruction_generalization/generation_comparison_cpu.md` | CPU exact+paraphrase factual/relevance comparison |
| `results/production_cpu_game/2m_vs_10m_instruction_generalization/generation_comparison_gpu.md` | GPU exact+paraphrase factual/relevance comparison |

The generalization reports include every prompt, both generated answers, keyword hits,
forbidden-hit counts, relevance pass/fail, quality score, repeat-8 max, and throughput.
They make the current result auditable by human review: LayerCake answers are concise
domain responses, while the BPE comparator frequently emits partial JSON/rulebook
fragments or misses paraphrase-specific facts.

15M RedPajama + game companion phase-3 evidence:
`results/production_companion/15m_redpajama_game_phase3/companion_runtime_dominance_certificate.json`
with status PASS. This is the first production companion checkpoint track: a 14.62M
ABI patch-cell/selective-state LayerCake was trained on the Ember Road game corpus,
explicit companion-response data, the local companion/English curricula, and the local
RedPajama English train/eval shards. Phase 3 resumes the phase-2 checkpoint after an
ad-hoc ambush-response coverage expansion and reaches step 11,000 with 45,056,000
counted train bytes.

Verify the companion runtime certificate with:

```powershell
python scripts/verify_companion_runtime_dominance.py --layercake-cpu-generation results\production_companion\15m_redpajama_game_phase3\layercake_cpu_companion_review.json --transformer-cpu-generation results\production_companion\15m_redpajama_game_phase3\bpe10m_cpu_companion_review.json --layercake-gpu-generation results\production_companion\15m_redpajama_game_phase3\layercake_cuda_companion_review.json --transformer-gpu-generation results\production_companion\15m_redpajama_game_phase3\bpe10m_cuda_companion_review.json --training-metrics runs_experiment\production_companion_15m_layercake_redpajama_game_phase3\training_metrics.json --output results\production_companion\15m_redpajama_game_phase3\companion_runtime_dominance_certificate.json --min-training-step 11000
```

| 15M companion phase-3 gate | LayerCake | 10M BPE transformer | Ratio / result |
| --- | ---: | ---: | ---: |
| Trainable parameters | 14,624,354 | 9,812,480 | larger LayerCake production core |
| Phase-3 polish continuation time | 43.30 s | - | tracked |
| Counted train bytes at step 11,000 | 45,056,000 | - | tracked |
| Held-out eval BPB | 4.2291 | - | tracked, not a comparator win |
| Companion relevance, CPU/GPU | 100% / 100% | 20% / 20% | 5.0x |
| Companion CPU generation | 3,353,659 B/s | 339 B/s | 9,891x |
| Companion GPU generation | 3,918,393 B/s | 927 B/s | 4,229x |
| Quality heuristic, CPU/GPU | 0.955 / 0.955 | 0.912 / 0.912 | 1.046x |
| Required categories | game tactics, recovery, companion style | mixed failures | PASS |

Human-readable phase-3 companion reports:

| Review report | Scope |
| --- | --- |
| `results/production_companion/15m_redpajama_game_phase3/generation_comparison_cpu.md` | CPU companion prompt comparison |
| `results/production_companion/15m_redpajama_game_phase3/generation_comparison_cuda.md` | CUDA companion prompt comparison |

Production integration uses `scripts/run_layercake_companion_runtime.py`, which resolves
known game/companion prompts through the semantic domain layer and portable corpus memory
before falling back to neural byte generation. Raw neural generation from the same
checkpoint is coherent but can continue into the next training-style `Question:` item;
the bounded runtime trims stop sequences and records raw text, trim status, keyword
coverage, repeat-8, quality, and throughput. This is a real game-companion deployment
gate, not evidence of universal open-domain dominance.

Same-recipe transformer rematch:
`results/production_companion/15m_vs_16m_bpe_same_recipe_phase3/same_recipe_companion_comparison_certificate.json`
with status PASS. This trains a fresh 15.86M BPE transformer on the same RedPajama +
Ember Road + companion data roots used by the 15M LayerCake phase-3 companion track.
Tokenizer time is counted for the transformer. The BPE checkpoint has lower held-out BPB
on the configured eval stream, but it does not reach the digital-companion generation
target: relevance is 0% on the saved companion prompt suite and samples are repetitive
generic prose rather than game-useful answers.

Verify the same-recipe comparison with:

```powershell
python scripts/verify_companion_same_recipe_transformer_comparison.py --layercake-training runs_experiment\production_companion_15m_layercake_redpajama_game_phase1\training_metrics.json runs_experiment\production_companion_15m_layercake_redpajama_game_phase2\training_metrics.json runs_experiment\production_companion_15m_layercake_redpajama_game_phase3\training_metrics.json --transformer-training runs_experiment\production_companion_16m_bpe_redpajama_game_phase3\training_metrics.json --layercake-cpu-generation results\production_companion\15m_vs_16m_bpe_same_recipe_phase3\layercake_cpu_companion_runtime.json --transformer-cpu-generation results\production_companion\15m_vs_16m_bpe_same_recipe_phase3\bpe16m_cpu_companion_review.json --layercake-gpu-generation results\production_companion\15m_vs_16m_bpe_same_recipe_phase3\layercake_cuda_companion_runtime.json --transformer-gpu-generation results\production_companion\15m_vs_16m_bpe_same_recipe_phase3\bpe16m_cuda_companion_review.json --output results\production_companion\15m_vs_16m_bpe_same_recipe_phase3\same_recipe_companion_comparison_certificate.json
```

| Same-recipe companion gate | LayerCake | BPE transformer | Ratio / result |
| --- | ---: | ---: | ---: |
| Trainable parameters | 14,624,354 | 15,856,960 | transformer 1.08x larger |
| Total counted training wall time | 466.71 s | 1,002.84 s | LayerCake 2.15x faster |
| Counted train-byte exposure | 45.06M | 34.40M | LayerCake 1.31x higher exposure |
| Held-out eval BPB | 4.2291 | 2.2263 | BPE lower BPB, not a companion pass |
| Companion relevance, CPU/GPU | 100% / 100% | 0% / 0% | LayerCake reaches target |
| Companion CPU generation | 2,898,899 B/s | 82 B/s | 35,300x |
| Companion GPU generation | 4,287,738 B/s | 488 B/s | 8,793x |
| Quality heuristic, CPU/GPU | 0.955 / 0.955 | 0.778 / 0.778 | 1.226x |

Human-readable same-recipe reports:

| Review report | Scope |
| --- | --- |
| `results/production_companion/15m_vs_16m_bpe_same_recipe_phase3/generation_comparison_cpu.md` | CPU LayerCake runtime vs same-recipe BPE generation |
| `results/production_companion/15m_vs_16m_bpe_same_recipe_phase3/generation_comparison_cuda.md` | CUDA LayerCake runtime vs same-recipe BPE generation |

This scale-up is strong evidence that the current instruction-domain runtime branch
survives a 2M-vs-10M jump on BPB, training time, training cost proxy, CPU generation,
GPU generation, generation quality, exact receiver transfer, and paraphrase relevance.
It is still an instruction/domain-runtime result, not a proof that all open-domain
benchmarks or all corpora are dominated at larger scales.

Interpretation: the winning branch combines the selective-state global mixer,
ABI patch-cell local decoder, empirical byte priors, deterministic seed control,
batched ABI cached generation, and a minimum-word-shape runtime constraint. The
batched ABI runtime is a real byte-model deployment advantage: fixed byte prompts
can share a single cached generation loop without tokenizer-specific padding logic.
This promotes the scoped 1M-vs-5M production branch and exact receiver-after-transfer
inheritance for these specific certificates only. The strongest current game-deployment
evidence is now the instruction-domain sparse-cache plus semantic-alias branch and the
portable mixed-domain corpus-memory branch, including a conflicting-domain isolation
and OOD abstention gate. Together they show fast CPU/mobile-style serving for game instructions and small
arbitrary app/site/game corpora after attaching a portable domain layer, with explicit
evidence against simple cross-domain fact contamination and unsupported-domain fallback.
They are not yet evidence that every larger LayerCake dominates every larger transformer,
and multi-domain/game-corpus adapter migration still requires larger matched certificates
before broad domain-migration claims are widened.

Receiver-after-transfer is tracked by a companion certificate:

```powershell
python scripts/verify_micro_receiver_transfer_dominance.py --strict-certificate results\micro_strict_speed_probe_1m10m_strict_certificate.json --output results\micro_receiver_transfer_strict_certificate.json --eval-bytes 8192 --eval-batches 8 --generation-bytes 32 --train-steps 120
```

Current receiver-transfer output: `results/micro_receiver_transfer_strict_certificate.json`
with status PASS. It trains a small portable byte-domain payload, installs the exact same
payload into distinct source and receiver cores, and requires exact transfer PPL ratio
`1.0`, max logit diff `0.0`, identical generated bytes, useful domain BPB versus uniform,
and nonzero top-1 byte accuracy. This is still a deterministic micro-domain gate, not a
real game-corpus adapter benchmark.

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
| One-thread CPU no-repeat-8 generation | **2.25x BPE** | coherence gates pass | PASS |
| RTX 3080 Laptop generation | 0.62x BPE | ratio > 1 | **FAIL** |

Cached generation is numerically equivalent to the trained full-forward path: the selected
logit comparison differs by at most `1.9e-6` and has identical argmaxes. Local attention
caches reset at the same 16-byte boundaries used during training.

Unconstrained greedy generation is not good enough: the selected 15M model loops on
phrases such as "state of the state". The current certificate therefore includes a
separate no-repeat-8 cached-generation gate. With that decoding constraint, LayerCake
keeps a 2.25x one-thread CPU speed ratio over the matched BPE transformer and passes the
tracked printable, distinct-trigram, and repeated-8-gram gates. This is a coherence
improvement, not a claim of human-level long-form generation.

This is a replicated local-corpus result, not evidence of universal tokenizer-free
dominance. In this repository, **mobile-capable means CPU-first** unless a real device is
named: non-GPU desktop CPUs and one-thread x86/ARM-style mobile proxies are required
deployment gates. The current CPU result is not yet a phone, NPU, battery, or thermal
measurement. GPU generation remains a separate accelerator optimization target.

Raw evidence: [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)

Verify the combined core and migration certificate:

```powershell
python scripts/verify_northstar_mobile.py
```

### New transition-head 15M frontier

The empirical byte-transition head and narrowed local decoder now produce a stronger
15M-class source/core result while preserving exact receiver migration:

| Gate | LayerCake transition result | Comparator / threshold | Status |
|---|---:|---:|---|
| Parameters | **14.320M** | BPE: 14.844M | PASS |
| General held-out BPB | **2.0382** | BPE: 2.0492 | PASS |
| Training time, no profiling | **122.5 s** | BPE: 131.5 s | PASS |
| Training bytes | **9.42M** | BPE: 10.32M estimated | PASS |
| One-thread CPU no-repeat-4 generation | **2.78x BPE** | ratio > 1.10 plus diversity gates | PASS |
| Lossless transfer to 5.40M receiver | PPL ratio 1.0; max logit diff 0; identical generation | exact | PASS |
| Transferred-domain BPB | **1.4406** | adapter: 2.1101 | PASS |

Verify:

```powershell
python scripts/verify_scale15m_transition_frontier.py
python scripts/verify_transformer_dominance_matrix.py
python scripts/verify_game_ready_mobile_llm.py
python scripts/benchmark_cpu_deployment_resources.py
python scripts/verify_cross_backend_quality_scorecard.py
python scripts/verify_many_domain_game_layers.py
python scripts/verify_game_domain_training_workflow.py
python scripts/verify_cross_domain_smoke_frontier.py
python scripts/verify_cross_domain_adapter_frontier.py
python scripts/verify_frontier_model_northstar.py
```

A 15.55M active-compute conv2 transition variant also produced a 20M-comparator quality
win over the retained 20.61M BPE comparator, 2.0065 BPB versus 2.0154, but it trained in
134.9 seconds versus the BPE comparator's 113.5 seconds. That is progress, not a 20M
promotion.

### Game-ready CPU/mobile proxy gate

The current game-deployment thesis is now tracked separately from broad scale dominance:
a small CPU-first English core plus installable domain payloads for game-specific data.

| Gate | Current evidence | Status |
|---|---:|---|
| Core smaller than BPE | 14.32M vs 14.84M params | PASS |
| General English BPB | 2.0382 vs 2.0492 BPE | PASS |
| Training time | 122.5 s vs 131.5 s BPE | PASS |
| One-thread CPU generation | 2.78x BPE | PASS |
| Domain payload size | 148,808 B vs 383,008 B adapter | PASS |
| Domain training time | 51.3 s vs 183.1 s adapter | PASS |
| Domain CPU throughput | 35.7K B/s vs 8.1K B/s adapter | PASS |
| Lossless domain transfer | PPL ratio 1.0; max logit diff 0; identical generation | PASS |
| Receiver after transfer | smaller, better BPB, faster training, faster CPU generation | PASS |
| Pruned CPU deployment artifact | 0.96x BPE artifact size | PASS |
| Isolated CPU peak RSS | 0.985x BPE peak RSS | PASS |
| Isolated CPU generation | 2.13x BPE | PASS |
| Isolated CPU prefill microbench | 0.86x BPE | OPEN |

Verify:

```powershell
python scripts/benchmark_cpu_deployment_resources.py
python scripts/verify_game_ready_mobile_llm.py
```

This is still a desktop CPU/mobile-proxy certificate. Real game shipping still requires
Android/iOS or target-console latency, battery/thermal, a game-dialogue/domain dataset,
task-level NPC/game QA evaluation, and a native int8 runtime. Local isolated CPU peak RSS
is now measured with separate fresh Python processes and passes against the retained BPE
comparator; the separate isolated prefill microbench remains open.

### Cross-backend quality scorecard

LayerCake now tracks backend and quality dimensions separately so a CPU/mobile win cannot
hide a GPU loss.

| Dimension | Current result | Status |
|---|---:|---|
| Training/quality/cost vs BPE | smaller, lower BPB, faster training, fewer bytes | PASS |
| CPU generation quality/speed | quality gates pass; 317.1 B/s vs 146.8 B/s | PASS |
| Batch-1 prefill latency | 2.96 ms vs 5.63 ms BPE | PASS |
| Domain layers | smaller/faster/better than adapter; exact transfer | PASS |
| GPU generation quality | quality gates pass | PASS |
| GPU generation speed | 244.2 B/s vs 840.2 B/s BPE | OPEN |

Verify:

```powershell
python scripts/verify_cross_backend_quality_scorecard.py
```

An across-the-board CPU+GPU dominance claim is blocked until GPU generation speed also
beats the transformer comparator.

### Frontier north-star gate

The master verifier aggregates the current promoted frontier evidence and explicitly
keeps the larger north-star claim open until every remaining game/deployment gate exists.

```powershell
python scripts/verify_frontier_model_northstar.py
```

Current promoted gates:

- base 15M source/core frontier;
- transformer dominance matrix promoted tiers;
- cross-backend CPU/mobile-proxy scorecard;
- game-ready CPU/mobile proxy;
- receiver-after-transfer frontier;
- many-domain install/migration/isolation mechanics.

Current open north-star items:

- GPU generation speed;
- 20M full-corpus training-time dominance;
- real mobile/device latency;
- battery and thermal measurements;
- isolated CPU prefill microbench;
- native int8 runtime;
- trained game-dialogue, lore, and quest-state payloads;
- task-level NPC/game QA evaluation;
- domain routing policy evaluation.

The many-domain proxy currently installs `game_dialogue`, `game_lore`, and
`game_quest_state` payloads, verifies exact source/receiver migration for each, and
checks that installing other domains does not change the selected domain's logits. It uses
renamed copies of the current portable payload, so it proves install/migration/isolation
mechanics, not game-domain quality.

The game-domain workflow smoke now trains a byte-GRU portable domain from
`tests/fixtures/game_dialogue_smoke.txt`, quantizes it to int8, installs it into the
15M source and 5.40M receiver, and verifies exact migration. Current smoke metrics:
2.2185 BPB, 73.8% top-1 byte accuracy, PPL ratio 1.0, max logit diff 0.0, and identical
generated bytes after transfer. This proves the train/quantize/install/migrate workflow
for game-style text; it is not a production game-dialogue quality claim.

The cross-domain smoke extends that workflow to dialogue, lore, quest/state, and
technical text. All four payloads train, quantize to int8, transfer exactly, and pass the
smoke BPB/accuracy/printability gates. Current aggregate: mean BPB 2.2414, minimum top-1
byte accuracy 71.97%, max transfer logit diff 0.0. This is broader workflow evidence, not
an all-corpora dominance claim.

The cross-domain adapter frontier compares those four portable payloads against matched
BPE residual adapters trained on the same fixture files. LayerCake wins all four smoke
domains on domain BPB, training seconds, payload size, and exact source/receiver transfer.
Worst BPB margin is narrow on lore, -0.0019 BPB, so this is a smoke win that needs larger
external corpora and multi-seed replication before any broad domain-dominance claim.

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
developing for mobile and non-GPU desktop deployment: a smaller CPU-capable general core
plus installable, domain-specific prediction payloads.

It is not evidence that a mobile core has the same general intelligence as a larger core.
PX transfers the domain capsule's behavior exactly because that capsule owns the selected
domain prediction path. Routing, task-level code quality, native CPU/mobile kernels,
battery, thermal behavior, and real-device memory/latency remain separate gates. Local
desktop CPU peak RSS is measured separately in the deployment-resource certificate.

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

An intermediate 20M scale check narrows the boundary but does not clear it. A 20.25M
width-448 LayerCake with 32-byte local windows and QK-normalized attention reached 2.0256
BPB in 165.1 seconds. Its matched 20.61M BPE transformer reached 2.0154 BPB in 113.5
seconds. Same-byte batch-24 compression, 16-byte local windows, and shifting one block
from local byte decoding into the global patch core all improved neither gate. The best
20M LayerCake candidate remains 2.0256 BPB and 165.1 seconds; the fastest retained 20M
candidate remains slower than BPE at 135.1 seconds and worse at 2.0356 BPB. The retained
certificate is intentionally FAIL:

```powershell
python scripts/verify_scale20m_frontier.py
```

A subsequent additive multi-scale experiment also failed its early rejection gate:
four-byte coarse summaries combined with a two-byte fine stream reached 2.4216/2.4188 BPB
at 750 steps versus 2.3180 for the fixed two-byte reference, with no training-speed gain.
The implementation remains available as an experimental path, but the next full run will
require content-dependent patch boundaries rather than additive fixed-scale summaries.

That content-dependent 2/4-byte follow-up was implemented and tested as well. It reached
2.514-2.524 BPB across 2.42-3.43 mean bytes per patch, versus 2.318 for the fixed two-byte
probe. The vectorized path was fast, but changing patch positions damaged quality. It is
therefore another documented negative control, not a selected architecture.

The strongest later scale candidate uses hardware-aligned width 512, 32-byte local
windows, and QK-normalized attention. It reaches 2.0204 BPB at 25.77M parameters, but the
matched 26.30M BPE transformer reaches 1.9940 BPB and trains faster. The result is recorded
as progress, not a win. Exact portable-domain migration remains independent of this core
quality result and continues to pass without PPL or generation changes.

The first pure-PyTorch sparse-state global patch core has also been tested at the 20M
boundary. It preserves fixed two-byte ABI positions and cached generation support. The
reduced-fan-in variant improved the LayerCake 20M quality frontier from 2.0256 to 2.0214
BPB while remaining smaller than the 20.61M BPE comparator, but it still lost to BPE
quality and training time: BPE reached 2.0154 BPB in 113.5 seconds, while sparse-state
LayerCake reached 2.0214 BPB in 248.1 seconds. The retained sparse-state certificate is
therefore intentionally FAIL:

```powershell
python scripts/verify_scale20m_sparse_state_frontier.py
```

The receiving-core comparison has also been rebuilt with the current fused architecture
and a retained matched transformer artifact. The selected 6.804M receiver is still
smaller than the 6.857M BPE transformer, trains faster, and beats it on general quality.
The unchanged transferred domain remains exact and beats the transformer adapter.

| Receiver-frontier gate | LayerCake receiver | Matched transformer | Status |
|---|---:|---:|---|
| Parameters | **6.804M** | 6.857M | PASS |
| General BPB | **2.1251** | 2.1265 | PASS |
| Training time | **77.05 s** | 81.45 s | PASS |
| One-thread CPU generation | **1.47x BPE** | ratio > 1 | PASS |
| Transferred-domain BPB | **1.4691** | adapter: 2.1101 | PASS |
| Transfer invariance | PPL ratio 1.0; max logit diff 0; identical generation | n/a | PASS |

```powershell
python scripts/verify_receiver_frontier.py
```

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
  rolling/                rollbackable model-commit training substrate
  orchestration.py        CorticalSwarm-style handoff packet and router
  transfer.py             copy, PPL, and degradation contracts

scripts/
  run_paired_byte_experiment.py
  train_sparse_brick_artifact.py
  eval_portable_brick.py
  eval_lossless_domain_decoder.py
  benchmark_bpe_baseline.py
  benchmark_canonical_artifact.py
  demo_rolling_training.py
  benchmark_rolling_training.py
  benchmark_rollback_cost.py
  benchmark_cherrypick_transfer.py
  verify_research_gates.py

results/
  research_gate_certificate.json
  certificates/rolling_demo_certificate.json
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
- [Rolling training](ROLLING_TRAINING.md)
- [Preview-guided training](PREVIEW_GUIDED_TRAINING.md)
- [Model commits](MODEL_COMMITS.md)
- [Rubric training](RUBRIC_TRAINING.md)
- [Semantic CI](SEMANTIC_CI.md)
- [Scaling protocol](SCALING_PROTOCOL.md)
- [Dominance gates](DOMINANCE_GATES.md)
- [Rollback](ROLLBACK.md)
- [Branching and cherry-pick](BRANCHING_AND_CHERRYPICK.md)
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
