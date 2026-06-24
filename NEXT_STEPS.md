# Exact Next Scientific Runs

```powershell
pytest -q
python scripts/train_abi_aligned.py --steps 100 --output results/alignment_train_smoke.json
python scripts/smoke_byte_patch.py --output results/smoke_byte_patch.json
python scripts/eval_transfer_matrix.py --output results/transfer_matrix.json
python scripts/benchmark_domain_routing.py --iterations 100 --output results/domain_routing.json
```

Before making L4 or L6 claims:

1. pretrain paired tokenized/byte-patch cores on the same byte-normalized corpus;
2. freeze the evaluation split and PPL threshold before domain training;
3. report source/target domain PPL, general PPL, ABI drift, and confidence intervals;
4. compare against tokenizer baseline, LoRA, dense adapter, and no-brick controls;
5. repeat across at least seeds 42, 314, and 2718.

Full matched 25M-1B training remains compute- and dataset-limited, not code-path-limited.

The selected sub-million-parameter gates now pass. The next scale experiment should:

1. train 25M byte-patch and BPE baselines on at least 1B identical bytes;
2. use three independent seeds and a held-out domain not present in repository source;
3. freeze thresholds and evaluation hashes before training;
4. report BPB confidence intervals, wall-clock-to-quality, energy, and peak memory;
5. repeat cross-seed, cross-size, int8, and active-brick latency gates.

The intermediate 5.40M tier and matched 5,000-step 15.45M checkpoint are complete. The
15.45M checkpoint reached 2.430 general BPB versus 2.227 for the 25.75M byte baseline,
runs 2.45x faster in the selected CUDA benchmark, and passes PX exact transfer. Next:

1. train independent 15.45M seeds;
2. evaluate on an externally sourced, contamination-audited Python corpus;
3. add task-level code completion and execution benchmarks;
4. add Android/iOS or ARM CPU, memory, battery, and thermal measurements;
5. replace load-time int8 dequantization with native int8 execution.

## General-core frontier

The deployment definition is CPU-first. A candidate cannot be called mobile-ready because
it is merely small or GPU-fast; it must beat the matched transformer on non-GPU CPU
prefill/generation or an explicitly reported Android/iOS/ARM measurement.

The 15M mobile frontier now passes the combined certificate across two LayerCake seeds:

- 14.792M parameters versus 14.844M BPE;
- 2.0446/2.0457 versus 2.0492 general BPB;
- 121.4 s mean training time versus 131.5 s;
- 2.96 ms versus 5.63 ms batch-1 prefill;
- 2.91x/2.96x one-thread cached-generation speed;
- 2.25x one-thread cached-generation speed with no-repeat-8 decoding and passing
  repetition/coherence gates;
- exact domain migration from each 15M core into an independent 5.40M host;
- migrated Python BPB 1.4418/1.4436 versus adapter BPB 2.1101/2.0951.

Next:

1. reproduce on a second general-text split and a non-repository domain;
2. run ARM/Android/iOS latency, memory, battery, and thermal measurements;
3. close the current RTX generation gap (0.62x BPE);
4. scale the matched protocol to 25M and 60M before making broader claims;
5. add task-level generation and execution benchmarks;
6. replace the current no-repeat decoding constraint with a trained anti-repetition or
   contrastive decoding objective if it improves quality without sacrificing CPU speed.

Two first-step dense scaling candidates have already been rejected under the same sampled
byte budget:

- 20.25M width-448 LayerCake, window-32/QK norm: 2.0256 BPB and 165.1 s;
- 20.25M width-448 LayerCake, same-byte batch-24: 2.0356 BPB and 135.1 s;
- 20.25M width-448 LayerCake, 5 global / 3 local: 2.0321 BPB and 157.0 s;
- 20.25M width-448 LayerCake, window-16/QK norm: 2.0306 BPB and 169.4 s;
- 20.25M width-448 LayerCake, window-16/QK batch-24: 2.0472 BPB and 145.5 s;
- 20.25M width-448 probe, 6 global / 2 local: 2.3446 BPB at 750 steps;
- 20.25M width-448 probe, 7 global / 1 local: 2.4180 BPB at 750 steps;
- 20.61M BPE comparator: 2.0154 BPB and 113.5 s;
- 23.69M LayerCake, 5+5 blocks: 2.0299 BPB and 214.1 s;
- 25.24M LayerCake, 4+4 blocks: 2.0376 BPB and 204.6 s;
- 24.09M BPE comparator: 2.0035 BPB and 158.0 s.

Adding dense width or depth alone therefore does not preserve the full north-star win.
The next candidate should reduce effective global compute without destabilizing fixed
two-byte ABI positions: hardware-fused local/global blocks, structured sparse attention,
or a causal state-space/global mixer that preserves exact cached generation and PX
transfer. Dynamic/variable patching remains rejected until it can recover quality.
Shuffling dense blocks between the global patch core and local byte decoder is now also
rejected: reducing local depth below three layers collapses the early quality probe.

An additive two-resolution hierarchy was also rejected at the 750-step probe gate:

- one fine + three coarse global blocks: 2.4216 BPB, 34.8 s;
- two fine + two coarse global blocks: 2.4188 BPB, 34.6 s;
- fixed two-byte reference: 2.3180 BPB, 32.3 s.

Coarse four-byte summaries therefore do not substitute for fine two-byte global depth in
their current additive form. Do not launch a full run from this branch. The next patching
design must make boundaries content-dependent and keep exact byte-level causal decoding.

Every future full candidate must pass `scripts/verify_scale_candidate.py`; improvement on
GPU alone is insufficient. Mobile CPU, desktop CPU, and GPU prefill/generation are
tracked as distinct required gates so CPU deployment cannot be hidden behind accelerator
speed.

The first content-dependent 2/4-byte implementation is also rejected:

- unthresholded, 3.43 bytes/patch: 2.51395 BPB;
- 25% difficulty boundaries, 2.84 bytes/patch: 2.52407 BPB;
- 50% difficulty boundaries, 2.42 bytes/patch: 2.51268 BPB;
- fixed two-byte reference: 2.31795 BPB at the same 750-step probe.

Vectorization reduced the unthresholded probe to 24.6 seconds, but boundary tuning did not
recover quality. Variable patch positions currently destabilize the learned global
semantics. Preserve fixed two-byte semantic positions in the next branch and target cost
through fused kernels, recurrent/state-space global mixing, or structured sparsity.

Further fixed-position loops are now measured:

- gated causal convolution hybrid: 2.3851 BPB, 42.1 s;
- cuDNN GRU hybrid: 2.3840 BPB, 50.0 s;
- 20.26M asymmetric local width: 2.4053 BPB;
- 23.25M seven-global/four-local depth: 2.4056 BPB;
- 25.24M batch-8/equal-byte optimization: 2.4107 BPB;
- 26M width-512, window-32, QK-normalized full run: 2.0204 BPB,
  versus matched 26.30M BPE at 1.9940 BPB;
- pure-PyTorch sparse-state 20.25M core: 2.0214 BPB and 248.1 s,
  versus matched 20.61M BPE at 2.0154 BPB and 113.5 s;
- causal BLT-style encoder/decoder bridge: 2.4490-2.6268 BPB across
  two-, three-, and four-byte patches.

The 26M model is the best larger LayerCake quality result but fails quality and training
time against its matched transformer. Do not promote it. Dense rearrangements, simple
recurrent/convolution substitutions, local widening, longer windows, QK normalization,
dropout, and one-layer byte encoders are exhausted under this protocol.

Next architecture requirement: preserve fixed two-byte ABI positions while introducing
selective active compute that is actually efficient on the target backend. Pure-PyTorch
gathered sparse attention improved BPB but failed training cost, so the next branch should
be either a hardware-fused sparse kernel path or a selective state-space/global mixer that
avoids gather-heavy attention. Any implementation must retain exact cached generation and
PX transfer.

## Receiving-core frontier

A newly trained current-architecture receiver now establishes a passing receiver-frontier
certificate:

- LayerCake receiver: 6.804M parameters, 2.1251 BPB, 77.05 s;
- matched BPE receiver baseline: 6.857M parameters, 2.1265 BPB, 81.45 s;
- one-thread CPU receiver generation: 1.47x the matched BPE;
- unchanged PX payload after migration: PPL ratio 1.0, max logit diff 0,
  identical generation;
- transferred-domain BPB: 1.4691 versus transformer adapter 2.1101.

The receiver now wins size, general quality, training time, and transferred-domain
behavior. The earlier 5.999M receiver remains a useful negative control: it reached
2.1204 BPB but trained in 83.49 seconds, missing the BPE training-time gate by 2.04
seconds. Same-byte batch-24 compression on that smaller receiver trained in 62.90
seconds but regressed to 2.1430 BPB; batch-20 regressed to 2.1306 BPB. The selected
6.804M four-local-layer receiver keeps enough quality margin to survive batch-24
same-byte compression while still training faster than the matched BPE.
Verify with `python scripts/verify_receiver_frontier.py`.

Verify the frontier:

```powershell
python scripts/verify_northstar_mobile.py
```

## Rolling-training substrate

The next engineering phase is now represented by `layercake/rolling/`: a rollbackable,
ABI-preserving model-commit layer for staged training. It is intentionally separate from
the scale-dominance claim. Its job is to make future architecture loops auditable:

- every promoted model state is a `ModelCommit` with ABI, input-interface, byte-patch,
  module, artifact, rubric, and gate hashes;
- failed updates are preserved and can be rolled back to the last passing parent;
- rubrics define trainable/frozen modules, protected capabilities, gates, and rollback
  policy;
- branches, tags, diffs, cherry-picks, and regression bisection have smoke-tested
  Python APIs and CLI coverage;
- semantic certificates and a capability ledger make it clear which claims are actually
  backed by passing gates.

Run the smoke demo:

```powershell
python scripts/demo_rolling_training.py --smoke
```

Run rolling tests:

```powershell
pytest tests/test_rolling_cli.py tests/test_rolling_rubric.py tests/test_rolling_trainer.py tests/test_model_commit.py tests/test_module_registry.py tests/test_dataset_manifest.py tests/test_gates.py tests/test_rollback.py tests/test_branching.py tests/test_cherrypick.py tests/test_bisect.py -q
```

This does not change the evidence boundary above: dense block reshuffling remains
exhausted, the sparse-state 20M branch remains a failed cost/quality gate, and no public
scale-dominance claim should be promoted until source/core and receiver-after-transfer
certificates both pass at the relevant scale.
