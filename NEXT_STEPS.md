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

## Preview-guided scaling experiments

The next serious experiments should use the new Rubric Preview path instead of blind
launches:

1. Tier 1 local: 1M-5M parameter LayerCake blind versus preview-guided versus matched
   byte transformer on the same byte budget.
2. Tier 1 receiver: repeat with a receiver-after-transfer stage and require exact PX
   behavior after rollback/cherry-pick.
3. Tier 2 serious: 5M-25M, multiple seeds, 100M-1B bytes, source/core and receiver
   certificates both required.
4. Only after Tier 2 passes: reopen 60M rematch with preview-guided curriculum,
   training-diff reports, and dominance gates.

Use:

```powershell
python scripts/benchmark_preview_guided_training.py
python scripts/benchmark_curriculum_modes.py
python scripts/run_dominance_gates.py --run-id smoke
python scripts/benchmark_tier1_dominance.py --steps 4
python scripts/verify_tier1_dominance.py
python scripts/benchmark_tier1_dominance.py --steps 4 --d-model 64 --layers 2 --heads 2 --d-byte 16 --d-abi 32 --max-patches 256 --output results/dominance/tier1_local_276k_probe.json
python scripts/verify_tier1_local_frontier.py
```

The empirical transition-head prior and expanded equal-or-larger matched transformer
selection now clear the 276k, 474k, 735k, 1.15M, 2.7M, 5.8M, 8.8M, 10.4M, 12.8M,
19.4M, and 25.6M local probes. The next step is a controlled full-corpus
15M/20M/25M rematch using the same dominance certificates and receiver-after-transfer
checks.

The first full-corpus transition-head rematch is now promoted at 15M class:

- 14.32M transition-head LayerCake: 2.0382 BPB, 122.5 s, 9.42M train bytes;
- 14.84M BPE comparator: 2.0492 BPB, 131.5 s, 10.32M estimated train bytes;
- one-thread CPU no-repeat-4 generation: 2.78x BPE with stricter diversity gates;
- transfer into the 5.40M receiver: PPL ratio 1.0, max logit diff 0, identical generation;
- transferred-domain BPB: 1.4406 versus adapter BPB 2.1101.

The best current 20M-comparator candidate is the 15.55M active-compute conv2 transition
model. It reaches 2.0065 BPB against the 20.61M BPE's 2.0154, but still misses the
training-time gate at 134.9 s versus 113.5 s. The next architecture loop should target
global/local fused training cost or faster convergence, because conv substitution closes
part of the gap but does not yet win the 20M time gate.

Promotion now requires `python scripts/verify_transformer_dominance_matrix.py`. This
matrix keeps local-methodology, full-corpus source, and receiver-after-transfer gates
separate, and records the 20M result as OPEN rather than allowing a partial quality win to
be marketed as full transformer dominance.

For the game-ready north star, promotion also requires
`python scripts/verify_game_ready_mobile_llm.py`. It keeps the target concrete:
CPU/mobile-proxy English generation, installable domain payloads, exact receiver transfer,
and open requirements for real mobile hardware and task-level game evaluation. The next
domain-specific step is to replace the current Python-domain payload with a small
game-dialogue/game-state corpus and add NPC dialogue or game QA task gates.

CPU deployment resources are now measured with
`python scripts/benchmark_cpu_deployment_resources.py`. The benchmark exports a pruned
patch-runtime LayerCake artifact, runs LayerCake and BPE in separate fresh Python
processes, and records artifact size, parameter memory, peak RSS, prefill, and generation.
The current pruned LayerCake runtime is smaller, has lower peak RSS, and generates faster
than the retained BPE comparator; the isolated prefill microbench remains open and should
be treated as a local CPU optimization target.

Across-the-board backend claims require `python scripts/verify_cross_backend_quality_scorecard.py`.
That scorecard currently passes training/quality/cost, CPU generation quality/speed,
batch-1 prefill latency, and domain-layer gates, but keeps GPU generation speed OPEN.
The no-repeat cached decoder now uses a tensorized mask instead of Python `tolist`
candidate selection, improving the retained GPU no-repeat path while preserving quality.
The next GPU-specific architecture task is a cached generation fast path that further
reduces per-byte Python/kernel overhead or emits larger verified chunks without degrading
the quality gates.

The master frontier gate is `python scripts/verify_frontier_model_northstar.py`. It now
includes a many-domain proxy via `python scripts/verify_many_domain_game_layers.py`.
That proxy proves three game-named payloads can be installed, migrated exactly, and kept
isolated with zero selected-domain logit interference. The next non-proxy game task is to
train real `game_dialogue`, `game_lore`, and `game_quest_state` payloads and add task-level
NPC/game QA gates plus a routing-policy evaluator.

The custom game-domain training path is now smoke-tested:
`train_portable_domain_decoder.py` accepts `--domain-file`, `eval_lossless_domain_decoder.py`
accepts `--eval-file`, and `verify_game_domain_training_workflow.py` verifies a
train/quantize/install/migrate loop on a game-dialogue fixture. This makes the repo ready
to ingest a real game corpus, but the production claim still requires larger held-out
game data and task-level dialogue/quest evaluation.

`verify_cross_domain_smoke_frontier.py` now broadens that smoke from one fixture to four
text styles: game dialogue, game lore, game quest/state, and technical prose. The next
promotion step is not more tiny fixtures; it is larger external corpora, matched
transformer adapters per domain, multiple seeds, and task-level quality gates.

`verify_cross_domain_adapter_frontier.py` now supplies the first matched-adapter version
of that check. The four smoke domains beat BPE residual adapters on BPB, training time,
payload size, and exact transfer. The lore margin is very small, so the next promotion
still needs external corpora and multi-seed confidence intervals.
