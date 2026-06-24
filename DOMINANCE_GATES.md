# Dominance gates

Dominance gates are a framework for serious experiments, not a smoke-CI claim.

Tracked dimensions:

- lower training time to target BPB;
- lower final BPB at same budget;
- fewer trainable parameters updated;
- lower artifact size;
- faster CPU inference;
- lower memory;
- successful rollback after failed update;
- successful domain transfer/cherry-pick;
- lower cost to add domain;
- lower installed-vs-active compute ratio.

Smoke command:

```powershell
python scripts/run_dominance_gates.py --run-id smoke
python scripts/benchmark_tier1_dominance.py --steps 4
python scripts/verify_tier1_dominance.py
python scripts/benchmark_tier1_dominance.py --steps 4 --d-model 64 --layers 2 --heads 2 --d-byte 16 --d-abi 32 --max-patches 256 --output results/dominance/tier1_local_276k_probe.json
python scripts/verify_tier1_local_frontier.py
```

Outputs are written under `results/dominance/<run_id>.json`.

`benchmark_tier1_dominance.py` is stronger than the synthetic smoke gate. It trains:

- LayerCake blind;
- LayerCake preview-guided;
- closest matched-parameter tiny byte transformer.

It checks time-to-quality, final BPB, trainable parameters, artifact-size proxy,
cached CPU generation, generation printability, rollback/transfer placeholders, and
preview-guided improvement over blind. Passing this Tier 0/Tier 1 smoke harness still is
not a scale claim.

Current local frontier:

- 276k probe: passes.
- 474k probe: passes after empirical transition-head initialization.
- 735k probe: passes after empirical transition-head initialization.
- 1.15M probe: passes after empirical transition-head initialization.
- 2.7M probe: passes after empirical transition-head initialization.
- 5.8M probe: passes after empirical transition-head initialization.
- 8.8M probe: passes after empirical transition-head initialization.
- 10.4M probe: passes after empirical transition-head initialization and expanded equal-or-larger transformer matching.
- 12.8M probe: passes after empirical transition-head initialization and expanded equal-or-larger transformer matching.
- 19.4M probe: passes after empirical transition-head initialization and expanded equal-or-larger transformer matching.
- 25.6M probe: passes after empirical transition-head initialization and expanded equal-or-larger transformer matching.

The next rematch tier is full-corpus 15M/20M/25M using the same verifier discipline and
requiring receiver-after-transfer certificates.

Full-corpus transition-head update:

- 14.32M transition-head LayerCake passes the 15M source/core and receiver-transfer gate:
  2.0382 BPB versus 2.0492 BPE, 122.5 s versus 131.5 s, 2.78x one-thread CPU
  no-repeat-4 generation, exact transfer into the 5.40M receiver, and transferred-domain
  BPB 1.4406 versus adapter BPB 2.1101.
- 15.55M active-compute conv2 transition LayerCake reaches 2.0065 BPB versus the retained
  20.61M BPE at 2.0154, but trains in 134.9 s versus 113.5 s. This is a quality win and a
  training-time fail, so it is not promoted.

Verify the promoted transition-head 15M certificate:

```powershell
python scripts/verify_scale15m_transition_frontier.py
python scripts/verify_transformer_dominance_matrix.py
python scripts/benchmark_cpu_deployment_resources.py
python scripts/verify_game_ready_mobile_llm.py
python scripts/verify_cross_backend_quality_scorecard.py
python scripts/verify_many_domain_game_layers.py
python scripts/verify_game_domain_training_workflow.py
python scripts/verify_cross_domain_smoke_frontier.py
python scripts/verify_cross_domain_adapter_frontier.py
python scripts/verify_frontier_model_northstar.py
```

The transition verifier requires explicit margins: at least 2% fewer parameters, at least
0.5% better BPB, at least 1% faster training, at least 10% faster one-thread CPU
generation, no extra training bytes, printable/alpha-space/diverse generation, no repeated
4-gram or 8-gram in the checked sample, exact transfer PPL/logit/generation invariance,
and at least 10% better transferred-domain BPB than the transformer adapter.

The game-ready verifier adds a deployment-oriented CPU/mobile-proxy gate: at least 2x
one-thread CPU generation versus BPE, smaller/faster/better installable domain payloads
than transformer adapters, exact receiver transfer, and an isolated local CPU resource
certificate. The pruned LayerCake deployment artifact now beats the retained BPE on
artifact size, parameter memory, peak RSS, and generation speed. The isolated CPU prefill
microbench remains open, and real device latency, battery/thermal, game dialogue data,
task-level game evaluation, and native int8 runtime are still not promoted.

The cross-backend scorecard separates CPU, GPU, latency, generation quality, training,
and domain layers. Current CPU, latency, training, and domain gates pass; GPU generation
quality passes but GPU generation speed remains OPEN, so no across-the-board CPU+GPU
dominance claim is allowed.

The frontier north-star verifier is the master gate for the repository. It passes only
when the currently promoted evidence is internally consistent, and separately reports
open items that block the final claim: GPU generation speed, 20M training time, real
mobile/device measurements, isolated CPU prefill, native int8 runtime, trained
game-domain payloads, task-level NPC evaluation, and domain routing policy.

The game-domain workflow smoke proves custom game-style text can train a portable
byte-GRU payload, quantize to int8, install into separate LayerCake runtimes, and migrate
with exact PPL/logit/generation invariance. It is a workflow gate, not production
dialogue-quality evidence.

The cross-domain smoke extends the same portable-payload workflow across dialogue, lore,
quest/state, and technical prose. It must pass exact transfer and smoke quality gates for
every domain. It still does not replace large external corpora, matched transformer
adapter comparisons per domain, multi-seed runs, or task-level quality evaluation.

`verify_cross_domain_adapter_frontier.py` adds the matched-adapter smoke comparison:
LayerCake must beat a BPE residual adapter on BPB, training seconds, and payload size for
each domain while preserving exact source/receiver transfer.
