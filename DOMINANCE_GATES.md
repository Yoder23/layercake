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

The next rematch tier is 5M/15M/20M using the same verifier discipline.
