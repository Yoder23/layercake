# Rolling training

`layercake/rolling/` is the training-control substrate for future LayerCake iteration.
It is designed for small, reversible semantic updates instead of opaque checkpoint
overwrites.

Core contract:

- train from a declared `TrainingRubric`;
- commit every candidate as a content-addressed `ModelCommit`;
- run semantic gates before promotion;
- preserve failed commits;
- roll back trainable modules to the last passing parent when gates fail;
- keep ABI, input-interface, and byte-patch hashes stable unless a rubric explicitly
  targets an ABI migration.

Smoke demo:

```powershell
python scripts/demo_rolling_training.py --smoke
```

Benchmarks:

```powershell
python scripts/benchmark_rolling_training.py
python scripts/benchmark_rollback_cost.py
python scripts/benchmark_cherrypick_transfer.py
```

The current demo is intentionally tiny. It proves the mechanics: pass, fail, rollback,
safe follow-up, semantic certificate emission, capability ledger entry, and module
cherry-pick under compatible ABI hashes. It does not prove language-model quality.

The next layer is preview-guided rolling training. It inserts a non-destructive preview
and compiled syllabus before staged training. See [PREVIEW_GUIDED_TRAINING.md](PREVIEW_GUIDED_TRAINING.md).
