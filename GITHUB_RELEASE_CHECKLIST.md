# GitHub release checklist

## Required checks

```powershell
pytest -q
python scripts/verify_research_gates.py
python scripts/verify_scale5m_results.py
python -m compileall -q layercake scripts tests
git diff --check
```

Expected:

- 32 tests pass;
- selected small-scale certificate passes all gates;
- 5.40M certificate passes all architecture gates and reports BPE parity as open;
- no whitespace errors.

## Files intended for Git

- source under `layercake/`;
- reproducible scripts under `scripts/`;
- tests;
- public documentation;
- multi-size configs;
- selected JSON evidence named in `.gitignore`.

## Files intentionally local

- `runs_experiment/` checkpoints and generated tokenizer corpora;
- `.pt`, `.pth`, and `.ckpt` artifacts;
- exploratory JSON sweeps not selected by the evidence verifiers;
- raw/pre-tokenized datasets.

Model checkpoints should be published separately through Git LFS or a release artifact only
after checksums, licenses, and model cards are prepared.

## Public claim check

Before release, verify that public text says:

- small-scale general BPB parity by point estimate;
- larger-tier size, speed, adaptation, transfer, and int8 gates pass;
- larger-tier BPE general BPB remains better;
- cross-seed transfer requires canonical training contracts;
- no universal tokenizer or frontier-model dominance is claimed.
