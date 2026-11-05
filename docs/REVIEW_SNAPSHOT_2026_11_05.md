# November 2026 LayerCake review snapshot

This is a documentation-only update, not a new model or scientific release.
The intended annotated tag is `layercake-review-2026-11-05`. Its published Git
commit is the source identity; the immutable scientific tag stays unchanged.

## Validation

On the declared Windows/Python 3.10 development machine, with CUDA disabled in
the test process, the portable suite reports **695 passed, 14 skipped**. The
[JUnit receipt](../results/review_2026_11_05/portable-tests.xml) preserves exact
test names, failures, skips and timings. Warnings are retained in test output;
this is not a new GPU benchmark or a release recertification.

Documentation checks passed for 20 canonical documents and tracked local links.
The CLI help command succeeded, and both wheel and source distribution built
with the installed toolchain. Packaging emitted existing license-metadata and
dependency deprecation warnings; those are not silently treated as runtime
changes in this documentation snapshot.

```powershell
$env:CUDA_VISIBLE_DEVICES=''
python -m pytest -q
python scripts/check_docs.py
python -m layercake --help
```

The skipped exact-lineage and unavailable-device checks are not passes. Use the
exact scientific tag and retained assets for campaign replay, as described in
[Verification and limits](VERIFICATION_AND_LIMITS.md).

## Current handoff

The current host remains construct-certified and unsealed. One independently
validated external English artifact still needs same-artifact quality,
throughput, TTFT, memory, isolation, signed installation and hostile checks.
No acquisition result or certificate from another repository is imported by
this snapshot. See [project status](PROJECT_STATUS.md),
[handoff](ABI_HANDOFF_STATUS.md), and [versioning](VERSIONING.md).
