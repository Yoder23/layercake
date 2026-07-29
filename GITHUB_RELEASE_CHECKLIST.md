# GitHub Release Checklist

The Moonshot evidence release is already sealed locally at
`layercake-moonshot-final`. Remote publication is a separate authorized action
and is not inferred from this checklist.

## Required release-integrity checks

```powershell
git status --porcelain=v1
git cat-file -t layercake-moonshot-final
C:\Python310\python.exe -m layercake.moonshot_campaign verify-sealed 8
C:\Python310\python.exe -m layercake.moonshot_campaign verify-all
C:\Python310\python.exe -m layercake.moonshot_final verify
```

Expected results:

- a clean worktree;
- `layercake-moonshot-final` is an annotated `tag`;
- Phase 8 reports `SEALED`;
- the aggregate verifier reports `completed_phases_valid: true`;
- the final verifier reports `PROVEN`.

## Release contents

- source, tests, contracts, and documentation;
- Phase 0–8 selected raw evidence, certificates, and seals;
- the final report and release mirrors;
- preserved invalidated evidence required for auditability.

Do not publish private keys, restricted training data, unapproved checkpoints,
credentials, or executable package payloads. Check package licenses and model
licenses separately from campaign verification.

## Public-text check

Public release notes must state the exact final scope: sealed campaign,
declared CPU/GPU Qwen comparison, immutable signed packages, measured
portability/routing, and explicit exclusions for mobile, energy, GPU training,
latent fusion, universal quality, and remote-publication claims.
