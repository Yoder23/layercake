# LayerCake changelog

## 2026-11-14 — incomplete producer evidence is not host failure

- Clarified that resource-limited or missing-replay producer evaluations leave
  acceptance prerequisites unmet; they establish neither artifact-quality
  failure nor a native LayerCake regression.
- Updated the documentation snapshot/version map. Runtime, package interfaces,
  checkpoints, all scientific seals and external-artifact acceptance are unchanged.
- Acquisition details and raw results remain in the separate ABI repository.

## 2026-11-13 — external-core acceptance clarity

- Clarified the current acceptance checklist and separated acquisition quality,
  integration conformance and same-artifact host performance failures.
- Kept older interface notes explicitly historical and v25 construct-only.
  No external English artifact, runtime change or new scientific pass is claimed.
- External acquisition details remain in ABI, linked rather than copied here.

## 2026-11-05 — documentation and review snapshot

- Added an explicit versioning guide separating package version, host interface,
  Git review snapshot and scientific certification.
- Kept the default branch's unsealed status and the external English artifact
  acceptance gate visible from the README and documentation index.
- Preserved all historical evidence and the immutable `layercake-moonshot-final`
  release. No model, runtime, host interface, package or benchmark was changed.
- External capability acquisition remains owned by the separate ABI repository;
  no acquisition experiments or research claims are imported here.

Earlier development and scientific history remain in Git, the phase documents,
and their immutable decision records. This changelog does not retroactively
assign semantic versions or new certificates to those records.
