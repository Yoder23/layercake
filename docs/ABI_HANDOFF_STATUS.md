# External capability-artifact handoff

Status date: 2026-08-06

This document defines LayerCake's boundary for externally produced capability
artifacts. It does not import external code or evidence, and it does not expand
the sealed LayerCake certificate.

## Repository boundary

LayerCake owns its execution host, canonical interfaces, package lifecycle,
transfer, composition, routing, orchestration, and product-level CPU/GPU
certification.

Foreign-teacher extraction, labeling, normalization, minimization, and source
information accounting belong to the separate
[ABI repository](https://github.com/Yoder23/abi). Its research status and
evidence must be read there, not inferred from this repository.

An ABI acquisition bundle is not a deployable cake. LayerCake accepts only an
independently validated, non-executable, signed, content-addressed artifact with
complete provenance, license, deletion lineage, and imported-information
accounting.

## Acceptance rule

The post-release `lc-direct-neural-core/2` interface is construct-certified as
the current English-core handoff target. It uses Unicode-atomic fixed and
pointer actions, strict UTF-8 input/output validation, signed safetensors-only
packages, persistent incremental state, and zero receiver learning. It passed
the same deterministic package on CPU and CUDA. The older v1 interface failed
its UTF-8 conformance audit and must not receive new external English artifacts.

This host readiness is not an acquisition result. No ABI-derived English
artifact has passed v2, and no external artifact is part of the sealed release.
See `POSTRELEASE_UNICODE_DIRECT_NEURAL_CORE_V2.md` for the exact construct scope.

An external artifact never inherits this release's quality, speed, TTFT,
memory, portability, routing, or verification evidence. Integrating it requires
a governed successor lineage and every recertification required by
`moonshot/invalidation_matrix.yaml`, including quality, isolation, identity,
teacher absence, selected-only execution, CPU/GPU behavior, memory, latency,
and hostile verification on the same final candidate.

Until such a successor is sealed, `layercake-moonshot-final` remains unchanged
and no external capability artifact is part of this release.
