# External capability-artifact handoff

Status date: 2026-08-03

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

An external artifact never inherits this release's quality, speed, TTFT,
memory, portability, routing, or verification evidence. Integrating it requires
a governed successor lineage and every recertification required by
`moonshot/invalidation_matrix.yaml`, including quality, isolation, identity,
teacher absence, selected-only execution, CPU/GPU behavior, memory, latency,
and hostile verification on the same final candidate.

Until such a successor is sealed, `layercake-moonshot-final` remains unchanged
and no external capability artifact is part of this release.
