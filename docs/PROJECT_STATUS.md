# LayerCake project status

Status date: 2026-11-17

LayerCake has a sealed scientific snapshot and a newer host-development branch.
They must not be presented as one artifact lineage.

The November review update is documentation-only. No new external English core
has been accepted and no scientific gate is promoted. See [versioning](VERSIONING.md)
and the [changelog](../CHANGELOG.md) for exact review identities and boundaries.
Incomplete or resource-limited producer evaluations do not change the host's
scientific status or supply a qualified external artifact.
Likewise, qualification of an external source/control is not qualification or
host acceptance of its recipient. Those require the actual same-artifact quality,
conformance and performance evidence in the [handoff checklist](ABI_HANDOFF_STATUS.md).

## Status summary

| Surface | Identity | Status | What may be said |
| --- | --- | --- | --- |
| Sealed research release | `layercake-moonshot-final` at `0537cbb9e93cd7ebd4ba01c0bf641414ecebb1c3` | Locally revalidated; immutable release boundary | Only the bounded claims in `CLAIMS.md` and `docs/VERIFICATION_AND_LIMITS.md` |
| Default branch | Post-release HEAD | Unsealed development lineage | Mechanical construct results backed by their individual decision records |
| Latest route-isolated host construct | `lc-direct-neural-core/25` | Construct-certified | Five physically isolated rank-16 routes, selected-route execution, signed-package lifecycle, and fixture CPU/CUDA identity |
| Authoritative-destination extension | Post-release additive extension | Construct-certified | Immutable outer destination selects only its registered package and fails closed otherwise |
| External English artifact | None accepted into a successor release | Open | No external artifact quality, speed, memory, TTFT, or acquisition claim |
| Foreign-teacher acquisition | Separate ABI repository | Outside LayerCake | No ABI research result is inherited here |

## Why current HEAD is unsealed

Post-release changes touched registered package/runtime components. The frozen
invalidation matrix therefore rejects inheritance of the old
`precision_contract` and dependent Phase 2-8 seals. This is expected behavior,
not evidence that the immutable tag regressed.

Use the exact tag for release verification. Use current HEAD for development
and its narrowly scoped construct tests. A future successor release must run
the governed recertification required by `moonshot/invalidation_matrix.yaml`.
Exact-lineage evidence tests skip their positive replay on an incompatible
development checkout and instead require the stale evidence to fail closed.

## Sealed campaign disposition

Phases 0, 1, 2, 4, 5, 6, 7, and 8 have sealed scientific results for the exact
release lineage. Phase 3 is sealed as `RETIRED_BY_GOVERNANCE`; its original
faster-training objective was retired with zero headline training-efficiency
claims. It was not skipped, and it must not be described as proof of faster
training or external capability acquisition.

See [Phase status](PHASE_STATUS.md) for the evidence map.

## Availability boundary

The evidence certifies the exact local tag and retained assets described by the
campaign. It does not itself assert that every tag, checkpoint, or release
asset is published remotely. Users should distinguish source availability,
artifact availability, local reproducibility, and independent reproduction.

## Next product gate

The post-release host constructs need one real, independently validated,
immutable external English artifact. The same final payload must then pass
output conformance, English quality, collapse, CPU/GPU, throughput, TTFT,
memory, package-identity, and hostile-verification gates before any successor
release claim changes.

Acquisition diagnostics do not satisfy this acceptance gate. A model can execute
correctly yet lack the required English behavior; conversely, a capable artifact
can expose a host integration defect. These require separate evidence. The
[handoff checklist](ABI_HANDOFF_STATUS.md#current-acceptance-checklist) states what
must cross the boundary and how failures are attributed. No new artifact was
accepted in this documentation update; runtime, interface and model bytes are
unchanged.
