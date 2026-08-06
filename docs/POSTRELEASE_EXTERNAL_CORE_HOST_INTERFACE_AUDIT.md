# Post-release external English-core host-interface audit

Status date: 2026-08-06

The sealed LayerCake release remains valid and unchanged. A read-only audit of
its existing public interfaces found a narrower post-release scope gap: no
canonical interface currently accepts an independently validated signed direct
neural artifact *as the English core*.

This is not a LayerCake quality regression. `python -m
layercake.moonshot_campaign verify-all` still validates all eight sealed phases.
It is also not evidence about any external extraction or labeling method.

## Existing interfaces

| Interface | What is established | Why it is not the external English-core interface |
| --- | --- | --- |
| `lc-semantic-gpt2-768/1` | Canonical same-shape semantic residual | Historical branches fit teacher-forced targets nearly perfectly but failed autonomous realization; it is not certified for an external English core. |
| `lc-direct-neural-decoder/1` | Signed, byte-facing, self-causal capability package; exact CPU/CUDA transfer; no receiver learning | Its locked scope is one selected capability cake, not replacement or supply of the English core. |
| Ad hoc hidden-state hooks | None | They are not a canonical LayerCake package or core interface and inherit no host claims. |

The successful primitive is therefore the direct decoder's byte-facing,
self-causal, persistent-state package path—not another hidden-state residual or
block-hook bridge. The required future repair is a separately versioned direct
neural core artifact interface that reuses the existing package safety and
incremental execution primitives.

That construct repair is now complete under `lc-direct-neural-core/1`; see
`POSTRELEASE_DIRECT_NEURAL_CORE_HOST.md`. The remaining gate is acceptance and
full recertification of a real independently validated English artifact.

That future interface must independently recertify core-only English behavior,
artifact immutability, teacher absence, CPU and CUDA execution, persistent
state, same-artifact CPU speed/TTFT/RSS/memory, and compatibility with future
domain packages. It may not inherit any of those results from the sealed core
or the existing capability packages.

The original audit and its failure remain historical. The current successor is
`moonshot/postrelease_external_capability_host_interface_audit_successor_v3.json`,
with raw result
`results/moonshot/postrelease/external_capability_host_interface_audit_v2.json`.
