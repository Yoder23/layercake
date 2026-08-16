# LayerCake Claim and Evidence Map

The sealed moonshot release is defined by `layercake-moonshot-final`. Every row
below is bounded by the exact release lineage and raw evidence paths.

| Claim | Evidence anchor | Certified result |
| --- | --- | --- |
| Campaign completion | `moonshot/campaign.yaml`, Phase 8 seal | All eight phases sealed; `verify-all` validates the completed campaign. |
| Phase 3 disposition | `results/moonshot/phase3/release_certificate.json` | `RETIRED_BY_GOVERNANCE`; zero headline claims and no training-efficiency, acquisition, ABI, or host-certification proof. |
| Faster full-core training | `TRAINING_NORTHSTAR.md` | OPEN; no training-dominance claim. |
| Independent reproduction | `results/moonshot/phase8/raw_runs/cleanroom_environment.json` | Detached Phase 7 checkout, 597 passing tests, all prior typed gates recomputed. |
| CPU performance | `phase8/raw_runs/reproduction_performance.json` | 9.91x LayerCake/Qwen output-byte throughput; 0.0179x median latency ratio. |
| GPU performance | same | 8.25x LayerCake/Qwen output-byte throughput. |
| CPU versus transformer GPU | same | 7.67x output-byte throughput; 0.0222x median latency ratio. |
| Functional quality | same | LayerCake 100/100 on CPU and GPU; paired bootstrap lower bounds +0.68 and +0.67. |
| Domain retention | `phase8/raw_runs/domain_retention.json` | 384/384 CPU, 384/384 GPU, and 384/384 identical outputs. |
| Portability | `phase8/raw_runs/lifecycle_portability.json` | Exact archives survive three-host install/verify/remove/reinstall with zero receiver learning. |
| Routing and catalog | `phase8/raw_runs/routing_catalog.json` | 1,980/1,980 correct rows and a 500-entry management catalog. |
| Attack resistance | `phase8/raw_runs/adversarial_falsification.json` | 32 attacks across 24 categories reached and resolved their intended boundaries. |

## Claims deliberately excluded

This release does not claim physical mobile performance, energy dominance,
GPU-training dominance, general faster foundation training, latent cake fusion,
universal model superiority, arbitrary host compatibility, or remote tag
publication. See [docs/VERIFICATION_AND_LIMITS.md](docs/VERIFICATION_AND_LIMITS.md).

All earlier claim tables are historical research context. They do not modify
this evidence map or survive changes that trigger the invalidation matrix.

## External ABI research is not an inherited claim

Capability acquisition from foreign teachers is researched in the separate
[ABI repository](https://github.com/Yoder23/abi). No external ABI result
changes or inherits this release's claims. See `docs/ABI_HANDOFF_STATUS.md` for
the repository and recertification boundary.

## Post-release host-interface audit

The additive authoritative-destination extension is construct-certified by
`moonshot/postrelease_authoritative_destination_control_decision_v95.json`.
Its narrow mechanical claim is fail-closed selection from an immutable outer
label with no core fallback and no prompt-text override. The sealed direct
orchestrator remains byte-identical; 9 focused tests, all 693 repository tests,
and the unchanged campaign verifier pass. This is not evidence for an external
artifact, acquisition, English or specialist quality, semantic purity, speed,
memory, TTFT, or superiority.

The latest isolated host construct is `lc-direct-neural-core/25`, certified by
`moonshot/postrelease_route_isolated_v25_clarification_route_decision_v89.json`.
Its bounded claim is mechanical: one signed five-route package installs on CPU
and CUDA, the original four routes remain tensor-exact, clarification physically
selects route 4, all other mappings remain unchanged, storage adoption is
allocation-bounded, and receiver learning is zero. It does not certify an
external artifact, English quality, an information minimum, performance, or ABI
superiority.

The read-only audit in
`docs/POSTRELEASE_EXTERNAL_CORE_HOST_INTERFACE_AUDIT.md` establishes only that
the sealed product originally had no canonical signed direct neural English-
core artifact role. The isolated `lc-direct-neural-core/1` extension now passes
its mechanical host construct on CPU and CUDA. It does not invalidate the
release or certify English, external acquisition, speed, memory, TTFT, domain
compatibility, or ABI superiority. See
`docs/POSTRELEASE_DIRECT_NEURAL_CORE_HOST.md`.
