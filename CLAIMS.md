# LayerCake Claim and Evidence Map

The sealed moonshot release is defined by `layercake-moonshot-final`. Every row
below is bounded by the exact release lineage and raw evidence paths.

| Claim | Evidence anchor | Certified result |
| --- | --- | --- |
| Campaign completion | `moonshot/campaign.yaml`, Phase 8 seal | All eight phases sealed; `verify-all` validates the completed campaign. |
| Phase 3 disposition | `results/moonshot/phase3/release_certificate.json` | `RETIRED_BY_GOVERNANCE`; zero headline claims and no training-efficiency, acquisition, ABI, or host-certification proof. |
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
