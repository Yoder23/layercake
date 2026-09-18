# LayerCake repository map

LayerCake contains a runtime library, experimental implementations, frozen
contracts, raw evidence, and large model artifacts. This map identifies which
surface to use for each task.

## Top-level directories

| Path | Purpose | Use it when |
| --- | --- | --- |
| `layercake/` | Installable Python library | Integrating the host, package manager, routing, runtime, training, or evaluation APIs |
| `layercake/cake/` | Package manifests, signatures, archives, installer, and registry | Building or validating cake lifecycle behavior |
| `layercake/models/` | Core and capability model implementations | Working on neural execution contracts |
| `layercake/routing/` | Selection policies, routers, catalog cache, and orchestrators | Implementing request-to-capability selection |
| `layercake/runtime/` | CPU, CUDA, native, and export implementations | Working on execution performance or device support |
| `layercake/training/` | Research and fallback training workflows | Training a core or cake under a declared experiment |
| `layercake/evaluation/` | Quality, portability, performance, and campaign verifiers | Recomputing a result or adding a falsifiable evaluation |
| `layercake_extensions/` | Isolated post-release host constructs | Inspecting development interfaces that are not in the sealed release |
| `examples/` | Small untrained cake lifecycle fixtures | Trying discovery, install, verify, and removal without claiming quality |
| `tests/` | Unit, integration, construct, and campaign tests | Validating a change |
| `configs/` | Current and historical experiment configurations | Reproducing a specifically identified run; do not choose by filename alone |
| `scripts/` | Experiment, benchmark, packaging, and verification entry points | Following an exact protocol or evidence command |
| `moonshot/` | Governance, frozen phase contracts, invalidation rules, and campaign state | Determining whether work is authorized or a phase claim remains valid |
| `results/moonshot/` | Raw observations, failures, certificates, and phase seals | Auditing the sealed scientific campaign |
| `artifacts/` | Checkpoints, packages, exports, and experiment state | Reproducing an artifact-bound run; many files are research history |
| `data/` | Small fixtures, manifests, and retained evaluation/training material | Running the exact workflow that names the data hash |
| `archive/` | Explicitly archived material | Historical investigation only |
| `docs/` | Canonical user, architecture, security, status, and evidence guides | Understanding or integrating the project |

## Common tasks

| Task | Code | Documentation |
| --- | --- | --- |
| Install or verify a cake | `layercake/cake/installer.py`, `layercake/cake/registry.py` | [Quickstart](../DEPLOYMENT_QUICKSTART.md), [registry specification](CAKE_REGISTRY_SPEC.md) |
| Sign or package a cake | `layercake/cake/signing.py`, `layercake/cake/package.py` | [Cake authoring](CAKE_AUTHORING.md), [threat model](CAKE_THREAT_MODEL.md) |
| Execute a request | `layercake/__main__.py`, `layercake/routing/orchestrator.py` | [Quickstart](../DEPLOYMENT_QUICKSTART.md#run-inference-with-your-artifacts) |
| Add routing behavior | `layercake/routing/` | [Architecture](../ARCHITECTURE.md) |
| Inspect the sealed result | `layercake/moonshot_campaign.py` | [Phase status](PHASE_STATUS.md), [claims](../CLAIMS.md) |
| Understand external artifacts | Host code under `layercake_extensions/` | [ABI handoff](ABI_HANDOFF_STATUS.md) |

## Evidence navigation

Do not begin by reading every file in `results/`. Follow this order:

1. [Project status](PROJECT_STATUS.md)
2. [Claims](../CLAIMS.md)
3. [Phase status](PHASE_STATUS.md)
4. the named phase certificate and seal
5. the certificate's content-addressed raw evidence
6. the exact verifier implementation and tests

Negative and superseded evidence remains preserved intentionally. A high file
count is not an invitation to average or combine results across lineages.

## Historical research

The following root-level document families predate or sit outside the current
canonical product documentation:

- `NORTHSTAR_*`
- `BYTE_*`
- `BREAKTHROUGH_*`
- `MICRO_FRONTIER_*`
- `FAST_*` and `PRODUCTION_TRAINING_PLAN.md`
- `ROLLING_TRAINING.md`, `RUBRIC_TRAINING.md`, and related training notes
- `EXPERIMENT_RESULTS.md`
- `docs/MOONSHOT_V2_REPORT.md`

They remain useful scientific history and negative controls. They are not the
current release source of truth unless an active certificate explicitly names
and verifies them.

## Repository size

This is an evidence-heavy research repository, not a minimal SDK checkout.
Tracked history contains many checkpoints and optimizer states. For code-only
exploration, use the partial sparse-clone command in the
[deployment quickstart](../DEPLOYMENT_QUICKSTART.md#requirements). Moving
immutable scientific payloads into hash-addressed release storage or Git LFS is
future repository-maintenance work and must preserve every evidence binding
before history is changed.
