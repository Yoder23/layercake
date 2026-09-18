# LayerCake

LayerCake is a research implementation of a modular language-model host. A
LayerCake deployment combines an English core with independently packaged
capabilities called **cakes**, then loads and executes only the capabilities a
request selects.

[Get started](DEPLOYMENT_QUICKSTART.md) | [Documentation](docs/README.md) |
[Architecture](ARCHITECTURE.md) | [Verified claims](CLAIMS.md) |
[Contributing](CONTRIBUTING.md)

> **Project status:** the scientific release is the immutable local tag
> `layercake-moonshot-final` at commit
> `0537cbb9e93cd7ebd4ba01c0bf641414ecebb1c3`. The default branch is a newer,
> unsealed host-development lineage. It does not inherit the release's Phase
> 2-8 certificates. See [Project status](docs/PROJECT_STATUS.md) before using
> benchmark numbers or making release claims.

## Why LayerCake?

Traditional model extension often couples every capability to one model
checkpoint. LayerCake instead makes the host/package boundary explicit:

- cakes are immutable, content-addressed, non-executable archives;
- installation performs no receiver training or core mutation;
- package eligibility and automatic routing fail closed;
- inactive installed cakes do not receive proportional neural execution;
- request and response boundaries are UTF-8;
- persistent incremental state avoids recomputing completed context; and
- package identity, semantic behavior, and performance are measured as
  separate claims.

LayerCake hosts and executes capability artifacts. Extracting knowledge from a
foreign teacher is a separate problem owned by the
[ABI project](https://github.com/Yoder23/abi); ABI code and evidence are not
vendored here.

## What can I do from this checkout?

| Goal | Starting point | Important boundary |
| --- | --- | --- |
| Inspect, install, verify, and remove a local cake | [Five-minute quickstart](DEPLOYMENT_QUICKSTART.md#five-minute-package-lifecycle) | Bundled example cakes are untrained lifecycle fixtures, not useful specialists. |
| Understand the system | [Concepts](docs/CONCEPTS.md) and [architecture](ARCHITECTURE.md) | The sealed release and current development host are distinct lineages. |
| Integrate a compatible core and cake | [Inference interface](DEPLOYMENT_QUICKSTART.md#run-inference-with-your-artifacts) | You must supply artifacts matching the selected host interface. |
| Author a cake | [Cake authoring](docs/CAKE_AUTHORING.md) | A new cake earns no quality or portability claim automatically. |
| Recompute the sealed campaign | [Release verification](DEPLOYMENT_QUICKSTART.md#verify-the-sealed-research-release) | Use a detached exact-tag checkout and the required retained assets. |
| Navigate code, evidence, and historical experiments | [Repository map](docs/REPOSITORY_MAP.md) | `results/` is evidence; `artifacts/` is model/package material; neither is the library API. |

## Five-minute package lifecycle

LayerCake requires Python 3.10 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m layercake --help
```

Exercise the package manager with the bundled untrained Python fixture:

```powershell
python -m layercake cake --registry .cache/demo-registry --catalog examples/catalog.json search python
python -m layercake cake --registry .cache/demo-registry install examples/python.cake --trusted-local
python -m layercake cake --registry .cache/demo-registry list
python -m layercake cake --registry .cache/demo-registry verify python
python -m layercake cake --registry .cache/demo-registry remove python
```

These commands demonstrate discovery, compatibility checking, content-addressed
storage, integrity verification, and removal. They do not demonstrate model
quality. Continue with the [deployment quickstart](DEPLOYMENT_QUICKSTART.md)
when you have a compatible core and cake.

## Architecture at a glance

```text
UTF-8 request
  -> LayerCake execution host + persistent state
  -> authenticated registry and package-bound routing policy
  -> core-only, one selected cake, or an explicit orchestration plan
  -> selected neural modules only
  -> UTF-8 response + auditable execution trace
```

The primary implementation surfaces are:

- `layercake/cake/`: package schema, signing, installation, and registry;
- `layercake/models/`: cores, portable decoders, and canonical interfaces;
- `layercake/routing/`: policies, routers, catalogs, and orchestration;
- `layercake/runtime/`: CPU, CUDA, native, and export paths;
- `layercake/training/`: research and fallback training workflows; and
- `layercake/evaluation/`: quality, portability, performance, and campaign
  verification.

See the [repository map](docs/REPOSITORY_MAP.md) before navigating the larger
research and evidence surfaces.

## Scientific release in one table

The following are bounded results for the exact tagged release, artifacts,
benchmark suite, comparator deployment, and laptop hardware named by the Phase
8 evidence. They are not universal performance claims.

| Area | Tagged-release result |
| --- | ---: |
| Functional suite | LayerCake 100/100 on CPU and GPU; locked Qwen comparator 23/100 CPU and 24/100 GPU |
| CPU output-byte throughput ratio | 9.91x |
| GPU output-byte throughput ratio | 8.25x |
| LayerCake CPU vs comparator GPU | 7.67x |
| Held-out domain retention | 384/384 CPU, 384/384 GPU, identical on 384/384 |
| Routing checks | 1,980/1,980 |
| Hostile checks | 32 attacks across 24 categories |

The release does **not** establish universal language-model superiority,
faster foundation training, physical mobile performance, calibrated energy
dominance, arbitrary third-party host compatibility, latent multi-cake neural
fusion, or external-laboratory independence. Read
[Verification and limits](docs/VERIFICATION_AND_LIMITS.md) for the complete
claim boundary.

## Release tracks

LayerCake currently has two intentionally separate tracks:

1. **Sealed research release** - `layercake-moonshot-final`, the immutable
   evidence lineage used for the Phase 0-8 claims.
2. **Post-release host development** - the default branch, including newer
   signed direct-neural-core host constructs. These constructs prove mechanical
   host properties only until a successor campaign certifies a real artifact on
   the same lineage.

Phase 3 retired its original faster-training objective by governance. It is a
sealed lifecycle disposition with zero training-efficiency claims, not proof of
faster training and not a skipped phase.

Full-core training speed is a separate, currently open gate. LayerCake retains
ordinary research and fallback training, but this release makes no training-
dominance claim.

## Documentation paths

- **New user:** [Deployment quickstart](DEPLOYMENT_QUICKSTART.md)
- **Integrator:** [Concepts](docs/CONCEPTS.md), [architecture](ARCHITECTURE.md), and [registry specification](docs/CAKE_REGISTRY_SPEC.md)
- **Cake author:** [Authoring guide](docs/CAKE_AUTHORING.md) and [threat model](docs/CAKE_THREAT_MODEL.md)
- **Research reviewer:** [Claims](CLAIMS.md), [phase status](docs/PHASE_STATUS.md), and [verification limits](docs/VERIFICATION_AND_LIMITS.md)
- **Contributor:** [Contributing guide](CONTRIBUTING.md) and [repository map](docs/REPOSITORY_MAP.md)
- **ABI integrator:** [External artifact handoff](docs/ABI_HANDOFF_STATUS.md)

The numerous North Star, byte-model, training, and architecture-search files at
the repository root are preserved research history. They are indexed in the
[repository map](docs/REPOSITORY_MAP.md) and are not the current product or
release source of truth.

## License and citation

LayerCake is licensed under Apache-2.0. See [LICENSE](LICENSE). Citation
metadata is available in [CITATION.cff](CITATION.cff).
