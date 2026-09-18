# LayerCake documentation

This page is the documentation front door. Choose the path that matches what
you are trying to do instead of beginning in the experiment archive.

## Start here

| Audience | Read in this order |
| --- | --- |
| New user | [Project README](../README.md) -> [Deployment quickstart](../DEPLOYMENT_QUICKSTART.md) -> [Concepts](CONCEPTS.md) |
| Application integrator | [Architecture](../ARCHITECTURE.md) -> [Registry specification](CAKE_REGISTRY_SPEC.md) -> [Threat model](CAKE_THREAT_MODEL.md) |
| Cake author | [Cake authoring](CAKE_AUTHORING.md) -> [Registry specification](CAKE_REGISTRY_SPEC.md) -> [Contributing](../CONTRIBUTING.md) |
| Scientific reviewer | [Project status](PROJECT_STATUS.md) -> [Claims](../CLAIMS.md) -> [Phase status](PHASE_STATUS.md) -> [Verification and limits](VERIFICATION_AND_LIMITS.md) |
| ABI artifact integrator | [External artifact handoff](ABI_HANDOFF_STATUS.md) -> [Project status](PROJECT_STATUS.md) |
| Contributor or coding agent | [Repository map](REPOSITORY_MAP.md) -> [Contributing](../CONTRIBUTING.md) -> [AGENTS.md](../AGENTS.md) |

## Canonical documents

- [Project status](PROJECT_STATUS.md) distinguishes the sealed release from
  current post-release development.
- [Concepts](CONCEPTS.md) defines core, host, cake, registry, router, ABI, and
  evidence terminology.
- [Architecture](../ARCHITECTURE.md) maps concepts to implementation modules.
- [Deployment quickstart](../DEPLOYMENT_QUICKSTART.md) contains commands that
  can be exercised from a checkout.
- [Claims](../CLAIMS.md) maps bounded claims to evidence.
- [Verification and limits](VERIFICATION_AND_LIMITS.md) states what has and
  has not been established.
- [Repository map](REPOSITORY_MAP.md) explains the source, evidence, artifact,
  configuration, and historical-research trees.

## Source-of-truth order

When documents disagree, use this order:

1. immutable Git tag and content hashes;
2. `moonshot/campaign.yaml`, frozen contracts, and phase seals;
3. raw observations under `results/moonshot/`;
4. generated certificates derived from those observations;
5. current canonical documentation listed above; and
6. historical research notes.

Prose never promotes a scientific claim by itself. Current development code
never inherits a tagged-release claim merely because an interface name is
similar.

## Research history

LayerCake preserves unsuccessful branches and superseded experiments as part
of its scientific record. Files named `NORTHSTAR_*`, `BYTE_*`,
`BREAKTHROUGH_*`, older training plans, and `docs/MOONSHOT_V2_REPORT.md` are
historical unless a current certificate explicitly imports their evidence.
Use the [repository map](REPOSITORY_MAP.md#historical-research) to find them.
