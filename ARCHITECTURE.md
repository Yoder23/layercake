# LayerCake architecture

LayerCake is organized around an immutable artifact boundary: a host validates
and executes an English core plus explicitly selected capability packages. It
does not treat acquisition, packaging, routing, and evidence as one operation.

## Request path

```text
UTF-8 request
  -> application supplies an explicit destination or routing mode
  -> routing policy filters the authenticated installed catalog
  -> host selects core-only, one cake, or a structured execution plan
  -> core / selected neural modules update persistent incremental state
  -> UTF-8 response and execution trace
```

The host fails closed when a package, signature, ABI, permission, or
archive-bound routing profile is missing or inconsistent. Installed but
inactive packages must not receive proportional neural execution.

## Components and code

| Component | Responsibility | Primary implementation |
| --- | --- | --- |
| Package manifest | Declares identity, ABI, tensors, permissions, provenance, and output contract | `layercake/cake/manifest.py` |
| Package archive | Loads closed-schema, non-executable package contents and verifies hashes | `layercake/cake/package.py` |
| Signing | Creates and verifies Ed25519 package signatures | `layercake/cake/signing.py` |
| Installer | Checks trust, host compatibility, dependencies, and lifecycle operations | `layercake/cake/installer.py` |
| Registry | Stores immutable blobs and active version records atomically | `layercake/cake/registry.py` |
| Models | Defines core, portable decoder, fusion, and direct-host neural surfaces | `layercake/models/` and `layercake_extensions/` |
| Routing | Selects eligible packages and supports explicit orchestration modes | `layercake/routing/` |
| Runtime | Provides reference and optimized CPU/CUDA execution paths | `layercake/runtime/` |
| Evaluation | Recomputes quality, portability, performance, and release gates | `layercake/evaluation/` |

See [Concepts](docs/CONCEPTS.md) for terminology and the
[repository map](docs/REPOSITORY_MAP.md) for the broader tree.

## Architectural invariants

1. UTF-8 is the public request and response boundary.
2. Certified cores and package archives are immutable.
3. Cakes are signed, non-executable, content-addressed artifacts.
4. Installation performs no receiver training, calibration, or core mutation.
5. Automatic execution is archive-profile-bound, permission-filtered, and
   fail-closed.
6. Only selected packages execute.
7. Incremental state persists; continuation decoding does not recompute
   completed prompt context.
8. Package identity, functional behavior, and runtime performance are verified
   separately on the same artifact lineage.
9. A changed ABI, runtime, model, package, data set, or precision contract does
   not inherit an earlier certificate.

## Package lifecycle

```text
author tensors + declarative manifest
  -> build deterministic .cake archive
  -> sign package digest
  -> inspect and verify against host capabilities
  -> store archive by SHA-256
  -> activate registry record
  -> route only if package and profile remain eligible
  -> verify / update / rollback / remove
```

Network transport is outside the v1 registry. A remote catalog can locate a
package, but cannot bypass local package verification.

## Sealed architecture and development constructs

The exact sealed product architecture is described in
[Sealed Moonshot architecture](docs/MOONSHOT_ARCHITECTURE.md). Current HEAD
also contains newer isolated direct-neural-core constructs under
`layercake_extensions/`. Those constructs have narrow mechanical evidence and
are not a successor release. See [Project status](docs/PROJECT_STATUS.md).

Former V1/V2, byte-patch, canonical-ABI, and North Star designs remain research
controls. They are not the current source of truth unless a current certificate
imports and validates their raw evidence.
