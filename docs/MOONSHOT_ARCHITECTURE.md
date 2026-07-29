# Sealed Moonshot Architecture

This page describes the product architecture certified by the sealed campaign.
It does not authorize a new architecture branch or expand the release scope.

## System boundary

```text
UTF-8 request
  -> sealed English host
  -> package eligibility and archive-bound routing
  -> core-only / selected cake / top-k / structured multidomain plan
  -> persistent incremental neural execution
  -> UTF-8 response + auditable trace
```

The external boundary is UTF-8 bytes in and out. The promoted attachment path
is `lc-direct-neural-decoder/1`: an explicitly selected, signed neural capability
package consumes and returns byte-facing state through the frozen host contract.
It is not a latent-residual fusion claim.

## English host

The English host is immutable for the sealed Phase 4–8 product lineage. It
maintains persistent incremental state: prompt context is prefetched once and
continuation work updates only current state. The runtime records selected
modules, prompt/request timing, memory scope, and output-byte measurements.

The release claim is not about a preferred internal representation. Byte,
token, and hybrid research lines remain historical controls. The promoted host
is defined by its hashes, ABI, runtime numerics, checkpoints, and certificates,
not by a generic architectural label.

## Capability packages

Promoted capabilities are immutable `.cake` archives. They are:

- non-executable and schema-closed;
- content-addressed and signed;
- bound to a canonical ABI and package manifest;
- verified before installation and before automatic routing;
- installable, removable, and reinstallable without changing the English core;
- evaluated separately for exact archive identity and semantic retention.

The sealed product includes real Python, SQL, and regular-expression packages.
Package bytes are identical across certified compatible hosts. Installation runs
no receiver-side training or target-domain calibration.

## Routing and orchestration

The registry discovers installed eligible packages. An archive-bound profile
maps a request to the current catalog only after authentication, permission,
and profile-hash checks. The supported certified modes are core-only, automatic
top-1, automatic top-k, explicit manual selection, structured multidomain
orchestration, and abstention.

Only selected real packages may be loaded, prefetched, or decoded. Catalog-only
entries are management descriptors and never become functional capabilities by
being counted in a scale test. A missing or mismatched profile fails closed.

## Hardware and performance boundary

The certified CPU and CUDA runtimes share the same sealed host, package bytes,
functional suite, and output contract. The Phase 8 evidence establishes device
retention and CPU/GPU output identity for the held-out domain set. It does not
claim ARM/mobile behavior, calibrated energy behavior, or a custom-kernel
advantage outside the declared laptop configuration.

## Security boundary

The host treats package inputs, manifests, profiles, archives, data references,
and release evidence as untrusted until validated. The Phase 8 hostile suite
includes path traversal, signature and manifest mutation, profile mismatch,
permission escalation, training/API backdoors, inactive-package execution,
catalog manipulation, lineage mixing, and claim-boundary attacks. See
[the threat model](CAKE_THREAT_MODEL.md) for the package-level contract.
