# Cake registry specification v1

The local registry is rooted at `$LAYERCAKE_REGISTRY` or
`$LAYERCAKE_HOME/cakes`. Archives are immutable blobs at
`blobs/sha256/<archive-sha256>.cake`; `registry.json` maps a cake identifier to its active
version, archive/content/payload hashes, publisher, ABI, trust mode, domains, permissions,
and bounded rollback history.

Writers acquire `.registry.lock` with exclusive creation, write canonical JSON to a
same-directory temporary file, and use atomic replace. Blob writes follow the same rule.
An interrupted update therefore leaves either the previous or new complete record.

Catalogs are JSON arrays of declarative records containing at least `cake_id` and `path`.
Network transport is intentionally outside v1. A future remote catalog must authenticate
its own index and still pass the complete local package verification path.

Package content identity (`manifest.package_hash`) covers the canonical manifest with its
own package-hash field blank plus the exact safetensors bytes. The detached Ed25519
signature authenticates that digest. Archive identity hashes the final ZIP bytes and is
used for local content addressing. Both identities are checked during `cake verify`.
