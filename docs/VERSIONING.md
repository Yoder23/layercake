# Versions and release boundaries

## Choose the right identity

| Identity | Current value | What it means |
| --- | --- | --- |
| Installable Python package | `0.1.0` | Alpha API metadata; not a scientific certificate |
| Scientific release | `layercake-moonshot-final` | Exact sealed campaign lineage at `0537cbb9e93cd7ebd4ba01c0bf641414ecebb1c3` |
| Latest route-isolated host construct | `lc-direct-neural-core/25` | A specific compatibility contract; construct-only |
| Documentation/review snapshot | `layercake-review-2026-11-05` | Named source snapshot; not a new certified runtime |
| Default branch | `master` | Moving development branch; unsealed |

The review snapshot changes documentation only. Package and host-interface
versions therefore stay unchanged. A tag is available only after its remote
publication is verified; do not infer publication from this table alone.
The [review record](REVIEW_SNAPSHOT_2026_11_05.md) contains validation scope and
the portable-test receipt.

## Reproduce an identity

```bash
git fetch origin --tags
git show --no-patch layercake-review-2026-11-05
git rev-parse 'layercake-review-2026-11-05^{commit}'
```

Use a separate checkout of `layercake-moonshot-final` for scientific release
replay. Do not move a tag, replace an artifact under the same name, or use a
newer host to satisfy an old source hash. See [verification and
limits](VERIFICATION_AND_LIMITS.md) for the asset and scope requirements.

## Change policy

- Public Python API changes receive a package version and changelog entry.
- Incompatible package or execution contracts receive a new canonical interface
  identity; a package must declare the interface it targets.
- Research and documentation snapshots receive immutable annotated Git tags.
- A scientific promotion additionally requires preregistration, raw evidence,
  same-artifact checks, adversarial verification and governed recertification.
  A package version, tag, green CI run or release page cannot supply those gates.

## External artifacts

LayerCake's next product gate is one independently validated external English
artifact, accepted and measured as the same payload for quality, CPU/GPU
throughput, TTFT, memory, persistence, isolation and package lifecycle.
Acquisition research lives in [ABI](https://github.com/Yoder23/abi). Follow
the [handoff contract](ABI_HANDOFF_STATUS.md); neither repository inherits the
other's scientific claims.
