# LayerCake Release Quickstart

This guide is for inspecting and verifying the sealed Moonshot release. It does
not promise that a generic game, production stack, mobile device, or new domain
inherits the certified result.

## Verify the checkout

```powershell
git status --porcelain=v1
git tag -l layercake-moonshot-final
C:\Python310\python.exe -m layercake.moonshot_campaign verify-sealed 8
C:\Python310\python.exe -m layercake.moonshot_campaign verify-all
```

The worktree must be clean. `verify-all` must report
`completed_phases_valid: true`.

## Understand the package boundary

Promoted capabilities are signed, non-executable `.cake` archives. They install
through the registry after manifest, signature, ABI, archive-hash, and
permission validation. Installation must not train, calibrate, mutate the host,
or accept executable payloads.

Read these documents before integrating a package:

- [Cake authoring](docs/CAKE_AUTHORING.md)
- [Registry specification](docs/CAKE_REGISTRY_SPEC.md)
- [Threat model](docs/CAKE_THREAT_MODEL.md)
- [Verified limits](docs/VERIFICATION_AND_LIMITS.md)

## Integration constraints

Use the exact certified host, package bytes, and routing profiles when making a
claim based on this release. A modified model, new package, new device class,
or different runtime is a new evidence lineage. Start governed recertification
before describing it as part of the sealed Moonshot.
