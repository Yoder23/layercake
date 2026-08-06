# Next Steps After the Sealed Moonshot

## Verify before changing anything

```powershell
C:\Python310\python.exe -m layercake.moonshot_campaign verify-sealed 8
C:\Python310\python.exe -m layercake.moonshot_campaign verify-all
C:\Python310\python.exe -m layercake.moonshot_final verify
```

The expected aggregate result is `completed_phases_valid: true` with Phase 8
`SEALED` at `layercake-moonshot-final`.

## Safe next actions

1. Review [docs/PHASE_STATUS.md](docs/PHASE_STATUS.md) and
   [docs/VERIFICATION_AND_LIMITS.md](docs/VERIFICATION_AND_LIMITS.md) before
   planning a release, integration, or new capability.
2. Use the documented package authoring and registry contracts for a new cake.
3. Keep ABI extraction in its separate repository and campaign.
4. Start a governed recertification before changing a sealed dependency.
5. For an external English core, use the post-release host-interface audit as
   the ownership boundary; do not attach ad hoc hidden-state hooks to the
   sealed core.

## Actions that require new authority

- Pushing or publishing `layercake-moonshot-final` to a remote;
- promoting a new hardware, mobile, energy, training, or universal-quality claim;
- changing any certified artifact or runtime path;
- importing ABI extraction artifacts into LayerCake;
- representing a management-only catalog entry as a functional capability.

The final release is evidence-backed, not self-extending. A new result becomes a
new claim only after governed recertification.
