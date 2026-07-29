# LayerCake Architecture

The release architecture is documented in
[docs/MOONSHOT_ARCHITECTURE.md](docs/MOONSHOT_ARCHITECTURE.md). It is the
current description of the sealed host, capability packages, routing, and
runtime boundaries.

## Architectural invariants

1. UTF-8 is the public request and response boundary.
2. The core and promoted package archives are immutable after certification.
3. Packages are signed, non-executable, content-addressed artifacts.
4. Installation performs no receiver training, calibration, or core mutation.
5. Automatic execution is archive-profile-bound, permission-filtered, and
   fail-closed.
6. Only selected packages may execute; inactive installed packages do not
   receive proportional neural work.
7. Incremental state is persistent and continuation decoding does not recompute
   completed prompt context.
8. Product claims are tied to exact artifacts and raw evidence through the
   invalidation matrix.

The former V1/V2, byte-patch, canonical-ABI, and North Star descriptions in
repository history are valuable research controls. They are not the source of
truth for the sealed release unless a current certificate imports and validates
their raw evidence.
