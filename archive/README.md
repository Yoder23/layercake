# Historical experiment archive policy

The root `scripts/`, `results/`, and `runs_experiment/` trees contain historical experiments
needed to reproduce earlier certificates. They are not the canonical moonshot package and
are intentionally retained in place because existing certificates and tests reference
their paths. New architecture, package, routing, runtime, and evaluation work lives under
the focused `layercake/` subpackages documented in `docs/MOONSHOT_ARCHITECTURE.md`.

Generated Inductor/Triton caches formerly tracked under `.cache`, `.i`, `.t`, and related
roots are removed from source control and ignored. Historical evidence is preserved;
transient compiler output is not.
