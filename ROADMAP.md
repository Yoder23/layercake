# LayerCake Post-Release Roadmap

The eight-phase Moonshot campaign is complete and sealed. There is no automatic
"Phase 9" and no permission to alter the certified product under the release
claim. Work after `layercake-moonshot-final` belongs to one of the following
explicit tracks.

## 1. Release stewardship

- Re-run `verify-all` on clean supported environments.
- Publish the annotated tag only through an explicit release decision; remote
  publication is not currently certified.
- Maintain the package signing, registry, and vulnerability-response process.
- Record environment-specific verification results without claiming that a new
  host inherits the declared performance ratios.

## 2. Capability ecosystem

- Author new signed, non-executable packages using the canonical cake ABI.
- Evaluate each package for package identity, semantic retention, core
  immutability, routing eligibility, and selected-only execution.
- Treat Python, SQL, and regex as sealed examples—not as a claim that any new
  domain is automatically useful or portable.

## 3. ABI as a separate campaign

ABI extraction owns source-model capability extraction, minimization, imported
information accounting, and independent artifact validation. ABI code and
evidence do not belong in this LayerCake repository. A future ABI artifact may
enter only as an independently validated, signed, content-addressed package.

## 4. Governed recertification

Any change to the host, ABI, package format, package payload, router, runtime
numerics, checkpoint, data, evaluation suite, benchmark protocol, precision, or
hardware claim invokes `moonshot/invalidation_matrix.yaml`. The change must:

1. create a versioned governance amendment;
2. preserve earlier evidence as historical;
3. declare affected phase gates before experiments;
4. rerun every required dependent phase; and
5. receive a new annotated seal before a public claim changes.

## 5. Bounded research controls

Older byte, token, hybrid, training-efficiency, and architecture experiments can
continue as research only. They must not consume the sealed release identity or
be described as an unfinished Moonshot phase. Faster from-scratch English
training remains a control, not a LayerCake product gate.
