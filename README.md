# LayerCake

**Moonshot campaign status: SEALED.**

The eight-phase LayerCake campaign is locally sealed at the annotated Git tag
`layercake-moonshot-final` (`0537cbb9e93cd7ebd4ba01c0bf641414ecebb1c3`).
The tag is the release boundary. The canonical campaign state, raw evidence, and
machine validation are under `moonshot/` and `results/moonshot/`.

All phase lifecycle entries are sealed. Phase 3 has the explicit disposition
`RETIRED_BY_GOVERNANCE`: it preserved the bounded training controls and retired
the original faster-training requirement with zero headline scientific claims.
It was not skipped, but it also must not be described as a training-efficiency
or host-certification proof.

LayerCake is a capability host: an English core executes selected, signed,
non-executable capability packages over a byte-facing interface. It is designed
for fast CPU and GPU inference, immutable packages, persistent incremental
state, selected-only execution, lossless package transfer, and safe dynamic
routing. ABI extraction is a separate project and is not bundled into this
repository or release.

## Certified release scope

The final independent clean-room verification reran the sealed Phase 7 product
from a detached checkout and passed every Phase 8 gate.

| Area | Certified result |
| --- | --- |
| Campaign lifecycle | Phases 0 through 8 are sealed and verify as one ordered campaign. |
| Clean-room regression | 597 tests passed in the detached Phase 7 checkout. |
| Functional quality suite | LayerCake passed 100/100 CPU and 100/100 GPU prompts; the locked Qwen comparator passed 23/100 and 24/100. |
| Quality uncertainty | Paired bootstrap 95% lower bounds for the LayerCake-minus-Qwen success delta were +0.68 on CPU and +0.67 on GPU. |
| CPU comparison | 9.91x LayerCake/Qwen output-byte throughput; LayerCake median latency ratio 0.0179x. |
| GPU comparison | 8.25x LayerCake/Qwen output-byte throughput. |
| CPU versus transformer GPU | 7.67x output-byte throughput; LayerCake median latency ratio 0.0222x. |
| Domain retention | 384/384 held-out cases passed on CPU, 384/384 on GPU, with 384/384 identical outputs. |
| Core safety | 100/100 core-only abstentions; no receiver training or calibration. |
| Package portability | Three fresh hosts installed, verified, removed, and reinstalled identical package bytes. |
| Routing and catalog | 1,980/1,980 routing rows passed; the management catalog exercised 500 entries. |
| Adversarial verification | 32 hostile checks across 24 categories reached their intended boundaries and were resolved. |

These are deliberately bounded claims for the exact sealed hardware, runtime,
packages, checkpoints, data, and Qwen deployment digest. They do not establish
physical mobile performance, calibrated energy dominance, GPU-training
dominance, latent neural fusion, external-laboratory independence, or faster
from-scratch English acquisition.

## Phase map

| Phase | Objective | Release status |
| --- | --- | --- |
| 0 | Permanent governance and campaign state | Sealed |
| 1 | Benchmark truth | Sealed |
| 2 | Matched-quality CPU core performance | Sealed |
| 3 | Governed retirement of the original training-efficiency objective and preservation of its controls | Sealed as `RETIRED_BY_GOVERNANCE`; no training-efficiency or host-certification claim |
| 4 | One useful lossless portable Python capability | Sealed |
| 5 | Generic multi-domain extensibility | Sealed |
| 6 | Safe routing and catalog scalability | Sealed |
| 7 | Integrated CPU, GPU, and CPU-versus-GPU evidence | Sealed |
| 8 | Independent hostile verification and release | Sealed at `layercake-moonshot-final` |

See [the phase-status document](docs/PHASE_STATUS.md) for evidence anchors and
the governing tags.

## Verify the release

Use the declared Python 3.10 environment. These commands are read-only except
for normal temporary test caches:

```powershell
C:\Python310\python.exe -m layercake.moonshot_campaign verify-sealed 8
C:\Python310\python.exe -m layercake.moonshot_campaign verify-all
C:\Python310\python.exe -m layercake.moonshot_final verify
```

`verify-all` must report `completed_phases_valid: true`. The final tag is local
in this checkout; remote publication is intentionally not claimed by the
verifier.

## Architecture at a glance

```text
UTF-8 request
  -> sealed English host with persistent incremental state
  -> authenticated package registry and archive-bound routing profile
  -> core-only abstention, one selected cake, top-k, or structured multidomain plan
  -> selected neural capability execution only
  -> UTF-8 response and auditable execution trace
```

The release uses a frozen direct neural-decoder attachment path for the promoted
packages. That is a byte-facing capability-host contract, not a claim of latent
semantic fusion. See [the architecture guide](docs/MOONSHOT_ARCHITECTURE.md)
and [the cake registry specification](docs/CAKE_REGISTRY_SPEC.md).

## Documentation

- [Phase status and evidence map](docs/PHASE_STATUS.md)
- [Verified claims and explicit limits](docs/VERIFICATION_AND_LIMITS.md)
- [Architecture and runtime](docs/MOONSHOT_ARCHITECTURE.md)
- [Benchmark and reproduction protocol](BENCHMARKS.md)
- [Capability authoring](docs/CAKE_AUTHORING.md)
- [Registry and package format](docs/CAKE_REGISTRY_SPEC.md)
- [Threat model](docs/CAKE_THREAT_MODEL.md)
- [Post-release operating plan](ROADMAP.md)

Older North Star, tokenizer, byte-patch, and V2 development documents remain in
the repository as historical research. They are not the release source of truth
unless a current phase certificate imports and recomputes their raw evidence.
