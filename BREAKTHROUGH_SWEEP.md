# LayerCake Breakthrough Sweep

This is the locked engineer-facing gate for promoting the LayerCake breakthrough claim.

The sweep is intentionally strict: the top-level certificate passes only when every
promoted gate in `breakthrough_sweep.yaml` passes. Partial wins are reported as blockers
and must not be promoted as the breakthrough.

## One-command verifier

Run from `layercake_release`:

```powershell
python scripts\verify_breakthrough_sweep.py
```

The verifier writes:

```text
results/breakthrough_equal/layercake_breakthrough_sweep_certificate.json
```

Exit code semantics:

- `0`: every promoted gate passed.
- `1`: one or more blockers remain.

## Current promoted tracks

- `fair_neural`: raw neural LayerCake evidence only. Structured schema heads, direct answer
  caches, train/eval overlap, and alias runtimes are blockers.
- `product_runtime`: portable domains, structured transducers, bounded task runtimes, and
  portable memory are allowed, but must remain explicitly labeled as product-runtime claims.

## Promotion standard

Promoted gates require:

- CPU generation speed >= 5x transformer.
- GPU generation speed >= 5x transformer.
- CPU/GPU prefill speed >= 5x transformer.
- Heldout BPB no worse than transformer.
- Task relevance, exactness, parseability, and similarity no worse than transformer and above
  absolute floors where the sub-certificate defines them.
- Exact domain migration evidence.
- Runtime footprint no worse than transformer.
- Pinned local transformer runtime evidence.
- Bounded multi-domain routing with one active specialist model per prompt.
- Hashed CPU/phone deployment bundle plus real phone runtime evidence.

The certificate is fail-closed: missing artifacts, missing metric paths, malformed JSON, and
sub-threshold metrics all become blockers.

## Evidence hygiene

Run a tiny real paired train/eval smoke before trusting a long campaign:

```powershell
python scripts\benchmark_micro_scale_curriculum_frontier.py --steps 1 --seq 32 --batch 2 --eval-batches 1 --train-bytes 20000 --eval-bytes 4000 --vocab 512 --output results/breakthrough_equal/tiny_real_train_sweep_smoke.json
python scripts\verify_tiny_real_train_smoke.py
```

This smoke may fail domination gates. Its job is narrower: prove both LayerCake and
the BPE transformer trained, evaluated on a hashed heldout split, emitted raw samples,
and retained the pass/fail result for the master sweep.

## Product-runtime artifacts

Generate the bounded domain-orchestrator certificate:

```powershell
python scripts\verify_domain_orchestrator.py
```

Generate the CPU/phone deployment bundle:

```powershell
python scripts\build_cpu_phone_deployment_bundle.py
```

That bundle remains `OPEN` until `--phone-evidence` points at a complete schema-v1,
same-device Android or iOS ARM64 measurement. A boolean hardware declaration is not
evidence and is rejected. The validator requires the tested LayerCake artifact hash to
match the bundle, at least 20 raw latency samples per model, an identical hashed prompt
pack, noninferior quality, at least 5x median and five-minute sustained speed, no larger
artifact or peak RSS, plus battery and thermal measurements without detected throttling.
Use `--core artifacts/layercake_v22_patch_int8.ts` when validating the packaged v22
runtime.

Generate the local transformer runtime comparison:

```powershell
python scripts\verify_local_transformer_runtime_comparison.py
```

That comparison remains `OPEN` until `--evidence` supplies exact local runtime metadata,
identical prompt-pack confirmation, raw generation artifacts, noninferior quality, and
CPU/GPU generation ratios of at least `5.0`.
