# North Star v23: routed domain cakes

North Star v23 locks a migration-compatible sparse training architecture while
preserving the v22 deployment path.

## Architecture

The 15.19M-parameter model contains three shared causal foundation layers and
five selectable one-layer domain cakes. A domain-homogeneous batch pins one
route. AdamW receives the selected cake plus explicitly unfrozen shared state;
inactive cakes receive no gradients or optimizer state.

```text
byte patches -> frozen shared layers 0..2 -> one selected domain cake
             -> frozen autoregressive patch decoder -> byte-span loss
```

The compatibility migration maps v22 global layers 0..2 into the shared
foundation, global layer 3 into route 0, and the four old local layers into
routes 1..4. Route 0 remains the default. The legacy local next-byte decoder
can reuse routes 1..4 to preserve the old general-BPB path without adding a
second block bank.

## Locked evidence

The fail-closed certificate is
`results/breakthrough_equal/northstar_v23_release_certificate.json`.

| Gate | Result |
|---|---:|
| v22 -> v23 migration | next-byte logits, ABI, patch logits, and generated patches bit-exact |
| Parameters | 15,193,137 LayerCake vs 14,950,848 transformer |
| Schema/relevance heldout generation | 100% exact on CPU and GPU |
| CPU/GPU generation | minimum locked split remains above 5x; at least 95% of v22 throughput |
| Domain-cake training, one-thread CPU | median 5.32x; minimum 5.21x |
| Domain-cake training, GPU | median 5.70x; minimum 5.39x |
| Sparse optimizer | 1,772,544 parameters (11.67% of the model) |
| Route-4 training | logged loss 0.1112 -> 0.0323 |
| Default generation path after route-4 training | bit-exact |
| Portable 15M -> 5M domain payload | logit diff 0, PPL ratio 1, identical generation on CPU/GPU |
| Full repository regression | 314/314 passed |

The CPU training path uses dynamic INT8 only for the frozen three-layer
foundation. The active cake and decoder gradient path remain FP32. CUDA uses
AMP FP16 and fused AdamW. Each repeat includes zeroing gradients, forward,
loss, backward, clipping, and AdamW. Initialization, checkpoint I/O,
evaluation, and warmup are excluded.

## Reproduce

```powershell
python scripts/migrate_v22_to_sparse_routed_cake.py

python scripts/benchmark_northstar_training_speed.py `
  --devices cpu,cuda --cpu-threads 1 --cpu-batch-size 16 `
  --gpu-batch-size 128 --raw-sequence-bytes 256 `
  --layercake-mode shared3_routed_tail_int8_foundation `
  --warmup-steps 6 --measured-steps 20 --repeats 3 `
  --output results/breakthrough_equal/northstar_v23_domain_cake_training_speed.json

python scripts/train_byte_core_from_config.py `
  --config configs/northstar_v23_route4_schema_training.json
python scripts/verify_northstar_v23_route_isolation.py
python scripts/run_northstar_release_tests.py `
  --output results/breakthrough_equal/northstar_v23_pytest_summary.json
python scripts/verify_northstar_v23_release.py
```

The CPU/GPU schema, relevance, and transfer commands are the v22 commands in
`NORTHSTAR_V22_RELEASE.md` with the migrated checkpoint and v23 output names.

## Claim boundary

The >5x training result is selected-domain-cake fine-tuning with a frozen
foundation and portable decoder versus full equal-capacity transformer
training. It is not evidence that a new LayerCake foundation trains from
random initialization 5x faster, and it is not a time-to-quality comparison.
The dense full-foundation gate remains open at the v22 audit result.

Route isolation covers the deployed autoregressive generation path. Because
the compatibility-only local next-byte decoder reuses routes 1..4, modifying
one of those routes can change that legacy decoder. The promoted migrated
route-0 checkpoint itself remains bit-exact on the previously measured
general-BPB logits.
