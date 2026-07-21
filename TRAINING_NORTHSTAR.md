# Training North Star audit

## v23 selected-domain-cake gate

V23 now passes a narrower, production-relevant sparse update gate. The model
has three frozen foundation layers and five selectable one-layer domain cakes;
only one 1,772,544-parameter cake is optimizer-resident. With the exact v23
autoregressive decoder objective, the locked certificate measures 5.32x CPU
median (5.21x minimum) and 5.70x GPU median (5.39x minimum). Route-4 training
reduced logged loss from 0.1112 to 0.0323 without changing the default deployed
generation path.

This does not close the full-core gate below. It is sparse domain-cake
fine-tuning from a published frozen foundation versus full transformer
training, not random-initialization foundation pretraining or time-to-quality.
See [NORTHSTAR_V23_ROUTED_CAKES.md](NORTHSTAR_V23_ROUTED_CAKES.md).

The 5x full-core training target is **open**. North Star v22 proves bounded
quality, generation speed, compact task deployment, and exact portable-domain
transfer; it does not prove that a new LayerCake foundation trains faster than
an equal-size tokenizer transformer.

This distinction matters. The final v21 LayerCake phase updates only a small
pointer module, while the strengthened transformer phase updates the full
model. Comparing those phase timers would not answer how quickly someone can
train a LayerCake foundation from scratch.

## Locked paired measurement

The new audit compares the 14,804,448-parameter LayerCake foundation recipe
with the 14,950,848-parameter strengthened transformer. Each measured step
includes `zero_grad`, forward pass, loss, backward pass, gradient clipping,
and AdamW update. The raw-byte volume is matched within 1%, CPU runs use one
thread and FP32, and GPU runs use AMP FP16 with fused AdamW. Three repeats
alternate which architecture runs first.

The transformer's 74-token sequence is derived from its measured
3.4641-byte/token training-corpus ratio and represents the same logical
256-byte sequence volume as LayerCake. Random IDs are sufficient for this
dense-compute throughput measurement; convergence is evaluated separately.

| Full-core measurement | LayerCake | Transformer | LayerCake/transformer |
|---|---:|---:|---:|
| Recipe CPU median | 1,007.8 bytes/s | 1,388.7 bytes/s | 0.722x |
| Recipe GPU median | 104,145 bytes/s | 97,233 bytes/s | 1.045x |
| Recipe GPU minimum repeat | - | - | 0.971x |
| Favorable core-only CPU median | 1,115.8 bytes/s | 1,554.3 bytes/s | 0.750x |
| Favorable core-only GPU median | 120,879 bytes/s | 119,648 bytes/s | 1.010x |
| Favorable core-only GPU minimum repeat | - | - | 0.981x |

The favorable lower bound removes the auxiliary patch loss and excludes
dormant patch-generator/ABI parameters from AdamW. It is intentionally biased
toward LayerCake and still does not approach 5x.

GPU peak allocated memory is also not yet a win. The recipe used 642,919,936
bytes for LayerCake versus 495,049,728 bytes for the transformer. The favorable
core-only path reduced LayerCake to 619,802,624 bytes, still above the baseline.
Persistent parameter/gradient/optimizer tensor state is nearly tied: 235.68 MB
for the recipe LayerCake versus 239.21 MB for the transformer.

The earlier equal-size, equal-quality convergence certificate reports only a
1.107x transformer-time/LayerCake-time ratio, not 5x. It remains useful as a
time-to-quality cross-check, but it is a historical run rather than the new
CPU/GPU optimizer-step matrix.

## Reproduce and verify

```powershell
python scripts/benchmark_northstar_training_speed.py `
  --devices cpu,cuda --layercake-mode recipe `
  --cpu-threads 1 --cpu-batch-size 1 --gpu-batch-size 16 `
  --raw-sequence-bytes 256 --warmup-steps 3 `
  --measured-steps 12 --repeats 3 `
  --output results/breakthrough_equal/northstar_v22_training_speed_recipe.json

python scripts/benchmark_northstar_training_speed.py `
  --devices cpu,cuda --layercake-mode next_byte_only `
  --cpu-threads 1 --cpu-batch-size 1 --gpu-batch-size 16 `
  --raw-sequence-bytes 256 --warmup-steps 3 `
  --measured-steps 12 --repeats 3 `
  --output results/breakthrough_equal/northstar_v22_training_speed_favorable_lower_bound.json

python scripts/verify_northstar_training_audit.py
```

The verifier independently recomputes every repeat ratio and separates
`measurement_status=PASS` from `training_northstar_status=OPEN`. A valid
measurement is not relabeled as a successful performance gate.

## What must change to reach 5x

The current dense path updates essentially the same parameter and optimizer
state volume as the transformer, so patching alone cannot produce a credible
5x full-training result. The next architecture needs all of the following:

- a quality-validated conditional or sparse update path with at most 20% of
  parameters active per optimizer step;
- optimizer state and gradient allocation that follow the sparse activation
  rather than touching the entire model;
- fused patch/local training kernels, with CPU and GPU implementations;
- a locked multi-seed time-to-quality protocol using the same corpus, heldout
  data, parameter window, and quality threshold;
- separately reported one-thread CPU, all-core CPU, and batch-tuned GPU
  measurements, including peak memory and initialization/tokenization costs.

Until those gates pass, the repository must not claim 5x training dominance.
If 5x training is a condition for publication rather than a future milestone,
the v22 release should not be pushed yet.
