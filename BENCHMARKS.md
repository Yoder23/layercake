# Benchmark Protocol

Smoke commands emit JSON suitable for automation:

```powershell
python scripts/smoke_byte_patch.py --output results/smoke_byte_patch.json
python scripts/eval_lossless_copy.py --output results/lossless_copy_v2.json
python scripts/eval_abi_alignment.py --output results/abi_alignment_smoke.json
python scripts/eval_tokenizer_independent_transfer.py --output results/interface_transfer_smoke.json
python scripts/benchmark_training_cost.py --output results/training_cost_static.json
python scripts/benchmark_byte_patch.py --output results/byte_patch_benchmark.json
python scripts/benchmark_domain_routing.py --output results/domain_routing_benchmark.json
python scripts/verify_northstar_mobile.py
```

Required real-run metrics are trainable/total parameters, steps, wall time, bytes or
tokens processed, loss curve, domain/general validation PPL, memory, installed and active
brick counts, routing overhead, and patch compression ratio.

Smoke numbers validate code paths only. They are not model-quality evidence.

The selected 15M-class mobile certificate additionally requires two core seeds, exact
stateful cached-generation BPB, one-thread generation, unchanged cross-host domain
migration, and a matched domain-adapter comparison. GPU generation is reported but is not
a passing gate because the current implementation loses that benchmark.

## Full north-star promotion contract

Larger candidates must provide every field consumed by
`layercake.northstar.NorthStarMetrics` and pass:

```powershell
python scripts/verify_scale_candidate.py `
  --metrics results/<candidate>_northstar_metrics.json `
  --output results/<candidate>_northstar_certificate.json
```

Required gates are smaller parameter count, better heldout BPB, no additional sampled
training bytes, lower training wall time, faster mobile and desktop prefill/generation,
faster GPU prefill/generation, exact migration PPL/logits, and better migrated-domain BPB.
Missing measurements are not treated as passes.

Measured paired-training and transfer results are recorded in
[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md). Raw JSON artifacts are under `results/`,
and reproducible model/brick artifacts are under `runs_experiment/paired_seed*.pt`.
