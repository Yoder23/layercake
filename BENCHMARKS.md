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

## North Star v23 routed-cake verification

```powershell
python scripts/migrate_v22_to_sparse_routed_cake.py
python scripts/benchmark_northstar_training_speed.py --devices cpu,cuda --cpu-threads 1 --cpu-batch-size 16 --gpu-batch-size 128 --raw-sequence-bytes 256 --layercake-mode shared3_routed_tail_int8_foundation --warmup-steps 6 --measured-steps 20 --repeats 3 --output results/breakthrough_equal/northstar_v23_domain_cake_training_speed.json
python scripts/train_byte_core_from_config.py --config configs/northstar_v23_route4_schema_training.json
python scripts/verify_northstar_v23_route_isolation.py
python scripts/run_northstar_release_tests.py --output results/breakthrough_equal/northstar_v23_pytest_summary.json
python scripts/verify_northstar_v23_release.py
```

This measures selected-domain-cake fine-tuning with the foundation and decoder
frozen. It is not the full-core pretraining protocol below.

## North Star v22 release verification

The v22 comparison uses the same corrected training sources, heldout-safe combinations,
and a transformer continuation with greater byte exposure. Reproduce the final validation:

```powershell
python scripts/train_bpe_transformer_from_config.py --config configs/northstar_v22_fair_corrected_bpe.json
python scripts/eval_schema_action_generation.py --questions data/schema_action_domain/eval_questions.json --layercake runs_experiment/northstar_v21_semantic_pointer/latest.pt --layercake-metrics runs_experiment/northstar_v21_semantic_pointer/training_metrics.json --bpe runs_experiment/northstar_v22_fair_corrected_bpe/latest.pt --bpe-metrics runs_experiment/northstar_v22_fair_corrected_bpe/training_metrics.json --device cpu --cpu-threads 1 --repeats 3 --max-new-bytes 128 --layercake-neural-mode patch --stop-after-json --benchmark-mode fair_neural --output results/breakthrough_equal/northstar_v22_schema_patch_cpu.json
python scripts/export_northstar_v22_runtime.py
python scripts/run_northstar_release_tests.py
python scripts/verify_northstar_v22_release.py
```

GPU, compositional, INT8, resource, and transfer commands are listed in
[NORTHSTAR_V22_RELEASE.md](NORTHSTAR_V22_RELEASE.md). The verifier exits nonzero when any
required artifact is missing or any quality, speed, exposure, footprint, regression, or
transfer gate fails.

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
