# Scaling protocol

Every serious LayerCake scaling run must compare:

- LayerCake blind;
- LayerCake preview-guided;
- matched byte transformer;
- matched BPE transformer where available;
- LayerCake with rollback enabled;
- LayerCake without rollback.

Metrics:

- BPB/loss;
- time-to-BPB;
- training seconds;
- bytes/sec;
- parameters;
- trainable parameters;
- memory;
- CPU generation;
- GPU generation if available;
- domain adaptation cost;
- rollback recovery;
- ABI drift;
- patch compression;
- transfer result.

Tiers:

- Tier 0 smoke: tiny model, tiny data, CPU, CI-compatible.
- Tier 1 local: 1M-25M params, local CPU/GPU, minutes; methodology evidence only.
- Tier 2 serious: 5M-25M params, 100M-1B bytes, multiple seeds.
- Tier 3 research: 25M-150M params, multi-GPU if available.
- Tier 4 moonshot: 150M+, large byte corpus, multiple domains, multiple seeds.

Promotion requires both source/core and receiver-after-transfer certificates.

Current Tier 0/Tier 1 smoke command:

```powershell
python scripts/benchmark_tier1_dominance.py --steps 4
python scripts/verify_tier1_dominance.py
```

The smoke certificate is useful for methodology regressions only. Tier 1 local has now
advanced through 25M-class parameter counts on the tiny fixed file. It must still move to
larger held-out byte streams and repeated seeds before any public efficiency claim is
upgraded.

Tier 1 local validation commands:

```powershell
python scripts/benchmark_tier1_dominance.py --steps 4 --d-model 64 --layers 2 --heads 2 --d-byte 16 --d-abi 32 --max-patches 256 --output results/dominance/tier1_local_276k_probe.json
python scripts/benchmark_tier1_dominance.py --steps 3 --d-model 96 --layers 2 --heads 2 --d-byte 16 --d-abi 32 --max-patches 256 --output results/dominance/tier1_local_474k_probe.json
python scripts/benchmark_tier1_dominance.py --steps 2 --d-model 128 --layers 2 --heads 2 --d-byte 16 --d-abi 32 --max-patches 256 --output results/dominance/tier1_local_735k_probe.json
python scripts/benchmark_tier1_dominance.py --steps 2 --d-model 160 --layers 2 --heads 2 --d-byte 16 --d-abi 32 --max-patches 256 --output results/dominance/tier1_local_1m_probe.json
python scripts/benchmark_tier1_dominance.py --steps 1 --d-model 256 --layers 2 --heads 4 --d-byte 16 --d-abi 64 --max-patches 256 --output results/dominance/tier1_local_27m_probe.json
python scripts/benchmark_tier1_dominance.py --steps 1 --d-model 384 --layers 2 --heads 8 --d-byte 24 --d-abi 64 --max-patches 256 --output results/dominance/tier1_local_58m_probe.json
python scripts/benchmark_tier1_dominance.py --steps 1 --d-model 416 --layers 3 --heads 8 --d-byte 24 --d-abi 64 --max-patches 256 --output results/dominance/tier1_local_9m_probe.json
python scripts/benchmark_tier1_dominance.py --steps 1 --d-model 448 --layers 3 --heads 8 --d-byte 24 --d-abi 64 --max-patches 256 --output results/dominance/tier1_local_10m_probe.json
python scripts/benchmark_tier1_dominance.py --steps 1 --d-model 448 --layers 4 --heads 8 --d-byte 24 --d-abi 64 --max-patches 256 --output results/dominance/tier1_local_128m_probe.json
python scripts/benchmark_tier1_dominance.py --steps 1 --d-model 512 --layers 5 --heads 8 --d-byte 24 --d-abi 64 --max-patches 256 --output results/dominance/tier1_local_20m_probe.json
python scripts/benchmark_tier1_dominance.py --steps 1 --d-model 512 --layers 7 --heads 8 --d-byte 24 --d-abi 64 --max-patches 256 --output results/dominance/tier1_local_25m_probe.json
python scripts/verify_tier1_local_frontier.py
python scripts/verify_source_receiver_dominance.py
python scripts/verify_scale15m_transition_frontier.py
```

Tier 2 preparation commands:

```powershell
python scripts/replicate_northstar_15m.py
python scripts/verify_northstar_15m_replication.py
python scripts/replicate_receiver_frontier.py
python scripts/verify_receiver_frontier.py
python scripts/benchmark_domain_adaptation_dominance.py
python scripts/verify_domain_adaptation_dominance.py
```

These Tier 2 commands currently verify locked artifacts. Full Tier 2 retraining requires
new 3-seed source/core and receiver runs under the same verifier contracts.

Current promoted full-corpus transition-head command:

```powershell
python scripts/train_patch_only.py --steps 2300 --seed 6250 --seq 256 --batch 16 --general-bytes 8000000 --patch-size 2 --d-byte 32 --d-model 448 --d-abi 128 --layers 4 --heads 7 --local-decoder window_transformer --local-layers 4 --local-width 280 --local-window 32 --qk-norm --modern-blocks --fused-attention --patch-prediction --patch-prediction-weight 0.5 --lr 0.0006 --lr-schedule late_cosine --warmup-steps 100 --empirical-transition-head --artifact runs_experiment/scale15m_transition_lw280_2300_noprofile.pt --output results/scale15m_transition_lw280_2300_noprofile.json
python scripts/benchmark_generation.py --layercake runs_experiment/scale15m_transition_lw280_2300_noprofile.pt --bpe runs_experiment/scale15m_bpe_matched.pt --new-bytes 96 --layercake-mode stateful_cached --device cpu --cpu-threads 1 --no-repeat-ngram 4 --output results/scale15m_transition_lw280_2300_generation_cpu1_norepeat4.json
python scripts/eval_lossless_domain_decoder.py --decoder runs_experiment/portable_python_gru148k_v1.pt --source-core runs_experiment/scale15m_transition_lw280_2300_noprofile.pt --target-core runs_experiment/scale5m_seed4242_continued.pt --generation-bytes 64 --output results/lossless_domain_transition15m_2300_to_5m.json
python scripts/verify_scale15m_transition_frontier.py
python scripts/verify_transformer_dominance_matrix.py
python scripts/benchmark_cpu_deployment_resources.py
python scripts/verify_game_ready_mobile_llm.py
python scripts/verify_cross_backend_quality_scorecard.py
python scripts/verify_many_domain_game_layers.py
python scripts/train_portable_domain_decoder.py --domain-id game_dialogue_smoke_gru --architecture byte_gru --hidden 128 --embedding-width 32 --domain-file tests\fixtures\game_dialogue_smoke.txt --seq 32 --batch 4 --steps 200 --lr 0.003 --domain-bytes 20000 --eval-batches 2 --artifact runs_experiment\game_dialogue_smoke_gru.pt --output results\game_dialogue_smoke_gru_train.json
python scripts/quantize_portable_domain.py --input runs_experiment\game_dialogue_smoke_gru.pt --artifact runs_experiment\game_dialogue_smoke_gru_int8.pt --output results\game_dialogue_smoke_gru_int8.json
python scripts/eval_lossless_domain_decoder.py --decoder runs_experiment\game_dialogue_smoke_gru_int8.pt --source-core runs_experiment\scale15m_transition_lw280_2300_noprofile.pt --target-core runs_experiment\scale5m_seed4242_continued.pt --eval-file tests\fixtures\game_dialogue_smoke.txt --eval-bytes 512 --batches 1 --generation-bytes 32 --output results\game_dialogue_smoke_gru_lossless_transfer.json
python scripts/verify_game_domain_training_workflow.py
python scripts/verify_cross_domain_smoke_frontier.py
python scripts/verify_cross_domain_adapter_frontier.py
python scripts/verify_frontier_model_northstar.py
```

25M/20M rematch template:

```powershell
python scripts/train_patch_only.py --steps 750 --seed 6250 --seq 256 --batch 24 --general-bytes 8000000 --patch-size 2 --d-byte 32 --d-model 448 --d-abi 128 --layers 4 --heads 7 --local-decoder window_transformer --local-layers 4 --local-window 16 --modern-blocks --fused-attention --empirical-transition-head --artifact runs_experiment/scale20m_transition_probe.pt --output results/scale20m_transition_probe.json
python scripts/benchmark_bpe_baseline.py --steps 750 --seed 6250 --seq 128 --batch 24 --d-model 448 --layers 7 --heads 7 --general-bytes 8000000 --artifact runs_experiment/scale20m_bpe_transition_probe.pt --output results/scale20m_bpe_transition_probe.json
```

Promotion requires a verifier that checks quality, training time, CPU generation,
printable/coherent generation, parameter count, and receiver-after-transfer behavior.
