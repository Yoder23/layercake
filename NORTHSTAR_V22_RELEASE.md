# CPU/GPU North Star v22 release

North Star v22 is a bounded, reproducible same-size CPU/GPU generation architecture result. It does
not claim that every transformer workload or every open-domain benchmark has been solved, and it
does not claim faster full-core training.

## Locked result

- LayerCake parameters: 15,190,449.
- BPE transformer parameters: 14,950,848.
- LayerCake/transformer parameter ratio: 1.0160.
- LayerCake general BPB: 1.9088; transformer: 2.7149.
- LayerCake schema and compositional held-out exactness: 100%.
- Transformer schema/compositional held-out exactness: 87.5% / 60%.
- Dense CPU/GPU speed ratios: 19.51x through 24.58x.
- Pruned Linear-INT8 patch runtime: 8.73 MB state / 9.00 MB TorchScript package.
- INT8 CPU speed ratios: 11.59x isolated-process; 11.81x through 12.57x task splits.
- Exact 15M-to-5M portable-domain transfer on CPU and GPU.
- Full regression: 304 passed, zero failures/errors/skips.
- Full-core 5x training gate: OPEN; 0.722x CPU and 1.045x GPU recipe medians.
- Measured host: 12th Gen Intel Core i9-12900H and NVIDIA GeForce RTX 3080
  Laptop GPU (16 GiB), PyTorch 2.7.1, CUDA 11.8.

The authoritative aggregate is
`results/breakthrough_equal/northstar_v22_release_certificate.json`. It contains SHA-256
hashes for every input evidence artifact.

## Fairness controls

The final transformer is not the earlier underexposed comparator. It strictly resumes its
original 14.95M checkpoint and serialized SentencePiece model, uses an equal-byte weighted
corpus mixer, and receives:

- 159.37M cumulative training bytes versus LayerCake's reported 143.36M;
- at least 111.74M corrected-task bytes versus LayerCake's 73.73M corrected lineage;
- the same corrected relevance and schema/action sources;
- the same general-language held-out evaluation.

Every generation artifact uses `benchmark_mode=fair_neural`. Structured schema heads and
direct domain caches are disabled. LayerCake emits each 80-byte candidate with one neural
generation call. Both models stop after a complete JSON object for answer latency.

## Evidence files

- `northstar_v22_schema_patch_cpu.json`
- `northstar_v22_schema_patch_cuda_graph_gpu.json`
- `northstar_v22_relevance_patch_cpu.json`
- `northstar_v22_relevance_patch_cuda_graph_gpu.json`
- `northstar_v22_schema_patch_dynamic_int8_cpu.json`
- `northstar_v22_relevance_patch_dynamic_int8_cpu.json`
- `northstar_v22_deployment_resources_dynamic_int8_cpu.json`
- `northstar_v22_lossless_transfer_15m_to_5m_cpu.json`
- `northstar_v22_lossless_transfer_15m_to_5m_gpu.json`
- `northstar_v22_pytest_summary.json`
- `northstar_v22_release_certificate.json`
- `northstar_v22_training_speed_recipe.json`
- `northstar_v22_training_speed_favorable_lower_bound.json`
- `northstar_v22_training_audit.json`

All JSON files are under `results/breakthrough_equal/`.

## Reproduction

Install the package and benchmark dependencies:

```powershell
python -m pip install -e ".[dev,tokenizer]"
```

### Included versus external inputs

The Git repository includes the locked JSON evidence, the 9.00 MB TorchScript task
runtime, all corrected task corpora, configs, evaluation code, and tests. It deliberately
does not put large PyTorch checkpoints or the 2.4 GB general corpus in Git.

Running the packaged demo and verifying the committed certificate are self-contained.
Re-exporting or re-running evaluation additionally requires these release assets at the
paths used below:

- `runs_experiment/northstar_v21_semantic_pointer/latest.pt` and its
  `training_metrics.json`;
- `runs_experiment/northstar_v22_fair_corrected_bpe/latest.pt` and its
  `training_metrics.json`;
- the three portable-transfer artifacts named in the transfer command.

Continuing the transformer training recipe also requires
`runs_experiment/northstar_v6_patch4_grounded_mix_bpe/latest.pt`. The general corpus is an
external RedPajama-derived JSONL input and must be placed at the path declared in the
configs (or the config path must be changed):

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| `redpajama_english_train.jsonl` | 2,401,369,486 | `38d7c52e9a41f0d674134fa18d0594bd37782934ab01d4a9b5df0ff9c77462d5` |
| `redpajama_english_eval.jsonl` | 24,060,082 | `8b5d1f16571758c45a95967bc1c8e762045cf4fc0a562b94af9bd75b0953f6c4` |

Those hashes identify the exact local corpus used; they do not grant redistribution
rights. Obtain compatible source data under its own license.

Train the strengthened transformer comparator:

```powershell
python scripts/train_bpe_transformer_from_config.py `
  --config configs/northstar_v22_fair_corrected_bpe.json
```

CPU schema benchmark:

```powershell
python scripts/eval_schema_action_generation.py `
  --questions data/schema_action_domain/eval_questions.json `
  --layercake runs_experiment/northstar_v21_semantic_pointer/latest.pt `
  --layercake-metrics runs_experiment/northstar_v21_semantic_pointer/training_metrics.json `
  --bpe runs_experiment/northstar_v22_fair_corrected_bpe/latest.pt `
  --bpe-metrics runs_experiment/northstar_v22_fair_corrected_bpe/training_metrics.json `
  --device cpu --cpu-threads 1 --repeats 3 --max-new-bytes 128 `
  --layercake-neural-mode patch --stop-after-json `
  --benchmark-mode fair_neural `
  --output results/breakthrough_equal/northstar_v22_schema_patch_cpu.json
```

Use `--device cuda` for GPU and add `--dynamic-int8` for the symmetric CPU INT8 run.
Swap the questions file to `data/question_relevance/eval_questions.json` for the
compositional suite.

Transfer verification:

```powershell
python scripts/eval_lossless_domain_decoder.py `
  --decoder runs_experiment/portable_python_gru148k_seed6061_int8.pt `
  --source-core runs_experiment/scale15m_transition_lw280_2300_noprofile.pt `
  --target-core runs_experiment/scale5m_seed4242.pt `
  --eval-bytes 20000 --eval-file tests/fixtures/technical_text_smoke.txt `
  --batches 16 --generation-bytes 64 --device cpu `
  --output results/breakthrough_equal/northstar_v22_lossless_transfer_15m_to_5m_cpu.json
```

Package, test, and certify:

```powershell
python scripts/export_northstar_v22_runtime.py
python scripts/benchmark_northstar_deployment_resources.py --dynamic-int8 `
  --output results/breakthrough_equal/northstar_v22_deployment_resources_dynamic_int8_cpu.json
python scripts/run_northstar_release_tests.py
python scripts/verify_northstar_v22_release.py
python scripts/verify_northstar_training_audit.py
```

Run the included task runtime:

```powershell
python scripts/run_northstar_v22_runtime.py `
  "Question: A user says move the Login button to the top left of the app. What edit action should be taken? Answer: "
```

## CUDA setup accounting

The CUDA patch head uses a captured graph. Its one-time setup took 18.81-18.84 seconds and
is stored in both GPU artifacts. It is excluded from steady-state answer timing and is not
hidden. The deployment is advantageous for repeated requests; cold-start-sensitive systems
must budget or prewarm this setup.

## Boundaries

- Full-core training dominance is not established. The independently validated training
  measurement is valid, but its 5x performance gate is OPEN. See `TRAINING_NORTHSTAR.md`.
- No Android/iOS ARM phone, NPU, battery, or thermal measurement has been performed.
- The separate phone publication gate is fail-closed: it validates raw same-device
  LayerCake/baseline samples, artifact hashes, quality, memory, sustained speed, battery,
  and thermal data. A self-declared hardware boolean cannot pass it.
- The packaged INT8 runtime contains only the global autoregressive patch path used by the
  locked tasks. It excludes the full general byte-LM local decoder.
- General-language evidence is held-out BPB, not a comprehensive knowledge/reasoning suite.
- The exact transfer contract is a core-independent portable domain decoder. It does not
  imply arbitrary weights can be pasted losslessly between unrelated neural networks.
