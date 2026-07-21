# Ember Road Byte-Domain Workflow

This workflow uses the tokenizer-free byte-patch LayerCake path only.

## 1) Train the 500M byte core

Run from layercake_release:

```powershell
c:\Python310\python.exe scripts/train_byte_core_from_config.py --config configs/byte_500m_core.json
```

Selected model shape in configs/byte_500m_core.json is:
- patch_size=2
- d_model=1280
- d_abi=320
- layers=24
- heads=16
- local window-transformer decoder

This is approximately 502.2M trainable parameters in CausalBytePatchLM.

## 2) Convert Ember Road JSON into domain training JSONL

The source rulebook is already included at:
- data/game_domains/ember-road.rulebook.json

Prepare domain data:

```powershell
c:\Python310\python.exe scripts/prepare_game_domain_data.py --game-dir data/game_domains --output data/game_domains/ember_road_training.jsonl
```

## 3) Train a portable domain layer from the rulebook

Train the domain artifact (byte-level, tokenizer-free):

```powershell
c:\Python310\python.exe scripts/train_portable_domain_decoder.py --architecture byte_gru --domain-id ember_road --domain-file data/game_domains/ember_road_training.jsonl --d-abi 320 --hidden 512 --embedding-width 96 --seq 256 --batch 32 --steps 6000 --artifact runs_experiment/portable_ember_road_gru.pt --output results/portable_ember_road_gru.json
```

This creates a portable domain layer artifact that can be installed on compatible byte cores.

## 4) Optional migration check (source vs receiver)

```powershell
c:\Python310\python.exe scripts/eval_lossless_domain_decoder.py --decoder runs_experiment/portable_ember_road_gru.pt --source-core runs_experiment/byte_500m_core/latest.pt --target-core runs_experiment/byte_500m_core/latest.pt --eval-file data/game_domains/ember_road_training.jsonl --eval-source-label ember-road --generation-bytes 64 --device cuda --output results/ember_road_transfer_check.json
```

## 5) Required long-run path for fluency + game knowledge

Use the staged configs below. They are designed to avoid short-run undertraining and to force persistent game-domain exposure.

Phase 1 (fluency first):

```powershell
c:\Python310\python.exe scripts/train_byte_core_from_config.py --config configs/byte_500m_core_phase1_fluency.json
```

Phase 2 (game blend continuation):

```powershell
c:\Python310\python.exe scripts/train_byte_core_from_config.py --config configs/byte_500m_core_phase2_gameblend.json
```

Current throughput probe on this machine (500M, seq=2048, micro=2, accum=16) is about 0.103 steps/s.

Projected wall-clock at this throughput:
- Phase 1 target (200000 steps): ~22.47 days, ~12.21 GiB byte exposure
- Full target after Phase 2 (260000 total steps): ~29.22 days, ~15.87 GiB byte exposure
- Phase 2 increment only (200000 -> 260000): ~6.75 days, ~3.66 GiB additional exposure

Storage controls in the staged configs:
- `keep_last_n=1` and `save_optimizer=false` to reduce checkpoint growth
- `save_interval=2000` to reduce checkpoint write frequency

Quality gate policy (recommended):
- Re-run generation evaluation every 10000 steps during phase 1 and every 5000 steps during phase 2
- Require improvements in alpha ratio, unique-word ratio, and repetition before declaring completion
- Keep the portable Ember domain layer attached for game-specific prompting checks

## Notes

- This workflow never uses SentencePiece or token IDs.
- Domain knowledge is trained as a portable ABI-space artifact, not hardcoded into tokenizer-dependent layers.
- For a distinct receiver core, replace --target-core with a different byte-core checkpoint that shares ABI compatibility.
