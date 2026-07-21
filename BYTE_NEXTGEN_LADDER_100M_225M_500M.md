# Byte NextGen Ladder: 100M -> 225M -> 500M

This plan locks in fast iteration first, then scale only when quality gates require it.

## Core principle

- Keep core English-only and reusable.
- Add game knowledge as a portable domain layer.
- Promote model size only if quality gates fail.

## Stage 1: 100M core (primary target)

Configs:
- configs/byte_100m_core_phaseA_curriculum.json
- configs/byte_100m_core_phaseB_redpajama.json

Run:

```powershell
c:\Python310\python.exe scripts/train_byte_core_from_config.py --config configs/byte_100m_core_phaseA_curriculum.json
c:\Python310\python.exe scripts/train_byte_core_from_config.py --config configs/byte_100m_core_phaseB_redpajama.json
```

Fast quality check (core-only):

```powershell
c:\Python310\python.exe scripts/eval_ember_road_generation.py --core-checkpoint runs_experiment/byte_100m_core_phaseB_redpajama/latest.pt --out results/eval_100m_core_only.json --device cuda
```

Promote to game layer only if core passes fluency gate.

## Stage 1b: game layer on top of 100M core

Prepare game training jsonl if needed:

```powershell
c:\Python310\python.exe scripts/prepare_game_domain_data.py --game-dir data/game_domains --output data/game_domains/ember_road_training.jsonl
```

Train domain layer with ABI 256 (compatible with 100M and 225M):

```powershell
c:\Python310\python.exe scripts/train_portable_domain_decoder.py --architecture byte_gru --domain-id ember_road --domain-file data/game_domains/ember_road_training.jsonl --d-abi 256 --hidden 512 --embedding-width 64 --seq 256 --batch 32 --steps 6000 --artifact runs_experiment/portable_ember_road_gru_abi256.pt --output results/portable_ember_road_gru_abi256.json
```

Evaluate combined behavior:

```powershell
c:\Python310\python.exe scripts/eval_ember_road_generation.py --core-checkpoint runs_experiment/byte_100m_core_phaseB_redpajama/latest.pt --domain-artifact runs_experiment/portable_ember_road_gru_abi256.pt --out results/eval_100m_plus_domain.json --device cuda
```

## Stage 2 fallback: 225M

Use existing fast configs if 100M fails gates:
- configs/byte_225m_core_phaseA_curriculum.json
- configs/byte_225m_core_phaseB_polish.json

Reuse the same ABI-256 domain artifact at this stage if compatible behavior is acceptable.

## Stage 3 fallback: 500M

Only escalate if 225M still fails acceptance gates.
Use:
- configs/byte_500m_core_phase1_fluency.json
- configs/byte_500m_core_phase2_gameblend.json (optional, if you intentionally want game mixed into core)

## Quality gates (must pass twice in a row)

Core fluency gate:
- coherent, on-topic responses for companion prompts
- low repetition in generated text
- stable response formatting and actionability

Companion + game gate:
- clear tactical suggestions
- correct rulebook-grounded answers
- no JSON fragment leakage in domain responses

## Size promotion policy

- Start and aim to finish at 100M.
- Move to 225M only if 100M fails after curriculum + domain-layer tuning.
- Move to 500M only if 225M still fails.

This keeps iteration fast while preserving a strong scale-up path.
