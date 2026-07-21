# Fast 225M Curriculum Plan (Days-Scale)

This plan is designed for fast iteration and companion quality in days, not weeks.

## Why this pivot

- 500M core is still valid research, but it is too slow for current product iteration goals.
- A 225M byte model is a better fit for current data volume and faster turnaround.
- Core remains game-clean so it can be reused for other projects.

## Architecture

- Model: ~225M parameters
- Configs:
  - configs/byte_225m_core_phaseA_curriculum.json
  - configs/byte_225m_core_phaseB_polish.json

## Curriculum strategy

- Phase A flashes structured school-style language curriculum and companion dialogue style.
- Phase B expands on English corpus while preserving concise companion behavior.
- Game rulebook is not mixed into this core path.

Curriculum files:
- data/curriculum/english_school_curriculum.txt
- data/curriculum/companion_dialogue_curriculum.txt

## Byte budget

- Phase A bytes per step: 512 * 2 * 8 = 8192 bytes
- Phase A total: 20000 steps = 163,840,000 bytes (~0.15 GiB)

- Phase B bytes per step: 1024 * 2 * 16 = 32768 bytes
- Phase B incremental: 60000 steps = 1,966,080,000 bytes (~1.83 GiB)

- Full plan total (A+B): 2,129,920,000 bytes (~1.98 GiB)

## Run commands

From layercake_release:

```powershell
c:\Python310\python.exe scripts/train_byte_core_from_config.py --config configs/byte_225m_core_phaseA_curriculum.json
```

Then:

```powershell
c:\Python310\python.exe scripts/train_byte_core_from_config.py --config configs/byte_225m_core_phaseB_polish.json
```

## Recommended quality gates

Evaluate every 5000 steps in Phase B and stop early if all conditions pass twice in a row:

1. Conversational fluency
- Average alphabetic ratio >= 0.72
- Max token repeat <= 10
- Responses are coherent and on-topic for all prompts

2. Companion behavior
- Actionable plan appears in each prompt response
- No looping phrases or filler blocks

3. Rulebook readiness policy
- Keep rulebook out of the core
- Attach or train portable domain artifact after core is fluent

## Clean core and game companion split

- Reusable clean core checkpoint:
  - runs_experiment/byte_225m_core_phaseB_polish/latest.pt
- Game behavior should be layered using portable domain decoder artifacts, not baked into core.
