# Preview-guided training

Rubric Preview is a non-destructive pass before a rolling training stage. It inspects the
data slice and, when available, the current model. The output is a machine-readable
artifact under `results/previews/<preview_id>.json`.

The preview contains byte entropy, histogram summary, rare byte rate, Unicode rate,
code-symbol rate, sequence lengths, fixed byte-patch compression, current model BPB/loss,
optional transformer baseline BPB, ABI activation summaries, estimated steps/cost,
difficulty buckets, recommended trainable/frozen modules, loss weights, gates, and
warnings.

The syllabus compiler turns a rubric plus preview into `results/syllabi/<id>.json`.
Implemented curriculum modes are:

- `easy_to_hard`;
- `entropy_balanced`;
- `rehearsal_interleaved`;
- `hard_to_easy` for benchmarks.

Smoke commands:

```powershell
python scripts/preview_rubric.py rubrics/07_preview_guided_smoke.yaml
python scripts/demo_preview_guided_layercake_training.py --smoke
python scripts/benchmark_preview_guided_training.py
python scripts/benchmark_curriculum_modes.py
```

The current smoke result proves the control loop and tiny CPU LayerCake integration. It
does not prove transformer dominance at scale.
