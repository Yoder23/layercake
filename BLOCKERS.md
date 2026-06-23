# Experimental Blockers

The software acceptance path and selected small-scale research gates pass. The following
broader scientific claims remain blocked by resources or evidence:

- scaling results beyond sub-million-parameter cores require substantially larger GPU
  training budgets and multi-seed runs;
- fp8/native-int8 kernel performance requires target deployment hardware and kernels;
- direct tokenized-ABI to byte-patch-ABI transfer still requires a paired tokenized core;
- L7 orchestration quality needs CorticalSwarm transport and task-level evaluations.
- Any claim of beating tokenizer transformers needs matched-data, matched-compute training
  at useful scale. Current evidence is parity/slight BPB superiority on one small local run.

The exact continuation controls are in `NEXT_STEPS.md`.
