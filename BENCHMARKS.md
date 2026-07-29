# Benchmark and Reproduction Protocol

The authoritative benchmark protocol is the sealed campaign, not the older
smoke or North Star scripts. The final Phase 8 verifier recomputes all promoted
metrics from raw evidence and fails closed on missing, stale, mismatched, or
mixed-lineage artifacts.

## Release commands

```powershell
C:\Python310\python.exe -m layercake.moonshot_campaign verify-sealed 8
C:\Python310\python.exe -m layercake.moonshot_campaign verify-all
C:\Python310\python.exe -m layercake.moonshot_final verify
```

The release is valid only when the final tag is annotated, the worktree is
clean, `verify-sealed 8` returns `SEALED`, and `verify-all` reports
`completed_phases_valid: true`.

## Final performance protocol

The Phase 8 clean room evaluates the same sealed integrated LayerCake lineage
against the declared Qwen 2.5 0.5B runtime on the declared laptop CPU and RTX
3080 Laptop GPU. Cross-model throughput is completed-request UTF-8 output bytes
per wall second. Transformer token counts are retained as authoritative runtime
metadata, but token units are not treated as LayerCake work units.

The promoted matrix contains:

- 100 distinct frozen functional prompts;
- 20 repeated observations for each headline configuration;
- paired prompt-level comparisons and bootstrap confidence intervals;
- CPU-to-CPU, GPU-to-GPU, and LayerCake-CPU-to-transformer-GPU comparisons;
- one real cold streaming request after an unload control, including load,
  time-to-first-output, and complete-request timing;
- matched quality, repetition, coherence, grounding, memory, and selected-only
  execution checks from the same product lineage.

The final derived performance values are 9.91x CPU output-byte throughput,
8.25x GPU output-byte throughput, and 7.67x LayerCake-CPU-to-transformer-GPU
output-byte throughput. See
`results/moonshot/phase8/raw_runs/reproduction_performance.json` for all raw
observations and `results/moonshot/phase8/release_report.md` for the summary.

## Functional, portability, and routing protocol

The final verifier also reruns:

- 384 held-out domain cases on CPU and on GPU;
- 100 core-only abstention cases;
- install, verify, remove, and reinstall across three fresh hosts;
- 1,980 routing rows across core-only, top-1, top-k, manual, structured
  multidomain, and abstention modes;
- a 500-entry catalog stress test that distinguishes management descriptors
  from promoted neural packages;
- 32 hostile attacks across 24 categories.

## Historical benchmark scripts

Older smoke, tokenizer, byte-patch, North Star, and V2 commands remain useful
for regression research. Their numbers are historical controls unless imported
and recomputed by a current phase verifier. They must not be cited as replacing
the Phase 8 result or as extending its hardware, task, or model scope.
