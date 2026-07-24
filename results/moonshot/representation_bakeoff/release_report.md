# LayerCake representation bake-off release report

Status: **CONTINUATION REQUIRED**

No representation is promoted. Phase 2 remains open and Phase 3 remains locked.

| Candidate | Validation BPB | 128-byte CPU ratio | 1024-byte CPU ratio | Core adherence | Topic recall | Process RSS | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5 0.5B CPU | n/a | 1.00x | 1.00x | 0.60 | 0.87 | 214990848 | locked comparator |
| Byte sliding-cosine | 1.6917 | 2.52x | n/a | failed | failed | 364808192 | reject |
| Byte pointer gate | 1.6985 | 1.83x | 1.70x | 0.00 | 0.05 | 366436352 | reject |
| Shared tokenizer | 1.6619 | 2.41x | 2.01x | 0.00 | 0.225 | 393781248 | reject |
| Hybrid token-byte | 1.6348 | 2.15x | 1.91x | 0.00 | 0.040 | 392294400 | reject |

The byte pointer run and every negative checkpoint are preserved. No nearby
byte, gate, pointer, auxiliary-loss, hidden-size, or decoder variant was
launched after the authorized pointer run.

The next campaign must test the falsifiable multi-slot prompt-state diagnosis;
it may not restart any closed representation search under a new name.
