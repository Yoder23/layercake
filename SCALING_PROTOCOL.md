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
- Tier 1 local: 1M-5M params, local CPU/GPU, minutes.
- Tier 2 serious: 5M-25M params, 100M-1B bytes, multiple seeds.
- Tier 3 research: 25M-150M params, multi-GPU if available.
- Tier 4 moonshot: 150M+, large byte corpus, multiple domains, multiple seeds.

Promotion requires both source/core and receiver-after-transfer certificates.

Current Tier 0/Tier 1 smoke command:

```powershell
python scripts/benchmark_tier1_dominance.py --steps 4
python scripts/verify_tier1_dominance.py
```

The smoke certificate is useful for methodology regressions only. Tier 1 local must move
from the tiny fixed file to 1M-5M parameters, larger held-out byte streams, and repeated
seeds before any public efficiency claim is upgraded.
