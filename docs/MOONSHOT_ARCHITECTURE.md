# Moonshot architecture

Status: executable research implementation; the complete moonshot is not yet proven.

## Foundation path

`LayerCakeFoundation` consumes UTF-8 bytes directly. A left-padded depthwise local block
computes byte-rate features. Fixed causal patches are projected and processed at patch
rate by a GRU. A completed patch is shifted by one before it is expanded back to bytes,
so no byte in a patch can leak into an earlier prediction. The global representation
crosses a fixed-width ABI before the local and global paths are combined.

The default foundation bank contains 16 SwiGLU cakes. Routing is hard top-1 per row or
explicitly pinned per domain-homogeneous microbatch. Only selected experts execute or
receive gradients. `SparseOptimizerFactory` constructs AdamW over shared parameters and
one pinned expert, so inactive experts allocate no optimizer state. The default parameter
report measures an active fraction between 10% and 20%.

```text
bytes -> causal local byte mixer ------------------------+
    \-> causal patch summaries -> GRU -> top-1 cake -> ABI -> shifted patch context
                                                         |
                                      local + global -> byte logits
```

## Baseline and controls

The paired baseline learns byte-pair merges from the training split only and uses
pre-normalized fused causal attention, SwiGLU feed-forward layers, tied embeddings, and
AdamW. A deterministic architecture search matches total parameters within 5%. Every
step uses the same raw byte chunk, execution order is randomized, and tokenizer building,
tokenization, evaluation, and optimizer work are recorded separately and in the complete
wall path.

## Cake classes

- `portable_decoder` consumes only deterministic causal byte anchors. The host core is
  absent from its declared exact path, so the same verified FP32 CPU payload can produce
  bit-identical outputs across compatible hosts.
- `host_residual` consumes a host ABI state. Its tensor weights are portable, but its
  meaning is host-conditioned; shape compatibility is never treated as semantic proof.

## Runtime and orchestration

The router scores installed declarative metadata, applies trust and permission policy,
penalizes cold loads, supports top-k/forced/multidomain routes, and abstains on prompt
control phrases. The orchestrator maintains installed-versus-loaded state, prefetches,
loads under a memory budget, evicts least-recently-used cakes, traces decisions, and can
escalate low-confidence outputs to a verifier.

CPU, CUDA, and TorchScript export paths share normalized generated-byte metrics. Physical
mobile claims remain `NOT_RUN_NO_HARDWARE` until measured on ARM devices.
