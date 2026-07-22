# Moonshot architecture

Status: V2 development implementation; the complete moonshot is not yet proven.

The production research path is now:

```text
UTF-8 bytes
    | local causal convolution state
    +-> learned 4-byte summaries -> fast recurrent state --+
    +-> learned 16-byte summaries -> slow recurrent state -+-> shared trunk
                                                            |
                                           top-1 routed expert (16 installed)
                                                            |
                                  English byte logits + canonical ABI/2
                                                            |
                                 portable_fusion residual cake
                                                            |
                                             final byte distribution
```

`prefill()` consumes the prompt once. `decode_step()` subsequently updates only the
local convolution window, incomplete 4/16-byte patches, recurrent summaries, selected
expert, canonical state, optional cake state, and sampler accounting. It does not retain
or recompute the prompt. Serialized state uses safetensors plus inert JSON metadata and
is bound to a fingerprint of the model weights.

The 64-coordinate `lc-canonical-byte-semantics/2` contract is semantic rather than only
shape-compatible. Coordinates 0-31 encode a host prediction projected through a fixed
byte-semantic codebook. Coordinates 32-63 encode a deterministic causal byte anchor.
The anchor is exact across supported seeds and host sizes; the prediction block is
tolerance-based and is allowed to reflect host quality. The combination rule is a
bounded additive byte-logit residual.

`portable_fusion` is a 44,705-parameter recurrent residual module. It consumes the host
logits, canonical state, and causal raw bytes, and relies on the host for general language.
It is 1.27% of the source core's installed parameters and 7.99% of its active parameters.
The legacy `portable_decoder` and host-conditioned residual classes remain available,
but they are not used as evidence for V2 core-conditioned portability.

## Legacy V1 foundation path

`LayerCakeFoundation` consumes UTF-8 bytes directly. A left-padded depthwise local block
computes byte-rate features. Fixed causal patches are projected and processed at patch
rate by a GRU. A completed patch is shifted by one before it is expanded back to bytes,
so no byte in a patch can leak into an earlier prediction. The global representation
crosses a fixed-width ABI before the local and global paths are combined.

Both V1 and V2 route hard top-1 per row or use an explicitly pinned homogeneous
microbatch. Only selected experts execute or receive gradients. V1's
`SparseOptimizerFactory` can construct an optimizer over one pinned expert. V2 registers
all experts with AdamW so routes may change, but PyTorch allocates moment tensors lazily;
only experts that receive gradients gain optimizer state. Installed inactive weights
still occupy checkpoint and resident model memory, so native cache/memory-traffic proof
remains an open gate.

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
- `portable_fusion` consumes host logits, the semantic ABI, and raw causal bytes. It is
  the V2 production candidate; package identity is exact while functional behavior uses
  a predeclared tolerance/task criterion.

## Runtime and orchestration

The retained lexical router supplies the fast heuristic path. The V2 compact learned
router handles paraphrases, top-k/multidomain/no-domain/missing-cake prompts, quoted
keywords, and prompt-control attacks. Trust and permission filtering occurs before its
domain-to-package mapping. The orchestrator maintains installed-versus-loaded state,
prefetches, loads under a memory budget, evicts least-recently-used cakes, traces
decisions, and can escalate low-confidence outputs to a verifier.

CPU, CUDA, and TorchScript export paths share normalized generated-byte metrics. Physical
mobile claims remain `NOT_RUN_NO_HARDWARE` until measured on ARM devices.
