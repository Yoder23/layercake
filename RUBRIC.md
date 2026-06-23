# LayerCake Transfer Rubric

LayerCake reports transfer at separate levels. Passing a lower level never implies a
higher one.

| Level | Contract | Current status |
|---|---|---|
| L0 Weight Lossless | Brick tensors copy exactly; `max_diff = 0`. | Proven by tests. |
| L1 Function Lossless | Equal ABI input gives equal brick output. | Proven by tests. |
| L2 Generation Lossless | Equal core, interface, decoding policy, and pasted brick give identical generated sequences. | Proven for the v1 same-core tokenized path. |
| L3 Cross-Size Lossless | Equal/canonically mapped ABI input preserves brick function across `d_model` sizes. | Structurally proven; end-to-end cross-size PPL is not proven. |
| L4 Cross-Seed Semantic Transfer | Independently trained cores preserve domain behavior within a declared PPL/task bound. | Small-scale bounded PASS with deterministic anchors/canonical head. |
| L5 Quantized Bounded Transfer | A named quantization contract stays within declared degradation bounds. | Small-scale int8 bounded PASS. |
| L6 Tokenizer-Independent Transfer | Tokenized/byte or byte/byte-patch interfaces produce compatible canonical ABI behavior within bounds. | Small-scale byte-to-byte-patch bounded PASS; BPE is evaluated as an external baseline, not an ABI endpoint. |
| L7 Orchestrated Bounded Transfer | A routed LayerCake swarm preserves state/task quality within bounds. | Packet/router stub implemented; scientific claim not proven. |

## Orthogonal exact portable-domain contract

`PX` is deliberately not a replacement for L4-L7. A PX payload owns its byte-prediction
path and does not add a residual to host-core logits.

| Contract | Requirement | Current status |
|---|---|---|
| PX Exact Portable Domain | Verified unchanged artifact produces identical logits, PPL, top-1 byte accuracy, and deterministic generation across independent LayerCake hosts. | PASS through 15.45M -> 5.40M; fp32 and int8-storage artifacts. |

PX proves that domain behavior can be trained once and installed without host retraining.
It does not prove that the receiving host core acquired the payload's knowledge internally,
and it does not establish task-level generation quality.

## Generation/PPL gate

Every semantic transfer run must report source and target domain PPL, general PPL, and
`target_ppl / source_ppl`. The default smoke gate is `<= 1.05`; real experiments must
declare their threshold before training. Exact copying with a failed PPL gate is reported
as `L0/L1 pass, PPL regression`, never as lossless semantic transfer.
