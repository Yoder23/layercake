# Byte-Patch LayerCake

The v2 path is:

`UTF-8 bytes -> byte embeddings -> causal patches -> patch core -> canonical ABI -> sparse brick -> byte logits`

The ABI sits above perception. Bricks therefore bind to an ABI version and coordinate
contract, not tokenizer IDs.

Implemented now:

- exact UTF-8 byte encode/decode;
- fixed and whitespace patchers with stable boundary metadata;
- patch compression-ratio tracking;
- byte-patch model smoke forward pass;
- byte reconstruction logits expanded from patch states;
- ABI-compatible low-rank and sparse low-rank bricks.

Not implemented or proven at useful scale:

- trained entropy/difficulty patching;
- 25M+ tokenizer-independent transfer;
- superiority over BLT or tokenizer-based transformers at scale.

The selected sub-million-parameter fixed-patch experiment passes bounded byte/byte-patch
transfer gates and reaches a 2.4165 BPB point estimate versus 2.4243 for a trained
byte-fallback BPE baseline. This is local parity evidence, not a scale claim.

BLT-style systems target tokenizer removal and dynamic byte compute. LayerCake targets
portable ABI-space domain operators. The current experiment shows those ideas can compose
at small scale; larger matched-compute replication remains necessary.
