# Phase 2 grounding diagnosis

Checkpoint: `8b15cc67295e60d8a77f080f97ce5518f117a0307daa1b3f9b4c1f6ee770d269`

## Findings

- Autonomous topic-phrase retention across diagnostic prompts: 0.000.
- Exact supplied-entity retention: 0.000.
- Persistent prompt tensors stayed byte-identical at every measured horizon: True.
- Mean Jensen-Shannon contribution of directly ablating both fixed prompt context and copy bias: 0.01304088 nats.
- Fresh-process peak RSS: 393580544 bytes.
- PyTorch/native runtime component: 317644800 bytes.
- Model/tokenizer component: 45301760 bytes.

## Falsifiable diagnosis

The fast control does not lose or overwrite its encode-once prompt vector: the
cached condition remains exactly stable while KV state grows incrementally.
Instead, it compresses entities, topic, instruction, style, and format into one
fused 288-value vector plus a static bag-of-prompt-token bias. Those features
are not independently addressable, and direct prompt ablation has only a small
effect on next-unit probabilities after local generation begins. Autonomous
generation is therefore dominated by local language dynamics.

The absolute RSS failure is separate: the measured PyTorch/native runtime alone
exceeds the 214,990,848-byte final gate. A passing lineage requires a minimal
native runtime; model-only quantization cannot make the Python reference
process pass.

## Architecture consequence

Do not repeat slot-count, pointer-width, or gate-weight sweeps. The next
authorized hypothesis must expose independently recoverable instruction,
entity, constraint, format, and content records through an encode-once bounded
memory with an explicitly supervised neural read. Its autonomous outputs—not
its auxiliary loss—must clear continuation gates before native-runtime work is
promoted.
