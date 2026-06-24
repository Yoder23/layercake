# Transformer baselines

Every promoted LayerCake claim must compare against a matched transformer baseline.

Baseline rules:

- same dataset and split;
- same byte budget;
- matched or disclosed parameter count;
- same hardware;
- same seed count at the declared tier;
- same evaluation script;
- same wall-clock accounting;
- cache state disclosed for generation.

Current implemented baselines:

- `TinyByteTransformer` for Tier 0/Tier 1 smoke;
- locked 15M BPE transformer artifacts for the north-star CPU/mobile certificate;
- locked receiver BPE transformer artifacts for receiver-frontier verification;
- locked transformer adapter artifacts for Python-domain adaptation comparison.

Tier 2 must use serious matched BPE and byte-transformer baselines with repeated seeds.
