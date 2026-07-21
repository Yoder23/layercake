# Production LayerCake Training Plan

## Overview
Train a fully operational LayerCake model for game integration with fluent English and modular game FAQ learning.

## Strategy: Core + Fluency Layers

Instead of trying to achieve full fluency at 25M, we adopt a **two-tier approach**:

1. **Tier 1: Core Model (250M-500M-1B parameters)**
   - Train on redpajama_english_train.jsonl (2.4GB English corpus)
   - Achieves baseline fluency on general English
   - Serves as the foundation for game-specific augmentation

2. **Tier 2: Fluency Augmentation Layers (Portable Domains)**
   - Additional portable domain layers for specialized English subsets
   - Can be dynamically installed without retraining core
   - Each layer enhances fluency in specific domain (technical, creative, dialogue, etc.)

3. **Tier 3: Game Domain (Portable, Learnable)**
   - Portable domain trained on your game data
   - Preserved for FAQ learning and game interaction
   - Can learn continuously from player interactions via the modular framework
   - Attached to core via exact lossless transfer (proven to work)

## Available Corpora

### Primary Training Data
- **redpajama_english_train.jsonl** (2.4GB)
  - General English, web-scale, diverse
  - Sufficient for 250M-500M training
  - Already tokenized for LayerCake

- **redpajama_english_eval.jsonl** (24MB)
  - Validation set
  - For perplexity tracking during training

### Existing Specialized Data
- **chess_pretraining_tokens.npy** (domain data)
- **hf_cache/** (55 files) - Hugging Face cached models/data
- **raw_hf/** (55 files) - Raw HF downloads
- **moa_pretraining/** (4 files) - MOA domain pretraining

## Training Pipeline

### Phase 1: Scale Selection & Core Training (Days 1-3)
```
OPTION A: 250M Core (faster, prove model works)
  - Smaller, trains faster
  - Still captures general English fluency
  - Ideal for immediate game integration

OPTION B: 500M Core (balanced)
  - 2x parameters, 4-6x training time
  - Better fluency
  - Better transfer to game domains

OPTION C: 1B Core (maximum)
  - Largest, most fluent
  - Longest training (weeks)
  - Overkill if layers augment fluency anyway
```

**Recommendation**: Start with **250M core**, measure fluency, then add layers.

### Phase 2: Core Training
1. Initialize LayerCake model: 250M parameters (patch_size=2, local_window adjustable)
2. Train on redpajama_english_train.jsonl with:
   - AdamW optimizer (lr=1e-4, weight decay=0.01)
   - Mixed precision (AMP) for speed
   - Gradient checkpointing for memory
   - Batch size: scaled to fill GPU memory (~128-256 depending on hardware)
   - Sequence length: 2048 bytes (matched to training infrastructure)
3. Target: 5-10 epochs over corpus (~$100-300 compute depending on scale)
4. Validation: Track perplexity on redpajama_english_eval.jsonl

### Phase 3: Fluency Assessment
- Generate 100 samples (~1KB each) from prompt set
- Evaluate:
  - Perplexity on heldout eval
  - Generation fluency (human spot-check or automated metric)
  - Lack of repetition/hallucination
  - Grammatical correctness

### Phase 4: Fluency Layer Creation (if needed)
If core fluency is insufficient:
- Slice redpajama_english_train by domain (e.g., Wikipedia, news, books, code)
- Train lightweight portable domains (~10-50M parameters each) on subsets
- Layer 1: Technical/formal writing
- Layer 2: Creative/narrative writing
- Layer 3: Dialogue/conversational
- These layers can be mixed at inference time or stacked

### Phase 5: Game Domain Training
1. Format game data as JSONL (one JSON doc per game passage)
2. Train portable domain decoder on game data
3. Store as **portable_game_domain.pt** with payload preservation
4. Register in LayerCakeRuntime so it can be:
   - Installed alongside core
   - Queried for game FAQ
   - Updated with player interaction data over time

### Phase 6: Integration & Deployment
1. Package core + game domain as single model
2. Deploy to game engine:
   - Core handles general English understanding
   - Game domain handles FAQ and game-specific context
   - Both run together with exact modularity guarantees
3. Set up FAQ learning loop:
   - Player asks question → game domain predicts FAQ answer
   - Measure confidence → if low, tag for human review
   - Periodically retrain game domain on accumulated interactions

## Modularity Guarantees (Already Proven)

✅ **Exact Transfer**: Game domain can be trained on core → receiver and transferred losslessly
✅ **1-Domain vs 10-Domain**: Core + game domain behaves identically whether alone or with other domains installed
✅ **Preservation**: Game domain payload never corrupted; FAQ state preserved
✅ **CPU Speed**: 3.38x faster generation than BPE (no tokenizer overhead)

## Scale Recommendations

| Scale | Training Time | Fluency | Game Integration | Recommendation |
|-------|---------------|---------|------------------|-----------------|
| 250M  | 2-3 days      | Good    | Excellent        | **START HERE** |
| 500M  | 5-7 days      | Very Good | Excellent        | If time allows |
| 1B    | 2-3 weeks     | Excellent | Excellent        | Overkill, use layers instead |

## Next Steps

1. **Confirm game data location** — Where are your game files? (JSONL, TXT, CSV?)
2. **Choose core scale** — 250M, 500M, or 1B?
3. **Pick training hardware** — GPU type/count available?
4. **Initiate Phase 1** — Begin 250M core training on redpajama_english_train

Once core is trained, adding fluency layers is **weeks** of work vs. **months** for a larger core alone.

## Files Ready to Use

- ✅ redpajama_english_train.jsonl (2.4GB training corpus)
- ✅ redpajama_english_eval.jsonl (24MB eval corpus)
- ✅ Portable domain decoder training pipeline (scripts/train_portable_domain_decoder.py)
- ✅ Modular runtime (LayerCakeRuntime for multi-domain inference)
- ✅ Transfer verification (lossless_domain_decoder tests proven PASS)
