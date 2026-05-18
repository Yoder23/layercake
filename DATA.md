# Data Preparation

LayerCake training scripts expect pre-tokenized corpora stored as 1D NumPy arrays of
integer token IDs (`.npy` files). This document explains how to prepare the data.

---

## Required Files

| Variable in scripts | Purpose | Recommended source |
|--------------------|---------|-------------------|
| `data/tokens/c4_train.npy` | Main training data | C4 (`allenai/c4`) |
| `data/tokens/c4_val.npy` | C4 validation | C4 validation split |
| `data/tokens/wikitext2.npy` | Held-out eval | WikiText-2 test split |
| `data/tokens/<domain>_train.npy` | Domain training | Domain-specific corpus |
| `data/tokens/<domain>_val.npy` | Domain eval | Domain-specific corpus |

---

## Tokenizer

LayerCake uses a 16,000-vocab SentencePiece BPE tokenizer. Train one on your corpus:

```bash
python -c "
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='<your_text_file.txt>',
    model_prefix='tokenizer/layercake_sp',
    vocab_size=16000,
    character_coverage=0.9999,
    model_type='bpe',
)
"
```

Or use any tokenizer that maps text → integer IDs in `[0, 16000)`.

---

## Tokenizing a Corpus

```python
import sentencepiece as spm
import numpy as np

sp = spm.SentencePieceProcessor()
sp.Load("tokenizer/layercake_sp.model")

# Read raw text file, tokenize, flatten to 1D array
tokens = []
with open("raw_corpus.txt", "r", encoding="utf-8") as f:
    for line in f:
        ids = sp.EncodeAsIds(line.strip())
        tokens.extend(ids)

arr = np.array(tokens, dtype=np.int32)
np.save("data/tokens/c4_train.npy", arr)
print(f"Saved {len(arr):,} tokens")
```

---

## Token Array Format

- Shape: `(N,)` — 1D array of integers
- dtype: `int32` or `int64`
- Values: in range `[0, vocab_size)`
- Minimum length: `seq_len + 1` (e.g., 257 for seq_len=256)

The `LM1DDataset` in `data.py` builds sliding windows from this array automatically.
There is no special end-of-document token required (though adding one is recommended
to prevent cross-document context bleed in training).

---

## Minimum Viable Test (no real data needed)

For testing paste and architecture (not training), you can generate synthetic tokens:

```python
import numpy as np
arr = np.random.randint(0, 16000, size=100_000, dtype=np.int32)
np.save("data/tokens/synthetic_train.npy", arr)
np.save("data/tokens/synthetic_val.npy", arr[:10000])
```

This is sufficient to run `tests/test_paste_lossless.py` and verify the architecture.
It is NOT useful for PPL benchmarks or domain training.
