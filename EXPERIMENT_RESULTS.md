# Measured Byte-Patch Results

## Strict same-PPL domain transfer

The unchanged additive sparse brick fails the strict absolute-PPL contract:

| Pair | Source PPL | Target PPL | Target/source |
|---|---:|---:|---:|
| seed 2028 -> seed 314 | 56.8248 | 98.9076 | 1.7406 |
| 5.40M -> 2.19M | 40.6288 | 84.5737 | 2.0816 |

Diagnostics show ABI cosine above 0.91 but only 50-60% exact top-k routing agreement,
36-66% correction relative error, and 54-71% base-logit relative error.

The core-independent portable decoder passes:

| Pair | Parameters | Source PPL | Target PPL | Max logit diff | Ratio |
|---|---:|---:|---:|---:|---:|
| seed 2028 -> seed 314, context 128 | 148,736 | 2.8553 | 2.8553 | 0.0 | 1.0 |
| 5.40M -> 2.19M, context 256 | 148,736 | 2.7143 | 2.7143 | 0.0 | 1.0 |
| 15.45M -> 5.40M, context 256 | 148,736 | 2.7143 | 2.7143 | 0.0 | 1.0 |
| int8 15.45M -> 5.40M | 148,736 | 2.7165 | 2.7165 | 0.0 | 1.0 |

Both gates use evaluation stream SHA-256
`a361e0a0beeb680dc277c6f1b8cb6cab35a89f6f1e79e7791ded09e69a553238`.

The fp32 recurrent payload is 594,944 bytes. The symmetric per-tensor int8 artifact is
148,808 bytes (25.0%) and degrades PPL by 0.083%. It is dequantized at load time; native
int8 execution is not claimed.

On a filesystem-disjoint 100,000-byte slice from the Python 3.10 standard library, the
int8 artifact measures PPL 5.8296 and 57.72% top-1 byte accuracy on both the 15.45M and
5.40M hosts. The evaluation SHA-256 is
`f1c760b52e52b90efa2ccfb34532f96189795f46e67c26a544eb606036cbc47f`.
This is a stronger distribution-shift check, not a full contamination audit.

## Mobile domain deployment versus transformer adapter

Matched baseline:

- 14.84M-parameter BPE transformer;
- rank-16 residual adapters after eight transformer blocks;
- 95,752 trainable adapter parameters;
- same local Python domain;
- adapter disabled to recover exact base behavior outside the domain.

| Metric | LayerCake portable domain | BPE adapter |
|---|---:|---:|
| Domain BPB | 1.4418 | 2.1101 |
| Training wall time | 51.3 s | 183.1 s |
| Trainable parameters | 148,736 | 95,752 |
| Artifact bytes | 148,808 int8 | 383,008 fp32 |
| One-thread x86 CPU bytes/s | 35,744.7 | 8,075.0 |
| RTX 3080 Laptop bytes/s | 153,637.3 | 214,802.6 |
| Cross-host exactness | ratio 1.0, max logit diff 0 | model-specific |

The LayerCake domain path wins the declared mobile CPU/domain gates. The transformer wins
GPU prefill and the general-core quality comparison. See
`results/mobile_domain_win_certificate.json`.

The domain-quality ordering repeats on a second adaptation seed:

- LayerCake int8 PX BPB: 1.4436;
- transformer rank-16 adapter BPB: 2.0951;
- LayerCake transfer ratio: 1.0 with zero logit difference.

The second runs shared the GPU concurrently, so their wall-clock times are excluded.

## Paired byte/byte-patch training

Data: 8 MB local RedPajama text for general training/evaluation and 2 MB local Python
source for domain training/evaluation. Models use 128-byte sequences, fixed four-byte
patches, a 64-dimensional ABI, and strictly causal local decoding.

With alignment weight 1.0 and general-preservation weight 6.0:

| Seed | ABI MSE at step 1500 | Byte domain PPL base → brick | Byte general ratio | L6 gate |
|---|---:|---:|---:|---|
| 42 | 0.01101 | 152.10 → 146.21 | 1.0451 | PASS |
| 314 | 0.01055 | 126.73 → 119.71 | 1.0161 | PASS |

The unchanged brick was trained through the byte-patch source path and evaluated through
the byte target path. These are bounded smoke results at small scale, not superiority
claims.

## Cross-seed transfer

| Source → target | Target domain PPL base → transferred | General ratio | Result |
|---|---:|---:|---|
| 42 → 314 | 126.734 → 126.653 | 1.00013 | bounded pass, negligible gain |
| 314 → 42 | 152.098 → 153.079 | 1.00948 | FAIL: domain regression |

Historical conclusion from this failed loop: pairwise same-seed alignment alone was
insufficient. This failure motivated deterministic external anchors and a canonical output
head; the later selected result below supersedes the earlier L4 status.

## Canonical-head and deterministic-anchor result

The failure above led to two protocol changes:

1. every core is trained against deterministic causal byte-prefix anchors;
2. brick deltas use a fixed ABI-to-byte-logit head shared across seeds and sizes.

These changes produce bounded bidirectional transfer. The final selected compact system
uses a 96-wide, two-layer patch core, fixed four-byte patches, a continuous local GRU,
`d_abi=64`, and a sparse rank-16 brick with eight installed experts and top-2 activation.

| Gate | Measured result | Status |
|---|---:|---|
| General BPB versus 2,048-piece byte-fallback BPE | 2.4165 vs 2.4243 | PASS |
| Patch parameters versus BPE | 349,888 vs 629,376 | PASS |
| Patch parameters versus byte transformer | 349,888 vs 693,952 | PASS |
| Active-brick throughput versus byte base | 1.601M vs 1.458M bytes/s | PASS |
| Source patch domain PPL | 213.70 → 55.59 | PASS |
| Source patch general PPL ratio | 1.0251 | PASS |
| Cross-seed domain/general ratio | 0.7084 / 1.0445 | PASS |
| Cross-size domain/general ratio | 0.4837 / 1.0489 | PASS |
| Int8 domain/general ratio | 0.7027 / 1.0450 | PASS |

The valid brick does not beat the BPE baseline on Python BPB under the full cross-interface
preservation gate. An invalid lower-preservation run approaches it but exceeds the 5%
general regression threshold. Current evidence therefore supports small-scale general
BPB parity, smaller size, faster inference, and portable bounded domain adaptation—not
universal tokenizer dominance.

Run `python scripts/verify_research_gates.py` to validate the selected JSON artifacts.

## 5.40M patch-core scaling checkpoint

The first larger tier uses:

- 5,396,608 patch-core parameters;
- 14,566,048 byte-baseline parameters;
- 6,901,760 byte-fallback BPE parameters;
- `d_abi=96`;
- 20 MB general-text stream;
- 256-byte context;
- an independent 2.19M patch target for transfer.

| Gate | Result | Status |
|---|---:|---|
| Patch size vs byte | 5.40M vs 14.57M | PASS |
| Patch size vs BPE | 5.40M vs 6.90M | PASS |
| Patch base throughput | 243.6K vs 122.1K bytes/s | PASS |
| Active-brick throughput | 232.0K vs 122.1K bytes/s | PASS |
| Source domain PPL | 157.03 -> 40.94 | PASS |
| Source general ratio | 1.0105 | PASS |
| Cross-size/seed domain ratio | 0.5326 | PASS |
| Cross-size/seed general ratio | 1.0210 | PASS |
| Int8 domain/general ratio | 0.5319 / 1.0214 | PASS |
| Patch general BPB vs BPE | 2.2612 vs 2.0747 | FAIL / target open |

At this tier the architecture's deployment and portability advantages scale, but raw
general BPB parity does not yet scale under the measured protocol. The BPE baseline leads
by about 9%. The patch model received additional patch-only optimization and still did not
close the gap. This falsifies a scale-independent parity claim and makes improved patching,
optimization, or local decoding the next quality target.

Run `python scripts/verify_scale5m_results.py`.
