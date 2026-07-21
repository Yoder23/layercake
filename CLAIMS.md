# LayerCake claim and evidence map

This file separates current evidence, historical negative controls, and research targets.
Passing a small-scale gate does not imply the same result at larger scale.

## North Star v23 routed-cake evidence

| Claim | Evidence | Result |
|---|---|---|
| Lossless architectural migration | v23 migration artifact | Full next-byte logits, ABI, patch logits, and generation bit-exact |
| Selected domain-cake training speed | v23 training artifact | CPU 5.32x median / 5.21x min; GPU 5.70x median / 5.39x min |
| Sparse optimizer | v23 route-isolation artifact | 1.773M optimizer params; 11.67% of 15.193M |
| No deployed generation regression | v23 CPU/GPU artifacts | 100% exact; at least 95% v22 throughput |
| Route training isolation | v23 route-isolation artifact | Default ABI, patch logits, and generated patches bit-exact |
| Portable-domain transfer | v23 transfer artifacts | logit diff 0; PPL ratio 1; identical CPU/GPU generation |

The >5x row is domain-cake fine-tuning with a frozen foundation versus full
transformer training. Full-foundation pretraining and time-to-quality remain
open and must not be inferred from that row.

## North Star v22 evidence

The current promoted result is the fail-closed
`results/breakthrough_equal/northstar_v22_release_certificate.json` certificate.

| Claim | Evidence | Result |
|---|---|---|
| Equal-size parameter comparison | v22 certificate | 15.190M vs 14.951M; 1.016x ratio |
| Better held-out general BPB | v22 certificate | 1.9088 vs 2.7149 |
| Exact schema/compositional grounding | v22 CPU/GPU artifacts | 100% LayerCake; 87.5%/60% transformer |
| Faster CPU/GPU task generation | v22 CPU/GPU artifacts | minimum 19.51x across locked dense splits |
| Smaller/faster INT8 task deployment | v22 INT8/resource artifacts | 8.73 MB; minimum 11.59x; lower process peaks |
| Exact independent cross-size domain transfer | v22 transfer artifacts | logit diff 0; PPL ratio 1; identical CPU/GPU generation |
| Repository regression | v22 pytest summary | 304 passed |
| Faster full-core training | v22 training audit | OPEN: 0.722x CPU / 1.045x GPU recipe medians; historical equal-quality time ratio 1.107x |

The strengthened transformer receives 159.37M total bytes versus LayerCake's reported
143.36M and a conservative 111.74M corrected-task bytes versus LayerCake's 73.73M.
Structured heads and direct domain caches are forbidden in every quality/speed artifact.

This promotes a same-size local architecture result for the locked task and general-BPB
protocol. It does not establish universal open-domain superiority or real-phone dominance.
The INT8 deployment artifact excludes the full local byte-LM training decoder. It also does
not establish training dominance; `TRAINING_NORTHSTAR.md` defines the open 5x gate.

## Current v2 evidence

Current north-star certificate:

| Claim | Evidence | Result |
|---|---|---|
| Matched general quality, two LayerCake seeds | `results/northstar_mobile_certificate.json` | 2.0446/2.0457 vs BPE 2.0492 BPB |
| Smaller core | same | 14.792M vs 14.844M parameters |
| Lower fixed-budget mean training time | same | 121.4 s vs 131.5 s |
| Faster batch-1 prefill | same | 2.96 ms vs 5.63 ms |
| Better exact cached-generation quality | same | 1.9953/1.9836 vs 2.0492 BPB |
| Faster one-thread generation | same | 2.91x/2.96x BPE |
| Exact migration into independent smaller host | same | max logit diff 0; PPL ratio 1.0 |
| Migrated domain beats transformer adapter | same | 1.4418/1.4436 vs 2.1101/2.0951 BPB |

Protocol:

- 8 MB fixed local general-text stream;
- 2 MB fixed local Python-source stream;
- held-out tails used consistently for evaluation;
- `d_abi=64`;
- fixed four-byte patches;
- continuous causal local decoder;
- deterministic causal ABI anchors;
- fixed canonical ABI-to-byte-logit brick head;
- predeclared maximum general-PPL ratio of 1.05.

Selected evidence:

| Claim | Evidence | Result |
|---|---|---|
| General byte-patch quality reaches BPE parity | `results/research_gate_certificate.json` | 2.4165 vs 2.4243 BPB |
| Compact core | same | 349,888 vs 629,376 BPE parameters |
| Faster base inference | `results/final_inference_benchmark.json` | 2.089M vs 1.458M bytes/s |
| Faster active-brick inference | same | 1.601M vs 1.458M bytes/s |
| Source domain adaptation | `results/sparse_brick_continuous2028_r16_p2.json` | PPL 213.70 -> 55.59 |
| Bounded cross-seed transfer | `results/final_transfer_seed314.json` | domain ratio 0.708; general 1.044 |
| Bounded cross-size transfer | `results/final_transfer_large2718.json` | domain ratio 0.484; general 1.049 |
| Bounded int8 transfer | `results/final_transfer_seed314_int8.json` | domain ratio 0.703; general 1.045 |
| Sparse activation | selected brick config | 8 installed, top-2 active |
| Exact same-PPL lossless mode | `results/lossless_domain_small.json` | PPL 2.8553 on both cores; ratio 1.0 |
| Exact same-PPL cross-size mode | `results/lossless_domain_scale5m_to_2m.json` | PPL 2.7143 on both cores; ratio 1.0 |
| Exact transfer through 15.45M tier | `results/lossless_domain_scale15m_to_5m.json` | PPL 2.7143; logits/generation identical |
| Compact int8 transfer artifact | `results/lossless_domain_scale15m_to_5m_int8.json` | 148,808 bytes; PPL 2.7165; ratio 1.0 |
| Filesystem-disjoint Python transfer | `results/lossless_domain_external_python_int8.json` | PPL 5.8296; 57.72% byte accuracy; ratio 1.0 |
| Mobile CPU domain win vs transformer adapter | `results/mobile_domain_win_certificate.json` | Better BPB across two adaptation seeds; 3.57x faster isolated training, 2.57x smaller artifact, 4.43x CPU throughput |

Reproduce the certificate:

```powershell
python scripts/verify_research_gates.py
```

## What solved the original generalization failure

Legacy LayerCake bricks copied exactly but did not generalize across independently trained
cores. The failure was real and remains a useful negative control:

- tensor copy could have `max_diff=0`;
- the target core could still produce a different ABI distribution;
- target-side ABI decoding could assign a different meaning to the same delta;
- domain PPL could therefore regress catastrophically.

V2 adds two seed-independent contracts:

1. **Canonical input coordinates:** deterministic byte-prefix anchors supervise every core.
2. **Canonical output semantics:** brick deltas use a fixed ABI-to-byte-logit head.

It also fixes temporal alignment: byte state after a completed patch aligns with the
context for the following patch. With these changes, unchanged bricks pass bounded
cross-seed and cross-size tests locally.

They do not preserve absolute PPL. Strict evaluation measured target/source PPL ratios of
1.74 on the small pair and 2.08 on the 5.40M-to-2.19M pair. Router agreement was only
50-60%, and different base logits remained even when the same correction was forced.

The strict contract is implemented separately as a portable recurrent domain decoder
driven by raw bytes and deterministic causal anchors. It owns the domain logits, so
host-core size, seed, ABI width, and base predictions cannot change its output. This
148,736-parameter mode measured held-out Python PPL 2.71-2.86 and 72.6-73.8% top-1 byte
accuracy with bit-exact logits, generation, and PPL ratio 1.0 through the 15.45M tier.

The correct conclusion is not that arbitrary neural networks now share semantics. It is
that cores trained under this explicit canonical protocol can share a measured ABI.

## Transfer ladder

| Level | Contract | Status |
|---|---|---|
| L0 | Exact tensor copy | Proven |
| L1 | Exact brick function on equal ABI inputs | Proven |
| L2 | Same-core token-generation identity | Proven on legacy path |
| L3 | Cross-size structural/function portability | Proven; bounded v2 end-to-end local PASS |
| L4 | Cross-seed bounded semantic transfer | Small-scale PASS |
| L5 | Quantized bounded transfer | Small-scale int8 PASS |
| L6 | Byte/byte-patch tokenizer-independent bounded transfer | Small-scale PASS |
| PX | Exact core-independent portable-domain transfer | PASS through 15.45M tier |
| L7 | Orchestrated bounded transfer | Not yet task-validated |

Exact definitions and thresholds are in [RUBRIC.md](RUBRIC.md).

## Historical v1 evidence

The original tokenized model established:

- bit-exact domain tensor paste;
- exact domain-function output for equal ABI inputs;
- same-core generation identity;
- matched-parameter LM parity at the 48M class;
- domain adaptation with fewer trainable parameters than full fine-tuning.

Those results remain valid in their original scope. They do not substitute for the v2
cross-seed or tokenizer-free evidence.

| Historical claim | Result artifact |
|---|---|
| Exact structural paste | `results/paste_proof.json` |
| Same-core generation/function identity | `tests/test_paste_lossless.py` |
| 48M matched LM comparison | `results/fair_comparison.json` |
| Legacy domain adaptation | `results/domain_paste_functional.json` |

## Claims not made

| Not claimed | Reason |
|---|---|
| Universal tokenizer-free superiority | One small local point estimate is insufficient. |
| 25M-1B scaling validation | Those runs have not completed. |
| Arbitrary pretrained-model compatibility | Cores must implement the canonical ABI contract. |
| Exact additive-brick semantic losslessness | Additive outputs still depend on host ABI states and logits. |
| Host-assisted exact PPL equivalence | Current exact mode is deliberately core-independent. |
| Autonomous coding competence | Free-running held-out completion quality is not sufficient. |
| Production mobile performance | Current CPU result is an x86 proxy, not phone/NPU evidence. |
| GPU generation superiority | LayerCake reaches 0.62x BPE in the selected RTX benchmark. |
| Native int8 speedup | Current L5 artifact is quantize/dequantize evidence. |
| Dynamic BLT-quality patching | Current selected model uses fixed patches. |
| Production readiness | Distributed training, serving, and security hardening remain. |
| L7 swarm equivalence | Packet/router interfaces exist, but task evidence is pending. |
| Rolling-training substrate proves scale dominance | It only proves rollbackable training mechanics and auditability. |
| Preview-guided smoke demo proves transformer dominance | It only proves preview artifact generation, syllabus compilation, tiny CPU training, baseline harness execution, and rollback mechanics. |
| Tier-1 smoke dominance proves scale dominance | It is a tiny deterministic methodology gate; Tier 1 local and Tier 2 serious runs are still required. |

## Larger-tier evidence

The 5.40M patch-core checkpoint confirms that size, throughput, sparse adaptation,
cross-size/seed transfer, and int8 bounded transfer continue to work on a larger local
model. It does not match the 6.90M BPE baseline's general BPB:

- byte-patch: 2.2612 BPB;
- BPE: 2.0747 BPB.

Those models are historical negative controls. The newer 14.79M architecture reaches
2.0446/2.0457 BPB against the 14.84M BPE baseline at 2.0492. Validation beyond this local
15M-class protocol remains open.

## Required language for public discussion

Use:

- bit-exact structural paste;
- canonical ABI;
- bounded cross-seed transfer;
- bounded cross-size transfer;
- small-scale tokenizer-free parity by point estimate;
- portable sparse domain brick;
- cost-adjusted domain adaptation.
- replicated 15M-class mobile CPU win under the frozen local protocol;
- exact portable-domain migration into an independent smaller LayerCake host.
- rollbackable model commits and semantic CI for future architecture experiments.
- preview-guided rolling-training smoke path with tiny LayerCake and tiny byte-transformer comparison harness.
- Tier 0/Tier 1 smoke dominance certificate against a closest matched-parameter tiny byte transformer.
- CPU/mobile-proxy source and receiver dominance under locked 15M/6.8M certificates.

Do not use without larger evidence:

- universal lossless semantic transfer;
- tokenizer-free dominance;
- frontier-model replacement;
- mobile model has the same intelligence as a server model.
