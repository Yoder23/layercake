# Exact Next Scientific Runs

```powershell
pytest -q
python scripts/train_abi_aligned.py --steps 100 --output results/alignment_train_smoke.json
python scripts/smoke_byte_patch.py --output results/smoke_byte_patch.json
python scripts/eval_transfer_matrix.py --output results/transfer_matrix.json
python scripts/benchmark_domain_routing.py --iterations 100 --output results/domain_routing.json
```

Before making L4 or L6 claims:

1. pretrain paired tokenized/byte-patch cores on the same byte-normalized corpus;
2. freeze the evaluation split and PPL threshold before domain training;
3. report source/target domain PPL, general PPL, ABI drift, and confidence intervals;
4. compare against tokenizer baseline, LoRA, dense adapter, and no-brick controls;
5. repeat across at least seeds 42, 314, and 2718.

Full matched 25M-1B training remains compute- and dataset-limited, not code-path-limited.

The selected sub-million-parameter gates now pass. The next scale experiment should:

1. train 25M byte-patch and BPE baselines on at least 1B identical bytes;
2. use three independent seeds and a held-out domain not present in repository source;
3. freeze thresholds and evaluation hashes before training;
4. report BPB confidence intervals, wall-clock-to-quality, energy, and peak memory;
5. repeat cross-seed, cross-size, int8, and active-brick latency gates.

The intermediate 5.40M tier and matched 5,000-step 15.45M checkpoint are complete. The
15.45M checkpoint reached 2.430 general BPB versus 2.227 for the 25.75M byte baseline,
runs 2.45x faster in the selected CUDA benchmark, and passes PX exact transfer. Next:

1. train independent 15.45M seeds;
2. evaluate on an externally sourced, contamination-audited Python corpus;
3. add task-level code completion and execution benchmarks;
4. add Android/iOS or ARM CPU, memory, battery, and thermal measurements;
5. replace load-time int8 dequantization with native int8 execution.

## General-core frontier

The 15M mobile frontier now passes the combined certificate across two LayerCake seeds:

- 14.792M parameters versus 14.844M BPE;
- 2.0446/2.0457 versus 2.0492 general BPB;
- 121.4 s mean training time versus 131.5 s;
- 2.96 ms versus 5.63 ms batch-1 prefill;
- 2.91x/2.96x one-thread cached-generation speed;
- exact domain migration from each 15M core into an independent 5.40M host;
- migrated Python BPB 1.4418/1.4436 versus adapter BPB 2.1101/2.0951.

Next:

1. reproduce on a second general-text split and a non-repository domain;
2. run ARM/Android/iOS latency, memory, battery, and thermal measurements;
3. close the current RTX generation gap (0.62x BPE);
4. scale the matched protocol to 25M and 60M before making broader claims;
5. add task-level generation and execution benchmarks.

Two first-step dense scaling candidates have already been rejected under the same sampled
byte budget:

- 23.69M LayerCake, 5+5 blocks: 2.0299 BPB and 214.1 s;
- 25.24M LayerCake, 4+4 blocks: 2.0376 BPB and 204.6 s;
- 24.09M BPE comparator: 2.0035 BPB and 158.0 s.

Adding dense width or depth alone therefore does not preserve the full north-star win.
The next candidate should reduce global sequence length without returning to independent
factorized patch bytes: learned/dynamic patch boundaries, a stronger causal within-patch
decoder, and fused training kernels are the priority.

An additive two-resolution hierarchy was also rejected at the 750-step probe gate:

- one fine + three coarse global blocks: 2.4216 BPB, 34.8 s;
- two fine + two coarse global blocks: 2.4188 BPB, 34.6 s;
- fixed two-byte reference: 2.3180 BPB, 32.3 s.

Coarse four-byte summaries therefore do not substitute for fine two-byte global depth in
their current additive form. Do not launch a full run from this branch. The next patching
design must make boundaries content-dependent and keep exact byte-level causal decoding.

Every future full candidate must pass `scripts/verify_scale_candidate.py`; improvement on
mobile alone is insufficient. Desktop and GPU prefill/generation are now required gates.

Verify the frontier:

```powershell
python scripts/verify_northstar_mobile.py
```
