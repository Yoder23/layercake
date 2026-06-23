# Skeptics FAQ

## 2026 v2 update

Some objections below describe the original tokenized prototype and remain useful
historical controls. The statement that cross-seed transfer is simply unsolved no longer
describes the selected byte-patch architecture.

V2 trains participating cores against deterministic causal ABI anchors and applies brick
deltas through a fixed canonical byte-logit head. Under the fixed local protocol,
unchanged bricks pass bounded cross-seed, cross-size, cross-interface, and int8 gates.

This does not make arbitrary models compatible. Cores must implement the same ABI version
and canonical contracts. Verify the selected evidence with:

```powershell
python scripts/verify_research_gates.py
```

Anticipated objections and direct answers.

---

## "If paste is lossless, why does performance explode on different seeds?"

Because paste fidelity and functional compatibility are orthogonal.

The paste copies the domain module weights **exactly** — weight-by-weight, bit-for-bit,
`max_diff = 0.0` on all tensors, and token-for-token identical generation when both models
use the same core representations. This is proven by six independent tests in
`tests/test_paste_lossless.py`.

What the paste does **not** do is make the pasted module's inputs sensible. The domain module
was trained to interpret `h_abi` vectors produced by a specific core (e.g., seed9000's
`core_to_abi` projection). A different-seed core (seed6000) produces geometrically different
`h_abi` vectors — a completely different coordinate system. The module is correct; it just
receives inputs it was never trained on.

Analogy: copying a chess engine's weights to a system that feeds it Go board states. The copy
is perfect. The output is garbage. That is not a copy problem — it is an input problem.

Measured evidence:
| Paste target | Domain | PPL (no domain) | PPL (pasted) | Weights bit-exact? |
|---|---|---|---|---|
| seed6000 (same-size, different seed) | chess | 31.04 | 2618.89 | ✅ yes |
| seed7000 (same-size, different seed) | chess | 39.86 | 2877.22 | ✅ yes |

PPL explodes; weights are still perfect. These are consistent facts, not a contradiction.

ABI alignment (aligning `core_to_abi` projections across different-seed cores) resolves this.
It is left as future work. See [README.md — Relationship to ABI](README.md#relationship-to-abi).

---

## "Are you claiming task behavior transfers losslessly across any two LayerCake models?"

No. The lossless paste claim has a defined scope:

| What is claimed | Status |
|---|---|
| Weight tensors are bit-exact after paste | ✅ Proven |
| Forward-pass outputs are bit-identical (same core) | ✅ Proven (`max_diff = 0.0`) |
| Autoregressive generation is token-identical (same core) | ✅ Proven (50 tokens, no divergence) |
| Domain module function is preserved (cross-size, same d_abi input) | ✅ Proven |
| Functional task performance across different-seed cores | ❌ Not claimed |
| Functional task performance across different-size cores without alignment | ❌ Not claimed |

"Lossless" refers to the copy operation and the mathematical function it implements — not to
guaranteed downstream task performance in arbitrary environments.

---

## "Is this just an adapter / LoRA / prefix tuning?"

Not exactly, though there is overlap.

Adapters and LoRA inject learned parameters into an existing model, typically after training.
They are not designed for portability — an adapter trained on model A has no well-defined
relationship to model B's internal representations.

LayerCake's domain modules are different in one key way: they operate in a **fixed-dimension
interface (`d_abi=512`) that is shared by design across all model sizes**. This is a
architectural constraint, not a post-hoc modification. The consequence is that the module is
literally the same tensor whether the surrounding model is 48M, 150M, or 350M parameters.

LoRA modules are not portable across model sizes. LayerCake domain modules are — structurally,
and as functions, as proven by the six tests in `tests/test_paste_lossless.py`.

---

## "Does LayerCake outperform standard transformers?"

No, and we do not claim it does.

At 48M parameters, 20K training steps, with matched parameters and optimizer:

| Metric | LayerCake | Baseline | Δ |
|---|---|---|---|
| C4 val PPL | 45.01 | 44.89 | +0.27% (overhead) |
| WikiText2 PPL | 174.69 | 171.82 | +1.67% (overhead) |
| HellaSwag | 27.0% | 26.0% | +3.85% (advantage) |

LayerCake has a small PPL overhead on raw LM quality and a modest HellaSwag advantage.
Neither result is large enough to make strong claims at this scale.

The value of LayerCake is **modularity and structural portability**, not raw LM performance.
The architecture is at parity — it does not cost you quality to gain portability.

---

## "Why does ABI alignment matter if the paste is lossless?"

Because the paste being lossless is a statement about the copy operation, not about the
compatibility of the source and target models' representation spaces.

LayerCake, when all models are trained together with the same `d_abi`, gives you exact
structural portability for free. But when you have:
- Models trained independently with different random seeds
- Models from different training runs
- Models with different architectures but compatible `d_abi`

...the `core_to_abi` projections will have learned different geometric spaces. The module
is copied perfectly but operates on inputs from a different distribution. ABI alignment learns
a mapping between those spaces so the module receives inputs consistent with its training.

**LayerCake is the clean design principle.** ABI is what you use when the clean assumption breaks.

---

## "The 150M cross-size experiment is inconclusive — is cross-size portability actually proven?"

At the function level, yes. At the task-performance level, inconclusive.

What is proven (see `tests/test_paste_lossless.py`, `test_cross_size_forward_pass_lossless`):
- The domain module pasted to a 150M model produces **bit-identical outputs** given the same
  `d_abi=512` input vectors. `max_diff = 0.0`.
- 9 weight tensors are bit-exact across the 48M→150M paste.

What is not proven:
- End-to-end task performance on the 150M model, because the 150M core used in the experiment
  was only trained for 10K steps (baseline PPL ~374) — too weak to evaluate domain modules
  meaningfully. Structural paste is confirmed. Functional evaluation requires a fully-trained
  150M core.

This is explicitly stated as a limitation in [README.md — Honest Known Limitations](README.md#honest-known-limitations).
