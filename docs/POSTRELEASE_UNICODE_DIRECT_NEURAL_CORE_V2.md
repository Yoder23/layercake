# Post-release Unicode-safe direct neural core v2

Status date: 2026-08-06

LayerCake now has a construct-certified Unicode-safe successor to the v1
external English-core interface: `lc-direct-neural-core/2`.

V2 is not a relabel or patch of v1. It has a new canonical ABI hash, tokenizer
identity, token-plan identity, package contract, and host. Every fixed action
and every source-pointer action is a complete valid UTF-8 sequence. Unicode
normalization is not performed; exact input text is preserved. Invalid input is
rejected before model execution, and invalid output is rejected before crossing
the host boundary.

The deterministic signed construct passed:

- exact UTF-8 roundtrip and per-action validity for accented Latin, smart
  punctuation, CJK, combining marks, Arabic, Devanagari, and emoji/ZWJ text;
- strict invalid-input and invalid-output rejection;
- the same signed package and identical state on CPU and CUDA;
- persistent incremental state;
- exact install, verify, remove, and reinstall;
- zero receiver training and calibration;
- teacher absence;
- v1-package, wrong-role, and tamper rejection;
- 613 repository tests; and
- the unchanged original Phase 0–8 campaign verifier.

The first construct command wrote valid immutable evidence, then failed while
printing an emoji to a CP1252 console. That failure is preserved. The stdout-
only repair escaped non-ASCII JSON and deterministically reverified the existing
result without overwriting it.

This remains a host construct, not an English model certificate. A real external
English artifact must be independently trained or conformed against v2 and then
pass quality, teacher-relative, CPU/GPU, memory, TTFT, persistent-state, and
future-domain compatibility gates on that same artifact. V1 checkpoints cannot
inherit v2 validity.

Evidence:

- `moonshot/canonical_unicode_direct_neural_core_abi_v2.json`
- `moonshot/postrelease_unicode_direct_neural_core_implementation_v2.json`
- `moonshot/postrelease_unicode_direct_neural_core_execution_repair1_v3.json`
- `results/moonshot/postrelease/unicode_direct_neural_core_construct_v1.json`
- `moonshot/postrelease_unicode_direct_neural_core_decision_v4.json`
