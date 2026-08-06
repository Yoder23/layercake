# Post-release direct neural core UTF-8 audit

Status date: 2026-08-06

The `lc-direct-neural-core/1` construct remains valid for its original package,
identity, lifecycle, persistent-state, and zero-learning scope. Its UTF-8 output
conformance is now failed.

The locked local audit used U+201C, whose valid UTF-8 encoding is `e2809c`.
The v1 byte-regex tokenizer split it into three independently selectable actions:
`e2`, `80`, and `9c`. Each fragment is invalid UTF-8 alone, although the complete
ordered sequence round-trips exactly. A host fixture also proved that v1
`generate` returns an invalid byte without validating the declared UTF-8 output
boundary.

This is an action-atomicity and boundary-validation gap. It is not a change to
the sealed eight-phase release, an English-quality result, or imported ABI
evidence.

A repair must use a new interface and tokenizer identity, make every fixed and
pointer lexeme a complete valid UTF-8 sequence, validate input and output
strictly, reject v1 packages rather than relabel them, and remain isolated from
sealed model/runtime paths. Any external artifact must be trained or conformed
against the new action semantics; old checkpoints cannot inherit the repair.

Evidence:

- `moonshot/postrelease_direct_neural_core_utf8_audit_preregistration_v1.json`
- `results/moonshot/postrelease/direct_neural_core_utf8_audit_v1.json`
- `moonshot/postrelease_direct_neural_core_utf8_audit_decision_v1.json`
