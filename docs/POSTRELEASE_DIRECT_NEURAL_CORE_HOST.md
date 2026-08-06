# Post-release direct neural English-core host

Status date: 2026-08-06

LayerCake now has a construct-verified, separately versioned host role for an
unchanged signed byte-facing neural artifact to execute as the English core:
`lc-direct-neural-core/1`.

The extension is isolated under `layercake_extensions/`; it does not modify the
sealed model, runtime, package schema, installer, checkpoints, router, or prior
evidence. The original eight-phase campaign still verifies at the current
commit.

The construct used no external teacher and performed no training. The same
signed test package passed on CPU and CUDA with:

- exact archive, payload, and state-dict identity;
- authenticated installation and verification;
- persistent incremental decoder state;
- zero receiver training and zero calibration;
- exact remove and reinstall;
- rejection of a capability-cake role; and
- rejection of authenticated-content tampering.

This is a host-construct pass only. The test artifact is deliberately an
EOS-only mechanical fixture. It does not establish English quality, teacher
relative quality, external acquisition, CPU/GPU speed, TTFT, memory, domain
compatibility, or ABI superiority.

The next gate is an independently validated external English artifact packaged
against this exact interface. That same artifact must then pass core-only
English behavior, immutability, teacher absence, CPU/CUDA execution, persistent
state, CPU speed/TTFT/RSS/memory, and future domain compatibility. No evidence
may be borrowed from the sealed core or existing capability packages.

Post-release UTF-8 audit: the v1 package construct remains valid, but v1 UTF-8
output conformance is failed. Its byte-regex tokenizer can expose fragments of
a multibyte character as separate actions, and the host does not validate final
bytes. See `POSTRELEASE_DIRECT_NEURAL_CORE_UTF8_AUDIT.md`. External English
artifacts must target a separately certified Unicode-atomic successor rather
than v1.

That successor is now construct-certified as `lc-direct-neural-core/2`. See
`POSTRELEASE_UNICODE_DIRECT_NEURAL_CORE_V2.md`. V1 remains historical and
incompatible; no v1 checkpoint or package is automatically upgraded.

Evidence:

- `moonshot/canonical_direct_neural_core_abi_v1.json`
- `moonshot/postrelease_direct_neural_core_host_execution_repair1_v3.json`
- `results/moonshot/postrelease/direct_neural_core_host_construct_v1.json`
- `results/moonshot/postrelease/external_capability_host_interface_audit_v2.json`
