# External capability-artifact handoff

Status date: 2026-09-15

Current documentation note (2026-09-15): the latest construct-certified
route-isolated host is `lc-direct-neural-core/25`, not v19. It preserves the
four V24 routes and adds one physically isolated clarification route. This is
still construct-only: no real external English artifact has passed output
conformance, English quality, information-minimum, speed, memory, or TTFT
certification. The older interface notes below are retained as chronological
handoff history.

Current-host note (2026-08-16): the allocation-bounded streaming package and
active integrity-verification changes are post-release work. They touch the
registered `precision_contract`, so current HEAD is not the sealed LayerCake
release and does not inherit its Phase 2-8 claims. The immutable release tag
still verifies as `PROVEN` under V99. Any external artifact using the newer
host requires a governed successor recertification; ABI-side evidence is not a
substitute for a LayerCake release seal.

This document defines LayerCake's boundary for externally produced capability
artifacts. It does not import external code or evidence, and it does not expand
the sealed LayerCake certificate.

## Repository boundary

LayerCake owns its execution host, canonical interfaces, package lifecycle,
transfer, composition, routing, orchestration, and product-level CPU/GPU
certification.

Foreign-teacher extraction, labeling, normalization, minimization, and source
information accounting belong to the separate
[ABI repository](https://github.com/Yoder23/abi). Its research status and
evidence must be read there, not inferred from this repository.

An ABI acquisition bundle is not a deployable cake. LayerCake accepts only an
independently validated, non-executable, signed, content-addressed artifact with
complete provenance, license, deletion lineage, and imported-information
accounting.

## Acceptance rule

The post-release `lc-direct-neural-core/2` interface was the first construct-
certified Unicode-safe English-core handoff target. It uses Unicode-atomic fixed and
pointer actions, strict UTF-8 input/output validation, signed safetensors-only
packages, persistent incremental state, and zero receiver learning. It passed
the same deterministic package on CPU and CUDA. The older v1 interface failed
its UTF-8 conformance audit and must not receive new external English artifacts.

This host readiness is not an acquisition result. No ABI-derived English
artifact has passed v2, and no external artifact is part of the sealed release.
See `POSTRELEASE_UNICODE_DIRECT_NEURAL_CORE_V2.md` for the exact construct scope.

The later `lc-direct-neural-core/5` interface is also construct-certified for
selective-boundary UTF-8 BPE token plans. It provides stable pointer pieces for
digit-or-underscore identifiers without disabling ordinary raw-BPE
compression. This is a generic LayerCake hosting surface only: no external
artifact, English quality, or performance result is imported into this repo.

The separately versioned `lc-direct-neural-core/18` interface was later
construct-certified for one signed shallow-sparse English-core package.
It retains the model, sparse capability router, declarative tokenizers, guard,
and persistent KV-state boundary while accepting explicit route-indexed
`residual.down` and `residual.up` tensors and physically executing only the
selected route slice. The construct passes CPU/CUDA identity, five focused
tests, all 665 repository tests, and the unchanged sealed verifier. V17 remains
valid for its own declared schema, but a real immutable external candidate was
correctly rejected at its strict state-dict boundary; V18 repairs that interface
without reshaping or mutating the artifact. No acquisition code or evidence is
present here, and no real v18 artifact has yet passed package-output
conformance, English quality, or performance certification.

The separately versioned `lc-direct-neural-core/19` construct preserves those
v18 tensors and its ordinary neural-generation fallback. It adds an explicitly
bounded, zero-parameter mode for prompts that request the order of exactly
three unique literal bracketed spans. The frozen active model scores all six
permutations without evaluator or expected-answer access, reusing one
persistent prompt prefill and one batched candidate forward while one residual
route is active. The same signed fixture produces identical output on CPU and
CUDA; all 667 repository tests and independent evidence recomputation pass.
This constrained extractive neural mode is not evidence of broad autonomous
generation, and no real external artifact has yet passed v19 quality or
performance gates. The only authorized handoff is an immutable v18 tensor
payload repackaged under a signed v19 manifest without tensor mutation, then
tested on the same artifact for output conformance, quality, TTFT, throughput,
RSS, CPU/GPU behavior, and teacher absence.

An external artifact never inherits this release's quality, speed, TTFT,
memory, portability, routing, or verification evidence. Integrating it requires
a governed successor lineage and every recertification required by
`moonshot/invalidation_matrix.yaml`, including quality, isolation, identity,
teacher absence, selected-only execution, CPU/GPU behavior, memory, latency,
and hostile verification on the same final candidate.

Until such a successor is sealed, `layercake-moonshot-final` remains unchanged
and no external capability artifact is part of this release.
