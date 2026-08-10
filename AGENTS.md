# LayerCake Moonshot Gated Research Campaign Charter

## Current sealed release state

The eight-phase campaign is sealed locally at the annotated
`layercake-moonshot-final` tag. `moonshot/campaign.yaml`, the phase seals, and
`results/moonshot/phase8/` are the release source of truth. Treat the current
task as release stewardship unless a new governed recertification explicitly
opens affected work. Do not relabel older byte, token, V2, North Star, or
training experiments as an open Moonshot phase or use them to expand the sealed
claim.

The isolated `lc-direct-neural-core/5` selective-boundary BPE host now passes
its construct under
`moonshot/postrelease_selective_boundary_bpe_decision_v12.json`. It preserves
stable boundaries only around digit-or-underscore identifier units while
retaining raw concatenative BPE elsewhere. The exact signed package passes
CPU/CUDA identity, UTF-8 validity, persistent state, lifecycle, zero learning,
all 622 tests, and the unchanged Phase 0-8 verifier. Its next gate is one
independently validated external v5 artifact evaluated for quality and
performance as the same immutable payload. This remains construct-only.

The isolated `lc-direct-neural-core/6` tied decoder-only causal host now passes
its generic construct under `moonshot/postrelease_causal_core_decision_v14.json`.
It provides a decoder-aware external-token boundary, tied input/output
vocabulary, strict UTF-8 realization, and persistent per-layer incremental
state. The same signed package passes CPU/CUDA identity, lifecycle, zero
learning, all 625 tests, and the unchanged Phase 0-8 verifier. Its next gate is
one independently validated external v6 artifact evaluated for quality and
performance as the same immutable payload. This remains construct-only and
imports no external acquisition implementation or evidence.

The isolated `lc-direct-neural-core/7` structural causal host now passes its
generic construct under
`moonshot/postrelease_structural_causal_core_decision_v17.json`. It supplies an
untied Phi-compatible execution boundary with RMSNorm, rotary full-MHA,
SwiGLU, strict UTF-8 realization, and persistent per-layer rotary key/value
state. The exact 14,654,784-parameter target geometry, signed package
lifecycle, first- and second-step incremental identity, CPU/CUDA package
identity, all 629 tests, and the unchanged sealed verifier pass. The preserved
V15 failure was a verifier-stimulus error: its EOS-first fixture could not grow
the cache; V16 changed only that stimulus and retained the failed receipt. The
v7 result is construct-only and imports no ABI extraction code or evidence.

The isolated `lc-direct-neural-core/8` progressive replacement host has a
passing generic construct receipt under
`results/moonshot/postrelease_progressive_replacement_core_construct_v18.json`,
with corrected runtime-vocabulary accounting under
`results/moonshot/postrelease_progressive_replacement_core_construct_repair_v21.json`.
It preserves a full-width residual stream while executing compact rotary
causal replacement cakes, declares zero source transformer blocks, retains
persistent per-replacement key/value state, and accepts only immutable signed
packages with no receiver learning. The initial 253,836,288 count incorrectly
included all 32,064 source weight rows; the deployable 32,011-action tokenizer
plus four host specials yields the corrected 253,535,232-parameter target.
The original receipt and reporter failure are preserved. Fixture CPU/CUDA
package identity and incremental/full-forward identity pass. This remains construct-only: no external artifact, English
quality, physical performance, or ABI acquisition claim is inherited.

The isolated `lc-direct-neural-core/9` source-aligned successor has a passing
construct receipt at
`results/moonshot/postrelease_source_aligned_progressive_construct_v23.json`.
It removes the injected BOS action between prompt and response so the first
response distribution is computed directly from the final external prompt
action. The v8 construct remains historical and valid for its declared
boundary, but external source-model conformance work must use v9. This is still
construct-only and inherits no acquisition, quality, or performance claim.

The final release does not claim physical mobile performance, calibrated energy
dominance, GPU-training dominance, latent neural fusion, or faster from-scratch
English acquisition. ABI extraction remains a separate repository and evidence
lineage.

This repository is governed by the LayerCake Moonshot Gated Research Campaign. These
instructions apply to every human, agent, script, experiment, verifier, and release task
performed from this directory.

## Objective

The campaign may declare the LayerCake moonshot proven only when one integrated lineage
establishes all of the following:

1. A useful general-English LayerCake core is independently validated and
   deployable without a source teacher at inference.
2. Its general quality is no worse than a strong transformer.
3. Its routed mixed-domain quality is statistically better than the transformer.
4. Ordinary LayerCake training remains available as a research and fallback
   capability; faster training is measured when studied but is not a product
   promotion gate.
5. Useful domain artifacts install and retain matched functional quality without
   retraining or mutating the English core.
6. LayerCake is substantially faster than an optimized transformer on the same CPU.
7. LayerCake is faster than an optimized transformer on the same GPU.
8. LayerCake on CPU matches or exceeds an optimized transformer on GPU.
9. Every speed comparison uses matched useful quality.
10. LayerCake retains persistent incremental state and does not recompute completed context.
11. Installed but inactive domains consume no proportional neural compute.
12. A core can be deployed alone.
13. A core can be deployed with one selected domain and no router overhead.
14. Any number of independently authored domains can be installed.
15. Domain authoring is driven by configuration and data, not source edits.
16. Every domain package is non-executable, signed, versioned, authenticated, and safely installable.
17. Identical package bytes transfer between compatible independently trained hosts.
18. Installation performs no receiver training, target-domain calibration, or cake mutation.
19. Portable domains retain useful functional capabilities across compatible hosts.
20. An orchestrator dynamically discovers installed domains and routes requests.
21. Routing supports core-only, top-1, top-k, multidomain, manual selection, and abstention.
22. Active compute remains approximately core plus selected cakes, not all installed cakes.
23. LayerCake operates across optimized CPU and GPU runtimes.
24. Physical ARM/mobile performance is claimed only after testing physical hardware.
25. Every promoted claim is independently reproducible from raw evidence.

An unsuccessful or resource-limited experiment does not weaken these requirements.

## Machine-enforced phase gates

The canonical state is `moonshot/campaign.yaml`. The phases are strictly ordered:

0. governance and campaign state;
1. benchmark truth;
2. matched-quality CPU speed;
3. retired training-efficiency controls and lineage handoff;
4. one useful lossless portable domain artifact;
5. generic multi-domain extensibility;
6. orchestration and catalog scalability;
7. integrated CPU, GPU, CPU-versus-GPU, and device performance;
8. independent hostile verification and release.

Only `python -m layercake.moonshot_campaign verify-phase N` may change a phase to `PASS`
or unlock the next phase. Work on a locked phase is forbidden. A task that cannot finish
the current phase reports `PHASE N STATUS: CONTINUATION REQUIRED`. A verified compute,
storage, data, or hardware limit reports `PHASE N STATUS: BLOCKED BY VERIFIED HARD LIMIT`;
that status does not unlock another phase.

Do not conceal unfinished work behind status text. Do not start the next phase until the
current phase is committed, tagged, independently rechecked, and marked `PASS` by the
campaign verifier.

## Integrated-lineage rule

All final evidence must descend from one explicit lineage:

```text
final architecture
-> final data
-> final source core
-> final receiver cores
-> final domains
-> final packages
-> final router
-> final runtimes
-> final benchmarks
-> final certificate
```

Quality from one checkpoint may not be combined with speed from another checkpoint unless
the comparison explicitly evaluates both and the verifier accepts the lineage. Artifact
identity is established by cryptographic hashes, not filenames, labels, or prose.

## Invalidation rule

A change to any of these invalidates every dependent certificate:

* core architecture;
* canonical ABI;
* patching contract;
* output contract;
* routing computation;
* fusion contract;
* training data;
* evaluation data;
* precision contract;
* runtime numerical behavior.

Performance-only kernel changes require numerical-equivalence, performance, and memory
recertification. A prior pass is never inherited merely because a component keeps its name.
Apply `moonshot/invalidation_matrix.yaml` fail-closed.

## Matched-quality rule

No inference-speed claim passes when LayerCake is materially worse in held-out BPB,
functional-task quality, instruction following, invalid-output rate, repetition, coherence,
or domain success. Speed and quality must come from the same accepted checkpoint lineage.

## Neural-generation rule

Retrieval, stored answers, templates, corpus chunks, prompt prefixes, and deterministic
transducers are separate baselines. They cannot satisfy neural generation or neural domain
gates and must be labeled wherever measured.

## Raw-evidence rule

The verifier derives every headline value from immutable raw evidence. It never trusts or
copies a manually edited status, hard-coded winning number, README claim, or prior
certificate summary. Preserve raw samples, commands, environment, ordering, failures, and
artifact hashes. Certificates are derived outputs, not sources of truth.

## Test-isolation rule

Search, validation, and final-test splits are immutable and content-addressed. The final
test split cannot influence architecture, thresholds, baseline selection, stopping, or
promotion. Test access and contamination checks must be auditable.

## Failure-preservation rule

Preserve failed candidates, failed seeds, invalid evidence, benchmark changes, negative
results, and verifier rejections. Never delete or overwrite an inconvenient run. Superseded
evidence remains labeled and addressable.

## Hard-limit rule

A hard limit requires machine-readable evidence of available hardware, memory, storage,
remaining compute allowance where knowable, the blocked command, and an estimated resource
requirement based on measured runs. A disappointing metric, slow experiment, or unoptimized
implementation is not a hard limit.

## Permanent evidence rules

* Use the frozen contracts under `moonshot/`; modify them only through a versioned governance
  change that invalidates affected phases.
* Record exact commands, seeds, checkpoint hashes, data hashes, source commit, software,
  hardware, precision, thread/device settings, failures, and raw observations.
* At least three independent seeds are required whenever the phase contract requires them.
  Missing or selectively omitted seeds fail the gate.
* An optimized transformer baseline cannot be an eager Python decode loop. Baseline strength
  is a measured, certified property.
* Semantic portability requires a non-empty, frozen source-host success set. Structural or
  mathematical identity alone cannot satisfy it.
* Phase 4 may use either the locked same-shape semantic-residual ABI or the locked
  direct neural-decoder ABI. The latter is valid only for an explicitly selected
  one-cake path and cannot claim fusion, top-k composition, routing, or multi-domain
  proof. Both modes require autonomous neural generation, exact signed transfer,
  receiver behavior retention, and the same CPU inference gates.
* Synthetic tasks or packages may be used only where the contract explicitly permits them;
  they never become real functional evidence through relabeling.
* Never edit a generated release certificate to change a result. Regenerate it from raw data.
* Never commit secrets, private package keys, restricted data, or executable cake payloads.

## Required commands

```bash
python -m layercake.moonshot_campaign status
python -m layercake.moonshot_campaign verify-phase 0
python -m layercake.moonshot_campaign verify-all
```

Legacy Moonshot artifacts pre-dating this campaign are retained as historical evidence and
negative controls. They do not pass a campaign phase unless the new verifier imports their
raw records, validates their lineage, and recomputes every applicable gate.

## Repository and responsibility boundary

ABI is a separate product, repository, campaign, runtime dependency graph, and
evidence lineage. ABI work is outside this LayerCake task. Do not clone, vendor,
create, import, execute, or modify ABI extraction code or evidence here.

LayerCake owns safe hosting, execution, installation, transfer, composition,
sparse activation, routing, and orchestration. A future ABI artifact may cross
into LayerCake only as an independently validated, signed, content-addressed,
non-executable package. Any future core, ABI, fusion, or runtime change invokes
the invalidation matrix and must recertify every dependent phase.

From-scratch LayerCake training remains available as bounded research and
fallback control, but faster training is not a product promotion gate and must
not consume additional major campaign compute.

## Phase 4 acquisition-device rule

Phase 4 capability acquisition, cake training, bridge fitting, conformance
training, and certification preparation are GPU-first when a compatible GPU is
available. CPU execution is the required fallback and must remain supported,
but CPU-only acquisition is not a promotion gate.

This device policy does not relax the production inference contract. The same
promoted cake payload must still pass the locked optimized-CPU throughput,
latency, TTFT, active-memory, incremental-state, package-identity, semantic
retention, and lossless-transfer gates. Training or evaluating a different
payload on CPU cannot supply those inference results.

Package transfer remains device-independent: identical authenticated package
bytes and tensor hashes must install without receiver learning on both CPU-only
and GPU-capable compatible hosts. GPU use may accelerate acquisition and
certification, but it may not introduce a GPU-only package format, private
device state, or inference dependency.

## Phase 4 canonical attachment modes

Phase 4 certifies one useful Python cake through one of two explicit,
content-addressed interfaces:

* `lc-semantic-gpt2-768/1` for a same-shape residual fused before the frozen
  English-core head; or
* `lc-direct-neural-decoder/1` for one explicitly selected neural cake that
  accepts and returns UTF-8 bytes while preserving persistent incremental
  neural state.

The direct mode is not a semantic-residual claim and must never be relabeled as
one. It is nevertheless a valid LayerCake host path when the installed package
is non-executable, signed, immutable, autonomously neural, lossless at the byte
boundary, device-independent, and behavior-identical across certified
receivers. It does not satisfy later multi-domain fusion or routing gates;
Phases 5 and 6 remain responsible for those proofs.

## Phase 6 orchestration and catalog rule

Phase 6 operates above the sealed English core and the exact sealed Python,
SQL, and regular-expression package bytes. It may add routing, catalog, and
orchestration code and content-addressed routing metadata. It may not modify,
retrain, calibrate, repackage, or borrow quality from any sealed core or cake.

Automatic activation requires both an eligible authenticated installed package
and a campaign-authenticated routing profile bound to that package's exact
archive hash. A missing or mismatched profile fails closed to manual selection
or core abstention. The router output set must be discovered from the current
catalog; a fixed compiled domain head cannot satisfy dynamic discovery.

The required modes are core-only, automatic top-1, automatic top-k, structured
multidomain orchestration, explicit manual selection, and abstention. Because
the promoted direct-decoder packages do not declare a latent fusion contract,
multidomain operation means routing and executing explicitly separable
subrequests and returning a deterministic structured envelope. It is not a
claim of simultaneous neural fusion or of one decoder generating another
decoder's domain output.

Catalog stress entries without promoted neural behavior are management-only
descriptors. They may prove lookup and refresh scaling, but they may never be
counted as real capabilities, installed executable cakes, domain quality, or
functional success. Only selected real cakes may receive module-load, prefill,
or decode calls.

The immutable Phase 6 preregistration is
`moonshot/phase6_orchestration_preregistration.json`.

## Phase 7 integrated-performance rule

Phase 7 benchmarks the exact sealed Phase 6 product lineage. The English core,
three signed neural cakes, direct-decoder ABI, router profiles, and orchestration
semantics are immutable. A CPU result from one payload may not be paired with a
GPU result from another payload, and speed may not be promoted without the
device-matched functional-quality result.

The locked systems are routed LayerCake on CPU FP32, the same routed LayerCake
packages on CUDA FP32, and Qwen 2.5 0.5B Q4_K_M through Ollama's deployment
runtime on CPU and with physically verified GPU residency. Cross-model primary
throughput is UTF-8 output bytes per complete request wall second. Transformer
tokens are recorded authoritatively but are not used to equate tokenizer units
with LayerCake actions.

The promoted matrix uses 100 distinct frozen functional prompts plus 20
repeated observations for every headline system. CPU observations may be
imported only from the exact sealed Phase 6 raw artifact and must be fully
revalidated. GPU observations are new, prompt/trial paired, and require at
least three host/order seeds. Cold timing uses one real streaming request after
an explicit unload control, never a model-load probe.

General English noninferiority comes only from typed revalidation of the sealed
Phase 2 core and its exact hashes. Mixed-domain superiority requires actual
functional execution against the same Qwen comparator on CPU and GPU. All 384
promoted held-out domain cases must execute successfully on GPU and exactly
match the sealed CPU outputs, with zero inactive neural forward calls.

Performance claims are limited to the declared laptop CPU, RTX 3080 Laptop
GPU, exact runtimes, precisions, and model digests. Phase 7 does not claim
mobile hardware, GPU training dominance, or latent neural cake fusion.

The immutable preregistration is
`moonshot/phase7_integrated_performance_preregistration.json`.

## Phase 8 independent hostile-verification rule

Phase 8 may not improve or alter the sealed product. It treats every Phase 0
through Phase 7 claim as untrusted until the raw evidence, artifact hashes,
required gates, commit/tag history, and current physical executions have been
independently recomputed.

The first detached Phase 7 preflight failed before benchmarking and is
preserved as negative evidence. It found a nonexistent packaging backend,
platform-dependent line endings in byte-locked contracts, and three undeclared
ignored test fixtures. The only permitted repair base changes are the corrected
setuptools backend and deterministic Git text/binary attributes; architecture,
runtime, ABI, models, checkpoints, cakes, router, data, and thresholds remain
unchanged.

The controlling reproduction runs from a detached clean worktree at the exact
annotated `layercake-moonshot-phase8-repair-base-v3` tag. Its six ignored external
assets (three promoted checkpoints and three historical-control test fixtures)
must be copied from the local content-addressed cache and rehashed against the
v2 preregistration. It must validate dependency resolution, run
the complete clean-checkout test suite, run the campaign-wide verifier, and
recheck every data, checkpoint, package, ABI, router, and runtime identity.
Fresh execution must repeat the 100-distinct-plus-20-repeat CPU/GPU matrix,
all 384 held-out cases per device, 100 core abstentions, the complete 1,980-row
routing suite, a 500-entry catalog, and install/verify/remove/reinstall on
three fresh hosts.

Hostile verification must attempt every attack category locked in the Phase 8
preregistration. A rejected attack is evidence only when the attack actually
reaches the intended boundary. No unresolved finding of any severity may be
waived to complete the campaign.

Phase 3's training-efficiency objective remains retired by governance.
Its sealed certificate has zero headline claims and must not be described as
training-efficiency, ABI-bootstrapping, capability-acquisition, or host-
certification proof. It is a valid campaign-lifecycle retirement, not a skipped
phase and not a scientific pass of the retired objective.
Phase 8 must preserve its historical controls and must not claim faster
foundation or domain training. Physical mobile, calibrated-energy dominance,
latent fusion, and third-party laboratory independence also remain outside
the final claim unless separately measured.

The immutable preregistration is
`moonshot/phase8_independent_verification_preregistration.json`.

## Post-release external English-core host audit

The read-only V2 audit under
`moonshot/postrelease_external_capability_host_interface_audit_repair1_v2.json`
identified a host-interface scope gap without changing the sealed release. The
semantic-residual interface is not certified for external autonomous English,
and the direct neural decoder is certified only as a selected capability cake.
Ad hoc hidden-state hooks are not a canonical host interface.

Any repair must be a separately versioned signed direct neural *core* artifact
interface using the existing byte-facing, self-causal package primitives. It
must not import external acquisition code or evidence, mutate the sealed
lineage, claim semantic fusion, or inherit quality/performance. Implementation
requires a new preregistration and complete affected-gate recertification.

That construct is now implemented as the isolated `lc-direct-neural-core/1`
extension and passes CPU/CUDA package, identity, lifecycle, persistent-state,
role, tamper, and zero-learning gates. It is not an English or performance
certificate. The only authorized next gate is acceptance and same-artifact
recertification of an independently validated external English artifact. Do
not reopen hidden-state bridge or semantic-residual sweeps.

The read-only UTF-8 action-atomicity audit is complete failed under
`moonshot/postrelease_direct_neural_core_utf8_audit_decision_v1.json`. V1 splits
multibyte characters into independently selectable invalid byte fragments and
does not validate output before returning it. Preserve the original construct
scope, but do not claim v1 UTF-8 conformance or relabel a v1 package. A
separately preregistered v2 extension may make actions Unicode-atomic and add
strict input/output validation without editing sealed model/runtime paths.

That v2 implementation is now frozen under
`moonshot/postrelease_unicode_direct_neural_core_implementation_v2.json`.
Execute only its deterministic construct. A pass remains construct-only and
does not authorize an old checkpoint, inherit English quality/performance, or
import ABI research into this repository.

The first v2 construct wrote its immutable evidence successfully, then CP1252
stdout failed on an emoji. Preserve that attempt. Repair V3 changes only stdout
to ASCII-escaped JSON and authorizes verification of the existing result; it
does not authorize overwriting or a new scientific execution.

V2 is now construct-certified under
`moonshot/postrelease_unicode_direct_neural_core_decision_v4.json`. It passes
Unicode-atomic action validity, strict input/output rejection, same-package
CPU/CUDA identity, lifecycle, persistent state, v1/role/tamper rejection, zero
learning, all 613 tests, and the unchanged Phase 0-8 verifier. The only next
gate is a real independently validated external artifact targeting v2. Do not
claim English quality, performance, or ABI acquisition before that same
artifact is certified.

The separately versioned `lc-direct-neural-core/3` BPE host construct is now
preregistered. It is generic LayerCake hosting work and imports no external
acquisition evidence or tokenizer. Execute only its construct gates; a pass may
open external-artifact conformance but inherits no English quality or speed.

That v3 construct now passes under
`moonshot/postrelease_bpe_direct_neural_core_decision_v8.json`: 616 tests, the
unchanged Phase 0-8 verifier, exact UTF-8 BPE concatenation, same-package CPU/
CUDA identity, persistent state, lifecycle, and zero learning. Its next gate is
one real independently validated external v3 artifact; it remains construct-
only and inherits no acquisition, English-quality, or performance result.

A separate ABI-side read-only audit identified a measured interface need:
exact teacher-native actions require a declarative tokenizer graph with
normalization, post-processing, sequence decoding, and byte fallback, which v3
correctly rejects. The isolated `lc-direct-neural-core/4` decoder-aware host now
passes its construct under
`moonshot/postrelease_decoder_direct_neural_core_decision_v10.json`. It shifts
every external token ID by the four reserved host actions, prohibits pointer
actions, passes the same-package CPU/CUDA, identity, lifecycle, persistent-
state, zero-learning, 619-test, and unchanged sealed-verifier gates. Its next
gate is one independently validated external v4 artifact. This remains
construct-only and imports no ABI evidence, English-quality, or performance
claim.
