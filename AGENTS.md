# LayerCake Moonshot Gated Research Campaign Charter

This repository is governed by the LayerCake Moonshot Gated Research Campaign. These
instructions apply to every human, agent, script, experiment, verifier, and release task
performed from this directory.

## Objective

The campaign may declare the LayerCake moonshot proven only when one integrated lineage
establishes all of the following:

1. A useful general-English LayerCake core is trained from scratch.
2. Its general quality is no worse than a strong transformer.
3. Its routed mixed-domain quality is statistically better than the transformer.
4. LayerCake reaches matched general quality faster during foundation training.
5. A LayerCake domain reaches matched functional quality faster than transformer adaptation.
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
3. foundation-training speed;
4. one useful lossless portable domain;
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
