# LayerCake Moonshot: Sealed Phase Status

This document is the human-readable companion to `moonshot/campaign.yaml`.
The campaign file, phase certificates, raw evidence, and Git tags are
authoritative; prose never promotes a claim by itself.

## Release identity

| Field | Value |
| --- | --- |
| Campaign | `layercake-moonshot` |
| Final status | `SEALED` |
| Final tag | `layercake-moonshot-final` |
| Final tag commit | `0537cbb9e93cd7ebd4ba01c0bf641414ecebb1c3` |
| Final release-evidence commit | `ebf594073c7364df9249e75ee9833861a7804992` |
| Phase 8 framework commit | `22501a3a4ef16c3cbb390aef565ec502391d8768` |
| Governing Phase 8 contract | `moonshot/phase8_independent_verification_preregistration.json` |

Run `C:\Python310\python.exe -m layercake.moonshot_campaign verify-all` from a
detached checkout of `layercake-moonshot-final` to recompute the sealed
campaign-level lifecycle state. A valid release reports
`completed_phases_valid: true`. Current post-release HEAD intentionally fails
because its package changes trigger `precision_contract` invalidation; see V96
and the successful exact-tag revalidation V99.

## Eight sealed phases

| Phase | Sealed result or disposition | Primary evidence | Seal tag |
| --- | --- | --- | --- |
| 0 | Permanent governance, contracts, invalidation rules, and campaign state | `results/moonshot/phase0/` | `layercake-moonshot-phase0` |
| 1 | Auditable benchmark truth with matched quality/speed accounting | `results/moonshot/phase1/` | `layercake-moonshot-phase1` |
| 2 | One integrated English-host lineage cleared the locked matched-quality CPU gate | `results/moonshot/phase2_recertification/` | `layercake-moonshot-phase2-r3` |
| 3 | Original training-efficiency objective retired by governance; bounded controls and negative evidence preserved | `results/moonshot/phase3/` | `layercake-moonshot-phase3` |
| 4 | A useful signed Python capability package transferred losslessly without receiver learning | `results/moonshot/phase4/` | `layercake-moonshot-phase4` |
| 5 | Three real capability packages established generic multi-domain extension | `results/moonshot/phase5/` | `layercake-moonshot-phase5` |
| 6 | Archive-bound routing, selected-only execution, multi-mode orchestration, and catalog scaling | `results/moonshot/phase6/` | `layercake-moonshot-phase6` |
| 7 | Locked CPU, GPU, and CPU-versus-GPU performance for the exact integrated product | `results/moonshot/phase7/` | `layercake-moonshot-phase7` |
| 8 | Detached clean-room reproduction, hostile falsification, and final release integrity | `results/moonshot/phase8/` | `layercake-moonshot-final` |

### Phase 3 is retired, not skipped

Phase 3 ran bounded training-efficiency experiments and preserved their positive
and negative evidence. Its final certificate has scope
`governance_retirement_no_training_efficiency_claim`, disposition
`RETIRED_BY_GOVERNANCE`, zero headline claims, and
`scientific_training_efficiency_passed: false`. The five promoted records prove
only that the retirement was authorized, historical evidence was preserved, no
scientific training pass was claimed, the Phase 2 parent was sealed, and a future
ABI artifact would require recertification.

Therefore Phase 3 is a valid sealed campaign transition, not a skipped lifecycle
step and not a scientific pass of faster training, English acquisition, ABI
bootstrapping, or host certification. Ordinary LayerCake training remains a
research and fallback capability.

## Final Phase 8 gate summary

The final certificate derives three required metrics from fresh raw evidence:

| Required gate | Result |
| --- | ---: |
| Prior required-gate retention | 1.0 |
| Clean-room reproduction | 1.0 |
| Adversarial findings resolved | 1.0 |

The detached verifier reran a 597-test clean checkout, all Phase 2-7 typed
gates, a 100-distinct-prompt plus 20-repeat four-system performance matrix,
384 held-out domain cases per device, 100 core abstentions, 1,980 routing rows,
a 500-entry catalog, and a three-host package lifecycle. It also executed 32
hostile checks spanning 24 categories.

The release certificate is
`results/moonshot/phase8/independent_verification_certificate.json`; the
human-readable report is `results/moonshot/phase8/release_report.md`.

## Evidence lineage and changes after release

The sealed phase tags are immutable historical boundaries. Documentation,
tooling, runtime, ABI, package, checkpoint, data, evaluation, or precision
changes made after a seal do not inherit the corresponding scientific claim.
Use `moonshot/invalidation_matrix.yaml` and open a governed recertification
before modifying a component that affects a sealed result.

Failed and invalidated Phase 8 attempts are deliberately preserved under
`results/moonshot/phase8/history/`. They are negative systems evidence, not
release evidence.
