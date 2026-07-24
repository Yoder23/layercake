"""Close the two-hypothesis Phase 2 grounding campaign without false promotion."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/moonshot/phase2_grounding_campaign"
TASK = ROOT / "results/moonshot/phase2_recertification/task_state.json"
LEDGER = ROOT / "results/moonshot/phase2_recertification/experiment_ledger.jsonl"
PHASE4_LOCK = ROOT / "moonshot/phase4_cpu_cake_training_lock.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_row(
    branch: str,
    hypothesis: str,
    checkpoint: Path,
    screen_path: Path,
    audit_path: Path,
    development_gates: dict,
    final_gates: dict,
) -> tuple[dict, dict]:
    screen = _read(screen_path)
    audit = _read(audit_path)
    metadata = _read(checkpoint / "metadata.json")
    tokenizer_document = _read(checkpoint / "tokenizer.json")
    aggregates = screen["aggregates"]
    semantic = audit["systems"]["layercake"]["aggregates"]
    checkpoint_sha = _sha(checkpoint / "model.safetensors")
    gates = {
        "validation_bpb": (
            screen["quality"]["validation"]["bits_per_byte"]
            <= development_gates["validation_bpb_max"]
        ),
        "topic_recall": (
            semantic["topic_token_recall"]
            >= development_gates["topic_recall_min"]
        ),
        "core_adherence": (
            semantic["core_adherence_pass"]
            >= development_gates["core_adherence_min"]
        ),
        "cpu_throughput": (
            screen["comparison"]["transformer_relative_median_decode_throughput"]
            >= development_gates["transformer_relative_cpu_throughput_min"]
        ),
        "ttft": (
            aggregates["median_time_to_first_output_seconds"]
            <= development_gates["transformer_ttft_seconds_max"]
        ),
        "no_repetition_collapse": (
            aggregates["repetition_rate"]
            <= development_gates["repetition_rate_max"]
        ),
        "memory_trend": (
            aggregates["median_resident_memory_bytes_after"]
            < final_gates["process_resident_memory_bytes_max_exclusive"]
        ),
    }
    row = {
        "branch": branch,
        "hypothesis": hypothesis,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "representation": {
            "class": metadata["representation"]["class"],
            "tokenizer_format": tokenizer_document.get(
                "format", "layercake-shared-bpe/1"
            ),
            "tokenizer_sha256": metadata["tokenizer"]["sha256"],
            "external_input": "UTF-8 bytes",
            "external_output": "UTF-8 bytes",
        },
        "validation_bpb": screen["quality"]["validation"]["bits_per_byte"],
        "topic_recall": semantic["topic_token_recall"],
        "topic_phrase_present_rate": semantic["topic_phrase_present"],
        "core_adherence_rate": semantic["core_adherence_pass"],
        "category_structure_rate": semantic["category_structure_pass"],
        "median_cpu_bytes_per_second": aggregates["median_bytes_per_second_decode"],
        "transformer_relative_cpu_throughput": screen["comparison"][
            "transformer_relative_median_decode_throughput"
        ],
        "median_ttft_seconds": aggregates["median_time_to_first_output_seconds"],
        "median_process_resident_memory_bytes": aggregates[
            "median_resident_memory_bytes_after"
        ],
        "repetition_rate": aggregates["repetition_rate"],
        "valid_utf8_rate": aggregates["valid_utf8"],
        "distinct_frozen_prompts": screen["distinct_prompts"],
        "development_gates": gates,
        "development_pass": all(gates.values()),
        "final_long_output_benchmark": (
            "NOT_RUN_BECAUSE_DEVELOPMENT_BEHAVIOR_GATES_FAILED"
        ),
        "systems_optimization_authorized": False,
        "systems_optimization_reason": (
            "The locked bounded optimization is available only after behavioral "
            "quality passes; this candidate failed topic/core adherence."
        ),
        "screen": {
            "path": screen_path.relative_to(ROOT).as_posix(),
            "sha256": _sha(screen_path),
            "payload_sha256": screen["screen_sha256"],
        },
        "semantic_audit": {
            "path": audit_path.relative_to(ROOT).as_posix(),
            "sha256": _sha(audit_path),
            "payload_sha256": audit["audit_sha256"],
        },
    }
    records = screen["records"]
    checks = {
        f"{branch}_100_distinct_prompts": (
            len(records) == 100
            and len({record["prompt_id"] for record in records}) == 100
        ),
        f"{branch}_checkpoint_bound": (
            checkpoint_sha
            == screen["checkpoint_sha256"]
            == audit["checkpoint_sha256"]
            == metadata["checkpoint"]["sha256"]
        ),
        f"{branch}_audit_binds_screen_file": (
            audit["screen_sha256"] == _sha(screen_path)
        ),
        f"{branch}_same_frozen_prompt_manifest": (
            audit["prompt_manifest_sha256"] == screen["prompt_manifest_sha256"]
        ),
        f"{branch}_autonomous_neural_decode": all(
            all(
                value == 0
                for value in record["execution"][
                    "external_path_counters"
                ].values()
            )
            and record["execution"][
                "maximum_active_experts_per_generated_token"
            ] == 1
            for record in records
        ),
        f"{branch}_output_hashes": all(
            hashlib.sha256(bytes.fromhex(record["generated_hex"])).hexdigest()
            == record["generated_sha256"]
            for record in records
        ),
        f"{branch}_no_test_access": (
            screen["test_accessed"] is False
            and screen["quality"]["test_accessed"] is False
        ),
        f"{branch}_failed_development": row["development_pass"] is False,
    }
    return row, checks


def main() -> int:
    development_gates = {
        "validation_bpb_max": 1.75,
        "topic_recall_min": 0.65,
        "core_adherence_min": 0.40,
        "transformer_relative_cpu_throughput_min": 1.75,
        "transformer_ttft_seconds_max": 0.0213021,
        "repetition_rate_max": 0.50,
    }
    final_gates = {
        "validation_bpb_max": 1.7174,
        "topic_recall_min": 0.82,
        "core_adherence_min": 0.55,
        "transformer_relative_cpu_throughput_128_min": 2.0,
        "transformer_relative_cpu_throughput_1024_min": 2.0,
        "transformer_ttft_seconds_max": 0.0213021,
        "process_resident_memory_bytes_max_exclusive": 214_990_848,
        "independent_seeds_min": 3,
    }
    definitions = [
        (
            "A_ENCODE_ONCE_STRUCTURED_MEMORY",
            "encode-once separately addressable role/token prompt memory",
            ROOT
            / "artifacts/moonshot/phase2_grounding_hypothesis_a"
            / "continuation10m-seed-9824",
            ROOT
            / "results/moonshot/phase2_grounding_hypothesis_a"
            / "functional_screen_640.json",
            ROOT
            / "results/moonshot/phase2_grounding_hypothesis_a"
            / "semantic_audit_640.json",
        ),
        (
            "B_WORD_BYTE_HYBRID_STRUCTURED_MEMORY",
            "whole-lexical-unit plus raw-byte fallback and persistent prompt memory",
            ROOT
            / "artifacts/moonshot/phase2_grounding_hypothesis_b"
            / "instruction10m-seed-9824",
            ROOT
            / "results/moonshot/phase2_grounding_hypothesis_b"
            / "functional_screen_640.json",
            ROOT
            / "results/moonshot/phase2_grounding_hypothesis_b"
            / "semantic_audit_640.json",
        ),
    ]
    rows = []
    checks: dict[str, bool] = {}
    for definition in definitions:
        row, row_checks = _candidate_row(
            *definition, development_gates, final_gates
        )
        rows.append(row)
        checks.update(row_checks)

    control_screen_path = (
        ROOT
        / "results/moonshot/phase2_multislot_prompt_state"
        / "functional_screen_seed9824_640.json"
    )
    control_audit_path = (
        ROOT
        / "results/moonshot/phase2_multislot_prompt_state"
        / "functional_semantic_audit_seed9824_640.json"
    )
    benchmark_128_path = (
        ROOT
        / "results/moonshot/phase2_multislot_prompt_state"
        / "benchmark_128x20.json"
    )
    benchmark_1024_path = (
        ROOT
        / "results/moonshot/phase2_multislot_prompt_state"
        / "benchmark_1024x20.json"
    )
    control_screen = _read(control_screen_path)
    control_audit = _read(control_audit_path)
    benchmark_128 = _read(benchmark_128_path)
    benchmark_1024 = _read(benchmark_1024_path)
    control_semantic = control_audit["systems"]["layercake"]["aggregates"]
    control = {
        "checkpoint_sha256": control_screen["checkpoint_sha256"],
        "validation_bpb": control_screen["quality"]["validation"]["bits_per_byte"],
        "topic_recall": control_semantic["topic_token_recall"],
        "core_adherence_rate": control_semantic["core_adherence_pass"],
        "cpu_128_transformer_relative": benchmark_128[
            "transformer_relative_median_decode_throughput"
        ],
        "cpu_1024_transformer_relative": benchmark_1024[
            "transformer_relative_median_decode_throughput"
        ],
        "ttft_1024_seconds": benchmark_1024["summary"][
            "time_to_first_output_seconds"
        ]["median"],
        "rss_1024_bytes": benchmark_1024["summary"][
            "resident_memory_bytes_after"
        ]["median"],
        "repetitions_each_headline_configuration": 20,
        "screen": {
            "path": control_screen_path.relative_to(ROOT).as_posix(),
            "sha256": _sha(control_screen_path),
        },
        "semantic_audit": {
            "path": control_audit_path.relative_to(ROOT).as_posix(),
            "sha256": _sha(control_audit_path),
        },
        "benchmark_128": {
            "path": benchmark_128_path.relative_to(ROOT).as_posix(),
            "sha256": _sha(benchmark_128_path),
        },
        "benchmark_1024": {
            "path": benchmark_1024_path.relative_to(ROOT).as_posix(),
            "sha256": _sha(benchmark_1024_path),
        },
    }
    diagnosis_paths = [
        ROOT / "results/moonshot/phase2_grounding_diagnosis" / name
        for name in (
            "prompt_probe_results.json",
            "state_drift.json",
            "entity_retention.json",
            "constraint_retention.json",
            "logit_contribution.json",
            "memory_profile.json",
            "diagnosis.md",
        )
    ]
    diagnosis = {
        path.name: {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha(path),
        }
        for path in diagnosis_paths
    }
    memory = _read(diagnosis_paths[-2])
    checks.update({
        "same_frozen_prompt_manifest": (
            len({
                _read(definition[3])["prompt_manifest_sha256"]
                for definition in definitions
            }) == 1
        ),
        "exactly_two_materially_distinct_hypotheses": len(rows) == 2,
        "both_hypotheses_failed_development": all(
            row["development_pass"] is False for row in rows
        ),
        "no_ineligible_systems_optimization": all(
            row["systems_optimization_authorized"] is False for row in rows
        ),
        "runtime_memory_bottleneck_physically_measured": (
            memory["components_rss_bytes"]["torch_and_native_runtime"]
            > final_gates["process_resident_memory_bytes_max_exclusive"]
        ),
        "phase4_lock_file_hash": (
            _sha(PHASE4_LOCK)
            == "2cdbe292fc0126aed898495dda4481a216211e111b6e24d0ba6a3098059410d2"
        ),
        "phase4_lock_status": (
            _read(PHASE4_LOCK)["status"] == "LOCKED_PENDING_PHASE3"
        ),
    })
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError("adversarial checks failed: " + ", ".join(failures))

    decision = {
        "format": "layercake-phase2-grounding-campaign-decision/1",
        "status": "CONTINUATION_REQUIRED",
        "locked_development_gates": development_gates,
        "locked_final_gates": final_gates,
        "speed_and_distributional_quality_control": control,
        "hypothesis_rows": rows,
        "winner": None,
        "selected_architecture": None,
        "three_seed_promotion_authorized": False,
        "phase2_status": "OPEN_RESEARCH_DIRECTION_REQUIRED",
        "phase2_r3_tag_created": False,
        "phase3_status": "LOCKED",
        "phase3_execution_authorized": False,
        "test_accessed": False,
        "reason": (
            "Hypothesis A failed topic recall, core adherence, CPU development "
            "throughput, and RSS. Hypothesis B recovered the >=2x CPU frontier "
            "but failed final BPB, topic recall, core adherence, UTF-8 product "
            "surface, and RSS. Neither behavior-passing candidate exists for "
            "the bounded profiler-directed runtime optimization or three-seed "
            "promotion, so Phase 3 remains scientifically locked."
        ),
        "diagnosis": {
            "prompt_state_overwrite_detected": False,
            "topic_phrase_retention": 0.0,
            "entity_retention": 0.0,
            "mean_prompt_ablation_js_nats": 0.01304088,
            "fresh_process_peak_rss_bytes": memory["stages_rss_bytes"][
                "decode_peak"
            ],
            "torch_and_native_runtime_rss_bytes": memory[
                "components_rss_bytes"
            ]["torch_and_native_runtime"],
            "model_and_tokenizer_rss_bytes": memory[
                "components_rss_bytes"
            ]["model_and_tokenizer"],
            "falsified_mechanism": (
                "The fixed prompt state remains intact, but fused prompt "
                "features have weak causal influence on later logits and local "
                "generation dynamics dominate."
            ),
            "runtime_consequence": (
                "A minimal native runtime is mandatory once a lineage first "
                "passes behavioral quality; Python/Torch overhead alone exceeds "
                "the absolute RSS gate."
            ),
            "evidence": diagnosis,
        },
        "historical_representation_label_note": (
            "The immutable Hypothesis B functional screen records the legacy "
            "generic label 'shared_tokenizer'. Its bound tokenizer and metadata "
            "identify layercake-word-byte-hybrid/1. Raw evidence was preserved "
            "rather than silently rewritten."
        ),
        "phase4_cpu_cake_training_contract": {
            "path": PHASE4_LOCK.relative_to(ROOT).as_posix(),
            "sha256": _sha(PHASE4_LOCK),
            "status": "LOCKED_PENDING_PHASE3",
        },
    }
    decision["decision_sha256"] = _canonical_sha(decision)
    decision_path = OUT / "branch_decision.json"
    _write(decision_path, decision)
    verifier = {
        "format": "layercake-phase2-grounding-adversarial-verifier/1",
        "status": "PASS",
        "decision": decision_path.relative_to(ROOT).as_posix(),
        "decision_file_sha256": _sha(decision_path),
        "decision_payload_sha256": decision["decision_sha256"],
        "checks": checks,
        "failures": [],
    }
    verifier["verifier_sha256"] = _canonical_sha(verifier)
    verifier_path = OUT / "adversarial_verifier.json"
    _write(verifier_path, verifier)

    task = _read(TASK)
    label = (
        "two-hypothesis grounding campaign closed "
        f"{decision['decision_sha256'][:12]}"
    )
    if label not in task["completed_batches"]:
        task["completed_batches"].append(label)
    task.update({
        "active_candidate": None,
        "current_stage": "PHASE2_GROUNDING_TWO_HYPOTHESES_REFUTED",
        "latest_batch": decision_path.relative_to(ROOT).as_posix(),
        "latest_batch_sha256": _sha(decision_path),
        "phase2_status": "OPEN_RESEARCH_DIRECTION_REQUIRED",
        "phase3_status": "LOCKED",
        "phase2_r3_tag_created": False,
        "representation_winner": None,
        "continuation_command": (
            "C:\\Python310\\python.exe "
            "scripts\\verify_phase2_grounding_campaign.py"
        ),
        "continuation_prompt": decision_path.relative_to(ROOT).as_posix(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    task["representation_branches"].update({
        "encode_once_structured_prompt_memory": "CLOSED_NEGATIVE",
        "word_byte_hybrid_structured_memory": "CLOSED_NEGATIVE",
    })
    task["remaining_gates"] = [
        "authorize one new materially distinct falsifiable Phase 2 architecture",
        "one checkpoint passes BPB and autonomous semantic behavior",
        "one checkpoint passes 128-byte and 1024-byte CPU throughput >=2x",
        "one checkpoint passes absolute RSS in a minimal native runtime",
        "canonical representation-agnostic cake ABI",
        "three-seed Phase 2 r3 recertification",
        "Phase 3 CPU-only matched-quality training-speed proof",
    ]
    _write(TASK, task)
    ledger_event = {
        "event": "phase2_grounding_two_hypotheses_closed",
        "status": "NEGATIVE_EVIDENCE_PRESERVED",
        "decision": decision_path.relative_to(ROOT).as_posix(),
        "decision_sha256": decision["decision_sha256"],
        "verifier": verifier_path.relative_to(ROOT).as_posix(),
        "verifier_sha256": verifier["verifier_sha256"],
        "winner": None,
        "phase2_status": task["phase2_status"],
        "phase3_status": task["phase3_status"],
        "phase4_contract_sha256": _sha(PHASE4_LOCK),
        "test_accessed": False,
    }
    ledger_lines = [
        line
        for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("event")
        != "phase2_grounding_two_hypotheses_closed"
    ]
    ledger_lines.append(json.dumps(ledger_event, sort_keys=True))
    LEDGER.write_text(
        "\n".join(ledger_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": decision["status"],
        "decision": decision_path.relative_to(ROOT).as_posix(),
        "decision_sha256": decision["decision_sha256"],
        "verifier": verifier_path.relative_to(ROOT).as_posix(),
        "verifier_sha256": verifier["verifier_sha256"],
        "phase2_status": task["phase2_status"],
        "phase3_status": task["phase3_status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
