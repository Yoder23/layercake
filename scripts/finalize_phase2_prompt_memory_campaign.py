"""Seal the bounded Phase 2 prompt-memory campaign without promoting failures."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/moonshot/phase2_prompt_memory"
TASK = ROOT / "results/moonshot/phase2_recertification/task_state.json"
LEDGER = ROOT / "results/moonshot/phase2_recertification/experiment_ledger.jsonl"


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


def main() -> int:
    definitions = {
        "A_FIXED_SEMANTIC_SLOTS": (
            "candidate_a_10m",
            ROOT / "artifacts/moonshot/phase2_prompt_memory/candidate_a_10m/seed-9824",
        ),
        "B_HIERARCHICAL_CHUNK_SUMMARIES": (
            "candidate_b_10m",
            ROOT / "artifacts/moonshot/phase2_prompt_memory/candidate_b_10m/seed-9824",
        ),
        "C_COADAPTED_TOKEN_EFFICIENT_CORE": (
            "candidate_c_10m",
            ROOT / "artifacts/moonshot/phase2_prompt_memory/candidate_c_10m/seed-9824",
        ),
    }
    continuation = {
        "validation_bpb_max": 1.75,
        "topic_token_recall_min": 0.65,
        "core_adherence_pass_rate_min": 0.40,
        "transformer_relative_cpu_throughput_min": 1.75,
        "repetition_rate_collapse_max": 0.50,
        "process_resident_memory_trend_max_bytes": 394_829_824,
    }
    final = {
        "validation_bpb_max": 1.7174,
        "topic_token_recall_min": 0.82,
        "core_adherence_pass_rate_min": 0.55,
        "transformer_relative_cpu_throughput_128_min": 2.0,
        "transformer_relative_cpu_throughput_1024_min": 2.0,
        "process_resident_memory_bytes_max_exclusive": 214_990_848,
        "independent_seeds_min": 3,
    }
    rows = []
    prompt_manifests = set()
    checks = {}
    for branch, (stem, checkpoint) in definitions.items():
        screen_path = OUT / f"{stem}_screen.json"
        audit_path = OUT / f"{stem}_semantic_audit.json"
        screen = _read(screen_path)
        audit = _read(audit_path)
        metadata = _read(checkpoint / "metadata.json")
        semantic = audit["systems"]["layercake"]["aggregates"]
        aggregate = screen["aggregates"]
        prompt_manifests.add(screen["prompt_manifest_sha256"])
        gates = {
            "validation_bpb": (
                screen["quality"]["validation"]["bits_per_byte"]
                <= continuation["validation_bpb_max"]
            ),
            "topic_token_recall": (
                semantic["topic_token_recall"]
                >= continuation["topic_token_recall_min"]
            ),
            "core_adherence": (
                semantic["core_adherence_pass"]
                >= continuation["core_adherence_pass_rate_min"]
            ),
            "cpu_throughput": (
                screen["comparison"][
                    "transformer_relative_median_decode_throughput"
                ]
                >= continuation["transformer_relative_cpu_throughput_min"]
            ),
            "no_repetition_collapse": (
                aggregate["repetition_rate"]
                <= continuation["repetition_rate_collapse_max"]
            ),
            "memory_trend": (
                aggregate["median_resident_memory_bytes_after"]
                <= continuation[
                    "process_resident_memory_trend_max_bytes"
                ]
            ),
        }
        checkpoint_sha = _sha(checkpoint / "model.safetensors")
        row = {
            "branch": branch,
            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "validation_bpb": screen["quality"]["validation"][
                "bits_per_byte"
            ],
            "topic_token_recall": semantic["topic_token_recall"],
            "core_adherence_pass_rate": semantic["core_adherence_pass"],
            "topic_phrase_present_rate": semantic["topic_phrase_present"],
            "category_structure_pass_rate": semantic[
                "category_structure_pass"
            ],
            "repetition_rate": aggregate["repetition_rate"],
            "word_diversity": aggregate["word_diversity"],
            "valid_utf8_rate": aggregate["valid_utf8"],
            "median_cpu_bytes_per_second": aggregate[
                "median_bytes_per_second_decode"
            ],
            "transformer_relative_cpu_throughput": screen["comparison"][
                "transformer_relative_median_decode_throughput"
            ],
            "median_time_to_first_output_seconds": aggregate[
                "median_time_to_first_output_seconds"
            ],
            "median_process_resident_memory_bytes": aggregate[
                "median_resident_memory_bytes_after"
            ],
            "distinct_frozen_prompts": screen["distinct_prompts"],
            "continuation_gates": gates,
            "continuation_pass": all(gates.values()),
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
        rows.append(row)
        records = screen["records"]
        checks[f"{branch}_100_distinct_prompts"] = (
            len(records) == 100
            and len({record["prompt_id"] for record in records}) == 100
        )
        checks[f"{branch}_checkpoint_bound"] = (
            checkpoint_sha == screen["checkpoint_sha256"]
            == audit["checkpoint_sha256"]
            == metadata["checkpoint"]["sha256"]
        )
        checks[f"{branch}_audit_bound"] = (
            audit["screen_sha256"] == _sha(screen_path)
        )
        checks[f"{branch}_neural_autonomous_decode"] = all(
            all(value == 0 for value in record["execution"][
                "external_path_counters"
            ].values())
            and record["execution"][
                "maximum_active_experts_per_generated_token"
            ] == 1
            for record in records
        )
        checks[f"{branch}_output_hashes"] = all(
            hashlib.sha256(bytes.fromhex(record["generated_hex"])).hexdigest()
            == record["generated_sha256"]
            for record in records
        )
        checks[f"{branch}_no_test_access"] = (
            screen["test_accessed"] is False
            and screen["quality"]["test_accessed"] is False
        )

    memory_path = OUT / "candidate_c_memory_profile.json"
    memory = _read(memory_path)
    curriculum_manifest_path = (
        ROOT
        / "data/moonshot/phase2/instruction_curriculum_prompt_memory_v1.manifest.json"
    )
    curriculum_manifest = _read(curriculum_manifest_path)
    phase3_lock = _read(ROOT / "moonshot/phase3_training_efficiency_lock.json")
    checks.update({
        "same_frozen_prompt_manifest": len(prompt_manifests) == 1,
        "all_candidates_failed_continuation": not any(
            row["continuation_pass"] for row in rows
        ),
        "candidate_c_memory_profile_bound": (
            memory["checkpoint_sha256"] == rows[-1]["checkpoint_sha256"]
        ),
        "candidate_c_memory_state_fixed": (
            memory["active_state"]["prompt_memory_shapes"][0] == [1, 128]
            and memory["active_state"]["prompt_memory_shapes"][3]
            == [1, 16, 288]
        ),
        "runtime_memory_bottleneck_measured": (
            memory["components_rss_bytes"]["torch_and_native_runtime"]
            > final["process_resident_memory_bytes_max_exclusive"]
        ),
        "curriculum_disjoint_and_verified": (
            curriculum_manifest["status"] == "PASS"
            and curriculum_manifest["exact_phase1_prompt_overlap"] == 0
            and curriculum_manifest["frozen_answers_used"] is False
        ),
        "phase3_lock_intact": (
            phase3_lock["status"] == "LOCKED_PENDING_PHASE2_R3"
        ),
    })
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError("adversarial checks failed: " + ", ".join(failures))

    decision = {
        "format": "layercake-phase2-prompt-memory-decision/1",
        "status": "CONTINUATION_REQUIRED",
        "campaign": (
            "bounded semantic memory / hierarchical summary / co-adapted "
            "token-efficient core"
        ),
        "locked_continuation_gates": continuation,
        "locked_final_gates": final,
        "pareto_rows": rows,
        "winner": None,
        "three_seed_promotion_authorized": False,
        "phase2_status": "OPEN_RESEARCH_DIRECTION_REQUIRED",
        "phase2_r3_tag_created": False,
        "phase3_status": "LOCKED",
        "phase3_execution_authorized": False,
        "test_accessed": False,
        "reason": (
            "A and B failed prompt adherence; C learned fluent non-collapsed "
            "instruction form but failed topic/core adherence, the 1.75x "
            "continuation speed floor, and the memory trend gate. No candidate "
            "may be promoted or advanced to Phase 3."
        ),
        "measured_limiting_factors": {
            "quality": (
                "unseen-topic binding remains absent despite low held-out "
                "instruction loss"
            ),
            "speed": (
                "best observed frozen-suite median is 1.626x the optimized "
                "transformer, below both continuation and final gates"
            ),
            "memory": {
                "fresh_process_peak_rss_bytes": memory[
                    "stages_rss_bytes"
                ]["decode_peak"],
                "torch_and_native_runtime_rss_bytes": memory[
                    "components_rss_bytes"
                ]["torch_and_native_runtime"],
                "model_and_tokenizer_rss_bytes": memory[
                    "components_rss_bytes"
                ]["model_and_tokenizer"],
                "conclusion": (
                    "the Python/Torch runtime alone exceeds the locked "
                    "transformer RSS; a native deployment runtime is required"
                ),
            },
        },
        "evidence": {
            "memory_profile": {
                "path": memory_path.relative_to(ROOT).as_posix(),
                "sha256": _sha(memory_path),
            },
            "curriculum_manifest": {
                "path": curriculum_manifest_path.relative_to(ROOT).as_posix(),
                "sha256": _sha(curriculum_manifest_path),
            },
        },
    }
    decision["decision_sha256"] = _canonical_sha(decision)
    decision_path = OUT / "branch_decision.json"
    _write(decision_path, decision)
    verifier = {
        "format": "layercake-phase2-prompt-memory-adversarial-verifier/1",
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
    label = f"bounded prompt-memory campaign closed {decision['decision_sha256'][:12]}"
    if label not in task["completed_batches"]:
        task["completed_batches"].append(label)
    task.update({
        "active_candidate": None,
        "current_stage": "PHASE2_PROMPT_MEMORY_CAMPAIGN_REFUTED",
        "latest_batch": decision_path.relative_to(ROOT).as_posix(),
        "latest_batch_sha256": _sha(decision_path),
        "phase2_status": "OPEN_RESEARCH_DIRECTION_REQUIRED",
        "phase3_status": "LOCKED",
        "phase2_r3_tag_created": False,
        "representation_winner": None,
        "continuation_command": (
            "C:\\Python310\\python.exe "
            "scripts\\verify_phase2_prompt_memory_campaign.py"
        ),
        "continuation_prompt": (
            "results/moonshot/phase2_prompt_memory/branch_decision.json"
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    task["representation_branches"].update({
        "fixed_semantic_memory": "CLOSED_NEGATIVE",
        "hierarchical_prompt_memory": "CLOSED_NEGATIVE",
        "coadapted_token_efficient_core": "CLOSED_NEGATIVE",
    })
    _write(TASK, task)
    ledger_event = {
        "event": "bounded_prompt_memory_campaign_closed",
        "status": "NEGATIVE_EVIDENCE_PRESERVED",
        "decision": decision_path.relative_to(ROOT).as_posix(),
        "decision_sha256": decision["decision_sha256"],
        "verifier": verifier_path.relative_to(ROOT).as_posix(),
        "verifier_sha256": verifier["verifier_sha256"],
        "winner": None,
        "phase2_status": task["phase2_status"],
        "phase3_status": task["phase3_status"],
        "test_accessed": False,
    }
    ledger_text = LEDGER.read_text(encoding="utf-8")
    if decision["decision_sha256"] not in ledger_text:
        with LEDGER.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(ledger_event, sort_keys=True) + "\n")
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
