from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _get(row: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = row
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _all_gates_pass(row: dict[str, Any]) -> bool:
    gates = row.get("gates", {})
    return bool(gates) and all(bool(value) for value in gates.values())


def _status_pass(row: dict[str, Any]) -> bool:
    return row.get("status") == "PASS"


def _ratio(row: dict[str, Any], key: str) -> float:
    return float(_get(row, f"ratios.{key}", 0.0))


def _metric(row: dict[str, Any], path: str, default: Any = 0.0) -> Any:
    return _get(row, f"metrics.{path}", default)


def _gate(row: dict[str, Any], name: str) -> bool:
    return bool(_get(row, f"gates.{name}", False))


def _failed(gates: dict[str, bool]) -> list[str]:
    return [name for name, passed in gates.items() if not passed]


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1e-12)


def verify(
    *,
    source_certificate: dict[str, Any],
    transfer_certificate: dict[str, Any],
    instruction_generalization_certificate: dict[str, Any],
    portable_mixed_certificate: dict[str, Any],
    conflicting_isolation_certificate: dict[str, Any],
    ood_abstention_certificate: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source = source_certificate
    transfer = transfer_certificate
    instruction = instruction_generalization_certificate
    mixed = portable_mixed_certificate
    conflict = conflicting_isolation_certificate
    abstain = ood_abstention_certificate
    source_lc_params = float(_metric(source, "layercake.params", 0.0))
    source_tx_params = float(_metric(source, "transformer.params", 0.0))
    source_lc_train_seconds = float(_metric(source, "layercake.train_seconds", 0.0))
    source_tx_train_seconds = float(_metric(source, "transformer.train_seconds", 0.0))
    source_lc_train_bytes = float(_metric(source, "layercake.train_bytes", 0.0))
    source_tx_train_bytes = float(_metric(source, "transformer.train_bytes", 0.0))
    source_train_cost_ratio = _safe_ratio(
        source_tx_params * source_tx_train_seconds,
        source_lc_params * source_lc_train_seconds,
    )
    source_train_byte_ratio = _safe_ratio(source_tx_train_bytes, source_lc_train_bytes)
    source_cpu_generation_cost_ratio = (
        _ratio(source, "parameter_ratio_transformer_over_layercake")
        * _ratio(source, "cpu_generation_speed_ratio")
    )
    source_gpu_generation_cost_ratio = (
        _ratio(source, "parameter_ratio_transformer_over_layercake")
        * _ratio(source, "gpu_generation_speed_ratio")
    )

    gates = {
        "source_certificate_pass": _status_pass(source),
        "source_all_gates_pass": _all_gates_pass(source),
        "source_transformer_at_least_5x_params": (
            _ratio(source, "parameter_ratio_transformer_over_layercake")
            >= args.min_param_ratio
        ),
        "source_bpb_noninferior": _gate(source, "bpb_non_inferior"),
        "source_training_speed_met": (
            _ratio(source, "training_speed_ratio")
            >= args.min_training_speed_ratio
        ),
        "source_no_more_training_bytes": _gate(source, "no_more_training_bytes"),
        "source_train_cost_proxy_met": (
            source_train_cost_ratio >= args.min_training_cost_ratio
        ),
        "source_train_byte_efficiency_met": (
            source_train_byte_ratio >= args.min_training_byte_ratio
        ),
        "source_cpu_generation_5x_met": (
            _ratio(source, "cpu_generation_speed_ratio")
            >= args.min_source_cpu_generation_speed_ratio
        ),
        "source_gpu_generation_noninferior": (
            _ratio(source, "gpu_generation_speed_ratio")
            >= args.min_source_gpu_generation_speed_ratio
        ),
        "source_cpu_quality_noninferior": (
            _ratio(source, "cpu_quality_ratio") >= args.min_quality_ratio
        ),
        "source_gpu_quality_noninferior": (
            _ratio(source, "gpu_quality_ratio") >= args.min_quality_ratio
        ),
        "source_cpu_generation_cost_proxy_met": (
            source_cpu_generation_cost_ratio >= args.min_generation_cost_ratio
        ),
        "source_gpu_generation_cost_proxy_met": (
            source_gpu_generation_cost_ratio >= args.min_gpu_generation_cost_ratio
        ),
        "transfer_certificate_pass": _status_pass(transfer),
        "transfer_all_gates_pass": _all_gates_pass(transfer),
        "transfer_receiver_inherits_cpu_win": _gate(
            transfer, "receiver_inherits_cpu_generation_win"
        ),
        "transfer_receiver_inherits_gpu_win": _gate(
            transfer, "receiver_inherits_gpu_generation_win"
        ),
        "transfer_receiver_inherits_training_win": _gate(
            transfer, "receiver_inherits_training_win"
        ),
        "transfer_receiver_inherits_quality_win": _gate(
            transfer, "receiver_inherits_quality_win"
        ),
        "transfer_ppl_exact": _metric(transfer, "transfer_ppl_ratio", None) == 1.0,
        "transfer_logit_exact": _metric(transfer, "transfer_max_logit_diff", None)
        == 0.0,
        "transfer_abi_exact": _metric(transfer, "transfer_max_abi_diff", None) == 0.0,
        "transfer_generation_exact": _metric(
            transfer, "transfer_generation_exact", False
        )
        is True,
        "instruction_generalization_pass": _status_pass(instruction),
        "instruction_all_gates_pass": _all_gates_pass(instruction),
        "instruction_cpu_generation_5x_met": (
            _ratio(instruction, "cpu_generation_speed_ratio")
            >= args.min_domain_cpu_generation_speed_ratio
        ),
        "instruction_gpu_generation_noninferior": (
            _ratio(instruction, "gpu_generation_speed_ratio")
            >= args.min_domain_gpu_generation_speed_ratio
        ),
        "instruction_cpu_quality_noninferior": (
            _ratio(instruction, "cpu_quality_ratio") >= args.min_quality_ratio
        ),
        "instruction_gpu_quality_noninferior": (
            _ratio(instruction, "gpu_quality_ratio") >= args.min_quality_ratio
        ),
        "instruction_cpu_relevance_noninferior": (
            _ratio(instruction, "cpu_relevance_ratio") >= args.min_relevance_ratio
        ),
        "instruction_gpu_relevance_noninferior": (
            _ratio(instruction, "gpu_relevance_ratio") >= args.min_relevance_ratio
        ),
        "instruction_exact_and_paraphrase_full": (
            _gate(instruction, "cpu_layercake_exact_relevance_full")
            and _gate(instruction, "cpu_layercake_paraphrase_relevance_full")
            and _gate(instruction, "gpu_layercake_exact_relevance_full")
            and _gate(instruction, "gpu_layercake_paraphrase_relevance_full")
        ),
        "portable_mixed_pass": _status_pass(mixed),
        "portable_mixed_all_gates_pass": _all_gates_pass(mixed),
        "portable_mixed_cpu_generation_5x_met": (
            _ratio(mixed, "cpu_generation_speed_ratio")
            >= args.min_domain_cpu_generation_speed_ratio
        ),
        "portable_mixed_gpu_generation_noninferior": (
            _ratio(mixed, "gpu_generation_speed_ratio")
            >= args.min_domain_gpu_generation_speed_ratio
        ),
        "portable_mixed_quality_noninferior": (
            _ratio(mixed, "cpu_quality_ratio") >= args.min_quality_ratio
            and _ratio(mixed, "gpu_quality_ratio") >= args.min_quality_ratio
        ),
        "portable_mixed_relevance_noninferior": (
            _ratio(mixed, "cpu_relevance_ratio") >= args.min_relevance_ratio
            and _ratio(mixed, "gpu_relevance_ratio") >= args.min_relevance_ratio
        ),
        "portable_mixed_memory_full": (
            _metric(mixed, "layercake_cpu_generation.portable_memory_match_rate")
            >= args.min_portable_memory_match
            and _metric(mixed, "layercake_gpu_generation.portable_memory_match_rate")
            >= args.min_portable_memory_match
        ),
        "conflicting_isolation_pass": _status_pass(conflict),
        "conflicting_isolation_all_gates_pass": _all_gates_pass(conflict),
        "conflicting_cpu_generation_5x_met": (
            _ratio(conflict, "cpu_generation_speed_ratio")
            >= args.min_domain_cpu_generation_speed_ratio
        ),
        "conflicting_gpu_generation_noninferior": (
            _ratio(conflict, "gpu_generation_speed_ratio")
            >= args.min_domain_gpu_generation_speed_ratio
        ),
        "conflicting_quality_noninferior": (
            _ratio(conflict, "cpu_quality_ratio") >= args.min_quality_ratio
            and _ratio(conflict, "gpu_quality_ratio") >= args.min_quality_ratio
        ),
        "conflicting_relevance_noninferior": (
            _ratio(conflict, "cpu_relevance_ratio") >= args.min_relevance_ratio
            and _ratio(conflict, "gpu_relevance_ratio") >= args.min_relevance_ratio
        ),
        "conflicting_no_forbidden": (
            _gate(conflict, "cpu_samples_no_forbidden")
            and _gate(conflict, "gpu_samples_no_forbidden")
        ),
        "ood_abstention_pass": _status_pass(abstain),
        "ood_abstention_all_gates_pass": _all_gates_pass(abstain),
        "ood_required_prompts_present": (
            int(_metric(abstain, "layercake_cpu_generation.abstention_required_count", 0))
            > 0
            and int(
                _metric(abstain, "layercake_gpu_generation.abstention_required_count", 0)
            )
            > 0
        ),
        "ood_abstentions_all_pass": (
            _metric(abstain, "layercake_cpu_generation.samples_abstentions_pass", False)
            is True
            and _metric(
                abstain, "layercake_gpu_generation.samples_abstentions_pass", False
            )
            is True
        ),
        "ood_in_domain_memory_full": (
            _metric(
                abstain,
                "layercake_cpu_generation.portable_memory_match_rate_effective",
            )
            >= args.min_portable_memory_match
            and _metric(
                abstain,
                "layercake_gpu_generation.portable_memory_match_rate_effective",
            )
            >= args.min_portable_memory_match
        ),
        "ood_cpu_generation_5x_met": (
            _ratio(abstain, "cpu_generation_speed_ratio")
            >= args.min_domain_cpu_generation_speed_ratio
        ),
        "ood_gpu_generation_noninferior": (
            _ratio(abstain, "gpu_generation_speed_ratio")
            >= args.min_domain_gpu_generation_speed_ratio
        ),
        "ood_quality_noninferior": (
            _ratio(abstain, "cpu_quality_ratio") >= args.min_quality_ratio
            and _ratio(abstain, "gpu_quality_ratio") >= args.min_quality_ratio
        ),
    }
    failed = _failed(gates)
    result = {
        "status": "PASS" if not failed else "FAIL",
        "scope": (
            "CPU/GPU platform-slice dominance gate for the current 1M-vs-5M "
            "LayerCake branch. This aggregates trained source dominance, exact "
            "receiver transfer, instruction paraphrase generalization, portable "
            "mixed-domain serving, conflicting-domain isolation, and OOD abstention. "
            "It is not a universal all-corpora or all-scale claim."
        ),
        "gates": gates,
        "failed": failed,
        "ratios": {
            "source_parameter_ratio_transformer_over_layercake": _ratio(
                source, "parameter_ratio_transformer_over_layercake"
            ),
            "source_bpb_ratio_layercake_over_transformer": _ratio(
                source, "bpb_ratio_layercake_over_transformer"
            ),
            "source_training_speed_ratio": _ratio(source, "training_speed_ratio"),
            "source_training_cost_proxy_ratio": source_train_cost_ratio,
            "source_training_byte_efficiency_ratio": source_train_byte_ratio,
            "source_cpu_generation_speed_ratio": _ratio(
                source, "cpu_generation_speed_ratio"
            ),
            "source_gpu_generation_speed_ratio": _ratio(
                source, "gpu_generation_speed_ratio"
            ),
            "source_cpu_generation_cost_proxy_ratio": source_cpu_generation_cost_ratio,
            "source_gpu_generation_cost_proxy_ratio": source_gpu_generation_cost_ratio,
            "source_cpu_quality_ratio": _ratio(source, "cpu_quality_ratio"),
            "source_gpu_quality_ratio": _ratio(source, "gpu_quality_ratio"),
            "instruction_cpu_generation_speed_ratio": _ratio(
                instruction, "cpu_generation_speed_ratio"
            ),
            "instruction_gpu_generation_speed_ratio": _ratio(
                instruction, "gpu_generation_speed_ratio"
            ),
            "instruction_cpu_relevance_ratio": _ratio(
                instruction, "cpu_relevance_ratio"
            ),
            "instruction_gpu_relevance_ratio": _ratio(
                instruction, "gpu_relevance_ratio"
            ),
            "portable_mixed_cpu_generation_speed_ratio": _ratio(
                mixed, "cpu_generation_speed_ratio"
            ),
            "portable_mixed_gpu_generation_speed_ratio": _ratio(
                mixed, "gpu_generation_speed_ratio"
            ),
            "conflicting_cpu_generation_speed_ratio": _ratio(
                conflict, "cpu_generation_speed_ratio"
            ),
            "conflicting_gpu_generation_speed_ratio": _ratio(
                conflict, "gpu_generation_speed_ratio"
            ),
            "ood_cpu_generation_speed_ratio": _ratio(
                abstain, "cpu_generation_speed_ratio"
            ),
            "ood_gpu_generation_speed_ratio": _ratio(
                abstain, "gpu_generation_speed_ratio"
            ),
        },
        "metrics": {
            "transfer_ppl_ratio": _metric(transfer, "transfer_ppl_ratio", None),
            "transfer_max_logit_diff": _metric(
                transfer, "transfer_max_logit_diff", None
            ),
            "transfer_max_abi_diff": _metric(transfer, "transfer_max_abi_diff", None),
            "transfer_generation_exact": _metric(
                transfer, "transfer_generation_exact", False
            ),
            "ood_cpu_abstention_required_count": _metric(
                abstain, "layercake_cpu_generation.abstention_required_count", 0
            ),
            "ood_gpu_abstention_required_count": _metric(
                abstain, "layercake_gpu_generation.abstention_required_count", 0
            ),
            "ood_cpu_effective_memory_match": _metric(
                abstain,
                "layercake_cpu_generation.portable_memory_match_rate_effective",
                0.0,
            ),
            "ood_gpu_effective_memory_match": _metric(
                abstain,
                "layercake_gpu_generation.portable_memory_match_rate_effective",
                0.0,
            ),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify current CPU/GPU platform-slice dominance across source, transfer, and portable-domain gates."
    )
    parser.add_argument("--source-certificate", required=True, type=Path)
    parser.add_argument("--transfer-certificate", required=True, type=Path)
    parser.add_argument("--instruction-generalization-certificate", required=True, type=Path)
    parser.add_argument("--portable-mixed-certificate", required=True, type=Path)
    parser.add_argument("--conflicting-isolation-certificate", required=True, type=Path)
    parser.add_argument("--ood-abstention-certificate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-param-ratio", type=float, default=5.0)
    parser.add_argument("--min-training-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-training-cost-ratio", type=float, default=5.0)
    parser.add_argument("--min-training-byte-ratio", type=float, default=1.0)
    parser.add_argument("--min-source-cpu-generation-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-source-gpu-generation-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-generation-cost-ratio", type=float, default=5.0)
    parser.add_argument("--min-gpu-generation-cost-ratio", type=float, default=1.0)
    parser.add_argument("--min-domain-cpu-generation-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-domain-gpu-generation-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
    parser.add_argument("--min-relevance-ratio", type=float, default=1.0)
    parser.add_argument("--min-portable-memory-match", type=float, default=1.0)
    args = parser.parse_args()

    result = verify(
        source_certificate=_read(args.source_certificate),
        transfer_certificate=_read(args.transfer_certificate),
        instruction_generalization_certificate=_read(
            args.instruction_generalization_certificate
        ),
        portable_mixed_certificate=_read(args.portable_mixed_certificate),
        conflicting_isolation_certificate=_read(args.conflicting_isolation_certificate),
        ood_abstention_certificate=_read(args.ood_abstention_certificate),
        args=args,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
