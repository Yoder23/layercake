"""Bounded validation-only experiment manager with a complete search ledger."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

from .data import sha256_file
from .foundation import train_english_core
from .promotion_gates import classify_foundation_failure


def _canonical_hash(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _pareto(rows: list[dict]) -> list[str]:
    frontier = []
    for row in rows:
        dominated = any(
            other is not row
            and other["mean_selection_bpb"] <= row["mean_selection_bpb"]
            and other["mean_wall_seconds"] <= row["mean_wall_seconds"]
            and other["active_parameters"] <= row["active_parameters"]
            and (
                other["mean_selection_bpb"] < row["mean_selection_bpb"]
                or other["mean_wall_seconds"] < row["mean_wall_seconds"]
                or other["active_parameters"] < row["active_parameters"]
            )
            for other in rows
        )
        if not dominated:
            frontier.append(row["candidate"])
    return frontier


def run_foundation_campaign(
    config_path: str | Path,
    output_path: str | Path,
    *,
    artifact_root: str | Path = "artifacts/final/foundation-search",
) -> dict:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = config["candidates"]
    seeds = [int(seed) for seed in config["seeds"]]
    budget = config["budget"]
    if len(candidates) > int(budget["maximum_candidates"]):
        raise ValueError("candidate budget exceeded")
    if len(candidates) * len(seeds) > int(budget["maximum_runs"]):
        raise ValueError("run budget exceeded")
    if int(config["training"]["steps"]) > int(budget["maximum_steps_per_run"]):
        raise ValueError("step budget exceeded")
    if bool(config["evaluation"].get("evaluate_test")):
        raise ValueError("architecture search may not access the final test split")
    started = time.perf_counter()
    artifacts = Path(artifact_root)
    generated = artifacts / "locked-configs"
    generated.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    for candidate in candidates:
        for seed in seeds:
            if time.perf_counter() - started > float(budget["maximum_wall_seconds"]):
                raise RuntimeError("campaign exhausted its predeclared wall-time budget")
            locked = {
                "format": "layercake-core-training-config/2",
                "scale_status": "final_routed_search_validation_only",
                "seed": seed,
                "device": config["device"],
                "precision": config["precision"],
                "model": copy.deepcopy(config["model"]),
                "training": copy.deepcopy(config["training"]),
                "evaluation": copy.deepcopy(config["evaluation"]),
                "data": copy.deepcopy(config["data"]),
            }
            locked["model"]["routing_mode"] = candidate["routing_mode"]
            locked["model"]["ablation"] = candidate["ablation"]
            locked["model"].update(candidate.get("model_overrides", {}))
            locked["training"]["route"] = candidate["route"]
            locked_hash = _canonical_hash(locked)
            locked_path = generated / f"{candidate['name']}-seed-{seed}-{locked_hash[:12]}.json"
            locked_path.write_text(json.dumps(locked, indent=2, sort_keys=True), encoding="utf-8")
            run_root = artifacts / candidate["name"] / f"seed-{seed}"
            try:
                evidence = train_english_core(locked_path, run_root)
                parameters = evidence["parameters"]
                routing = evidence["routing"]
                row = {
                    "candidate": candidate["name"],
                    "seed": seed,
                    "status": evidence["status"],
                    "config_sha256": sha256_file(locked_path),
                    "checkpoint_sha256": evidence["checkpoint"]["sha256"],
                    "selection_bpb": evidence["quality"]["architecture_selection"]["bits_per_byte"],
                    "validation_bpb_diagnostic_only": evidence["quality"]["validation"]["bits_per_byte"],
                    "test_accessed": evidence["quality"]["test_accessed"],
                    "wall_seconds": evidence["training"]["wall_seconds"],
                    "raw_bytes_seen": evidence["training"]["raw_bytes_seen"],
                    "total_parameters": parameters["total_parameters"],
                    "active_parameters": parameters["active_parameters"],
                    "active_fraction": parameters["active_fraction"],
                    "routed_candidate": candidate["ablation"] not in {"dense", "no_routed_experts"},
                    "routing": routing,
                    "peak_cuda_bytes": evidence["memory"]["cuda_peak_allocated_bytes"],
                    "artifact": str(run_root.resolve()),
                }
            except Exception as exc:  # preserve failures in the ledger
                row = {
                    "candidate": candidate["name"], "seed": seed, "status": "FAIL",
                    "failure": f"{type(exc).__name__}: {exc}", "selection_bpb": float("inf"),
                    "test_accessed": False, "routed_candidate": True,
                    "active_fraction": 1.0, "wall_seconds": 0.0,
                }
            runs.append(row)
    reference_candidate = config.get("reference_candidate", "fixed_expert")
    fixed_rows = [row for row in runs if row["candidate"] == reference_candidate and row["status"] == "PASS"]
    if not fixed_rows:
        raise ValueError(f"reference candidate did not complete: {reference_candidate}")
    fixed_bpb = statistics.fmean(row["selection_bpb"] for row in fixed_rows)
    for row in runs:
        row["failure_classes"] = classify_foundation_failure(
            row, fixed_bpb=fixed_bpb, policy=config["promotion"]
        )
        row["eligible"] = not row["failure_classes"] and row["routed_candidate"]
    summaries = []
    for candidate in candidates:
        candidate_rows = [row for row in runs if row["candidate"] == candidate["name"]]
        successful = [row for row in candidate_rows if row["status"] == "PASS"]
        summaries.append({
            "candidate": candidate["name"],
            "runs": len(candidate_rows),
            "successful_runs": len(successful),
            "eligible_all_seeds": bool(successful) and all(row["eligible"] for row in successful) and len(successful) == len(seeds),
            "mean_selection_bpb": statistics.fmean(row["selection_bpb"] for row in successful) if successful else float("inf"),
            "selection_bpb_stdev": statistics.stdev(row["selection_bpb"] for row in successful) if len(successful) > 1 else 0.0,
            "mean_wall_seconds": statistics.fmean(row["wall_seconds"] for row in successful) if successful else float("inf"),
            "total_parameters": successful[0]["total_parameters"] if successful else 0,
            "active_parameters": successful[0]["active_parameters"] if successful else 0,
            "mean_maximum_load": statistics.fmean(row["routing"]["maximum_load_fraction"] for row in successful) if successful else 1.0,
            "all_test_accessed_false": all(not row.get("test_accessed") for row in candidate_rows),
            "failure_classes": sorted({failure for row in candidate_rows for failure in row["failure_classes"]}),
        })
    eligible = [row for row in summaries if row["eligible_all_seeds"]]
    selected = min(
        eligible,
        key=lambda row: (row["mean_selection_bpb"], row["mean_wall_seconds"], row["active_parameters"]),
    ) if eligible else None
    result = {
        "format": "layercake-final-foundation-search/1",
        "status": "PASS" if selected else "FAIL",
        "source_config": str(config_path.resolve()),
        "source_config_sha256": sha256_file(config_path),
        "selection_split_only": True,
        "final_test_accessed": False,
        "fixed_expert_mean_selection_bpb": fixed_bpb,
        "reference_candidate": reference_candidate,
        "reference_mean_selection_bpb": fixed_bpb,
        "selected_candidate": selected,
        "promotion": config["promotion"],
        "budget": budget,
        "runs": runs,
        "summary": summaries,
        "pareto_frontier": _pareto([row for row in summaries if row["successful_runs"]]),
        "preserved_controls": config["preserved_controls"],
        "wall_seconds": time.perf_counter() - started,
        "failed_runs": [row for row in runs if row["status"] != "PASS"],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _confidence_interval_95(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "stdev": 0.0, "lower": mean, "upper": mean, "n": len(values)}
    stdev = statistics.stdev(values)
    # Student-t critical values for the campaign's supported small seed counts.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    half = critical * stdev / (len(values) ** 0.5)
    return {"mean": mean, "stdev": stdev, "lower": mean - half, "upper": mean + half, "n": len(values)}


def run_medium_foundation_campaign(
    config_path: str | Path,
    output_path: str | Path,
    *,
    artifact_root: str | Path = "artifacts/final/medium-cores",
) -> dict:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    search_path = Path(config["candidate_source"])
    search = json.loads(search_path.read_text(encoding="utf-8"))
    frozen_name = str(config["frozen_candidate"])
    if search.get("selected_candidate", {}).get("candidate") != frozen_name:
        raise ValueError("medium candidate is not the frozen validation-only search winner")
    source_config = json.loads(Path(search["source_config"]).read_text(encoding="utf-8"))
    candidate = next(row for row in source_config["candidates"] if row["name"] == frozen_name)
    model_config = copy.deepcopy(source_config["model"])
    model_config["routing_mode"] = candidate["routing_mode"]
    model_config["ablation"] = candidate["ablation"]
    model_config.update(candidate.get("model_overrides", {}))
    seeds = [int(seed) for seed in config["seeds"]]
    artifacts = Path(artifact_root)
    generated = artifacts / "locked-configs"
    generated.mkdir(parents=True, exist_ok=True)
    runs = []
    campaign_started = time.perf_counter()
    for seed in seeds:
        locked = {
            "format": "layercake-core-training-config/2",
            "scale_status": config["scale_status"],
            "seed": seed,
            "device": config["device"],
            "precision": config["precision"],
            "model": copy.deepcopy(model_config),
            "training": copy.deepcopy(config["training"]),
            "evaluation": copy.deepcopy(config["evaluation"]),
            "data": copy.deepcopy(config["data"]),
        }
        locked["training"]["route"] = candidate["route"]
        locked["training"]["resumable"] = True
        locked_hash = _canonical_hash(locked)
        locked_path = generated / f"{frozen_name}-seed-{seed}-{locked_hash[:12]}.json"
        if not locked_path.is_file():
            locked_path.write_text(json.dumps(locked, indent=2, sort_keys=True), encoding="utf-8")
        run_root = artifacts / f"seed-{seed}"
        try:
            evidence = train_english_core(locked_path, run_root)
            runs.append({
                "seed": seed,
                "status": evidence["status"],
                "config": str(locked_path.resolve()),
                "config_sha256": sha256_file(locked_path),
                "checkpoint_sha256": evidence["checkpoint"]["sha256"],
                "architecture_selection_bpb": evidence["quality"]["architecture_selection"]["bits_per_byte"],
                "validation_bpb": evidence["quality"]["validation"]["bits_per_byte"],
                "validation_byte_accuracy": evidence["quality"]["validation"]["byte_accuracy"],
                "test_accessed": evidence["quality"]["test_accessed"],
                "wall_seconds": evidence["training"]["wall_seconds"],
                "raw_bytes_seen": evidence["training"]["raw_bytes_seen"],
                "parameters": evidence["parameters"],
                "optimizer": evidence["optimizer"],
                "routing": evidence["routing"],
                "memory": evidence["memory"],
                "artifact": str(run_root.resolve()),
                "resume_metadata": evidence["training"]["resume_metadata"],
            })
        except Exception as exc:
            runs.append({"seed": seed, "status": "FAIL", "failure": f"{type(exc).__name__}: {exc}"})
    valid = [row for row in runs if row["status"] == "PASS"]
    integrity = (
        len(valid) == len(seeds)
        and all(not row["test_accessed"] for row in valid)
        and all(row["routing"]["all_experts_meaningfully_trained"] for row in valid)
        and all(not row["routing"]["router_collapsed"] for row in valid)
    )
    result = {
        "format": "layercake-final-medium-foundation/1",
        "status": "PASS" if integrity else "FAIL",
        "scale": config["scale_status"],
        "frozen_candidate": frozen_name,
        "frozen_architecture": model_config,
        "frozen_architecture_sha256": _canonical_hash(model_config),
        "search_evidence": str(search_path.resolve()),
        "search_evidence_sha256": sha256_file(search_path),
        "final_test_accessed": any(row.get("test_accessed", False) for row in runs),
        "data": {
            name: {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "sha256": sha256_file(path)}
            for name, path in config["data"].items()
        },
        "runs": runs,
        "validation_bpb_confidence_interval_95": _confidence_interval_95([
            row["validation_bpb"] for row in valid
        ]) if valid else None,
        "architecture_selection_bpb_confidence_interval_95": _confidence_interval_95([
            row["architecture_selection_bpb"] for row in valid
        ]) if valid else None,
        "failed_seeds": [row for row in runs if row["status"] != "PASS"],
        "all_experts_trained_all_seeds": bool(valid) and all(
            row["routing"]["all_experts_meaningfully_trained"] for row in valid
        ),
        "wall_seconds": time.perf_counter() - campaign_started,
        "continuation_command": "C:\\Python310\\python.exe -m layercake.moonshot_final train-core",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _time_to_thresholds(curves: list[dict], thresholds: list[float]) -> dict[str, float | None]:
    return {
        str(threshold): next((
            float(row["wall_seconds"])
            for row in curves
            if row.get("validation") is not None
            and float(row["validation"]["bits_per_byte"]) <= float(threshold)
        ), None)
        for threshold in thresholds
    }


def run_medium_transformer_campaign(
    config_path: str | Path,
    output_path: str | Path,
    *,
    artifact_root: str | Path = "artifacts/final/medium-transformers",
) -> dict:
    from .baseline import train_bpe_transformer

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    artifacts = Path(artifact_root)
    generated = artifacts / "locked-configs"
    generated.mkdir(parents=True, exist_ok=True)
    thresholds = [float(value) for value in config["comparison"]["time_to_quality_thresholds_bpb"]]
    runs = []
    campaign_started = time.perf_counter()
    for seed in seeds:
        locked = {key: copy.deepcopy(value) for key, value in config.items() if key not in {"seeds", "comparison"}}
        locked["format"] = "layercake-transformer-training-config/2"
        locked["seed"] = seed
        locked_hash = _canonical_hash(locked)
        locked_path = generated / f"transformer-seed-{seed}-{locked_hash[:12]}.json"
        if not locked_path.is_file():
            locked_path.write_text(json.dumps(locked, indent=2, sort_keys=True), encoding="utf-8")
        run_root = artifacts / f"seed-{seed}"
        try:
            evidence = train_bpe_transformer(locked_path, run_root)
            runs.append({
                "seed": seed,
                "status": evidence["status"],
                "config": str(locked_path.resolve()),
                "config_sha256": sha256_file(locked_path),
                "checkpoint_sha256": evidence["checkpoint"]["sha256"],
                "parameters": evidence["parameters"],
                "architecture_selection_bpb": evidence["quality"]["architecture_selection"]["bits_per_byte"],
                "validation_bpb": evidence["quality"]["validation"]["bits_per_byte"],
                "test_accessed": evidence["quality"]["test_accessed"],
                "training_wall_seconds": evidence["training"]["wall_seconds"],
                "tokenizer_training_seconds": evidence["tokenizer"]["training_seconds"],
                "end_to_end_training_seconds": evidence["training"]["wall_seconds"] + evidence["tokenizer"]["training_seconds"],
                "preprocessing_seconds_measured_within_training": evidence["training"]["preprocessing_seconds"],
                "raw_bytes_seen": evidence["training"]["raw_bytes_seen"],
                "time_to_validation_bpb": _time_to_thresholds(evidence["training"]["curves"], thresholds),
                "memory": evidence["memory"],
                "artifact": str(run_root.resolve()),
                "resume_metadata": evidence["training"]["resume_metadata"],
            })
        except Exception as exc:
            runs.append({"seed": seed, "status": "FAIL", "failure": f"{type(exc).__name__}: {exc}"})
    valid = [row for row in runs if row["status"] == "PASS"]
    layercake_path = Path(config["comparison"]["layercake_campaign"])
    layercake = json.loads(layercake_path.read_text(encoding="utf-8"))
    layercake_times = {}
    for row in layercake["runs"]:
        metadata = json.loads((Path(row["artifact"]) / "metadata.json").read_text(encoding="utf-8"))
        layercake_times[str(row["seed"])] = _time_to_thresholds(metadata["training"]["curves"], thresholds)
    comparisons = []
    for threshold in thresholds:
        key = str(threshold)
        lc_values = [row[key] for row in layercake_times.values() if row[key] is not None]
        tr_values = [row["time_to_validation_bpb"][key] for row in valid if row["time_to_validation_bpb"][key] is not None]
        comparisons.append({
            "threshold_bpb": threshold,
            "layercake_seeds_reached": len(lc_values),
            "transformer_seeds_reached": len(tr_values),
            "layercake_median_seconds": statistics.median(lc_values) if lc_values else None,
            "transformer_median_seconds": statistics.median(tr_values) if tr_values else None,
            "transformer_over_layercake_speed_ratio": (
                statistics.median(tr_values) / statistics.median(lc_values)
                if lc_values and tr_values else None
            ),
        })
    lc_params = int(layercake["runs"][0]["parameters"]["total_parameters"])
    tr_params = int(valid[0]["parameters"]) if valid else 0
    relative_delta = abs(lc_params - tr_params) / max(lc_params, tr_params, 1)
    transformer_ci = _confidence_interval_95([row["validation_bpb"] for row in valid]) if valid else None
    layercake_ci = layercake["validation_bpb_confidence_interval_95"]
    noninferior = bool(valid) and float(layercake_ci["upper"]) <= (
        float(transformer_ci["lower"]) + float(config["comparison"]["general_noninferiority_margin_bpb"])
    )
    common_complete = [row for row in comparisons if row["layercake_seeds_reached"] == len(seeds) and row["transformer_seeds_reached"] == len(seeds)]
    strongest = min(common_complete, key=lambda row: row["threshold_bpb"]) if common_complete else None
    training_speed_pass = bool(strongest) and float(strongest["transformer_over_layercake_speed_ratio"]) >= float(
        config["comparison"]["minimum_training_speedup_at_matched_quality"]
    )
    integrity = len(valid) == len(seeds) and all(not row["test_accessed"] for row in valid)
    result = {
        "format": "layercake-final-medium-transformer-comparison/1",
        "status": "PASS" if integrity else "FAIL",
        "final_test_accessed": any(row.get("test_accessed", False) for row in runs),
        "runs": runs,
        "failed_seeds": [row for row in runs if row["status"] != "PASS"],
        "transformer_validation_bpb_confidence_interval_95": transformer_ci,
        "layercake_validation_bpb_confidence_interval_95": layercake_ci,
        "total_parameters": {"layercake": lc_params, "transformer": tr_params, "relative_delta": relative_delta, "within_tolerance": relative_delta <= float(config["comparison"]["total_parameter_tolerance"])},
        "general_quality_noninferior": noninferior,
        "general_noninferiority_margin_bpb": config["comparison"]["general_noninferiority_margin_bpb"],
        "time_to_quality": comparisons,
        "strongest_common_threshold": strongest,
        "training_speed_gate_pass": training_speed_pass and noninferior,
        "training_speed_gate_reason": "requires both endpoint noninferiority and faster time to the strongest threshold reached by every seed",
        "wall_seconds": time.perf_counter() - campaign_started,
        "continuation_command": "C:\\Python310\\python.exe -m layercake.moonshot_final train-transformer",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
