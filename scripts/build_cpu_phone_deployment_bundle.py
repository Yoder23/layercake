from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/breakthrough_equal/cpu_phone_deployment_bundle.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def artifact_record(path: Path, *, kind: str, root: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "kind": kind,
        "path": str(path.relative_to(root) if exists and path.is_relative_to(root) else path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "sha256": sha256_file(path) if exists else None,
    }


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_samples(value: Any, *, minimum: int = 20) -> list[float]:
    if not isinstance(value, list) or len(value) < minimum:
        return []
    samples = [float(item) for item in value if _is_number(item) and float(item) > 0.0]
    return samples if len(samples) == len(value) else []


def validate_phone_evidence(
    evidence: dict[str, Any] | None,
    *,
    core_artifacts: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Fail-closed validation for a same-device LayerCake/mobile-baseline run."""

    evidence = evidence or {}
    device = evidence.get("device") if isinstance(evidence.get("device"), dict) else {}
    runtime = evidence.get("runtime") if isinstance(evidence.get("runtime"), dict) else {}
    protocol = evidence.get("protocol") if isinstance(evidence.get("protocol"), dict) else {}
    layercake = evidence.get("layercake") if isinstance(evidence.get("layercake"), dict) else {}
    baseline = evidence.get("baseline") if isinstance(evidence.get("baseline"), dict) else {}
    sustained = evidence.get("sustained") if isinstance(evidence.get("sustained"), dict) else {}

    layercake_latency = _positive_samples(layercake.get("latency_ms"))
    baseline_latency = _positive_samples(baseline.get("latency_ms"))
    layercake_median = median(layercake_latency) if layercake_latency else 0.0
    baseline_median = median(baseline_latency) if baseline_latency else 0.0
    speed_ratio = baseline_median / layercake_median if layercake_median else 0.0

    layercake_total = int(layercake.get("quality_total", 0) or 0)
    layercake_exact = int(layercake.get("quality_exact", 0) or 0)
    baseline_total = int(baseline.get("quality_total", 0) or 0)
    baseline_exact = int(baseline.get("quality_exact", 0) or 0)
    layercake_quality = layercake_exact / layercake_total if layercake_total else 0.0
    baseline_quality = baseline_exact / baseline_total if baseline_total else 0.0

    layercake_peak = float(layercake.get("peak_rss_bytes", 0.0) or 0.0)
    baseline_peak = float(baseline.get("peak_rss_bytes", 0.0) or 0.0)
    layercake_bytes = float(layercake.get("artifact_bytes", 0.0) or 0.0)
    baseline_bytes = float(baseline.get("artifact_bytes", 0.0) or 0.0)
    sustained_layercake_bps = float(
        sustained.get("layercake_bytes_per_second", 0.0) or 0.0
    )
    sustained_baseline_bps = float(
        sustained.get("baseline_bytes_per_second", 0.0) or 0.0
    )
    sustained_speed_ratio = (
        sustained_layercake_bps / sustained_baseline_bps
        if sustained_baseline_bps > 0.0
        else 0.0
    )
    layercake_battery_drop = float(
        sustained.get("layercake_battery_start_percent", 0.0) or 0.0
    ) - float(sustained.get("layercake_battery_end_percent", 0.0) or 0.0)
    baseline_battery_drop = float(
        sustained.get("baseline_battery_start_percent", 0.0) or 0.0
    ) - float(sustained.get("baseline_battery_end_percent", 0.0) or 0.0)

    artifact_hash = str(runtime.get("artifact_sha256", "")).lower()
    baseline_hash = str(baseline.get("artifact_sha256", "")).lower()
    committed_core_hashes = {
        str(item.get("sha256", "")).lower()
        for item in core_artifacts
        if item.get("exists") and item.get("sha256")
    }
    os_name = str(device.get("os", "")).lower()
    architecture = str(device.get("architecture", "")).lower()
    required_device_fields = ["manufacturer", "model", "os_version"]
    required_runtime_fields = ["engine", "engine_version"]

    gates = {
        "explicit_real_hardware_declaration": evidence.get("real_phone_hardware") is True,
        "phone_schema_v1": evidence.get("schema_version") == 1,
        "android_or_ios_arm64_device": (
            os_name in {"android", "ios"}
            and architecture in {"arm64", "aarch64", "arm64-v8a"}
            and all(str(device.get(field, "")).strip() for field in required_device_fields)
        ),
        "runtime_identity_complete": all(
            str(runtime.get(field, "")).strip() for field in required_runtime_fields
        ),
        "tested_artifact_hash_matches_bundle": (
            bool(SHA256_RE.fullmatch(artifact_hash))
            and artifact_hash in committed_core_hashes
        ),
        "same_prompt_protocol": (
            protocol.get("identical_prompt_pack") is True
            and bool(
                SHA256_RE.fullmatch(
                    str(protocol.get("prompt_pack_sha256", "")).lower()
                )
            )
            and int(protocol.get("prompt_count", 0) or 0) >= 20
            and int(protocol.get("warmup_runs", 0) or 0) >= 3
            and int(protocol.get("repetitions", 0) or 0) >= 3
        ),
        "raw_latency_samples_present": bool(
            layercake_latency
            and baseline_latency
            and len(layercake_latency) == len(baseline_latency)
        ),
        "phone_generation_speed_at_least_5x": speed_ratio >= 5.0,
        "layercake_quality_at_least_95_percent": (
            layercake_total >= 20 and layercake_quality >= 0.95
        ),
        "quality_noninferior_to_baseline": (
            baseline_total == layercake_total
            and baseline_total >= 20
            and layercake_quality >= baseline_quality
        ),
        "artifact_no_larger_than_baseline": (
            0.0 < layercake_bytes <= baseline_bytes
            and bool(SHA256_RE.fullmatch(baseline_hash))
            and str(baseline.get("name", "")).strip() != ""
            and str(baseline.get("version", "")).strip() != ""
        ),
        "peak_memory_no_higher_than_baseline": (
            0.0 < layercake_peak <= baseline_peak
        ),
        "five_minute_sustained_run": (
            float(sustained.get("duration_seconds", 0.0) or 0.0) >= 300.0
            and int(sustained.get("requests", 0) or 0) >= 100
        ),
        "sustained_speed_at_least_5x": sustained_speed_ratio >= 5.0,
        "thermal_measurement_without_throttling": (
            all(
                _is_number(sustained.get(field))
                for field in [
                    "thermal_start_celsius",
                    "thermal_end_celsius",
                    "thermal_peak_celsius",
                ]
            )
            and sustained.get("throttling_detected") is False
        ),
        "battery_measurement_noninferior": (
            layercake_battery_drop >= 0.0
            and baseline_battery_drop >= 0.0
            and layercake_battery_drop <= baseline_battery_drop
            and float(sustained.get("layercake_battery_start_percent", 0.0) or 0.0) > 0.0
            and float(sustained.get("baseline_battery_start_percent", 0.0) or 0.0) > 0.0
        ),
    }
    metrics = {
        "latency_samples_per_model": len(layercake_latency),
        "layercake_median_latency_ms": layercake_median,
        "baseline_median_latency_ms": baseline_median,
        "generation_speed_ratio": speed_ratio,
        "layercake_quality": layercake_quality,
        "baseline_quality": baseline_quality,
        "layercake_peak_rss_bytes": layercake_peak,
        "baseline_peak_rss_bytes": baseline_peak,
        "layercake_artifact_bytes": layercake_bytes,
        "baseline_artifact_bytes": baseline_bytes,
        "sustained_speed_ratio": sustained_speed_ratio,
        "layercake_battery_drop_percent": layercake_battery_drop,
        "baseline_battery_drop_percent": baseline_battery_drop,
    }
    return gates, metrics


def build_bundle(
    *,
    core_paths: list[Path],
    domain_paths: list[Path],
    output_path: Path,
    phone_evidence_path: Path | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    core_artifacts = [
        artifact_record(path, kind="layercake_core", root=root)
        for path in core_paths
    ]
    domain_artifacts = [
        artifact_record(path, kind="portable_domain", root=root)
        for path in domain_paths
    ]
    phone_evidence = _read_optional_json(phone_evidence_path)
    phone_gates, phone_metrics = validate_phone_evidence(
        phone_evidence,
        core_artifacts=core_artifacts,
    )
    real_phone_evidence = bool(phone_gates) and all(phone_gates.values())
    gates = {
        "core_artifacts_present": bool(core_artifacts)
        and all(item["exists"] for item in core_artifacts),
        "portable_domains_present": bool(domain_artifacts)
        and all(item["exists"] for item in domain_artifacts),
        "all_artifacts_hashed": all(
            bool(item["sha256"]) for item in core_artifacts + domain_artifacts
        ),
        "cpu_only_target_declared": True,
        "phone_target_declared": True,
        "real_phone_runtime_evidence": real_phone_evidence,
    }
    failed = [name for name, passed in gates.items() if not passed]
    status = "PASS" if not failed else ("OPEN" if failed == ["real_phone_runtime_evidence"] else "FAIL")
    bundle = {
        "status": status,
        "scope": (
            "CPU/phone deployment bundle manifest for moving LayerCake cores and "
            "portable domains to non-GPU desktop and phone targets. PASS requires "
            "real phone runtime evidence; otherwise a correctly hashed package remains OPEN."
        ),
        "required_gates": gates,
        "failed_required": failed,
        "target_devices": {
            "cpu_only_desktop": {
                "device_class": "x86_64_or_arm64_cpu_only",
                "gpu_required": False,
                "runtime": "python_torch_cpu_or_native_port",
            },
            "phone": {
                "device_class": "android_or_ios_arm64",
                "gpu_required": False,
                "runtime": "native_int8_or_cpu_runtime_required",
                "real_hardware_measurement_required_for_pass": True,
            },
        },
        "artifacts": {
            "cores": core_artifacts,
            "portable_domains": domain_artifacts,
        },
        "phone_evidence": {
            "path": str(phone_evidence_path) if phone_evidence_path else None,
            "present": phone_evidence is not None,
            "real_phone_runtime_evidence": real_phone_evidence,
            "gates": phone_gates,
            "failed": [name for name, passed in phone_gates.items() if not passed],
            "metrics": phone_metrics,
        },
        "output": str(output_path.relative_to(root) if output_path.is_relative_to(root) else output_path),
    }
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a hashed CPU/phone LayerCake deployment bundle manifest.")
    parser.add_argument(
        "--core",
        action="append",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--portable-domain",
        action="append",
        type=Path,
        default=None,
    )
    parser.add_argument("--phone-evidence", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    core_paths = args.core or [
        ROOT / "runs_experiment/scale15m_transition_lw280_2300_noprofile.pt"
    ]
    domain_paths = args.portable_domain or [
        ROOT / "runs_experiment/portable_python_gru148k_v1.pt"
    ]
    bundle = build_bundle(
        core_paths=[path if path.is_absolute() else ROOT / path for path in core_paths],
        domain_paths=[
            path if path.is_absolute() else ROOT / path
            for path in domain_paths
        ],
        phone_evidence_path=(
            args.phone_evidence
            if args.phone_evidence is None or args.phone_evidence.is_absolute()
            else ROOT / args.phone_evidence
        ),
        output_path=args.output,
        root=ROOT,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0 if bundle["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
