from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _run(command: list[str], cwd: Path, dry_run: bool) -> dict[str, Any]:
    rendered = " ".join(command)
    if dry_run:
        return {
            "command": rendered,
            "status": "SKIPPED_DRY_RUN",
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": rendered,
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _failed_buckets(certificate: dict[str, Any]) -> list[str]:
    buckets: list[str] = []
    gates = certificate.get("gates", {})
    if not gates.get("scale20_dominance", False):
        buckets.append("scale20")
    if not gates.get("scale24_dominance", False):
        buckets.append("scale24")
    if not gates.get("scale25_dominance", False):
        buckets.append("scale25")
    if not gates.get("transfer_no_damage_and_exact", False):
        buckets.append("transfer")
    return buckets


def _campaign_failed_buckets(certificate: dict[str, Any]) -> list[str]:
    gates = certificate.get("gates", {})
    buckets: list[str] = []
    if not gates.get("small_strict_dominance", False):
        buckets.append("scale20")
        buckets.append("scale24")
        buckets.append("scale25")
    if not gates.get("medium_48m_dominance", False):
        buckets.append("medium")
    if not gates.get("large_150m_dominance", False):
        buckets.append("large150")
    if not gates.get("large_350m_dominance", False):
        buckets.append("large350")
    if not gates.get("transfer_exactness_matrix", False):
        buckets.append("transfer")
    return sorted(set(buckets))


def _platform_failed_buckets(certificate: dict[str, Any]) -> list[str]:
    gates = certificate.get("gates", {})
    buckets: list[str] = []
    if not gates.get("cpu_quality_dominance", False):
        buckets.append("platform_cpu_quality")
    if not gates.get("cpu_training_efficiency_dominance", False):
        buckets.append("platform_cpu_training")
    if not gates.get("cpu_generation_dominance", False):
        buckets.append("platform_cpu_generation")
    if not gates.get("cpu_prefill_latency_dominance", False):
        buckets.append("platform_cpu_prefill")
    if not gates.get("memory_artifact_dominance", False):
        buckets.append("platform_memory")
    if not gates.get("transfer_exactness_dominance", False):
        buckets.append("transfer")
    if not gates.get("mobile_proxy_dominance", False):
        buckets.append("platform_mobile")
    if not gates.get("gpu_generation_dominance", False):
        buckets.append("platform_gpu_generation")
    if not gates.get("gpu_prefill_latency_dominance", False):
        buckets.append("platform_gpu_prefill")
    if not gates.get("real_mobile_hardware_evidence", False):
        buckets.append("platform_mobile_real_device")
    return sorted(set(buckets))


def _run_repairs(
    plan: dict[str, list[str]],
    buckets: list[str],
    python_exe: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    executions: list[dict[str, Any]] = []
    for bucket in buckets:
        for cmd in plan.get(bucket, []):
            if cmd.startswith("python "):
                argv = [python_exe] + cmd.split()[1:]
            else:
                argv = cmd.split()
            executions.append(_run(argv, ROOT, dry_run=dry_run))
    return executions


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Iterative research loop: verify 20M/24M/25M dominance and transfer "
            "invariance, then execute targeted repair commands until pass or max iterations."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--repair-plan",
        default="scripts/research_repair_plan.example.json",
    )
    parser.add_argument(
        "--execute-repairs",
        action="store_true",
        help="Run repair commands when gates fail.",
    )
    parser.add_argument(
        "--allow-proxy-25m-baseline",
        action="store_true",
        help="Allow nearest baseline for 25M if exact baseline is missing.",
    )
    parser.add_argument(
        "--output",
        default="results/research_loop_summary.json",
    )
    args = parser.parse_args()

    repair_plan_path = ROOT / args.repair_plan
    repair_plan = {
        "scale20": [],
        "scale24": [],
        "scale25": [],
        "medium": [],
        "large150": [],
        "large350": [],
        "platform_cpu_quality": [],
        "platform_cpu_training": [],
        "platform_cpu_generation": [],
        "platform_cpu_prefill": [],
        "platform_memory": [],
        "platform_mobile": [],
        "platform_gpu_generation": [],
        "platform_gpu_prefill": [],
        "platform_mobile_real_device": [],
        "transfer": [],
    }
    if repair_plan_path.exists():
        repair_plan.update(_load(repair_plan_path))

    history_path = RESULTS / "research_loop_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    iterations: list[dict[str, Any]] = []
    final_status = "FAIL"

    for iteration in range(1, args.max_iterations + 1):
        verify_cmd = [
            args.python,
            "scripts/verify_transformer_dominance_up_to_25m.py",
        ]
        if args.allow_proxy_25m_baseline:
            verify_cmd.append("--allow-proxy-25m-baseline")

        verify_exec = _run(verify_cmd, ROOT, dry_run=False)
        cert = _load(RESULTS / "dominance_up_to_25m_research_certificate.json")

        campaign_verify = _run(
            [args.python, "scripts/verify_sml_breakthrough_dominance.py"],
            ROOT,
            dry_run=False,
        )
        campaign_cert = _load(RESULTS / "sml_breakthrough_certificate.json")
        platform_verify = _run(
            [args.python, "scripts/verify_platform_benchmark_dominance.py"],
            ROOT,
            dry_run=False,
        )
        platform_cert = _load(
            RESULTS / "platform_benchmark_dominance_certificate.json"
        )

        loop_entry: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "verify": verify_exec,
            "campaign_verify": campaign_verify,
            "platform_verify": platform_verify,
            "status": cert.get("status", "FAIL"),
            "campaign_status": campaign_cert.get("status", "FAIL"),
            "platform_status": platform_cert.get("status", "FAIL"),
            "failed_buckets": sorted(
                set(
                    _failed_buckets(cert)
                    + _campaign_failed_buckets(campaign_cert)
                    + _platform_failed_buckets(platform_cert)
                )
            ),
            "certificate_path": "results/dominance_up_to_25m_research_certificate.json",
            "campaign_certificate_path": "results/sml_breakthrough_certificate.json",
            "platform_certificate_path": "results/platform_benchmark_dominance_certificate.json",
            "repairs": [],
        }

        if (
            cert.get("status") == "PASS"
            and campaign_cert.get("status") == "PASS"
            and platform_cert.get("status") == "PASS"
        ):
            final_status = "PASS"
            iterations.append(loop_entry)
            history_path.write_text(
                "\n".join(json.dumps(x, sort_keys=True) for x in iterations) + "\n",
                encoding="utf-8",
            )
            break

        if args.execute_repairs:
            repairs = _run_repairs(
                repair_plan,
                loop_entry["failed_buckets"],
                python_exe=args.python,
                dry_run=False,
            )
            loop_entry["repairs"] = repairs

            # Refresh certificates after repairs so this iteration captures post-repair truth.
            verify_exec = _run(verify_cmd, ROOT, dry_run=False)
            cert = _load(RESULTS / "dominance_up_to_25m_research_certificate.json")

            campaign_verify = _run(
                [args.python, "scripts/verify_sml_breakthrough_dominance.py"],
                ROOT,
                dry_run=False,
            )
            campaign_cert = _load(RESULTS / "sml_breakthrough_certificate.json")

            platform_verify = _run(
                [args.python, "scripts/verify_platform_benchmark_dominance.py"],
                ROOT,
                dry_run=False,
            )
            platform_cert = _load(
                RESULTS / "platform_benchmark_dominance_certificate.json"
            )

            loop_entry["verify"] = verify_exec
            loop_entry["campaign_verify"] = campaign_verify
            loop_entry["platform_verify"] = platform_verify
            loop_entry["status"] = cert.get("status", "FAIL")
            loop_entry["campaign_status"] = campaign_cert.get("status", "FAIL")
            loop_entry["platform_status"] = platform_cert.get("status", "FAIL")
            loop_entry["failed_buckets"] = sorted(
                set(
                    _failed_buckets(cert)
                    + _campaign_failed_buckets(campaign_cert)
                    + _platform_failed_buckets(platform_cert)
                )
            )

        iterations.append(loop_entry)

    summary = {
        "status": final_status,
        "max_iterations": args.max_iterations,
        "iterations_run": len(iterations),
        "execute_repairs": args.execute_repairs,
        "allow_proxy_25m_baseline": args.allow_proxy_25m_baseline,
        "repair_plan": str(repair_plan_path.relative_to(ROOT)).replace("\\", "/"),
        "history": iterations,
        "next_action": (
            "If FAIL, update repair-plan commands for failed buckets and rerun with --execute-repairs."
        ),
    }

    history_path.write_text(
        "\n".join(json.dumps(x, sort_keys=True) for x in iterations) + "\n",
        encoding="utf-8",
    )

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
