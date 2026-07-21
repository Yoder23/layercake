from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "dominance" / "matrix"

DEFAULT_SCALES: list[dict[str, Any]] = [
    {
        "name": "15m",
        "layercake": "runs_experiment/scale15m_transition_lw280_2300_noprofile.pt",
        "transformer": "runs_experiment/scale15m_bpe_matched.pt",
        "proxy_baseline": False,
    },
    {
        "name": "20m",
        "layercake": "runs_experiment/scale20m_lc448_w32_qk_seed6250.pt",
        "transformer": "runs_experiment/scale20m_bpe448_l7_seed6250.pt",
        "proxy_baseline": False,
    },
    {
        "name": "24m",
        "layercake": "runs_experiment/scale24m_layercake_seed7001.pt",
        "transformer": "runs_experiment/scale24m_bpe_seed7001.pt",
        "proxy_baseline": False,
    },
    {
        "name": "25m_proxy26m",
        "layercake": "runs_experiment/scale25m_layercake_4x4_seed7002.pt",
        "transformer": "runs_experiment/scale26m_bpe512_seed6250.pt",
        "proxy_baseline": True,
    },
]


def _run(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": " ".join(command),
        "exit_code": proc.returncode,
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-3000:],
    }


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_if_passed(command_result: dict[str, Any], path: Path) -> dict[str, Any] | None:
    if command_result.get("status") != "PASS":
        return None
    return _load(path)


def _scale_result(
    python_exe: str,
    scale: dict[str, Any],
    new_bytes: int,
    quality_bytes: int,
    quality_no_repeat_ngram: int,
    cpu_threads: int,
    run_gpu: bool,
) -> dict[str, Any]:
    name = str(scale["name"])
    layercake = str(scale["layercake"])
    transformer = str(scale["transformer"])

    scale_dir = RESULTS / name
    scale_dir.mkdir(parents=True, exist_ok=True)

    generation_cpu_path = scale_dir / "generation_cpu.json"
    generation_gpu_path = scale_dir / "generation_gpu.json"
    quality_path = scale_dir / "generation_quality_cpu.json"
    resources_path = scale_dir / "cpu_resources.json"

    commands = {
        "generation_cpu": _run(
            [
                python_exe,
                "scripts/benchmark_generation.py",
                "--layercake",
                layercake,
                "--bpe",
                transformer,
                "--layercake-mode",
                "stateful_cached",
                "--new-bytes",
                str(new_bytes),
                "--device",
                "cpu",
                "--cpu-threads",
                str(cpu_threads),
                "--output",
                str(generation_cpu_path.relative_to(ROOT)).replace("\\", "/"),
            ]
        ),
        "generation_quality_cpu": _run(
            [
                python_exe,
                "scripts/benchmark_generation_quality.py",
                "--layercake",
                layercake,
                "--bpe",
                transformer,
                "--new-bytes",
                str(quality_bytes),
                "--no-repeat-ngram",
                str(quality_no_repeat_ngram),
                "--device",
                "cpu",
                "--cpu-threads",
                str(cpu_threads),
                "--output",
                str(quality_path.relative_to(ROOT)).replace("\\", "/"),
            ]
        ),
        "cpu_resources": _run(
            [
                python_exe,
                "scripts/benchmark_cpu_deployment_resources.py",
                "--layercake",
                layercake,
                "--bpe",
                transformer,
                "--repeats",
                "3",
                "--output",
                str(resources_path.relative_to(ROOT)).replace("\\", "/"),
            ]
        ),
    }

    if run_gpu:
        commands["generation_gpu"] = _run(
            [
                python_exe,
                "scripts/benchmark_generation.py",
                "--layercake",
                layercake,
                "--bpe",
                transformer,
                "--layercake-mode",
                "stateful_cached",
                "--new-bytes",
                str(new_bytes),
                "--device",
                "cuda",
                "--output",
                str(generation_gpu_path.relative_to(ROOT)).replace("\\", "/"),
            ]
        )

    generation_cpu = _load_if_passed(commands["generation_cpu"], generation_cpu_path)
    generation_gpu = (
        _load_if_passed(commands["generation_gpu"], generation_gpu_path)
        if run_gpu and "generation_gpu" in commands
        else None
    )
    quality = _load_if_passed(commands["generation_quality_cpu"], quality_path)
    resources = _load_if_passed(commands["cpu_resources"], resources_path)

    gaps: list[str] = []
    if commands["generation_cpu"]["status"] != "PASS":
        gaps.append("cpu_generation_execution")
    if generation_cpu and generation_cpu.get("speed_ratio", 0.0) <= 1.0:
        gaps.append("cpu_generation_speed")
    if run_gpu and commands.get("generation_gpu", {}).get("status") != "PASS":
        gaps.append("gpu_generation_execution")
    if generation_gpu and generation_gpu.get("speed_ratio", 0.0) <= 1.0:
        gaps.append("gpu_generation_speed")
    if commands["cpu_resources"]["status"] != "PASS":
        gaps.append("cpu_resources_execution")
    if resources and resources.get("metrics", {}).get("prefill_speed_ratio", 0.0) <= 1.0:
        gaps.append("cpu_prefill_latency")
    if commands["generation_quality_cpu"]["status"] != "PASS":
        gaps.append("quality_execution")
    if quality:
        qg = quality.get("quality_gates", {})
        if not bool(qg.get("layercake_printable", False)):
            gaps.append("quality_printable")
        if not bool(qg.get("layercake_distinct_trigram_at_least_bpe", False)):
            gaps.append("quality_distinct_trigram")
        if not bool(qg.get("layercake_max_repeat_8gram_no_worse_than_bpe", False)):
            gaps.append("quality_repetition")

    return {
        "name": name,
        "proxy_baseline": bool(scale.get("proxy_baseline", False)),
        "artifacts": {
            "layercake": layercake,
            "transformer": transformer,
        },
        "commands": commands,
        "metrics": {
            "cpu_generation_speed_ratio": (
                generation_cpu.get("speed_ratio") if generation_cpu else None
            ),
            "gpu_generation_speed_ratio": (
                generation_gpu.get("speed_ratio") if generation_gpu else None
            ),
            "cpu_prefill_speed_ratio": (
                resources.get("metrics", {}).get("prefill_speed_ratio")
                if resources
                else None
            ),
            "cpu_resource_generation_speed_ratio": (
                resources.get("metrics", {}).get("generation_speed_ratio")
                if resources
                else None
            ),
            "quality_gates": quality.get("quality_gates") if quality else None,
        },
        "gaps": sorted(set(gaps)),
        "outputs": {
            "generation_cpu": str(generation_cpu_path.relative_to(ROOT)).replace("\\", "/"),
            "generation_gpu": (
                str(generation_gpu_path.relative_to(ROOT)).replace("\\", "/")
                if run_gpu
                else None
            ),
            "generation_quality_cpu": str(quality_path.relative_to(ROOT)).replace("\\", "/"),
            "cpu_resources": str(resources_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-scale LayerCake vs transformer matrix across speed, "
            "latency, and generation quality for CPU/GPU/mobile-proxy reporting."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--new-bytes", type=int, default=32)
    parser.add_argument("--quality-bytes", type=int, default=96)
    parser.add_argument("--quality-no-repeat-ngram", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument(
        "--output",
        default="results/dominance/multiscale_matrix_report.json",
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    cuda_available = _run(
        [
            args.python,
            "-c",
            "import torch; print('1' if torch.cuda.is_available() else '0')",
        ]
    )
    run_gpu = (not args.skip_gpu) and cuda_available["status"] == "PASS" and "1" in cuda_available["stdout_tail"]

    per_scale = [
        _scale_result(
            python_exe=args.python,
            scale=scale,
            new_bytes=args.new_bytes,
            quality_bytes=args.quality_bytes,
            quality_no_repeat_ngram=args.quality_no_repeat_ngram,
            cpu_threads=args.cpu_threads,
            run_gpu=run_gpu,
        )
        for scale in DEFAULT_SCALES
    ]

    mobile_proxy_exec = _run([args.python, "scripts/benchmark_cpu_mobile_proxy.py"])
    mobile_proxy = _load(ROOT / "results" / "dominance" / "mobile_cpu_proxy.json")

    all_gaps = sorted({gap for row in per_scale for gap in row["gaps"]})

    report = {
        "status": "PASS" if not all_gaps else "OPEN",
        "scope": (
            "Cross-scale matrix over LayerCake vs transformer speed/latency/quality "
            "for CPU and GPU plus mobile-proxy evidence."
        ),
        "run_gpu": run_gpu,
        "scales": per_scale,
        "mobile_proxy": {
            "exec": mobile_proxy_exec,
            "result": mobile_proxy,
        },
        "gaps": all_gaps,
        "next_focus": [
            "cpu_prefill_latency",
            "gpu_generation_speed",
            "quality_distinct_trigram",
            "quality_repetition",
        ],
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
