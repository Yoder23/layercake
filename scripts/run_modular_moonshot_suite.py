from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "moonshot_suite"


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
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _load(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Integrated moonshot suite for 25m standing, portable-domain transfer "
            "generation parity, CPU/GPU coverage, and multi-domain invariance."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output",
        default="results/moonshot_suite/modular_moonshot_report.json",
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    paths = {
        "lc25": "runs_experiment/scale25m_layercake_4x4_seed7002.pt",
        "tf26": "runs_experiment/scale26m_bpe512_seed6250.pt",
        "source_core": "runs_experiment/scale15m_transition_lw280_2300_noprofile.pt",
        "target_core": "runs_experiment/scale5m_seed4242.pt",
        "portable_python": "runs_experiment/portable_python_gru148k_seed6061.pt",
        "portable_game_dialogue": "runs_experiment/game_dialogue_smoke_gru.pt",
        "portable_game_lore": "runs_experiment/game_lore_smoke_gru.pt",
        "portable_game_quest": "runs_experiment/game_quest_state_smoke_gru.pt",
        "portable_technical": "runs_experiment/technical_text_smoke_gru.pt",
        "prompt_game": "tests/fixtures/game_dialogue_smoke.txt",
        "prompt_tech": "tests/fixtures/technical_text_smoke.txt",
    }

    commands: dict[str, dict[str, Any]] = {}

    # 25m standing on CPU/GPU + quality
    cpu_gen_path = RESULTS / "25m_generation_cpu.json"
    gpu_gen_path = RESULTS / "25m_generation_gpu.json"
    quality_path = RESULTS / "25m_generation_quality_cpu.json"
    commands["25m_generation_cpu"] = _run([
        args.python,
        "scripts/benchmark_generation.py",
        "--layercake", paths["lc25"],
        "--bpe", paths["tf26"],
        "--layercake-mode", "stateful_cached",
        "--new-bytes", "32",
        "--device", "cpu",
        "--cpu-threads", "1",
        "--output", str(cpu_gen_path.relative_to(ROOT)).replace("\\", "/"),
    ])
    commands["25m_generation_gpu"] = _run([
        args.python,
        "scripts/benchmark_generation.py",
        "--layercake", paths["lc25"],
        "--bpe", paths["tf26"],
        "--layercake-mode", "stateful_cached",
        "--new-bytes", "32",
        "--device", "cuda",
        "--output", str(gpu_gen_path.relative_to(ROOT)).replace("\\", "/"),
    ])
    commands["25m_generation_quality_cpu"] = _run([
        args.python,
        "scripts/benchmark_generation_quality.py",
        "--layercake", paths["lc25"],
        "--bpe", paths["tf26"],
        "--new-bytes", "64",
        "--no-repeat-ngram", "8",
        "--device", "cpu",
        "--cpu-threads", "1",
        "--output", str(quality_path.relative_to(ROOT)).replace("\\", "/"),
    ])

    # Transfer parity CPU/GPU using portable domain on two cores.
    transfer_cpu_path = RESULTS / "transfer_parity_cpu.json"
    transfer_gpu_path = RESULTS / "transfer_parity_gpu.json"
    for label, device, out_path in (
        ("transfer_parity_cpu", "cpu", transfer_cpu_path),
        ("transfer_parity_gpu", "cuda", transfer_gpu_path),
    ):
        commands[label] = _run([
            args.python,
            "scripts/eval_lossless_domain_decoder.py",
            "--decoder", paths["portable_python"],
            "--source-core", paths["source_core"],
            "--target-core", paths["target_core"],
            "--eval-file", paths["prompt_tech"],
            "--eval-source-label", f"portable-python-{device}",
            "--generation-bytes", "32",
            "--device", device,
            "--output", str(out_path.relative_to(ROOT)).replace("\\", "/"),
        ])

    # Multi-domain invariance CPU/GPU: alone vs many installed.
    invariance_cpu_path = RESULTS / "multi_domain_invariance_cpu.json"
    invariance_gpu_path = RESULTS / "multi_domain_invariance_gpu.json"
    other_domains = [
        paths["portable_game_lore"],
        paths["portable_game_quest"],
        paths["portable_technical"],
        paths["portable_python"],
    ]
    for label, device, out_path in (
        ("multi_domain_invariance_cpu", "cpu", invariance_cpu_path),
        ("multi_domain_invariance_gpu", "cuda", invariance_gpu_path),
    ):
        command = [
            args.python,
            "scripts/eval_multi_domain_runtime_invariance.py",
            "--target", paths["portable_game_dialogue"],
            "--prompt-file", paths["prompt_game"],
            "--prompt-bytes", "128",
            "--generation-bytes", "32",
            "--context-bytes", "128",
            "--device", device,
            "--output", str(out_path.relative_to(ROOT)).replace("\\", "/"),
        ]
        for item in other_domains:
            command.extend(["--others", item])
        commands[label] = _run(command)

    report = {
        "status": "PASS",
        "scope": (
            "25m standing, portable-domain transfer parity on CPU/GPU, and "
            "multi-domain modular invariance."
        ),
        "artifacts": paths,
        "commands": commands,
        "results": {
            "25m_generation_cpu": _load(cpu_gen_path),
            "25m_generation_gpu": _load(gpu_gen_path),
            "25m_generation_quality_cpu": _load(quality_path),
            "transfer_parity_cpu": _load(transfer_cpu_path),
            "transfer_parity_gpu": _load(transfer_gpu_path),
            "multi_domain_invariance_cpu": _load(invariance_cpu_path),
            "multi_domain_invariance_gpu": _load(invariance_gpu_path),
        },
    }

    failed = [name for name, item in report["commands"].items() if item["status"] != "PASS"]
    if failed:
        report["status"] = "FAIL"
        report["failed_commands"] = failed

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
