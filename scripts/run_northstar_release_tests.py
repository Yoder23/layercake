from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full release regression suite and emit compact evidence"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/breakthrough_equal/northstar_v22_pytest_summary.json"
        ),
    )
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as handle:
        junit_path = Path(handle.name)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"--junitxml={junit_path}",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter() - started
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    if junit_path.exists() and junit_path.stat().st_size:
        root = ET.parse(junit_path).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is not None:
            for key in counts:
                counts[key] = int(suite.attrib.get(key, 0))
    junit_path.unlink(missing_ok=True)
    passed = (
        completed.returncode == 0
        and counts["tests"] > 0
        and counts["failures"] == 0
        and counts["errors"] == 0
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "command": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--junitxml=<temporary-junit.xml>",
        ],
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "counts": counts,
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-4000:],
        "python": sys.version,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
