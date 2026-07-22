from _common import ROOT

import json
import re
import subprocess
import sys
import time

from layercake.moonshot import source_tree_hash


command = [sys.executable, "-m", "pytest", "-q"]
tree_before = source_tree_hash()
started = time.perf_counter()
completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
seconds = time.perf_counter() - started
combined = completed.stdout + completed.stderr
match = re.search(r"(\d+) passed", combined)
tree_after = source_tree_hash()
result = {
    "format": "layercake-pytest-evidence/2",
    "status": "PASS" if completed.returncode == 0 and tree_before == tree_after else "FAIL",
    "command": command,
    "exit_code": completed.returncode,
    "passed": int(match.group(1)) if match else 0,
    "seconds": seconds,
    "source_tree_sha256": tree_after,
    "source_unchanged_during_tests": tree_before == tree_after,
    "summary": combined[-12000:],
}
output = ROOT / "results/moonshot/v2/pytest_evidence.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["status"] == "PASS" else 1)
