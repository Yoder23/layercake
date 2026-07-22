"""Machine-enforced governance for the LayerCake Moonshot research campaign.

The campaign files use the JSON subset of YAML so the verifier has no optional parser
dependency.  Certificates are derived views: this module always recomputes values from
raw artifacts and never accepts a certificate's status or headline value as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = Path("moonshot")
RESULTS_DIR = Path("results/moonshot")
PHASE_KEYS = (
    "phase0_governance",
    "phase1_benchmark_truth",
    "phase2_cpu_quality_speed",
    "phase3_training_speed",
    "phase4_portable_domain",
    "phase5_multi_domain",
    "phase6_orchestration",
    "phase7_integrated_performance",
    "phase8_independent_verification",
)
CONTRACT_FILES = (
    "campaign.yaml",
    "claim_contract.yaml",
    "invalidation_matrix.yaml",
    "benchmark_contract.yaml",
    "data_contract.yaml",
    "security_contract.yaml",
)
STATIC_CONTRACT_FILES = tuple(name for name in CONTRACT_FILES if name != "campaign.yaml")
CONTRACT_FORMATS = {
    "claim_contract.yaml": "layercake-moonshot-claim-contract/1",
    "invalidation_matrix.yaml": "layercake-moonshot-invalidation-matrix/1",
    "benchmark_contract.yaml": "layercake-moonshot-benchmark-contract/1",
    "data_contract.yaml": "layercake-moonshot-data-contract/1",
    "security_contract.yaml": "layercake-moonshot-security-contract/1",
}
PHASE0_FORMAT = "layercake-moonshot-phase-certificate/1"
HANDOFF_FORMAT = "layercake-moonshot-handoff/1"
AUDIT_FORMAT = "layercake-moonshot-repository-audit/1"
TEST_FORMAT = "layercake-moonshot-test-results/1"


class CampaignVerificationError(RuntimeError):
    """A fail-closed campaign verification error."""


def _path(root: Path, relative: str | Path) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise CampaignVerificationError(f"path escapes repository: {relative}") from error
    return candidate


def read_document(path: Path) -> dict[str, Any]:
    """Read one JSON-compatible YAML object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CampaignVerificationError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CampaignVerificationError(
            f"{path} must use deterministic JSON-compatible YAML: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CampaignVerificationError(f"{path} must contain an object")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CampaignVerificationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def hash_named_files(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    for path in unique:
        try:
            relative = path.relative_to(root.resolve()).as_posix()
        except ValueError as error:
            raise CampaignVerificationError(f"hashed path escapes repository: {path}") from error
        raw = path.read_bytes()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def governed_source_paths(root: Path) -> list[Path]:
    paths = [root / "AGENTS.md", root / "pyproject.toml", root / ".gitignore"]
    for directory in ("layercake", "tests"):
        base = root / directory
        if base.is_dir():
            paths.extend(path for path in base.rglob("*.py") if "__pycache__" not in path.parts)
    paths.extend(root / CAMPAIGN_DIR / name for name in STATIC_CONTRACT_FILES)
    return [path for path in paths if path.is_file()]


def governed_source_hash(root: Path) -> str:
    """Hash code, tests, charter, and static contracts, excluding generated state/evidence."""

    return hash_named_files(root, governed_source_paths(root))


def _git(root: Path, *args: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise CampaignVerificationError(f"git {' '.join(args)} failed: {detail}")
    return process.stdout.strip()


def load_contracts(root: Path) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for name in CONTRACT_FILES:
        documents[name] = read_document(_path(root, CAMPAIGN_DIR / name))
    for name, expected in CONTRACT_FORMATS.items():
        if documents[name].get("format") != expected:
            raise CampaignVerificationError(f"invalid format in moonshot/{name}")
        if documents[name].get("campaign_version") != 1:
            raise CampaignVerificationError(f"unsupported campaign version in moonshot/{name}")
    validate_campaign_state(documents["campaign.yaml"], documents["claim_contract.yaml"])
    return documents


def validate_campaign_state(campaign: Mapping[str, Any], claim_contract: Mapping[str, Any]) -> None:
    if campaign.get("campaign") != "layercake-moonshot" or campaign.get("campaign_version") != 1:
        raise CampaignVerificationError("unsupported campaign identity or version")
    phases = campaign.get("phases")
    if not isinstance(phases, dict) or tuple(phases) != PHASE_KEYS:
        raise CampaignVerificationError("campaign phases are missing, reordered, or renamed")
    allowed = set(claim_contract.get("allowed_phase_statuses", []))
    statuses = [phases[key] for key in PHASE_KEYS]
    if any(status not in allowed for status in statuses):
        raise CampaignVerificationError("campaign contains an invalid phase status")
    current = campaign.get("current_phase")
    if not isinstance(current, int) or not 0 <= current <= 8:
        raise CampaignVerificationError("current_phase must be an integer in 0..8")
    first_not_pass = next((index for index, status in enumerate(statuses) if status != "PASS"), 9)
    if first_not_pass == 9:
        if current != 8:
            raise CampaignVerificationError("a completed campaign must remain at phase 8")
        return
    if current != first_not_pass:
        raise CampaignVerificationError("current_phase is inconsistent with the first non-PASS phase")
    if statuses[current] != "OPEN":
        raise CampaignVerificationError("the current phase must be OPEN")
    if any(status != "LOCKED" for status in statuses[current + 1 :]):
        raise CampaignVerificationError("a future phase is unlocked out of order")
    lineage = campaign.get("lineage")
    required_lineage = {
        "source_commit",
        "architecture_id",
        "architecture_hash",
        "abi_hash",
        "data_hashes",
        "core_checkpoint_hashes",
        "transformer_checkpoint_hashes",
        "cake_package_hashes",
        "router_hash",
        "runtime_hashes",
    }
    if not isinstance(lineage, dict) or set(lineage) != required_lineage:
        raise CampaignVerificationError("campaign lineage fields do not match the contract")


def _matching_files(root: Path, patterns: Sequence[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        if pattern.endswith("/**"):
            base = root / pattern[:-3]
            candidates = base.rglob("*") if base.is_dir() else []
        else:
            candidates = root.glob(pattern)
        for path in candidates:
            if path.is_file() and "__pycache__" not in path.parts:
                files.add(path)
    return sorted(files)


def component_hashes(root: Path, invalidation_matrix: Mapping[str, Any]) -> dict[str, str]:
    components = invalidation_matrix.get("components")
    if not isinstance(components, dict) or not components:
        raise CampaignVerificationError("invalidation matrix has no components")
    hashes: dict[str, str] = {}
    for name, policy in components.items():
        if not isinstance(policy, dict) or not isinstance(policy.get("paths"), list):
            raise CampaignVerificationError(f"invalid invalidation policy for {name}")
        hashes[name] = hash_named_files(root, _matching_files(root, policy["paths"]))
    return hashes


def changed_components(
    expected: Mapping[str, str], actual: Mapping[str, str]
) -> list[str]:
    """Return every missing, added, or changed governed component."""

    return sorted(
        name
        for name in set(expected) | set(actual)
        if expected.get(name) != actual.get(name)
    )


def prepare_phase0_audit(root: Path) -> dict[str, Any]:
    contracts = load_contracts(root)
    campaign = contracts["campaign.yaml"]
    source_commit = campaign["lineage"]["source_commit"]
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise CampaignVerificationError("Phase 0 requires a full 40-character source commit")
    _git(root, "cat-file", "-e", f"{source_commit}^{{commit}}")
    tracked = [line for line in _git(root, "ls-files").splitlines() if line]
    changed_paths = {
        line.replace("\\", "/")
        for command in (
            ("diff", "--name-only", source_commit, "--"),
            ("ls-files", "--others", "--exclude-standard"),
        )
        for line in _git(root, *command).splitlines()
        if line
    }
    governed_component_paths = {
        path.relative_to(root).as_posix()
        for policy in contracts["invalidation_matrix.yaml"]["components"].values()
        for path in _matching_files(root, policy["paths"])
    }
    legacy_certificates = sorted(
        path.relative_to(root).as_posix()
        for path in (root / RESULTS_DIR).glob("**/release_certificate.json")
        if "phase0" not in path.parts
    )
    audit = {
        "format": AUDIT_FORMAT,
        "campaign_version": 1,
        "source_commit": source_commit,
        "source_commit_subject": _git(root, "show", "-s", "--format=%s", source_commit),
        "source_commit_tree": _git(root, "show", "-s", "--format=%T", source_commit),
        "branch": _git(root, "branch", "--show-current"),
        "remote": _git(root, "remote", "get-url", "origin", check=False) or None,
        "tracked_file_count": len(tracked),
        "python_module_count": len(list((root / "layercake").rglob("*.py"))),
        "python_test_file_count": len(list((root / "tests").rglob("test_*.py"))),
        "governed_source_sha256": governed_source_hash(root),
        "component_hashes": component_hashes(root, contracts["invalidation_matrix.yaml"]),
        "legacy_release_certificates": legacy_certificates,
        "legacy_evidence_policy": "historical_only_not_inherited",
        "phase0_scope": [
            "campaign governance",
            "state machine",
            "contracts",
            "verifier",
            "verifier tests",
        ],
        "phase0_changed_paths": sorted(changed_paths),
        "architecture_or_training_changed": bool(changed_paths & governed_component_paths),
    }
    output = _path(root, RESULTS_DIR / "phase0/repository_audit.json")
    _atomic_write(output, audit)
    return audit


def _junit_totals(path: Path) -> dict[str, int | float]:
    try:
        document = ET.parse(path)
    except (OSError, ET.ParseError) as error:
        raise CampaignVerificationError(f"cannot parse JUnit evidence {path}: {error}") from error
    root = document.getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise CampaignVerificationError("JUnit evidence contains no testsuite")
    totals: dict[str, int | float] = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "duration_seconds": 0.0,
    }
    for suite in suites:
        totals["tests"] = int(totals["tests"]) + int(suite.attrib.get("tests", 0))
        totals["failures"] = int(totals["failures"]) + int(suite.attrib.get("failures", 0))
        totals["errors"] = int(totals["errors"]) + int(suite.attrib.get("errors", 0))
        totals["skipped"] = int(totals["skipped"]) + int(suite.attrib.get("skipped", 0))
        totals["duration_seconds"] = float(totals["duration_seconds"]) + float(
            suite.attrib.get("time", 0.0)
        )
    return totals


def record_test_results(root: Path, junit: Path, command: str) -> dict[str, Any]:
    junit = junit.resolve()
    try:
        relative_junit = junit.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise CampaignVerificationError("JUnit evidence must be inside the repository") from error
    totals = _junit_totals(junit)
    passed = int(totals["tests"]) - int(totals["failures"]) - int(totals["errors"]) - int(
        totals["skipped"]
    )
    result = {
        "format": TEST_FORMAT,
        "campaign_version": 1,
        "status": "PASS"
        if int(totals["tests"]) > 0
        and int(totals["failures"]) == 0
        and int(totals["errors"]) == 0
        else "FAIL",
        "command": command,
        "tests": int(totals["tests"]),
        "passed": passed,
        "failures": int(totals["failures"]),
        "errors": int(totals["errors"]),
        "skipped": int(totals["skipped"]),
        "duration_seconds": float(totals["duration_seconds"]),
        "junit_path": relative_junit,
        "junit_sha256": sha256_file(junit),
        "governed_source_sha256": governed_source_hash(root),
        "python": sys.version,
        "platform": platform.platform(),
    }
    _atomic_write(_path(root, RESULTS_DIR / "phase0/test_results.json"), result)
    return result


def validate_artifact_manifest(root: Path, manifest: Mapping[str, str]) -> None:
    if not manifest:
        raise CampaignVerificationError("artifact manifest is empty")
    for relative, expected_hash in manifest.items():
        path = _path(root, relative)
        if not path.is_file():
            raise CampaignVerificationError(f"required artifact is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected_hash:
            raise CampaignVerificationError(
                f"stale or modified artifact {relative}: expected {expected_hash}, got {actual}"
            )


def _lookup(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise CampaignVerificationError(f"raw field does not exist: {dotted_path}")
    return current


def _filtered_values(records: Sequence[Mapping[str, Any]], derivation: Mapping[str, Any]) -> list[float]:
    where = derivation.get("where", {})
    if not isinstance(where, dict):
        raise CampaignVerificationError("derivation where clause must be an object")
    field = derivation.get("field")
    if not isinstance(field, str):
        raise CampaignVerificationError("derivation requires a raw field")
    values: list[float] = []
    for record in records:
        if all(_lookup(record, key) == expected for key, expected in where.items()):
            raw = _lookup(record, field)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                raise CampaignVerificationError(f"raw metric {field} is not finite numeric evidence")
            values.append(float(raw))
    if not values:
        raise CampaignVerificationError("derivation selected no raw observations")
    return values


def recompute_derivation(raw_document: Mapping[str, Any], derivation: Mapping[str, Any]) -> float:
    """Recompute a supported headline derivation from raw records."""

    records = raw_document.get("records")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise CampaignVerificationError("raw evidence must contain an object-valued records list")
    operation = derivation.get("operation")
    if operation == "ratio":
        numerator = _filtered_values(records, derivation.get("numerator", {}))
        denominator = _filtered_values(records, derivation.get("denominator", {}))
        numerator_value = statistics.fmean(numerator)
        denominator_value = statistics.fmean(denominator)
        if denominator_value == 0:
            raise CampaignVerificationError("ratio denominator is zero")
        return numerator_value / denominator_value
    values = _filtered_values(records, derivation)
    if operation == "count":
        return float(len(values))
    if operation == "sum":
        return float(sum(values))
    if operation == "mean":
        return float(statistics.fmean(values))
    if operation == "median":
        return float(statistics.median(values))
    if operation == "min":
        return float(min(values))
    if operation == "max":
        return float(max(values))
    if operation == "nearest_rank_quantile":
        quantile = derivation.get("quantile")
        if not isinstance(quantile, (int, float)) or not 0 < quantile <= 1:
            raise CampaignVerificationError("quantile must be in (0, 1]")
        ordered = sorted(values)
        return ordered[max(0, math.ceil(float(quantile) * len(ordered)) - 1)]
    raise CampaignVerificationError(f"unsupported derivation operation: {operation}")


def verify_derived_claim(root: Path, phase: int, claim: Mapping[str, Any]) -> float:
    raw_path = claim.get("raw_artifact")
    if not isinstance(raw_path, str):
        raise CampaignVerificationError("headline claim has no raw artifact")
    required_prefix = f"results/moonshot/phase{phase}/raw_runs/"
    if not raw_path.replace("\\", "/").startswith(required_prefix):
        raise CampaignVerificationError("headline raw artifact is outside the phase raw_runs directory")
    path = _path(root, raw_path)
    expected_hash = claim.get("raw_sha256")
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise CampaignVerificationError("headline raw artifact hash is missing or stale")
    derivation = claim.get("derivation")
    if not isinstance(derivation, dict):
        raise CampaignVerificationError("headline value is hard-coded rather than derived")
    recomputed = recompute_derivation(read_document(path), derivation)
    claimed = claim.get("value")
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)):
        raise CampaignVerificationError("headline value must be numeric")
    tolerance = claim.get("absolute_tolerance", 1e-9)
    if not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise CampaignVerificationError("invalid headline tolerance")
    if not math.isclose(float(claimed), recomputed, rel_tol=0.0, abs_tol=float(tolerance)):
        raise CampaignVerificationError(
            f"headline value {claimed} does not recompute from raw evidence ({recomputed})"
        )
    return recomputed


def _threshold_passes(value: float, operator: str, threshold: float) -> bool:
    if operator == "lt":
        return value < threshold
    if operator == "le":
        return value <= threshold
    if operator == "eq":
        return math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12)
    if operator == "ge":
        return value >= threshold
    if operator == "gt":
        return value > threshold
    raise CampaignVerificationError(f"unsupported gate operator: {operator}")


def validate_required_gates(
    root: Path,
    phase: int,
    certificate: Mapping[str, Any],
    claim_contract: Mapping[str, Any],
) -> dict[str, float]:
    """Recompute each contract gate and reject missing, duplicate, or failed gates."""

    requirements = claim_contract["phase_requirements"][str(phase)].get("required_gates", [])
    claims = certificate.get("claims", [])
    if not isinstance(claims, list):
        raise CampaignVerificationError("certificate claims must be a list")
    by_gate: dict[str, Mapping[str, Any]] = {}
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("gate_id"), str):
            raise CampaignVerificationError("every phase claim requires a gate_id")
        if claim["gate_id"] in by_gate:
            raise CampaignVerificationError(f"duplicate gate claim: {claim['gate_id']}")
        by_gate[claim["gate_id"]] = claim
    required_ids = {gate["id"] for gate in requirements}
    missing = sorted(required_ids - set(by_gate))
    if missing:
        raise CampaignVerificationError(f"required gate claims are missing: {missing}")
    recomputed: dict[str, float] = {}
    for gate in requirements:
        claim = by_gate[gate["id"]]
        value = verify_derived_claim(root, phase, claim)
        threshold = gate["threshold"]
        if not isinstance(threshold, (int, float)):
            raise CampaignVerificationError(f"gate {gate['id']} has a non-numeric threshold")
        if not _threshold_passes(value, gate["operator"], float(threshold)):
            raise CampaignVerificationError(
                f"gate {gate['id']} failed: {value} {gate['operator']} {threshold} is false"
            )
        recomputed[gate["id"]] = value
    return recomputed


def validate_lineage_consistency(
    campaign: Mapping[str, Any], certificate: Mapping[str, Any], phase: int
) -> None:
    lineage = campaign["lineage"]
    certificate_lineage = certificate.get("lineage")
    if not isinstance(certificate_lineage, dict):
        raise CampaignVerificationError(f"Phase {phase} certificate has no lineage")
    if certificate_lineage.get("source_commit") != lineage.get("source_commit"):
        raise CampaignVerificationError("certificate source commit is outside the campaign lineage")
    for key, expected in lineage.items():
        if key == "source_commit" or expected in (None, {}, []):
            continue
        if certificate_lineage.get(key) != expected:
            raise CampaignVerificationError(f"certificate lineage differs for {key}")
    if phase >= 2:
        for key in ("architecture_id", "architecture_hash"):
            if not certificate_lineage.get(key):
                raise CampaignVerificationError(f"Phase {phase} requires lineage field {key}")
    if phase >= 4 and not certificate_lineage.get("abi_hash"):
        raise CampaignVerificationError(f"Phase {phase} requires an ABI hash")


def validate_component_snapshot(
    root: Path,
    phase: int,
    certificate: Mapping[str, Any],
    invalidation_matrix: Mapping[str, Any],
) -> None:
    expected = certificate.get("component_hashes")
    if not isinstance(expected, dict) or not expected:
        raise CampaignVerificationError("certificate has no governed component snapshot")
    actual = component_hashes(root, invalidation_matrix)
    changed = changed_components(expected, actual)
    invalidating = [
        name
        for name in changed
        if name not in invalidation_matrix["components"]
        or invalidation_matrix["components"][name]["invalidates_from_phase"] <= phase
    ]
    if invalidating:
        raise CampaignVerificationError(
            f"architecture, data, or runtime changes made the certificate stale: {invalidating}"
        )


def validate_seed_evidence(raw_documents: Sequence[Mapping[str, Any]], minimum_unique: int) -> list[int]:
    seeds: set[int] = set()
    failed_seeds: set[int] = set()
    for document in raw_documents:
        for record in document.get("records", []):
            if isinstance(record, dict) and isinstance(record.get("seed"), int):
                seeds.add(record["seed"])
        for seed in document.get("failed_seeds", []):
            if not isinstance(seed, int):
                raise CampaignVerificationError("failed seed identifiers must be integers")
            failed_seeds.add(seed)
    if len(seeds) < minimum_unique:
        raise CampaignVerificationError(
            f"missing seeds: required {minimum_unique}, observed {len(seeds)}"
        )
    if failed_seeds - seeds:
        raise CampaignVerificationError("failed seeds were not preserved in the raw records")
    return sorted(seeds)


def validate_baselines(certificate: Mapping[str, Any], claim_contract: Mapping[str, Any]) -> None:
    required = set(claim_contract["phase_requirements"]["1"]["required_baselines"])
    baselines = certificate.get("baselines")
    if not isinstance(baselines, list):
        raise CampaignVerificationError("Phase 1 certificate has no baseline inventory")
    by_id = {row.get("id"): row for row in baselines if isinstance(row, dict)}
    missing = sorted(required - set(by_id))
    if missing:
        raise CampaignVerificationError(f"required baselines are missing: {missing}")
    for identifier in ("optimized_cpu_transformer", "optimized_gpu_transformer"):
        baseline = by_id[identifier]
        runtime = baseline.get("runtime")
        if not isinstance(runtime, dict):
            raise CampaignVerificationError(f"{identifier} has no runtime provenance")
        if runtime.get("execution") == "eager_python":
            raise CampaignVerificationError(f"{identifier} is an invalid eager Python baseline")
        if runtime.get("deployment_grade") is not True or runtime.get("kv_cache") is not True:
            raise CampaignVerificationError(f"{identifier} is not a credible optimized baseline")
        if not runtime.get("name") or not runtime.get("version"):
            raise CampaignVerificationError(f"{identifier} runtime identity is incomplete")


def validate_matched_quality(
    certificate: Mapping[str, Any], benchmark_contract: Mapping[str, Any] | None = None
) -> None:
    speed_claims = [
        claim
        for claim in certificate.get("claims", [])
        if isinstance(claim, dict)
        and claim.get("kind") in {"throughput", "latency", "speed_ratio", "training_speed"}
        and claim.get("promoted") is True
    ]
    if not speed_claims:
        return
    match = certificate.get("quality_match")
    if not isinstance(match, dict):
        raise CampaignVerificationError("promoted speed claims lack matched-quality evidence")
    dimensions = (
        benchmark_contract.get("matched_quality_dimensions", [])
        if benchmark_contract is not None
        else []
    )
    if not dimensions:
        dimensions = [
            "heldout_bpb",
            "functional_task_quality",
            "instruction_following",
            "invalid_output_rate",
            "repetition",
            "coherence",
            "domain_success",
        ]
    failed = [name for name in dimensions if match.get(name) is not True]
    if failed:
        raise CampaignVerificationError(f"quality-unmatched speed claim: failed {failed}")
    if not match.get("layercake_checkpoint_sha256") or not match.get(
        "transformer_checkpoint_sha256"
    ):
        raise CampaignVerificationError("matched-quality evidence lacks checkpoint identity")


def validate_semantic_portability(certificate: Mapping[str, Any]) -> None:
    claims = certificate.get("claims", [])
    semantic = [
        claim
        for claim in claims
        if isinstance(claim, dict)
        and claim.get("kind") == "semantic_portability"
        and claim.get("promoted") is True
    ]
    if not semantic:
        return
    evidence = certificate.get("semantic_portability")
    if not isinstance(evidence, dict):
        raise CampaignVerificationError("semantic portability has no task evidence")
    source = evidence.get("source_success_task_ids")
    if not isinstance(source, list) or not source or not all(isinstance(item, str) for item in source):
        raise CampaignVerificationError("semantic portability requires source task successes")
    if len(source) != len(set(source)):
        raise CampaignVerificationError("source success task IDs are duplicated")
    receivers = evidence.get("receivers")
    if not isinstance(receivers, list) or not receivers:
        raise CampaignVerificationError("semantic portability has no receivers")
    source_set = set(source)
    for receiver in receivers:
        successes = receiver.get("success_task_ids") if isinstance(receiver, dict) else None
        if not isinstance(successes, list) or not source_set.issubset(set(successes)):
            raise CampaignVerificationError("a receiver lost one or more source task successes")
        if receiver.get("receiver_training_examples") != 0:
            raise CampaignVerificationError("receiver training invalidates semantic portability")
        if receiver.get("calibration_performed") is not False:
            raise CampaignVerificationError("receiver calibration invalidates semantic portability")


def _governance_hashes(root: Path) -> dict[str, str]:
    paths = [Path("AGENTS.md"), Path("layercake/moonshot_campaign.py")]
    paths.extend(CAMPAIGN_DIR / name for name in CONTRACT_FILES)
    return {path.as_posix(): sha256_file(_path(root, path)) for path in paths}


def _phase0_inputs(root: Path, contracts: Mapping[str, Mapping[str, Any]]) -> tuple[dict, dict]:
    audit_path = _path(root, RESULTS_DIR / "phase0/repository_audit.json")
    tests_path = _path(root, RESULTS_DIR / "phase0/test_results.json")
    audit = read_document(audit_path)
    tests = read_document(tests_path)
    if audit.get("format") != AUDIT_FORMAT:
        raise CampaignVerificationError("Phase 0 repository audit format is invalid")
    if tests.get("format") != TEST_FORMAT:
        raise CampaignVerificationError("Phase 0 test evidence format is invalid")
    source_hash = governed_source_hash(root)
    if audit.get("governed_source_sha256") != source_hash:
        raise CampaignVerificationError("repository audit is stale for the governed source")
    if tests.get("governed_source_sha256") != source_hash:
        raise CampaignVerificationError("test evidence is stale for the governed source")
    if tests.get("status") != "PASS" or tests.get("failures") != 0 or tests.get("errors") != 0:
        raise CampaignVerificationError("the complete regression suite is not green")
    if tests.get("tests", 0) <= 0 or tests.get("passed", -1) < 0:
        raise CampaignVerificationError("test evidence contains no executed tests")
    junit = _path(root, tests.get("junit_path", ""))
    if not junit.is_file() or sha256_file(junit) != tests.get("junit_sha256"):
        raise CampaignVerificationError("JUnit evidence is missing or stale")
    campaign = contracts["campaign.yaml"]
    if audit.get("source_commit") != campaign["lineage"]["source_commit"]:
        raise CampaignVerificationError("repository audit and campaign source commits differ")
    actual_components = component_hashes(root, contracts["invalidation_matrix.yaml"])
    changed = changed_components(audit.get("component_hashes", {}), actual_components)
    if changed:
        raise CampaignVerificationError(f"governed components changed after audit: {changed}")
    return audit, tests


def _completion_tag(root: Path, claim_contract: Mapping[str, Any], phase: int) -> tuple[str, str | None]:
    tag = claim_contract["phase_requirements"][str(phase)]["completion_tag"]
    commit = _git(root, "rev-list", "-n", "1", tag, check=False) or None
    return tag, commit


def _promote_phase0(
    root: Path,
    contracts: dict[str, dict[str, Any]],
    audit: Mapping[str, Any],
    tests: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = contracts["campaign.yaml"]
    if campaign["phases"][PHASE_KEYS[0]] != "OPEN":
        raise CampaignVerificationError("Phase 0 is not open for promotion")
    campaign["phases"][PHASE_KEYS[0]] = "PASS"
    campaign["phases"][PHASE_KEYS[1]] = "OPEN"
    campaign["current_phase"] = 1
    campaign.setdefault("phase_records", {})["phase0"] = {
        "certificate": "results/moonshot/phase0/release_certificate.json",
        "handoff": "results/moonshot/phase0/handoff.json",
        "completion_tag": "layercake-moonshot-phase0",
    }
    _atomic_write(_path(root, CAMPAIGN_DIR / "campaign.yaml"), campaign)
    contracts["campaign.yaml"] = campaign
    validate_campaign_state(campaign, contracts["claim_contract.yaml"])
    contract_hashes = {
        f"moonshot/{name}": sha256_file(_path(root, CAMPAIGN_DIR / name))
        for name in CONTRACT_FILES
    }
    artifact_hashes = {
        "results/moonshot/phase0/repository_audit.json": sha256_file(
            _path(root, RESULTS_DIR / "phase0/repository_audit.json")
        ),
        "results/moonshot/phase0/test_results.json": sha256_file(
            _path(root, RESULTS_DIR / "phase0/test_results.json")
        ),
        tests["junit_path"]: tests["junit_sha256"],
    }
    certificate = {
        "format": PHASE0_FORMAT,
        "campaign_version": 1,
        "phase": 0,
        "status": "PASS",
        "scope": "governance_only_no_model_or_training_claim",
        "source_commit": campaign["lineage"]["source_commit"],
        "governed_source_sha256": governed_source_hash(root),
        "campaign_file_hashes": contract_hashes,
        "governance_hashes": _governance_hashes(root),
        "artifact_hashes": artifact_hashes,
        "component_hashes": audit["component_hashes"],
        "tests": {
            key: tests[key]
            for key in ("command", "tests", "passed", "failures", "errors", "skipped", "duration_seconds")
        },
        "headline_claims": [],
        "legacy_evidence_inherited": False,
        "unlocked_phase": 1,
        "completion_tag": "layercake-moonshot-phase0",
    }
    certificate_path = _path(root, RESULTS_DIR / "phase0/release_certificate.json")
    _atomic_write(certificate_path, certificate)
    handoff = {
        "format": HANDOFF_FORMAT,
        "campaign_version": 1,
        "phase": 0,
        "source_commit": campaign["lineage"]["source_commit"],
        "campaign_file_hashes": contract_hashes,
        "verifier_sha256": sha256_file(_path(root, "layercake/moonshot_campaign.py")),
        "certificate_path": "results/moonshot/phase0/release_certificate.json",
        "certificate_sha256": sha256_file(certificate_path),
        "test_results": certificate["tests"],
        "exact_continuation_requirements": [
            "Validate the layercake-moonshot-phase0 tag and Phase 0 certificate before edits.",
            "Read AGENTS.md and every file under moonshot/.",
            "Execute Phase 1 only: establish benchmark truth without redesigning LayerCake.",
            "Use the six required baselines and freeze the general-English quality thresholds.",
            "Preserve all raw runs and let only the verifier unlock Phase 2.",
        ],
        "next_command": "python -m layercake.moonshot_campaign verify-phase 0",
        "next_phase": 1,
    }
    _atomic_write(_path(root, RESULTS_DIR / "phase0/handoff.json"), handoff)
    return certificate, handoff


def _validate_phase0_outputs(
    root: Path,
    contracts: Mapping[str, Mapping[str, Any]],
    certificate: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> None:
    if certificate.get("format") != PHASE0_FORMAT or certificate.get("phase") != 0:
        raise CampaignVerificationError("Phase 0 certificate identity is invalid")
    if certificate.get("status") != "PASS" or certificate.get("headline_claims") != []:
        raise CampaignVerificationError("Phase 0 certificate contains an invalid claim or status")
    if certificate.get("scope") != "governance_only_no_model_or_training_claim":
        raise CampaignVerificationError("Phase 0 certificate exceeds its allowed scope")
    if handoff.get("format") != HANDOFF_FORMAT or handoff.get("phase") != 0:
        raise CampaignVerificationError("Phase 0 handoff identity is invalid")
    if certificate.get("source_commit") != contracts["campaign.yaml"]["lineage"]["source_commit"]:
        raise CampaignVerificationError("Phase 0 certificate source lineage is stale")
    validate_artifact_manifest(root, certificate.get("campaign_file_hashes", {}))
    validate_artifact_manifest(root, certificate.get("governance_hashes", {}))
    validate_artifact_manifest(root, certificate.get("artifact_hashes", {}))
    if handoff.get("certificate_sha256") != sha256_file(
        _path(root, handoff.get("certificate_path", ""))
    ):
        raise CampaignVerificationError("handoff certificate hash is stale")
    if handoff.get("verifier_sha256") != sha256_file(_path(root, "layercake/moonshot_campaign.py")):
        raise CampaignVerificationError("handoff verifier hash is stale")
    if handoff.get("campaign_file_hashes") != certificate.get("campaign_file_hashes"):
        raise CampaignVerificationError("handoff and certificate campaign hashes differ")


def verify_phase0(root: Path, *, promote: bool = True) -> dict[str, Any]:
    contracts = load_contracts(root)
    audit, tests = _phase0_inputs(root, contracts)
    state = contracts["campaign.yaml"]["phases"][PHASE_KEYS[0]]
    if state == "OPEN":
        if not promote:
            return {"phase": 0, "status": "OPEN", "passed": False, "ready_for_promotion": True}
        certificate, handoff = _promote_phase0(root, contracts, audit, tests)
        contracts = load_contracts(root)
    elif state == "PASS":
        certificate = read_document(_path(root, RESULTS_DIR / "phase0/release_certificate.json"))
        handoff = read_document(_path(root, RESULTS_DIR / "phase0/handoff.json"))
    else:
        raise CampaignVerificationError(f"Phase 0 has invalid state {state}")
    _validate_phase0_outputs(root, contracts, certificate, handoff)
    tag, tag_commit = _completion_tag(root, contracts["claim_contract.yaml"], 0)
    head = _git(root, "rev-parse", "HEAD")
    clean = _git(root, "status", "--porcelain=v1") == ""
    sealed = tag_commit == head and clean
    return {
        "phase": 0,
        "status": "PASS" if sealed else "READY_FOR_COMMIT_AND_TAG",
        "passed": sealed,
        "phase1_unlocked": contracts["campaign.yaml"]["phases"][PHASE_KEYS[1]] == "OPEN",
        "completion_tag": tag,
        "tag_commit": tag_commit,
        "head_commit": head,
        "working_tree_clean": clean,
        "certificate_sha256": sha256_file(
            _path(root, RESULTS_DIR / "phase0/release_certificate.json")
        ),
    }


def _load_raw_documents(root: Path, phase: int, certificate: Mapping[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for claim in certificate.get("claims", []):
        if not isinstance(claim, dict) or not isinstance(claim.get("raw_artifact"), str):
            continue
        relative = claim["raw_artifact"]
        if relative not in seen:
            documents.append(read_document(_path(root, relative)))
            seen.add(relative)
        verify_derived_claim(root, phase, claim)
    return documents


def verify_later_phase(root: Path, phase: int) -> dict[str, Any]:
    contracts = load_contracts(root)
    campaign = contracts["campaign.yaml"]
    if campaign["phases"][PHASE_KEYS[phase]] != "PASS":
        raise CampaignVerificationError(f"Phase {phase} is not PASS")
    if any(campaign["phases"][PHASE_KEYS[index]] != "PASS" for index in range(phase)):
        raise CampaignVerificationError("a prior phase is not PASS")
    certificate_path = _path(root, RESULTS_DIR / f"phase{phase}/release_certificate.json")
    certificate = read_document(certificate_path)
    if certificate.get("phase") != phase:
        raise CampaignVerificationError(f"Phase {phase} certificate identity is invalid")
    raw_documents = _load_raw_documents(root, phase, certificate)
    validate_required_gates(root, phase, certificate, contracts["claim_contract.yaml"])
    minimum = contracts["claim_contract.yaml"]["phase_requirements"][str(phase)][
        "minimum_unique_seeds"
    ]
    validate_seed_evidence(raw_documents, minimum)
    if phase == 1:
        validate_baselines(certificate, contracts["claim_contract.yaml"])
    validate_matched_quality(certificate, contracts["benchmark_contract.yaml"])
    validate_semantic_portability(certificate)
    validate_lineage_consistency(campaign, certificate, phase)
    validate_component_snapshot(root, phase, certificate, contracts["invalidation_matrix.yaml"])
    validate_artifact_manifest(root, certificate.get("artifact_hashes", {}))
    return {"phase": phase, "status": "PASS", "passed": True}


def campaign_status(root: Path) -> dict[str, Any]:
    contracts = load_contracts(root)
    campaign = contracts["campaign.yaml"]
    phase0_tag, tag_commit = _completion_tag(root, contracts["claim_contract.yaml"], 0)
    return {
        "campaign": campaign["campaign"],
        "campaign_version": campaign["campaign_version"],
        "current_phase": campaign["current_phase"],
        "phases": campaign["phases"],
        "lineage": campaign["lineage"],
        "phase0_tag": phase0_tag,
        "phase0_tag_commit": tag_commit,
        "governed_source_sha256": governed_source_hash(root),
    }


def verify_all(root: Path) -> dict[str, Any]:
    campaign = load_contracts(root)["campaign.yaml"]
    results: list[dict[str, Any]] = []
    for phase, key in enumerate(PHASE_KEYS):
        status = campaign["phases"][key]
        if status != "PASS":
            results.append({"phase": phase, "status": status, "passed": None})
            continue
        results.append(verify_phase0(root, promote=False) if phase == 0 else verify_later_phase(root, phase))
    completed_valid = all(
        results[phase].get("passed") is True
        for phase, key in enumerate(PHASE_KEYS)
        if campaign["phases"][key] == "PASS"
    )
    return {"campaign": "layercake-moonshot", "completed_phases_valid": completed_valid, "phases": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m layercake.moonshot_campaign")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("status", help="show the validated campaign state")
    verify = subcommands.add_parser("verify-phase", help="verify and, when eligible, promote a phase")
    verify.add_argument("phase", type=int, choices=range(9))
    subcommands.add_parser("verify-all", help="verify all completed phases")
    subcommands.add_parser("prepare-phase0", help=argparse.SUPPRESS)
    record = subcommands.add_parser("record-tests", help=argparse.SUPPRESS)
    record.add_argument("--junit", type=Path, required=True)
    record.add_argument("--command-line", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "status":
            result = campaign_status(root)
        elif args.command == "prepare-phase0":
            result = prepare_phase0_audit(root)
        elif args.command == "record-tests":
            junit = args.junit if args.junit.is_absolute() else root / args.junit
            result = record_test_results(root, junit, args.command_line)
        elif args.command == "verify-phase":
            result = verify_phase0(root) if args.phase == 0 else verify_later_phase(root, args.phase)
        elif args.command == "verify-all":
            result = verify_all(root)
        else:  # pragma: no cover - argparse guarantees this branch is unreachable
            raise CampaignVerificationError(f"unsupported command: {args.command}")
    except CampaignVerificationError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "verify-phase" and result.get("passed") is not True:
        return 2
    if args.command == "verify-all" and result.get("completed_phases_valid") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
