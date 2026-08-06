"""Recompute the post-release UTF-8 action-atomicity audit for core ABI v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from layercake.portable_domain import canonical_json_hash
from layercake.portable_token_plan import GENERIC_TOKENIZER_FORMAT, LosslessLexemePointerTokenizer
from layercake_extensions.direct_neural_core import DirectNeuralCoreHost


class Utf8AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Utf8AuditError(f"expected JSON object: {path}")
    return value


class _InvalidFixture:
    def generate_bytes(self, prompt: bytes | str, *, maximum_actions: int | None = None) -> bytes:
        return b"\x9c"


def build_result(root: Path) -> dict[str, Any]:
    canonical_path = root / "moonshot/canonical_direct_neural_core_abi_v1.json"
    canonical = _json(canonical_path)
    if canonical.get("version") != "lc-direct-neural-core/1" or canonical.get("external_boundary", {}).get("output") != "UTF-8 bytes":
        raise Utf8AuditError("v1 canonical UTF-8 output contract changed")
    character = "“"
    encoded = character.encode("utf-8")
    pieces = LosslessLexemePointerTokenizer.split(character)
    if b"".join(pieces) != encoded:
        raise Utf8AuditError("v1 tokenizer no longer round-trips the audit character")
    tokenizer = LosslessLexemePointerTokenizer(sorted(set(pieces)), format_version=GENERIC_TOKENIZER_FORMAT)
    fragment_rows = []
    for piece in pieces:
        action = tokenizer.lexeme_to_id[piece]
        decoded = tokenizer.decode_actions([action], [])
        valid = True
        error = None
        try:
            decoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            valid = False
            error = f"{type(exc).__name__}: {exc.reason}"
        fragment_rows.append({"fragment_hex": piece.hex(), "action": action, "valid_utf8": valid, "decode_error": error})
    combined = tokenizer.decode_actions([tokenizer.lexeme_to_id[piece] for piece in pieces], [])
    combined_text = combined.decode("utf-8", errors="strict")

    host = DirectNeuralCoreHost.__new__(DirectNeuralCoreHost)
    host.module = _InvalidFixture()
    host.active_cake_id = "audit-fixture"
    raw = host.generate("valid prompt")
    host_rejected = False
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        pass
    else:
        host_rejected = True
    result: dict[str, Any] = {
        "format": "layercake-postrelease-direct-core-utf8-audit/1",
        "status": "FAIL_V1_UTF8_ACTION_ATOMICITY_AND_HOST_VALIDATION",
        "historical_release_changed": False,
        "canonical": {"path": str(canonical_path.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(canonical_path), "declared_output": "UTF-8 bytes"},
        "tokenizer_probe": {
            "character": character,
            "utf8_hex": encoded.hex(),
            "split_fragment_hex": [piece.hex() for piece in pieces],
            "fragment_count": len(pieces),
            "invalid_standalone_fragments": sum(not row["valid_utf8"] for row in fragment_rows),
            "fragments": fragment_rows,
            "combined_roundtrip_hex": combined.hex(),
            "combined_roundtrip_text": combined_text,
            "lossless_sequence_roundtrip": combined == encoded,
            "unicode_atomic_actions": all(row["valid_utf8"] for row in fragment_rows),
        },
        "host_probe": {
            "fixture_returned_hex": raw.hex(),
            "fixture_valid_utf8": False,
            "v1_host_rejected_invalid_utf8_before_boundary": host_rejected,
            "v1_host_passed_bytes_through": raw == b"\x9c",
        },
        "gates": {
            "every_action_realizes_valid_utf8": all(row["valid_utf8"] for row in fragment_rows),
            "host_fails_closed_before_invalid_output": host_rejected,
            "combined_declared_text_roundtrips": combined_text == character,
        },
        "decision": {
            "v1_construct_invalidated": False,
            "v1_utf8_output_conformance": "FAILED",
            "repair_required": True,
            "required_repair": "A separately versioned interface must make every fixed and pointer lexeme a complete valid UTF-8 sequence and must strictly validate output before it crosses the host boundary.",
            "external_artifact_retraining_required": True,
            "reason": "Changing action atomicity changes tokenizer identity and model action semantics; an old checkpoint cannot inherit v2 conformance."
        },
        "claim_boundary": "This local audit proves a UTF-8 conformance gap in the post-release v1 direct-core action surface. It does not import ABI evidence, assess English quality, or alter any sealed Phase 0-8 certificate."
    }
    result["evidence_sha256"] = canonical_json_hash(result)
    return result


def verify_result(root: Path, result_path: Path) -> dict[str, Any]:
    stored = _json(result_path)
    expected = build_result(root)
    if stored != expected:
        raise Utf8AuditError("stored UTF-8 audit differs from recomputation")
    return {"status": "PASS", "evidence_sha256": expected["evidence_sha256"], "v1_utf8_output_conformance": "FAILED"}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "verify"))
    parser.add_argument("--output", default="results/moonshot/postrelease/direct_neural_core_utf8_audit_v1.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    output = (root / args.output).resolve()
    if args.command == "audit":
        if output.exists():
            raise Utf8AuditError(f"audit output is immutable: {output}")
        result = build_result(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        result = verify_result(root, output)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
