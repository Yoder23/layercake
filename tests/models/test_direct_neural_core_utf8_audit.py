import json
from pathlib import Path

import pytest

from layercake.evaluation.direct_core_utf8_audit import Utf8AuditError, build_result, verify_result


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/moonshot/postrelease/direct_neural_core_utf8_audit_v1.json"


def test_v1_utf8_audit_recomputes_local_gap():
    result = build_result(ROOT)
    assert result["status"] == "FAIL_V1_UTF8_ACTION_ATOMICITY_AND_HOST_VALIDATION"
    assert result["tokenizer_probe"]["split_fragment_hex"] == ["e2", "80", "9c"]
    assert result["tokenizer_probe"]["invalid_standalone_fragments"] == 3
    assert result["tokenizer_probe"]["lossless_sequence_roundtrip"] is True
    assert result["gates"]["every_action_realizes_valid_utf8"] is False
    assert result["gates"]["host_fails_closed_before_invalid_output"] is False


def test_v1_utf8_audit_verifier_rejects_mutation(tmp_path):
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    value["gates"]["every_action_realizes_valid_utf8"] = True
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Utf8AuditError, match="differs from recomputation"):
        verify_result(ROOT, path)
