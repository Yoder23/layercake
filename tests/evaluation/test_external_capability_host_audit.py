from pathlib import Path

import pytest

from layercake.evaluation.external_capability_host_audit import (
    ExternalCapabilityHostAuditError,
    audit,
    load_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "moonshot/postrelease_external_capability_host_interface_audit_repair1_v2.json"


def test_sealed_host_interface_audit_identifies_scope_gap():
    result = audit(ROOT, PROTOCOL)
    assert result["status"] == "HOST_SCOPE_GAP_DIRECT_ENGLISH_CORE_ARTIFACT_INTERFACE_ABSENT"
    assert result["ownership"]["sealed_layercake_regression"] is False
    assert result["ownership"]["host_interface_scope_gap"] is True
    assert result["runtime_changed"] is False


def test_audit_protocol_fails_closed_on_mutation(tmp_path):
    protocol = PROTOCOL.read_text(encoding="utf-8").replace(
        '"training_allowed": false', '"training_allowed": true'
    )
    changed = tmp_path / "protocol.json"
    changed.write_text(protocol, encoding="utf-8")
    with pytest.raises(ExternalCapabilityHostAuditError):
        load_protocol(ROOT, changed)
