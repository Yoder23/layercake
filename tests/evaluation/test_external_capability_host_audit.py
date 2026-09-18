from pathlib import Path

import pytest

from layercake.evaluation.external_capability_host_audit import (
    ExternalCapabilityHostAuditError,
    audit,
    load_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "moonshot/postrelease_external_capability_host_interface_audit_successor_v3.json"


def _audit_bindings_match() -> bool:
    try:
        load_protocol(ROOT, PROTOCOL)
    except ExternalCapabilityHostAuditError:
        return False
    return True


AUDIT_BINDINGS_MATCH = _audit_bindings_match()


@pytest.mark.skipif(
    not AUDIT_BINDINGS_MATCH,
    reason="historical audit bindings do not apply to current development HEAD",
)
def test_sealed_host_interface_audit_identifies_repaired_construct_boundary():
    result = audit(ROOT, PROTOCOL)
    assert result["status"] == "HOST_CONSTRUCT_READY_EXTERNAL_ARTIFACT_NOT_YET_ACCEPTED"
    assert result["ownership"]["sealed_layercake_regression"] is False
    assert result["ownership"]["host_interface_scope_gap"] is False
    assert result["ownership"]["host_construct_ready"] is True
    assert result["runtime_changed"] is False


@pytest.mark.skipif(
    AUDIT_BINDINGS_MATCH,
    reason="historical audit bindings match this checkout",
)
def test_stale_host_interface_audit_fails_closed():
    with pytest.raises(ExternalCapabilityHostAuditError, match="binding changed"):
        audit(ROOT, PROTOCOL)


def test_audit_protocol_fails_closed_on_mutation(tmp_path):
    protocol = PROTOCOL.read_text(encoding="utf-8").replace(
        '"training_allowed": false', '"training_allowed": true'
    )
    changed = tmp_path / "protocol.json"
    changed.write_text(protocol, encoding="utf-8")
    with pytest.raises(ExternalCapabilityHostAuditError):
        load_protocol(ROOT, changed)
