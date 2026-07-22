from layercake.evaluation.moonshot_final_verifier import ALLOWED, REQUIRED_GATES


def test_final_certificate_has_exact_mandated_gate_count_and_statuses():
    assert len(REQUIRED_GATES) == 40
    assert REQUIRED_GATES[-1] == "overall_moonshot"
    assert ALLOWED == {
        "PASS", "FAIL", "OPEN", "INVALID_EVIDENCE",
        "NOT_RUN_NO_HARDWARE", "NOT_RUN_INSUFFICIENT_COMPUTE",
    }
