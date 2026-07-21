from scripts.verify_question_inference_guardrails import verify


def _comparison(speed_ratio=2.0, lc_repeat=2, tx_repeat=5, lc_distinct=0.9, tx_distinct=0.5):
    return {
        "summary": {"mean_speed_ratio_layercake_over_transformer": speed_ratio},
        "samples": [
            {
                "layercake": {
                    "printable_ratio": 1.0,
                    "distinct_word_trigram": lc_distinct,
                    "max_repeat_8gram": lc_repeat,
                },
                "transformer": {
                    "printable_ratio": 1.0,
                    "distinct_word_trigram": tx_distinct,
                    "max_repeat_8gram": tx_repeat,
                },
            }
        ],
    }


def _question_comparison(layercake_text):
    row = _comparison(speed_ratio=3.5)
    row["samples"][0]["name"] = "xml_json_schema"
    row["samples"][0]["layercake"]["text"] = layercake_text
    return row


def test_question_inference_guardrails_pass_for_cpu_and_gpu_dominance():
    result = verify(
        cpu=_comparison(speed_ratio=3.5),
        gpu=_comparison(speed_ratio=1.2),
        min_cpu_speed_ratio=3.0,
        min_gpu_speed_ratio=1.0,
        min_printable=0.95,
    )
    assert result["status"] == "PASS"
    assert result["gates"]["cpu_speed_ratio_met"] is True
    assert result["gates"]["gpu_repetition_no_worse"] is True


def test_question_inference_guardrails_fail_when_named_prompt_is_not_answered():
    result = verify(
        cpu=_question_comparison("the state of the students is available"),
        gpu=_question_comparison("the state of the students is available"),
        min_cpu_speed_ratio=3.0,
        min_gpu_speed_ratio=1.0,
        min_printable=0.95,
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["cpu_question_relevance_met"] is False
    assert result["gates"]["gpu_question_relevance_met"] is False


def test_question_inference_guardrails_fail_on_gpu_speed_or_quality_regression():
    result = verify(
        cpu=_comparison(speed_ratio=3.5),
        gpu=_comparison(speed_ratio=0.8, lc_repeat=10, tx_repeat=5),
        min_cpu_speed_ratio=3.0,
        min_gpu_speed_ratio=1.0,
        min_printable=0.95,
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["gpu_speed_ratio_met"] is False
    assert result["gates"]["gpu_repetition_no_worse"] is False
