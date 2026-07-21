from scripts.verify_tiny_real_train_smoke import verify


def _report() -> dict:
    return {
        "status": "FAIL",
        "scope": "tiny",
        "device": "cuda",
        "steps": 1,
        "train_bytes": 100,
        "eval_bytes": 20,
        "data_split": {
            "train_sha256": "a" * 64,
            "eval_sha256": "b" * 64,
            "disjoint_by_construction": True,
        },
        "summary_gates": {"all_scales_pass": False},
        "scales": [
            {
                "layercake": {
                    "train": {"elapsed_seconds": 0.1, "steps_per_second": 10.0},
                    "general_bpb": 8.0,
                },
                "baseline": {
                    "train": {"elapsed_seconds": 0.2, "steps_per_second": 5.0},
                    "general_bpb": 7.0,
                },
                "qa_samples": [
                    {
                        "layercake": {"text": "x"},
                        "baseline": {"text": "y"},
                    }
                ],
            }
        ],
    }


def test_tiny_real_train_smoke_integrity_passes_for_fresh_paired_evidence(tmp_path):
    result = verify(_report(), output_path=tmp_path / "out.json", root=tmp_path)
    assert result["status"] == "PASS"
    assert result["source_status"] == "FAIL"
    assert result["gates"]["dominance_result_retained"] is True


def test_tiny_real_train_smoke_integrity_fails_without_split_hashes(tmp_path):
    report = _report()
    report["data_split"] = {}
    result = verify(report, output_path=tmp_path / "out.json", root=tmp_path)
    assert result["status"] == "FAIL"
    assert result["gates"]["same_train_eval_split_declared"] is False


def test_tiny_real_train_smoke_integrity_fails_without_transformer_training(tmp_path):
    report = _report()
    report["scales"][0]["baseline"]["train"]["elapsed_seconds"] = 0.0
    result = verify(report, output_path=tmp_path / "out.json", root=tmp_path)
    assert result["status"] == "FAIL"
    assert result["gates"]["transformer_training_observed"] is False
