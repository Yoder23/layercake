from layercake.evaluation.quality import dataset_integrity


def test_dataset_integrity_detects_leakage():
    shared = b"this exact thirty two byte sequence is deliberately duplicated"
    result = dataset_integrity({
        "train": b"train-prefix " + shared,
        "validation": b"validation unique material",
        "test": b"test-prefix " + shared,
        "architecture_selection": b"architecture-only unique material",
    })
    assert result["status"] == "FAIL"
    assert result["cross_split_ngram_overlaps"]["test:train"] > 0
