from layercake.rolling.manifest import DatasetManifest


def test_dataset_manifest_hashes_files(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("abc", encoding="utf-8")
    manifest = DatasetManifest.from_path(data, name="toy")
    assert manifest.name == "toy"
    assert manifest.total_bytes == 3
    assert manifest.compute_hash() == DatasetManifest.from_json(manifest.to_json()).compute_hash()
