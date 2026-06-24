from layercake.rolling.commit import ModelCommit


def test_model_commit_save_load_verify_and_compare(tmp_path):
    parent = ModelCommit.create(
        parent_commit_id=None,
        branch="main",
        status="passed",
        model_family_id="toy",
        abi_hash="abi",
        input_interface_hash="input",
        byte_patch_hash="patch",
        module_hashes={"a": "1"},
        artifact_paths={"a": str(tmp_path / "a.pt")},
        rubric_hash="rubric",
        message="parent",
    )
    child = ModelCommit.create(
        parent_commit_id=parent.commit_id,
        branch="main",
        status="candidate",
        model_family_id="toy",
        abi_hash="abi",
        input_interface_hash="input",
        byte_patch_hash="patch",
        module_hashes={"a": "2"},
        artifact_paths={"a": str(tmp_path / "a2.pt")},
        rubric_hash="rubric2",
        message="child",
    )
    child.save(tmp_path)
    loaded = ModelCommit.load(tmp_path / f"{child.commit_id}.json")
    assert loaded.verify()
    assert loaded.compare_to_parent(parent)["changed_modules"] == ["a"]
    assert loaded.mark_passed().status == "passed"
    assert loaded.mark_failed().status == "failed"
