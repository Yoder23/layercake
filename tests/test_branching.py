from layercake.rolling.branching import BranchStore


def test_branch_store_create_checkout_tag(tmp_path):
    store = BranchStore(tmp_path)
    store.create_branch("main", "abc")
    store.checkout_commit("abc")
    store.tag_commit("abc", "v1")
    assert store.list_branches() == {"main": "abc"}
    assert store.list_commits("main") == ["abc"]
    assert (tmp_path / "HEAD").read_text(encoding="utf-8") == "abc"
