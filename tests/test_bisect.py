from layercake.rolling.bisect import bisect_regression


class _Commit:
    def __init__(self, commit_id, status):
        self.commit_id = commit_id
        self.eval_result_hashes = {"gate": status}


def test_bisect_returns_first_bad_commit():
    result = bisect_regression([_Commit("a", "pass"), _Commit("b", "fail"), _Commit("c", "fail")], "gate")
    assert result["first_bad_commit"] == "b"
