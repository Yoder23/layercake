from __future__ import annotations


def bisect_regression(commits: list, gate_name: str) -> dict:
    last_good = None
    first_bad = None
    for commit in commits:
        results = commit.eval_result_hashes
        passed = results.get(gate_name, "pass") == "pass"
        if passed:
            last_good = commit.commit_id
        else:
            first_bad = commit.commit_id
            break
    return {
        "gate_name": gate_name,
        "first_bad_commit": first_bad,
        "last_good_commit": last_good,
        "changed_modules": [],
        "changed_rubrics": [],
        "likely_regression_causes": ["first failing gate in ordered commit list"] if first_bad else [],
    }
