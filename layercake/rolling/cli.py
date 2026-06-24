from __future__ import annotations

import argparse
import json
from pathlib import Path

from .branching import BranchStore
from .bisect import bisect_regression
from .cherrypick import cherry_pick_module
from .commit import ModelCommit
from .common import load_structured
from .diff import diff_commits
from .rubric import TrainingRubric


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="layercake.rolling")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("init")
    sub.add_parser("status")
    sub.add_parser("log")
    branch = sub.add_parser("branch")
    branch.add_argument("name")
    branch.add_argument("--from-commit", default="root")
    checkout = sub.add_parser("checkout")
    checkout.add_argument("commit_id")
    diff = sub.add_parser("diff")
    diff.add_argument("commit_a")
    diff.add_argument("commit_b")
    run = sub.add_parser("run-rubric")
    run.add_argument("rubric")
    sub.add_parser("run-sequence").add_argument("sequence")
    sub.add_parser("rollback").add_argument("commit_id")
    cp = sub.add_parser("cherry-pick-module")
    cp.add_argument("source")
    cp.add_argument("target")
    cp.add_argument("module")
    sub.add_parser("verify").add_argument("commit_id")
    bisect = sub.add_parser("bisect")
    bisect.add_argument("branch")
    bisect.add_argument("gate_name")
    args = parser.parse_args(argv)
    store = BranchStore()
    if args.cmd == "init":
        store.root.mkdir(parents=True, exist_ok=True)
        print("initialized rolling store")
    elif args.cmd == "status":
        print(json.dumps({"branches": store.list_branches()}, indent=2))
    elif args.cmd == "log":
        commits = sorted(p.stem for p in store.root.glob("*.json") if p.name != "branches.json")
        print(json.dumps(commits, indent=2))
    elif args.cmd == "branch":
        store.create_branch(args.name, args.from_commit)
        print(f"branch {args.name} -> {args.from_commit}")
    elif args.cmd == "checkout":
        store.checkout_commit(args.commit_id)
        print(args.commit_id)
    elif args.cmd == "diff":
        a = ModelCommit.load(store.root / f"{args.commit_a}.json")
        b = ModelCommit.load(store.root / f"{args.commit_b}.json")
        print(json.dumps(diff_commits(a, b), indent=2))
    elif args.cmd == "run-rubric":
        rubric = TrainingRubric.from_file(args.rubric)
        print(json.dumps({"rubric_id": rubric.rubric_id, "hash": rubric.compute_hash()}, indent=2))
    elif args.cmd == "run-sequence":
        sequence = load_structured(args.sequence)
        rubrics = []
        for rubric_path in sequence.get("rubrics", []):
            rubric = TrainingRubric.from_file(rubric_path)
            rubrics.append({"rubric_id": rubric.rubric_id, "hash": rubric.compute_hash()})
        print(json.dumps({"sequence": sequence.get("sequence_id", args.sequence), "rubrics": rubrics, "status": "loaded"}, indent=2))
    elif args.cmd == "rollback":
        commit = ModelCommit.load(store.root / f"{args.commit_id}.json")
        print(json.dumps({"rollback": commit.commit_id, "status": "commit_loaded"}, indent=2))
    elif args.cmd == "cherry-pick-module":
        source = ModelCommit.load(store.root / f"{args.source}.json")
        target = ModelCommit.load(store.root / f"{args.target}.json")
        print(json.dumps(cherry_pick_module(source, target, args.module), indent=2))
    elif args.cmd == "verify":
        commit = ModelCommit.load(store.root / f"{args.commit_id}.json")
        print(json.dumps({"commit_id": commit.commit_id, "valid": commit.verify()}, indent=2))
    elif args.cmd == "bisect":
        commit_ids = store.list_commits(args.branch)
        if not commit_ids:
            commit_ids = sorted(p.stem for p in store.root.glob("*.json") if p.name != "branches.json")
        commits = [ModelCommit.load(store.root / f"{commit_id}.json") for commit_id in commit_ids]
        print(json.dumps(bisect_regression(commits, args.gate_name), indent=2))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
