from __future__ import annotations

import json
from pathlib import Path

import scripts.train_phase4_semantic_transition_identity_codec as channel


ROOT = Path(__file__).resolve().parents[1]


def _configure_branch() -> None:
    channel.PREREGISTRATION = (
        ROOT
        / "moonshot"
        / "phase4_semantic_transition_only_channel_preregistration.json"
    )
    channel.COPY_TRANSITION_REPLACES_LINEAR = True
    channel.TRAINING_FORMAT = (
        "layercake-phase4-semantic-transition-only-channel-training/1"
    )
    channel.LEXICAL_EVALUATION_FORMAT = (
        "layercake-phase4-semantic-transition-only-channel-lexical-evaluation/1"
    )
    channel.PYTHON_EVALUATION_FORMAT = (
        "layercake-phase4-semantic-transition-only-channel-python-evaluation/1"
    )


def main() -> None:
    _configure_branch()
    args = channel.parse_args()
    if args.command == "train":
        result = channel.train(args)
    elif args.command == "evaluate-lexical":
        result = channel.evaluate_lexical(args)
    else:
        result = channel.evaluate_python(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
