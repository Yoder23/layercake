from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = torch.load(args.input, map_location="cpu", weights_only=True)
    state = artifact["patch_model"]
    removed = [
        name for name in state if name.startswith("patch_prediction_heads.")
    ]
    artifact["patch_model"] = {
        name: tensor for name, tensor in state.items() if name not in removed
    }
    artifact["args"]["patch_prediction"] = False
    artifact["stripped_training_only_parameters"] = sum(
        state[name].numel() for name in removed
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(
        {
            "status": "PASS",
            "removed_tensors": len(removed),
            "removed_parameters": artifact[
                "stripped_training_only_parameters"
            ],
        }
    )


if __name__ == "__main__":
    main()
