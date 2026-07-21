from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import _common
from layercake.portable_domain import (
    artifact_payload_bytes,
    quantize_portable_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = torch.load(args.input, map_location="cpu", weights_only=True)
    quantized = quantize_portable_artifact(source)
    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized, artifact_path)
    result = {
        "status": "PASS",
        "quantization": quantized["spec"]["quantization"],
        "source_payload_bytes": artifact_payload_bytes(source),
        "quantized_payload_bytes": artifact_payload_bytes(quantized),
        "compression_ratio": artifact_payload_bytes(quantized)
        / artifact_payload_bytes(source),
        "spec_hash": quantized["spec_hash"],
        "payload_hash": quantized["payload_hash"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
