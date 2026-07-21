from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _first_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the packaged North Star v22 INT8 patch runtime"
    )
    parser.add_argument("prompt")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/layercake_v22_patch_int8.ts"),
    )
    args = parser.parse_args()

    artifact = args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    model = torch.jit.load(str(artifact), map_location="cpu").eval()
    payload = list(args.prompt.encode("utf-8"))[-256:]
    payload = ([32] * ((-len(payload)) % 4)) + payload
    prompt = torch.tensor([payload], dtype=torch.long)
    with torch.inference_mode():
        generated = model(prompt)[0].tolist()
    raw = bytes(generated).decode("utf-8", errors="replace")
    extracted = _first_json(raw)
    parsed = None
    parse_error = None
    if extracted is not None:
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
    result = {
        "raw": raw,
        "json": parsed,
        "parse_error": parse_error,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if parsed is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
