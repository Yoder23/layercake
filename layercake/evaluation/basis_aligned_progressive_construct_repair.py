"""Reporter-stimulus repair for the v11 basis-aligned construct."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Iterable

from layercake.basis_aligned_progressive_core import BasisAlignedProgressiveCore
from . import basis_aligned_progressive_construct as base


FORMAT = "layercake-postrelease-basis-aligned-progressive-reporter-repair/1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(root: Path, repair_path: Path) -> dict:
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    if repair.get("format") != FORMAT or repair.get("status") != "PREREGISTERED_REPORTER_STIMULUS_REPAIR":
        raise RuntimeError("basis-aligned reporter repair governance changed")
    for relative, expected in repair["bindings"].items():
        target = root / relative
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"basis-aligned reporter-repair binding changed: {relative}")
    result = base.execute(root, root / repair["base_protocol"])
    path = root / "tests/models/test_basis_aligned_progressive_core.py"
    spec = importlib.util.spec_from_file_location("basis_aligned_repair_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    tokenizer = fixture.DecoderAwareExternalTokenizer(fixture._doc())
    model = fixture._model(tokenizer)
    fixture_has_basis_and_mean = all(
        layer.mlp_output_projection.weight.shape == (model.full_width, model.bottleneck_width)
        and layer.mlp_residual_mean.shape == (model.full_width,)
        for layer in model.layers
    )
    production_is_rank_192 = BasisAlignedProgressiveCore.parameter_count_for_config(
        fixed_vocab_size=32_015,
        full_width=3_072,
        bottleneck_width=192,
        replacement_layers=32,
        intermediate_size=768,
    ) == 291_382_272
    result["checks"]["rank_192_basis_and_mean"] = fixture_has_basis_and_mean and production_is_rank_192
    result["status"] = "PASS" if all(result["checks"].values()) else "FAIL"
    result["repair"] = {
        "path": repair_path.name,
        "sha256": sha(repair_path),
        "scientific_fields_changed": False,
        "changed_check": "small fixture checks basis-and-mean shape; production rank remains static 192",
    }
    result["evidence_sha256"] = hashlib.sha256(
        (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol)
    output = root / args.output
    if args.command == "execute":
        if output.exists():
            raise RuntimeError("basis-aligned repaired result immutable")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif json.loads(output.read_text(encoding="utf-8")) != result:
        raise RuntimeError("stored repaired construct differs")
    print(json.dumps({"status": result["status"], "evidence_sha256": result["evidence_sha256"]}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
