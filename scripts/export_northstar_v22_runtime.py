from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from eval_schema_action_generation import _load_layercake, _score
from layercake.deployment import PatchGenerationDeployment


QUESTION_FILES = [
    ROOT / "data" / "schema_action_domain" / "eval_questions.json",
    ROOT / "data" / "question_relevance" / "eval_questions.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt_tensor(prompt: str) -> torch.Tensor:
    payload = list(prompt.encode("utf-8"))
    payload = ([32] * ((-len(payload)) % 4)) + payload
    return torch.tensor([payload], dtype=torch.long)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export and validate the North Star v22 INT8 patch runtime"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs_experiment/northstar_v21_semantic_pointer/latest.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/layercake_v22_patch_int8.ts"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/layercake_v22_patch_int8.manifest.json"),
    )
    args = parser.parse_args()

    checkpoint_path = (
        args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    checkpoint, full_model = _load_layercake(checkpoint_path, torch.device("cpu"))
    source_parameter_count = sum(
        parameter.numel() for parameter in full_model.parameters()
    )
    runtime = PatchGenerationDeployment(full_model).eval()
    deployment_parameter_count = sum(
        parameter.numel() for parameter in runtime.parameters()
    )
    runtime = torch.ao.quantization.quantize_dynamic(
        runtime,
        {torch.nn.Linear},
        dtype=torch.qint8,
    ).eval()

    example = torch.arange(128, dtype=torch.long).remainder(251).unsqueeze(0)
    traced = torch.jit.trace(
        runtime,
        example,
        strict=False,
        check_trace=False,
    )
    traced.save(str(output))
    loaded = torch.jit.load(str(output), map_location="cpu").eval()

    total = 0
    eager_script_equal = 0
    eager_script_json_equal = 0
    exact = 0
    per_file: dict[str, dict[str, int]] = {}
    for question_path in QUESTION_FILES:
        document = json.loads(question_path.read_text(encoding="utf-8-sig"))
        file_total = 0
        file_equal = 0
        file_json_equal = 0
        file_exact = 0
        for split in ("seen", "heldout"):
            for question in document.get(split, []):
                prompt = _prompt_tensor(question["prompt"])
                eager = runtime(prompt)
                scripted = loaded(prompt)
                equal = torch.equal(eager, scripted)
                eager_text = bytes(eager[0].tolist()).decode(
                    "utf-8", errors="replace"
                )
                text = bytes(scripted[0].tolist()).decode(
                    "utf-8", errors="replace"
                )
                eager_scored = _score(eager_text, question["expected"])
                scored = _score(text, question["expected"])
                json_equal = (
                    eager_scored["parseable_json"]
                    and scored["parseable_json"]
                    and eager_scored["parsed_json"] == scored["parsed_json"]
                )
                file_total += 1
                file_equal += int(equal)
                file_json_equal += int(json_equal)
                file_exact += int(scored["exact_json_match"])
        total += file_total
        eager_script_equal += file_equal
        eager_script_json_equal += file_json_equal
        exact += file_exact
        per_file[str(question_path.relative_to(ROOT)).replace("\\", "/")] = {
            "samples": file_total,
            "eager_script_equal": file_equal,
            "eager_script_json_equal": file_json_equal,
            "exact_json": file_exact,
        }

    passed = (
        total > 0
        and eager_script_json_equal == total
        and exact == total
    )
    manifest = {
        "status": "PASS" if passed else "FAIL",
        "format": "layercake-v22-patch-dynamic-int8-torchscript/1",
        "scope": "global autoregressive patch generation only",
        "source_checkpoint": str(checkpoint_path.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "source_checkpoint_format": checkpoint.get("format"),
        "source_parameter_count": source_parameter_count,
        "deployment_parameter_count_fp32_before_packing": (
            deployment_parameter_count
        ),
        "dynamic_int8_module_types": ["Linear"],
        "quantized_engine": torch.backends.quantized.engine,
        "artifact": str(output.relative_to(ROOT)).replace("\\", "/"),
        "artifact_bytes": output.stat().st_size,
        "artifact_sha256": _sha256(output),
        "validation": {
            "samples": total,
            "eager_script_equal": eager_script_equal,
            "eager_script_json_equal": eager_script_json_equal,
            "exact_json": exact,
            "files": per_file,
        },
        "runtime_contract": {
            "input": "batch-one, patch-aligned uint8 values in an int64 tensor",
            "output": "80 generated byte values",
            "patch_size": 4,
            "maximum_context_bytes": 256,
        },
        "limitations": {
            "backend": "Validated with the x86 PyTorch quantized backend.",
            "phone": "Not measured on Android/iOS ARM hardware.",
            "scope": "Does not include the full general byte-LM local decoder.",
        },
        "torch_version": torch.__version__,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
