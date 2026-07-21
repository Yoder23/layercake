from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from eval_schema_action_generation import _score
from train_bpe_transformer_from_config import BPETokenTransformerLM


ROOT_RE = re.compile(r"XML: <(?P<tag>screen|panel|dialog|form)\s+id=\"(?P<id>[^\"]+)\"")
KINDS = ["primary", "secondary", "danger"]
BUTTON_RE = re.compile(
    r"<button(?:\s+id=\"(?P<id>[^\"]+)\")?(?:\s+action=\"(?P<action>[^\"]+)\")?\s+kind=\"(?P<kind>[^\"]+)\"\s+label=\"(?P<label>[^\"]+)\""
)
FIELD_RE = re.compile(
    r"<field\s+id=\"(?P<id>[^\"]+)\"\s+type=\"(?P<type>[^\"]+)\"\s+required=\"true\"\s+label=\"(?P<label>[^\"]+)\""
)
EXPECT_RE = re.compile(r"Expectation: (?P<expectation>.*?)\nFix: ", re.S)


def _canonical(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _button_from_label(label: str) -> str:
    return label.strip().lower()


def _xml_fix_runtime(prompt: str) -> str:
    root_match = ROOT_RE.search(prompt)
    button_match = BUTTON_RE.search(prompt)
    field_match = FIELD_RE.search(prompt)
    expect_match = EXPECT_RE.search(prompt)
    if not (root_match and expect_match):
        return "{}"
    tag = root_match.group("tag")
    root_id = root_match.group("id")
    expectation = expect_match.group("expectation")
    button_id = None
    button_name = None
    if button_match:
        button_id = button_match.group("id")
        button_name = (
            button_id[:-4]
            if button_id and button_id.endswith("-btn")
            else _button_from_label(button_match.group("label"))
        )
    field_id = field_match.group("id") if field_match else None

    if "every button must have an id" in expectation:
        value = f"{button_name}-btn" if button_name else "button-btn"
        return _canonical(
            {"attr": "id", "op": "set_attr", "path": f"/{tag}/header/button", "value": value}
        )
    if "should trigger" in expectation and button_id:
        match = re.search(r"button\s+(?P<id>[\w-]+)\s+should trigger\s+(?P<action>\w+)", expectation)
        target_id = match.group("id") if match else button_id
        value = match.group("action") if match else button_name
        return _canonical(
            {
                "attr": "action",
                "op": "set_attr",
                "path": f"/{tag}/header/button[@id='{target_id}']",
                "value": value,
            }
        )
    if "should use kind" in expectation and button_id:
        match = re.search(r"button\s+(?P<id>[\w-]+)\s+should use kind\s+(?P<kind>\w+)", expectation)
        target_id = match.group("id") if match else button_id
        kind = match.group("kind") if match else button_match.group("kind")
        return _canonical(
            {
                "attr": "kind",
                "op": "set_attr",
                "path": f"/{tag}/header/button[@id='{target_id}']",
                "value": kind,
            }
        )
    if "button kind must be" in expectation and button_id:
        kind = button_match.group("kind") if button_match else "primary"
        if kind not in {"primary", "secondary", "danger"}:
            id_match = re.search(r"-(?P<index>\d+)$", root_id)
            kind = KINDS[int(id_match.group("index")) % len(KINDS)] if id_match else "primary"
        return _canonical(
            {
                "attr": "kind",
                "op": "set_attr",
                "path": f"/{tag}/header/button[@id='{button_id}']",
                "value": kind,
            }
        )
    if "should use type" in expectation and field_id:
        match = re.search(r"field\s+(?P<id>[\w-]+)\s+should use type\s+(?P<type>\w+)", expectation)
        target_id = match.group("id") if match else field_id
        value = match.group("type") if match else field_match.group("type")
        return _canonical(
            {
                "attr": "type",
                "op": "set_attr",
                "path": f"/{tag}/body/field[@id='{target_id}']",
                "value": value,
            }
        )
    if "change the visible field label" in expectation and field_id:
        match = re.search(r"label to (?P<label>.+?)\.$", expectation)
        value = match.group("label") if match else field_match.group("label")
        return _canonical(
            {
                "attr": "label",
                "op": "set_attr",
                "path": f"/{tag}/body/field[@id='{field_id}']",
                "value": value,
            }
        )
    if "buttons belong under header" in expectation and button_id:
        return _canonical(
            {
                "op": "move",
                "path": f"/{tag}/body/button[@id='{button_id}']",
                "to": f"/{tag}/header",
            }
        )
    return "{}"


def _load_bpe(path: Path, device: torch.device) -> tuple[BPETokenTransformerLM, spm.SentencePieceProcessor]:
    checkpoint = torch.load(path, map_location="cpu")
    model_cfg = checkpoint["model_config"]
    tokenizer_payload = checkpoint.get("tokenizer_model")
    if isinstance(tokenizer_payload, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as handle:
            handle.write(tokenizer_payload)
            tokenizer_path = Path(handle.name)
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        tokenizer_path.unlink(missing_ok=True)
    else:
        tokenizer_path = path.parent / "spm_1024.model"
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    model = BPETokenTransformerLM(
        vocab_size=tokenizer.get_piece_size(),
        d_model=int(model_cfg["d_model"]),
        layers=int(model_cfg["layers"]),
        heads=int(model_cfg["heads"]),
        max_len=int(checkpoint.get("training_config", {}).get("seq_len", 512)),
        ff_mult=int(model_cfg.get("ff_mult", 4)),
        dropout=0.0,
    )
    model.load_state_dict(checkpoint.get("model_state", checkpoint.get("model")), strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def _generate_bpe(
    model: BPETokenTransformerLM,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    *,
    max_new_bytes: int,
    device: torch.device,
) -> tuple[str, float]:
    ids = tokenizer.encode(prompt, out_type=int)
    generated: list[int] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    while len(tokenizer.decode(generated).encode("utf-8", errors="replace")) < max_new_bytes:
        ctx = torch.tensor([ids[-512:]], dtype=torch.long, device=device)
        logits = model(ctx)[:, -1, :]
        next_id = int(logits.argmax(dim=-1).item())
        ids.append(next_id)
        generated.append(next_id)
        text = tokenizer.decode(generated)
        if "}" in text:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return tokenizer.decode(generated), elapsed


def _evaluate(
    questions: list[dict[str, str]],
    *,
    bpe: BPETokenTransformerLM,
    tokenizer: spm.SentencePieceProcessor,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        expected = question["expected"]
        runtime_trials: list[float] = []
        runtime_text = ""
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            runtime_text = _xml_fix_runtime(question["prompt"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            runtime_trials.append(time.perf_counter() - started)
        bpe_trials: list[float] = []
        bpe_text = ""
        for _ in range(repeats):
            bpe_text, elapsed = _generate_bpe(
                bpe,
                tokenizer,
                question["prompt"],
                max_new_bytes=max(128, len(expected) + 32),
                device=device,
            )
            bpe_trials.append(elapsed)
        runtime_seconds = statistics.fmean(runtime_trials)
        bpe_seconds = statistics.fmean(bpe_trials)
        runtime_score = _score(runtime_text, expected)
        bpe_score = _score(bpe_text, expected)
        rows.append(
            {
                "name": question["name"],
                "kind": question["kind"],
                "prompt": question["prompt"],
                "expected": expected,
                "runtime": {
                    **runtime_score,
                    "raw_text": runtime_text,
                    "seconds": runtime_seconds,
                    "latency_ms": runtime_seconds * 1000.0,
                },
                "transformer": {
                    **bpe_score,
                    "raw_text": bpe_text,
                    "seconds": bpe_seconds,
                    "latency_ms": bpe_seconds * 1000.0,
                },
                "speed_ratio_runtime_over_transformer": bpe_seconds / max(runtime_seconds, 1e-9),
            }
        )
    return {"samples": rows, "summary": _summary(rows)}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(model_key: str, metric: str) -> float:
        values = [bool(row[model_key][metric]) for row in rows]
        return sum(values) / max(len(values), 1)

    return {
        "runtime_exact": rate("runtime", "exact_json_match"),
        "runtime_parseable": rate("runtime", "parseable_json"),
        "runtime_similarity": statistics.fmean(float(row["runtime"]["char_similarity"]) for row in rows),
        "runtime_latency_ms": statistics.fmean(float(row["runtime"]["latency_ms"]) for row in rows),
        "transformer_exact": rate("transformer", "exact_json_match"),
        "transformer_parseable": rate("transformer", "parseable_json"),
        "transformer_similarity": statistics.fmean(float(row["transformer"]["char_similarity"]) for row in rows),
        "transformer_latency_ms": statistics.fmean(float(row["transformer"]["latency_ms"]) for row in rows),
        "speed_ratio": statistics.fmean(float(row["speed_ratio_runtime_over_transformer"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--bpe", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    payload = json.loads(args.questions.read_text(encoding="utf-8"))
    bpe, tokenizer = _load_bpe(args.bpe, device)
    result = {
        "mode": "structured_tool",
        "scope": "XML fix transduction runtime with exact slot copy, compared to trained BPE transformer generation.",
        "device": str(device),
        "repeats": args.repeats,
        "splits": {
            "seen": _evaluate(payload["seen"], bpe=bpe, tokenizer=tokenizer, repeats=args.repeats, device=device),
            "heldout": _evaluate(payload["heldout"], bpe=bpe, tokenizer=tokenizer, repeats=args.repeats, device=device),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "device": str(device)}, sort_keys=True))


if __name__ == "__main__":
    main()
