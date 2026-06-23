from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F

import _common
from benchmark_bpe_baseline import BPETokenLM, evaluate
from run_paired_byte_experiment import batch, load_jsonl_bytes, load_python_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--domain-bytes", type=int, default=2_000_000)
    parser.add_argument("--general-bytes", type=int, default=20_000_000)
    parser.add_argument("--output-artifact")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.artifact, map_location="cpu")
    config = artifact["args"]
    model = BPETokenLM(
        artifact["vocab_size"],
        d_model=config["d_model"],
        layers=config["layers"],
        heads=config["heads"],
        max_len=max(config["seq"], args.seq),
    ).to(device)
    model.load_state_dict(artifact["model"])
    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as handle:
        handle.write(artifact["tokenizer_model"])
        tokenizer_path = Path(handle.name)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    tokenizer_path.unlink(missing_ok=True)

    root = Path(__file__).resolve().parents[1]
    domain = load_python_bytes(
        root.parent / "layercakeogwithdecoder", args.domain_bytes
    )
    general = load_jsonl_bytes(
        root.parent / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        args.general_bytes,
    )
    domain_train, domain_eval = domain[:-100_000], domain[-100_000:]
    general_eval = general[-200_000:]

    def encode(raw: torch.Tensor) -> torch.Tensor:
        text = bytes(raw.tolist()).decode("utf-8", errors="replace")
        return torch.tensor(tokenizer.encode(text, out_type=int), dtype=torch.long)

    domain_train_tokens = encode(domain_train)
    domain_eval_tokens = encode(domain_eval)
    general_eval_tokens = encode(general_eval)
    before = {
        "domain": evaluate(
            model,
            domain_eval_tokens,
            domain_eval.numel(),
            args.seq,
            min(args.batch, 32),
            30,
            device,
        ),
        "general": evaluate(
            model,
            general_eval_tokens,
            general_eval.numel(),
            args.seq,
            min(args.batch, 32),
            30,
            device,
        ),
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(6261)
    bytes_per_token = domain_train.numel() / domain_train_tokens.numel()
    history = []
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = batch(
            domain_train_tokens, args.seq, args.batch, generator, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 250 == 0:
            item = {"step": step, "loss": loss.item()}
            history.append(item)
            print(item, flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - started
    after = {
        "domain": evaluate(
            model,
            domain_eval_tokens,
            domain_eval.numel(),
            args.seq,
            min(args.batch, 32),
            30,
            device,
        ),
        "general": evaluate(
            model,
            general_eval_tokens,
            general_eval.numel(),
            args.seq,
            min(args.batch, 32),
            30,
            device,
        ),
    }
    if args.output_artifact:
        output_artifact = dict(artifact)
        output_artifact["model"] = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        }
        output_artifact["domain_finetune"] = vars(args)
        torch.save(output_artifact, args.output_artifact)
    result = {
        "status": "TRAINED",
        "device": str(device),
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "bytes_per_token": bytes_per_token,
        "estimated_bytes_per_update": args.batch * args.seq * bytes_per_token,
        "estimated_total_training_bytes": (
            args.steps * args.batch * args.seq * bytes_per_token
        ),
        "before": before,
        "after": after,
        "general_bpb_regression": (
            after["general"]["bpb"] - before["general"]["bpb"]
        ),
        "history": history,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
