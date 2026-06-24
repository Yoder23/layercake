from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import _common
from layercake.portable_domain import (
    PortableDomainDecoder,
    PortableDomainSpec,
    build_portable_artifact,
)
from run_paired_byte_experiment import batch, load_python_bytes


def load_domain_stream(args, root: Path) -> torch.Tensor:
    if args.domain_file:
        payload = bytearray()
        for item in args.domain_file:
            path = Path(item)
            if not path.is_absolute():
                path = root / path
            payload.extend(path.read_bytes())
            payload.extend(b"\n")
        if len(payload) < args.seq * 4:
            raise ValueError(
                "domain files are too small for the requested sequence length"
            )
        return torch.tensor(list(payload[: args.domain_bytes]), dtype=torch.long)
    return load_python_bytes(
        root.parent / "layercakeogwithdecoder", args.domain_bytes
    )


@torch.no_grad()
def evaluate(model, stream, seq, batch_size, batches, device):
    model.eval()
    generator = torch.Generator().manual_seed(991)
    losses = []
    for _ in range(batches):
        x, y = batch(stream, seq, batch_size, generator, device)
        logits = model(x)
        losses.append(
            F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
        )
    loss = sum(losses) / len(losses)
    return {"loss": loss, "ppl": math.exp(loss), "bpb": loss / math.log(2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-abi", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--architecture", choices=["anchor_mlp", "byte_gru"], default="anchor_mlp"
    )
    parser.add_argument("--embedding-width", type=int, default=64)
    parser.add_argument("--domain-id", default="python")
    parser.add_argument(
        "--domain-file",
        action="append",
        help=(
            "Text/JSONL file to train as the portable domain. May be passed "
            "multiple times. Defaults to the repository Python corpus."
        ),
    )
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--domain-bytes", type=int, default=2_000_000)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=6060)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[1]
    domain = load_domain_stream(args, root)
    eval_bytes = min(100_000, max(args.seq * args.batch * 2, domain.numel() // 10))
    if domain.numel() <= eval_bytes + args.seq + 1:
        raise ValueError("domain stream is too small after reserving eval bytes")
    train_stream, eval_stream = domain[:-eval_bytes], domain[-eval_bytes:]
    model = PortableDomainDecoder(
        feature_width=args.d_abi,
        hidden_width=args.hidden,
        architecture=args.architecture,
        embedding_width=args.embedding_width,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed + 1)
    history = []
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = batch(train_stream, args.seq, args.batch, generator, device)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            item = {"step": step, "loss": loss.item()}
            history.append(item)
            print(item, flush=True)

    evaluation = evaluate(
        model,
        eval_stream,
        args.seq,
        args.batch,
        args.eval_batches,
        device,
    )
    artifact = build_portable_artifact(
        model,
        PortableDomainSpec(
            domain_id=args.domain_id,
            feature_width=args.d_abi,
            hidden_width=args.hidden,
            architecture=args.architecture,
            embedding_width=args.embedding_width,
        ),
        training=vars(args),
        evaluation=evaluation,
    )
    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    result = {
        "status": "TRAINED",
        "mode": "core_independent_lossless",
        "parameters": model.parameter_count(),
        "elapsed_seconds": time.time() - started,
        "domain_files": args.domain_file or [],
        "train_bytes": int(train_stream.numel()),
        "eval_bytes": int(eval_stream.numel()),
        "spec_hash": artifact["spec_hash"],
        "payload_hash": artifact["payload_hash"],
        "evaluation": evaluation,
        "history": history,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
