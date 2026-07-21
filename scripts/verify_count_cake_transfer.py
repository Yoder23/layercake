"""Certify bit-exact full-state CountCake transfer to an independent receiver."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import (  # noqa: E402
    apply_causal_online_cache_to_observed,
    load_count_cake_bundle,
)
from layercake.count_cake_cpu import CountCakeCPUDecoder  # noqa: E402
from layercake.count_cake_triton import (  # noqa: E402
    CountCakeGPUDecoder,
    is_recurrent_cached_available,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _prepare(model, prompt, patches, device, decoder):
    state = model.begin_cached_generation(prompt.to(device))
    if isinstance(decoder, CountCakeGPUDecoder):
        decoder.prepare(state, generated_bytes=patches * model.patch_size)
    return state


def _generate(model, state, patches, decoder):
    if decoder is not None:
        return decoder.generate_cached(state, patches=patches)
    return model.generate_cached(state, patches=patches)


def _quality(model, payload: bytes, device: torch.device) -> dict:
    seq_len = 1056
    row_count = len(payload) // seq_len
    rows = np.frombuffer(
        payload[: row_count * seq_len], dtype=np.uint8
    ).reshape(row_count, seq_len).copy()
    chunks = []
    with torch.inference_mode():
        for offset in range(0, row_count, 128):
            batch = torch.from_numpy(rows[offset : offset + 128]).to(
                device=device, dtype=torch.long
            )
            chunks.append(model.target_log_probs(batch).exp().cpu().numpy())
    base = np.concatenate(chunks, axis=0).astype(np.float64)
    cached = (
        apply_causal_online_cache_to_observed(
            base,
            rows,
            start=model.prediction_start,
            specs=model.online_cache_specs,
            reset_each_row=False,
            window=model.online_cache_window,
            recent_specs=model.recent_cache_specs,
            normalized_specs=model.normalized_cache_specs,
            normalization=model.cache_normalization,
        )
        if model.cache_enabled
        else base
    )
    return {
        "evaluated_bytes": int(cached.size),
        "cached_bpb": float(-np.log(cached).mean() / math.log(2.0)),
        "probability_sha256": hashlib.sha256(
            cached.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender", required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--patches", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    sender_path = Path(args.sender)
    receiver_path = Path(args.receiver)
    sender_payload = sender_path.read_bytes()
    receiver_path.parent.mkdir(parents=True, exist_ok=True)
    transfer_started = time.perf_counter()
    receiver_path.write_bytes(sender_payload)
    transfer_seconds = time.perf_counter() - transfer_started
    receiver_payload = receiver_path.read_bytes()

    sender, sender_manifest = load_count_cake_bundle(sender_path, device=device)
    receiver, receiver_manifest = load_count_cake_bundle(receiver_path, device=device)
    sender.eval()
    receiver.eval()
    if device.type == "cpu":
        sender_decoder = CountCakeCPUDecoder(sender)
        receiver_decoder = CountCakeCPUDecoder(receiver)
    elif is_recurrent_cached_available(sender) and is_recurrent_cached_available(
        receiver
    ):
        sender_decoder = CountCakeGPUDecoder(sender)
        receiver_decoder = CountCakeGPUDecoder(receiver)
    else:
        sender_decoder = receiver_decoder = None
    exact_tensors = all(
        torch.equal(value, receiver.state_dict()[name])
        for name, value in sender.state_dict().items()
    )
    payload = Path(args.data).read_bytes()
    prompt_bytes = payload[: sender.patch_size * 4]
    prompt = torch.tensor(list(prompt_bytes), dtype=torch.long).unsqueeze(0)
    sender_output = _generate(
        sender,
        _prepare(sender, prompt, args.patches, device, sender_decoder),
        args.patches,
        sender_decoder,
    )
    receiver_output = _generate(
        receiver,
        _prepare(receiver, prompt, args.patches, device, receiver_decoder),
        args.patches,
        receiver_decoder,
    )

    def rate(model, decoder) -> float:
        if isinstance(decoder, CountCakeCPUDecoder):
            decoder.clear_cache()
        state = _prepare(model, prompt, args.patches, device, decoder)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        generated = _generate(model, state, args.patches, decoder)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return generated.numel() / (time.perf_counter() - started)

    # Compile/allocate both paths, then alternate order to limit thermal bias.
    rate(sender, sender_decoder)
    rate(receiver, receiver_decoder)
    sender_rates = []
    receiver_rates = []
    for repeat in range(args.repeats):
        order = ((sender, sender_decoder, sender_rates), (receiver, receiver_decoder, receiver_rates))
        if repeat % 2:
            order = tuple(reversed(order))
        for model, decoder, values in order:
            values.append(rate(model, decoder))
    speed_ratio = statistics.median(receiver_rates) / statistics.median(sender_rates)
    sender_quality = _quality(sender, payload, device)
    receiver_quality = _quality(receiver, payload, device)
    checks = {
        "artifact_sha256_exact": _sha256(sender_payload) == _sha256(receiver_payload),
        "manifest_exact": sender_manifest == receiver_manifest,
        "all_learned_tensors_exact": exact_tensors,
        "generation_exact": torch.equal(sender_output, receiver_output),
        "quality_exact": sender_quality == receiver_quality,
        "receiver_speed_within_10_percent": speed_ratio >= 0.9,
    }
    report = {
        "format": "layercake-count-cake-full-state-transfer/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "complete portable model/domain state; not an incremental merge",
        "device": args.device,
        "sender": {
            "path": str(sender_path),
            "bytes": len(sender_payload),
            "sha256": _sha256(sender_payload),
        },
        "receiver": {
            "path": str(receiver_path),
            "bytes": len(receiver_payload),
            "sha256": _sha256(receiver_payload),
        },
        "transfer_seconds": transfer_seconds,
        "generation": {
            "bytes": int(sender_output.numel()),
            "sender_bytes_per_second": sender_rates,
            "receiver_bytes_per_second": receiver_rates,
            "receiver_over_sender_median": speed_ratio,
        },
        "quality": {
            "sender": sender_quality,
            "receiver": receiver_quality,
        },
        "checks": checks,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
