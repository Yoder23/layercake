"""Paired cached-generation benchmark for matched v24 models."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys
import time

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402
from layercake.count_cake_cpu import CountCakeCPUDecoder  # noqa: E402
from layercake.count_cake_speculative import (  # noqa: E402
    CountCakeSpeculativeDecoder,
)
from layercake.count_cake_triton import (  # noqa: E402
    CountCakeGPUDecoder,
    fused_greedy_patch,
    is_available,
    is_recurrent_cached_available,
)
from scripts.train_bpe_transformer_from_config import (  # noqa: E402
    BPETokenTransformerLM,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CachedTransformerDecoder:
    """Exact KV cache for the norm-first PyTorch reference transformer."""

    def __init__(self, model: BPETokenTransformerLM) -> None:
        self.model = model.eval()

    @torch.no_grad()
    def step(
        self,
        token: torch.Tensor,
        position: int,
        caches: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        model = self.model
        hidden = model.emb(token.reshape(1, 1)) + model.pos(
            torch.tensor([position], device=token.device)
        ).reshape(1, 1, -1)
        for index, layer in enumerate(model.core.layers):
            normalized = layer.norm1(hidden)
            caches[index].append(normalized)
            key_value = torch.cat(caches[index], dim=1)
            attention, _ = layer.self_attn(
                normalized,
                key_value,
                key_value,
                need_weights=False,
            )
            hidden = hidden + layer.dropout1(attention)
            normalized = layer.norm2(hidden)
            hidden = hidden + layer.dropout2(
                layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(normalized)))
                )
            )
        return model.head(model.norm(hidden))[0, 0]

    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        caches: list[list[torch.Tensor]] = [
            [] for _ in self.model.core.layers
        ]
        logits = None
        for position, token in enumerate(tokens):
            logits = self.step(token, position, caches)
        if logits is None:
            raise ValueError("transformer prompt cannot be empty")
        return logits, caches

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        logits, caches = self.prefill(tokens)
        generated = []
        position = int(tokens.numel())
        for _ in range(count):
            token = logits.argmax()
            generated.append(token)
            logits = self.step(token, position, caches)
            position += 1
        return torch.stack(generated)


def _load_transformer(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = BPETokenTransformerLM(
        vocab_size=4096,
        d_model=int(config["d_model"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        # Comparator checkpoints use different context contracts.  Recover
        # the persisted positional-table length instead of silently assuming
        # the original short-context v24 shape.
        max_len=int(checkpoint["model"]["pos.weight"].shape[0]),
        ff_mult=int(config["ff_mult"]),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = spm.SentencePieceProcessor(
        model_proto=checkpoint["tokenizer_model"]
    )
    return model.eval(), tokenizer, checkpoint


def _summary(values: list[float]) -> dict:
    return {
        "values": values,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _device_benchmark(
    *,
    device: torch.device,
    count_bundle: Path,
    transformer_checkpoint: Path,
    prompt: bytes,
    repeats: int,
    count_patches: int,
    transformer_tokens: int,
    speculative_block_size: int,
) -> dict:
    count_model, count_manifest = load_count_cake_bundle(
        count_bundle, device=device
    )
    transformer, tokenizer, checkpoint = _load_transformer(
        transformer_checkpoint, device
    )
    count_rows = torch.tensor(
        list(prompt[:1024]), device=device, dtype=torch.long
    ).reshape(1, -1)
    token_ids = tokenizer.encode(
        prompt[:1024].decode("utf-8", errors="replace"), out_type=int
    )[-100:]
    transformer_prompt = torch.tensor(
        token_ids, device=device, dtype=torch.long
    )
    decoder = CachedTransformerDecoder(transformer)

    cached_logits, _ = decoder.prefill(transformer_prompt)
    full_logits = transformer(transformer_prompt.unsqueeze(0))[0, -1]
    cache_max_abs_error = float((cached_logits - full_logits).abs().max())
    cache_argmax_equal = bool(cached_logits.argmax() == full_logits.argmax())

    cpu_decoder = (
        CountCakeCPUDecoder(count_model) if device.type == "cpu" else None
    )
    speculative_decoder = None
    if (
        count_model.patch_size == 1
        and count_model.patch_core_type == "gru"
        and count_model.patch_layers == 1
        and count_model.local_decoder == "position"
        and not count_model.cache_enabled
    ):
        if cpu_decoder is not None:
            draft_decoder = cpu_decoder
        else:
            draft_model, _ = load_count_cake_bundle(
                count_bundle, device=torch.device("cpu")
            )
            draft_decoder = CountCakeCPUDecoder(draft_model)
        speculative_decoder = CountCakeSpeculativeDecoder(
            count_model, draft_decoder, block_size=speculative_block_size
        )
    gpu_decoder = (
        CountCakeGPUDecoder(count_model)
        if device.type == "cuda" and is_recurrent_cached_available(count_model)
        else None
    )
    fused_supported = bool(
        device.type == "cuda"
        and is_available()
        and count_model.count_cake.max_order == 4
        and count_model.patch_size == 32
        and not count_model.local_recurrent
        and not count_model.cache_enabled
    )
    count_backend = (
        "count_draft_speculative"
        if speculative_decoder is not None
        else "numpy_indexed"
        if device.type == "cpu"
        else "triton_certified_recurrent"
        if gpu_decoder is not None
        else "triton_fused"
        if fused_supported
        else "torch_reference"
    )

    def count_run() -> float:
        if speculative_decoder is not None:
            speculative_decoder.clear_cache()
        elif cpu_decoder is not None:
            cpu_decoder.clear_cache()
        state = count_model.begin_cached_generation(count_rows)
        if gpu_decoder is not None:
            gpu_decoder.prepare(
                state,
                generated_bytes=count_patches * count_model.patch_size,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        if speculative_decoder is not None:
            speculative_decoder.generate_cached(state, patches=count_patches)
        elif device.type == "cpu":
            cpu_decoder.generate_cached(state, patches=count_patches)
        elif gpu_decoder is not None:
            gpu_decoder.generate_cached(state, patches=count_patches)
        elif fused_supported:
            for _ in range(count_patches):
                generated = fused_greedy_patch(
                    count_model,
                    state["recurrent_state"].squeeze(0),
                    state["history"],
                )
                state["history"] = torch.cat(
                    [state["history"], generated[0]]
                )[-4:]
                feature = torch.tanh(
                    count_model.patch_projection(
                        count_model.byte_embedding(generated).flatten(-2)
                    )
                ).unsqueeze(1)
                _, state["recurrent_state"] = count_model.patch_core(
                    feature,
                    state["recurrent_state"],
                )
        else:
            count_model.generate_cached(state, patches=count_patches)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - started

    count_accelerator_audit = None
    if speculative_decoder is not None:
        reference_state = count_model.begin_cached_generation(count_rows)
        reference = count_model.generate_cached(
            reference_state, patches=count_patches
        )
        accelerated_state = count_model.begin_cached_generation(count_rows)
        speculative_decoder.clear_cache()
        accelerated = speculative_decoder.generate_cached(
            accelerated_state, patches=count_patches
        )
        differing = torch.nonzero(reference != accelerated).flatten()
        count_accelerator_audit = {
            "argmax_equal": not bool(differing.numel()),
            "first_difference": (
                None if not differing.numel() else int(differing[0].item())
            ),
            "audited_bytes": int(reference.numel()),
            "verification_rounds": int(
                accelerated_state["speculative_rounds"]
            ),
            "accepted_draft_bytes": int(
                accelerated_state["speculative_accepted_bytes"]
            ),
            "emitted_bytes": int(
                accelerated_state["speculative_emitted_bytes"]
            ),
            "output_sha256": hashlib.sha256(
                bytes(reference.reshape(-1).cpu().tolist())
            ).hexdigest(),
        }
        if not count_accelerator_audit["argmax_equal"]:
            raise RuntimeError("speculative CountCake output differs from reference")
    elif cpu_decoder is not None:
        reference_state = count_model.begin_cached_generation(count_rows)
        reference = count_model.generate_cached(
            reference_state, patches=count_patches
        )
        accelerated_state = count_model.begin_cached_generation(count_rows)
        cpu_decoder.clear_cache()
        accelerated = cpu_decoder.generate_cached(
            accelerated_state, patches=count_patches
        )
        differing = torch.nonzero(reference != accelerated).flatten()
        count_accelerator_audit = {
            "argmax_equal": not bool(differing.numel()),
            "first_difference": (
                None if not differing.numel() else int(differing[0].item())
            ),
            "audited_bytes": int(reference.numel()),
            "certified_bytes": int(accelerated_state["cpu_certified_bytes"]),
            "exact_fallback_bytes": int(accelerated_state["cpu_exact_bytes"]),
            "output_sha256": hashlib.sha256(
                bytes(reference.reshape(-1).cpu().tolist())
            ).hexdigest(),
        }
        if not count_accelerator_audit["argmax_equal"]:
            raise RuntimeError("accelerated CountCake output differs from reference")
    elif gpu_decoder is not None:
        reference_state = count_model.begin_cached_generation(count_rows)
        reference = count_model.generate_cached(
            reference_state, patches=count_patches
        )
        accelerated_state = count_model.begin_cached_generation(count_rows)
        gpu_decoder.prepare(
            accelerated_state,
            generated_bytes=count_patches * count_model.patch_size,
        )
        accelerated = gpu_decoder.generate_cached(
            accelerated_state, patches=count_patches
        )
        differing = torch.nonzero(reference != accelerated).flatten()
        count_accelerator_audit = {
            "argmax_equal": not bool(differing.numel()),
            "first_difference": (
                None if not differing.numel() else int(differing[0].item())
            ),
            "audited_bytes": int(reference.numel()),
            "certified_bytes": int(
                accelerated_state["gpu_certified_bytes"]
            ),
            "exact_fallback_bytes": int(
                accelerated_state["gpu_exact_bytes"]
            ),
            "certificate_launches": int(
                accelerated_state["gpu_certificate_launches"]
            ),
            "output_sha256": hashlib.sha256(
                bytes(reference.reshape(-1).cpu().tolist())
            ).hexdigest(),
        }
        if not count_accelerator_audit["argmax_equal"]:
            raise RuntimeError("accelerated CountCake output differs from reference")

    def transformer_generated_bytes() -> int:
        logits, caches = decoder.prefill(transformer_prompt)
        generated = []
        position = int(transformer_prompt.numel())
        for _ in range(transformer_tokens):
            token = logits.argmax()
            generated.append(int(token.item()))
            logits = decoder.step(token, position, caches)
            position += 1
        return len(tokenizer.decode(generated).encode("utf-8"))

    measured_transformer_bytes = transformer_generated_bytes()

    def transformer_run() -> float:
        logits, caches = decoder.prefill(transformer_prompt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        position = int(transformer_prompt.numel())
        for _ in range(transformer_tokens):
            token = logits.argmax()
            logits = decoder.step(token, position, caches)
            position += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - started

    # Compilation, index construction, cache prefill, and allocator warmup are
    # intentionally excluded from steady-state decode timing for both models.
    count_run()
    count_run()
    transformer_run()
    transformer_run()
    count_seconds = []
    transformer_seconds = []
    for repeat in range(repeats):
        order = ("count", "transformer")
        if repeat % 2:
            order = tuple(reversed(order))
        for name in order:
            if name == "count":
                count_seconds.append(count_run())
            else:
                transformer_seconds.append(transformer_run())

    count_bytes = float(count_patches * count_model.patch_size)
    transformer_bytes = float(measured_transformer_bytes)
    count_throughput = [count_bytes / value for value in count_seconds]
    transformer_throughput = [
        transformer_bytes / value for value in transformer_seconds
    ]
    paired_speedup = [
        count / baseline
        for count, baseline in zip(count_throughput, transformer_throughput)
    ]
    return {
        "device": str(device),
        "count_cake": {
            "backend": count_backend,
            "logical_parameters": count_model.logical_total_parameters,
            "logical_bytes": count_bytes,
            "seconds": _summary(count_seconds),
            "logical_bytes_per_second": _summary(count_throughput),
            "bundle_parameters": count_manifest["parameters"],
            "accelerator_exactness_audit": count_accelerator_audit,
        },
        "transformer": {
            "parameters": int(checkpoint["trainable_params"]),
            "logical_bytes": transformer_bytes,
            "generated_tokens": transformer_tokens,
            "byte_accounting": "exact UTF-8 bytes from decoded generated token IDs",
            "seconds": _summary(transformer_seconds),
            "logical_bytes_per_second": _summary(transformer_throughput),
            "cache_max_abs_logit_error": cache_max_abs_error,
            "cache_argmax_equal": cache_argmax_equal,
        },
        "paired_speedup": _summary(paired_speedup),
        "gate_5x": min(paired_speedup) >= 5.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count-bundle",
        default="runs_experiment/production_v24_1m_count_cake_bundle/model.npz",
    )
    parser.add_argument(
        "--transformer-checkpoint",
        default="runs_experiment/production_v24_1m_transformer/latest.pt",
    )
    parser.add_argument(
        "--transformer-metrics",
        default="runs_experiment/production_v24_1m_transformer/training_metrics.json",
    )
    parser.add_argument(
        "--prompt",
        default="runs_experiment/production_v24_corpus/eval.bin",
    )
    parser.add_argument(
        "--output",
        default="results/breakthrough_equal/v24_1m_generation.json",
    )
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--count-patches",
        type=int,
        default=20,
        help="number of LayerCake patches emitted per timed repeat",
    )
    parser.add_argument(
        "--transformer-tokens",
        type=int,
        default=200,
        help="number of BPE tokens emitted per timed repeat",
    )
    parser.add_argument("--speculative-block-size", type=int, default=32)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    count_bundle = Path(args.count_bundle)
    transformer_checkpoint = Path(args.transformer_checkpoint)
    transformer_metrics = Path(args.transformer_metrics)
    prompt_path = Path(args.prompt)
    prompt = prompt_path.read_bytes()
    requested_devices = [name.strip() for name in args.devices.split(",")]
    device_results = []
    for name in requested_devices:
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA generation requested but CUDA is unavailable")
        device_results.append(
            _device_benchmark(
                device=torch.device(name),
                count_bundle=count_bundle,
                transformer_checkpoint=transformer_checkpoint,
                prompt=prompt,
                repeats=args.repeats,
                count_patches=args.count_patches,
                transformer_tokens=args.transformer_tokens,
                speculative_block_size=args.speculative_block_size,
            )
        )
    result = {
        "format": "layercake-v24-generation-benchmark/1",
        "status": "PASS" if all(row["gate_5x"] for row in device_results) else "FAIL",
        "gate": {
            "minimum_speedup": 5.0,
            "all_devices_pass": all(row["gate_5x"] for row in device_results),
        },
        "artifacts": {
            "count_bundle": {
                "path": str(count_bundle),
                "bytes": count_bundle.stat().st_size,
                "sha256": _sha256(count_bundle),
            },
            "transformer_checkpoint": {
                "path": str(transformer_checkpoint),
                "bytes": transformer_checkpoint.stat().st_size,
                "sha256": _sha256(transformer_checkpoint),
            },
            "transformer_metrics": {
                "path": str(transformer_metrics),
                "bytes": transformer_metrics.stat().st_size,
                "sha256": _sha256(transformer_metrics),
            },
            "prompt": {
                "path": str(prompt_path),
                "bytes": prompt_path.stat().st_size,
                "sha256": _sha256(prompt_path),
            },
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_threads": args.cpu_threads,
        },
        "devices": device_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
