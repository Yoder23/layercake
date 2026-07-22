"""Train a separately installable portable-fusion specialist."""

from __future__ import annotations

from contextlib import nullcontext
import ast
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, load_package, sha256_bytes, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.models.portable_fusion import (
    PortableFusionCake,
    PortableFusionConfig,
    portable_fusion_manifest_architecture,
)
from .data import ByteCorpus, sha256_file
from .foundation import _config, evaluate_core, load_core_checkpoint


@torch.inference_mode()
def evaluate_fusion_bpb(
    core,
    cake,
    corpus: ByteCorpus,
    *,
    batch_size: int,
    sequence_bytes: int,
    batches: int,
    device: torch.device,
    route: int,
    disable_canonical: bool = False,
    disable_anchor: bool = False,
) -> dict:
    losses = []
    core_losses = []
    byte_count = 0
    for row in corpus.fixed_batches(
        batch_size=batch_size, sequence_bytes=sequence_bytes, batches=batches, device=device
    ):
        inputs, targets = row[:, :-1], row[:, 1:]
        _, aux = core(inputs, route=route, return_aux=True)
        canonical = aux["canonical"]
        if disable_canonical:
            canonical = torch.zeros_like(canonical)
        elif disable_anchor:
            canonical = canonical.clone()
            canonical[..., core.config.abi_width // 2:] = 0
        logits, _ = cake(aux["core_logits"], canonical, inputs)
        losses.append(F.cross_entropy(logits.flatten(0, 1), targets.flatten()).item())
        core_losses.append(F.cross_entropy(aux["core_logits"].flatten(0, 1), targets.flatten()).item())
        byte_count += targets.numel()
    divisor = 0.6931471805599453
    return {
        "cake_bits_per_byte": sum(losses) / len(losses) / divisor,
        "core_bits_per_byte": sum(core_losses) / len(core_losses) / divisor,
        "evaluated_bytes": byte_count,
    }


@torch.inference_mode()
def _syntax_tasks(core, cake, *, device: torch.device, route: int, generated_bytes: int = 96) -> dict:
    prompts = [
        "def add(left, right):\n    ",
        "def flatten(items):\n    \"\"\"Yield nested items.\"\"\"\n    ",
        "class Counter:\n    def __init__(self):\n        ",
        "async def gather_limited(tasks, limit):\n    ",
        "def safe_divide(a, b):\n    try:\n        ",
        "def fibonacci(n):\n    if n < 2:\n        ",
        "def chunks(values, size):\n    for index in range(0, len(values), size):\n        ",
        "def normalize_path(path):\n    ",
    ]
    rows = []
    for prompt in prompts:
        prompt_bytes = prompt.encode("utf-8")
        base_state = core.prefill(prompt_bytes, route=route, capture_generated=True)
        _, base_state = core.decode_many(base_state, generated_bytes)
        cake_state = core.prefill(
            prompt_bytes, route=route, fusion_cake=cake, capture_generated=True
        )
        _, cake_state = core.decode_many(cake_state, generated_bytes, fusion_cake=cake)
        base_text = bytes(base_state.generated_bytes[0].cpu().tolist()).decode("utf-8", errors="replace")
        cake_text = bytes(cake_state.generated_bytes[0].cpu().tolist()).decode("utf-8", errors="replace")
        def parses(completion: str) -> bool:
            try:
                ast.parse(prompt + completion)
                return True
            except (SyntaxError, ValueError):
                return False
        rows.append({
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "core_parse_success": parses(base_text),
            "cake_parse_success": parses(cake_text),
            "core_completion": base_text,
            "cake_completion": cake_text,
        })
    core_success = sum(row["core_parse_success"] for row in rows)
    cake_success = sum(row["cake_parse_success"] for row in rows)
    core_error = 1 - core_success / len(rows)
    cake_error = 1 - cake_success / len(rows)
    reduction = core_error / cake_error if cake_error else float("inf")
    return {
        "tasks": len(rows),
        "core_parse_success_rate": core_success / len(rows),
        "cake_parse_success_rate": cake_success / len(rows),
        "ordinary_error_reduction": reduction,
        "five_x_error_gate": "PASS" if reduction >= 5 else "FAIL",
        "rows": rows,
    }


def train_portable_fusion_cake(
    core_dir: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    *,
    domain: str | None = None,
    dataset_config_path: str | Path | None = None,
) -> dict:
    config = _config(config_path)
    configured_domain = str(config.get("domain", config.get("cake_id", "domain").split("-", 1)[0]))
    if domain is not None and domain != configured_domain:
        raise ValueError(
            f"requested domain {domain!r} does not match config domain {configured_domain!r}"
        )
    domain = configured_domain
    if dataset_config_path is not None:
        dataset = _config(dataset_config_path)
        if dataset.get("domain") not in {None, domain}:
            raise ValueError("dataset and cake domains differ")
        config["data"] = {
            key: dataset[key] for key in ("train", "validation", "test", "general_validation")
        }
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    core, core_metadata = load_core_checkpoint(core_dir, device=device)
    core_hash_before = sha256_file(Path(core_dir) / "model.safetensors")
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    cake = PortableFusionCake(PortableFusionConfig(**config["model"])).to(device)
    if cake.config.abi_width != core.config.abi_width:
        raise ValueError("cake and core canonical widths differ")
    optimizer = torch.optim.AdamW(
        cake.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    precision = config.get("precision", "fp32")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" and precision == "fp16" else nullcontext()
    )
    corpus = ByteCorpus(config["data"]["train"])
    route = int(core_metadata["route"])
    steps = int(config["training"]["steps"])
    batch_size = int(config["training"]["batch_size"])
    sequence_bytes = int(config["training"]["sequence_bytes"])
    curves = []
    started = time.perf_counter()
    for step, row in enumerate(corpus.batches(
        batch_size=batch_size, sequence_bytes=sequence_bytes, seed=seed,
        steps=steps, device=device,
    ), start=1):
        inputs, targets = row[:, :-1], row[:, 1:]
        with torch.no_grad(), autocast():
            _, aux = core(inputs, route=route, return_aux=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits, _ = cake(aux["core_logits"], aux["canonical"], inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(cake.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % int(config["training"]["evaluation_interval"]) == 0 or step == steps:
            curves.append({
                "step": step,
                "loss": float(loss.detach()),
                "raw_bytes_seen": step * batch_size * sequence_bytes,
                "wall_seconds": time.perf_counter() - started,
            })
    training_seconds = time.perf_counter() - started
    evaluation_kwargs = {
        "batch_size": int(config["evaluation"]["batch_size"]),
        "sequence_bytes": int(config["evaluation"]["sequence_bytes"]),
        "batches": int(config["evaluation"]["batches"]),
        "device": device,
        "route": route,
    }
    test_corpus = ByteCorpus(config["data"]["test"])
    domain = evaluate_fusion_bpb(core, cake, test_corpus, **evaluation_kwargs)
    no_canonical = evaluate_fusion_bpb(
        core, cake, test_corpus, disable_canonical=True, **evaluation_kwargs
    )
    no_anchor = evaluate_fusion_bpb(
        core, cake, test_corpus, disable_anchor=True, **evaluation_kwargs
    )
    random_cake = PortableFusionCake(cake.config).to(device).eval()
    random_control = evaluate_fusion_bpb(core, random_cake, test_corpus, **evaluation_kwargs)
    shuffled_cake = PortableFusionCake(cake.config).to(device)
    shuffled_cake.load_state_dict(cake.state_dict())
    with torch.no_grad():
        for parameter in shuffled_cake.parameters():
            flat = parameter.flatten()
            if flat.numel() > 1:
                parameter.copy_(flat[torch.randperm(flat.numel(), device=flat.device)].reshape_as(parameter))
    shuffled_control = evaluate_fusion_bpb(core, shuffled_cake, test_corpus, **evaluation_kwargs)
    syntax = _syntax_tasks(core, cake, device=device, route=route) if domain == "python" else {
        "tasks": 0, "core_parse_success_rate": None, "cake_parse_success_rate": None,
        "ordinary_error_reduction": None, "five_x_error_gate": "FAIL",
        "status": "NOT_RUN_NO_DOMAIN_TASK_SUITE", "rows": [],
    }
    general_corpus = ByteCorpus(config["data"]["general_validation"])
    general_core = evaluate_core(
        core, general_corpus, batch_size=evaluation_kwargs["batch_size"],
        sequence_bytes=evaluation_kwargs["sequence_bytes"], batches=evaluation_kwargs["batches"],
        device=device, route=route,
    )
    general_cake = evaluate_fusion_bpb(core, cake, general_corpus, **evaluation_kwargs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    private_pem, public_pem, key_id = generate_keypair()
    private_path = output_path.with_suffix(".private.pem")
    public_path = output_path.with_suffix(".public.pem")
    trust_store_path = output_path.with_name("trust-store.json")
    private_path.write_bytes(private_pem)
    public_path.write_bytes(public_pem)
    trust_store_path.write_text(
        json.dumps({key_id: public_path.name}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tensors = {name: value.detach().cpu().contiguous() for name, value in cake.state_dict().items()}
    tensor_copy = output_path.with_suffix(".safetensors")
    save_file(tensors, str(tensor_copy))
    manifest = CakeManifest(
        schema_version="1", cake_id=str(config["cake_id"]), name=str(config["name"]),
        description=str(config["description"]), version=str(config["version"]),
        publisher={"id": "layercake-research", "name": "LayerCake Research", "key_id": key_id},
        abi_version=core.canonical.config.version, abi_hash=core.canonical.config.abi_hash(),
        cake_type="portable_fusion",
        input_contract={"canonical_width": cake.config.abi_width, "raw_byte_context": True, "host_logits": 256},
        output_contract={
            "type": "logit_residual", "width": 256,
            "composition": cake.config.combination_rule,
        },
        architecture=portable_fusion_manifest_architecture(cake.config),
        supported_precisions=("fp32", "fp16", "int8"), supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"canonical_abi": core.canonical.config.version, "incremental": True},
        tensor_payload_hash="", tensor_shapes=tensor_specs(tensors), package_hash="",
        training_data_provenance={
            "path": str(Path(config["data"]["train"]).resolve()),
            "sha256": sha256_file(config["data"]["train"]),
            "bytes_seen": steps * batch_size * sequence_bytes,
            "receiver_data_used": False,
        },
        evaluation_evidence={
            "domain_bpb": domain["cake_bits_per_byte"],
            "core_bpb": domain["core_bits_per_byte"],
            "ordinary_error_reduction": syntax["ordinary_error_reduction"],
            "claim_status": "CANDIDATE_NOT_PROMOTED",
        },
        license=str(config["license"]), dependencies=(), parent_version=None,
        signature={"algorithm": "ed25519", "key_id": key_id}, domains=(domain,),
        keywords=tuple(config.get("keywords", (domain,))), permissions=("local-inference",),
    )
    build_package(output_path, manifest, tensors, private_key=private_path)
    package = load_package(output_path, trust_store={key_id: public_path})
    core_hash_after = sha256_file(Path(core_dir) / "model.safetensors")
    report = cake.parameter_report(
        core_metadata["parameters"]["total_parameters"],
        core_metadata["parameters"]["active_parameters"],
    )
    evidence = {
        "format": "layercake-portable-fusion-training/1",
        "status": "PASS" if core_hash_before == core_hash_after else "INVALID_EVIDENCE",
        "core": {
            "path": str(Path(core_dir).resolve()),
            "hash_before": core_hash_before,
            "hash_after": core_hash_after,
            "unchanged": core_hash_before == core_hash_after,
            "seed": core_metadata["seed"],
        },
        "cake": {
            "parameters": report,
            "training_seconds": training_seconds,
            "steps": steps,
            "raw_bytes_seen": steps * batch_size * sequence_bytes,
            "curves": curves,
        },
        "evaluation": {
            "heldout_domain": domain,
            "random_control": random_control,
            "shuffled_control": shuffled_control,
            "canonical_disabled": no_canonical,
            "byte_anchor_disabled": no_anchor,
            "general_core": general_core,
            "general_with_cake": general_cake,
            "syntax_tasks": syntax,
        },
        "package": {
            "path": str(output_path.resolve()),
            "archive_sha256": sha256_file(output_path),
            "content_hash": package.manifest.package_hash,
            "tensor_payload_hash": package.manifest.tensor_payload_hash,
            "signed": package.signed,
            "key_id": key_id,
            "public_key": str(public_path.resolve()),
            "trust_store": str(trust_store_path.resolve()),
            "safe_tensor_copy": str(tensor_copy.resolve()),
            "safe_tensor_sha256": sha256_file(tensor_copy),
        },
    }
    evidence_path = output_path.with_suffix(".evidence.json")
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence
