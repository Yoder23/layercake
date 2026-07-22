"""Real install/route/remove/reinstall demonstration over trained V2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import time
import zipfile

import torch
from safetensors.torch import load_file

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import load_package
from layercake.cake.registry import CakeRegistry
from layercake.models.portable_decoder import load_cake_module
from layercake.routing.learned_router import CompactSemanticRouter
from layercake.routing.orchestrator import LocalLayerCakeOrchestrator
from layercake.routing.policies import CakePermissionPolicy, RoutingPolicy
from layercake.training.cake import evaluate_fusion_bpb
from layercake.training.data import ByteCorpus, sha256_file
from layercake.training.foundation import evaluate_core, load_core_checkpoint


def _text(state) -> str:
    return bytes(state.generated_bytes[0].cpu().tolist()).decode("utf-8", errors="replace")


@torch.inference_mode()
def run_orchestration_demo(
    *,
    host_dirs: list[str | Path],
    cake_path: str | Path,
    public_key_path: str | Path,
    router_path: str | Path,
    domain_test_path: str | Path,
    registry_root: str | Path,
    output_path: str | Path,
) -> dict:
    cake_path = Path(cake_path).resolve()
    with zipfile.ZipFile(cake_path) as archive:
        manifest_data = json.loads(archive.read("manifest.json"))
    key_id = str(manifest_data["signature"]["key_id"])
    package_probe = load_package(
        cake_path,
        trust_store={key_id: Path(public_key_path)},
    )
    trust_store = {key_id: Path(public_key_path)}
    router = CompactSemanticRouter()
    router.load_state_dict(load_file(str(router_path)), strict=True)
    router.eval()
    cake_module = load_cake_module(package_probe).cpu().eval()
    corpus = ByteCorpus(domain_test_path)
    package_hash_before = sha256_file(cake_path)
    policy = RoutingPolicy(permissions=CakePermissionPolicy(
        allowed_permissions=frozenset({"local-inference"})
    ))
    host_rows = []
    source_lifecycle = None
    for index, host_dir in enumerate(host_dirs):
        host_dir = Path(host_dir)
        core, metadata = load_core_checkpoint(host_dir, device="cpu")
        route = int(metadata["route"])
        registry = CakeRegistry(Path(registry_root) / host_dir.name)
        installer = CakeInstaller(
            registry,
            HostCapabilities(
                abi_version=metadata["canonical_abi"]["version"],
                abi_hash=metadata["canonical_abi_hash"],
                precisions=("fp32", "fp16", "int8"),
                backends=("pytorch", "cuda", "torchscript"),
            ),
            trust_store=trust_store,
            strict_signatures=True,
        )
        started = time.perf_counter_ns()
        installed = installer.install(cake_path)
        install_ms = (time.perf_counter_ns() - started) / 1_000_000
        verified = installer.verify(package_probe.manifest.cake_id)
        before = evaluate_core(
            core, corpus, batch_size=4, sequence_bytes=128, batches=2,
            device=torch.device("cpu"), route=route,
        )
        after = evaluate_fusion_bpb(
            core, cake_module, corpus, batch_size=4, sequence_bytes=128, batches=2,
            device=torch.device("cpu"), route=route,
        )
        orchestrator = LocalLayerCakeOrchestrator(
            registry, policy=policy, trust_store=trust_store, device="cpu",
            semantic_router=router,
        )

        def core_handler(prompt: str) -> str:
            state = core.prefill(prompt, route=route, capture_generated=True)
            _, state = core.decode_many(state, 32)
            return _text(state)

        def cake_handler(prompt: str, modules: list[torch.nn.Module], _decision) -> str:
            state = core.prefill(
                prompt, route=route, fusion_cake=modules[0], capture_generated=True
            )
            _, state = core.decode_many(state, 32, fusion_cake=modules[0])
            return _text(state)

        general = orchestrator.execute(
            "Explain why careful measurement matters in science.",
            core_handler=core_handler, cake_handler=cake_handler,
        )
        specialist = orchestrator.execute(
            "Implement an asynchronous bounded-concurrency crawler in Python.",
            core_handler=core_handler, cake_handler=cake_handler,
        )
        row = {
            "host": host_dir.name,
            "seed": metadata["seed"],
            "architecture": metadata["architecture"],
            "checkpoint_sha256": sha256_file(host_dir / "model.safetensors"),
            "parameters": metadata["parameters"],
            "install_milliseconds": install_ms,
            "installation": installed,
            "verification": verified,
            "domain_bpb": {
                "core": before["bits_per_byte"],
                "with_cake": after["cake_bits_per_byte"],
                "improvement": before["bits_per_byte"] / after["cake_bits_per_byte"],
            },
            "general_generation": {"output": general.output, "trace": general.metrics()},
            "specialist_generation": {"output": specialist.output, "trace": specialist.metrics()},
        }
        if index == 0:
            removed = installer.remove(package_probe.manifest.cake_id)
            absent_route = orchestrator.route(
                "Implement an asynchronous bounded-concurrency crawler in Python."
            )
            reinstalled = installer.install(cake_path)
            reverified = installer.verify(package_probe.manifest.cake_id)
            source_lifecycle = {
                "removed": removed,
                "route_after_remove": {
                    "selected": list(absent_route.selected),
                    "core_fallback": absent_route.core_fallback,
                    "reason": absent_route.reason,
                },
                "specialist_bpb_after_remove": before["bits_per_byte"],
                "reinstalled": reinstalled,
                "reverified": reverified,
                "archive_hash_preserved": sha256_file(cake_path) == package_hash_before,
            }
        host_rows.append(row)
    evidence = {
        "format": "layercake-orchestration-demo/2",
        "status": "PASS" if (
            len(host_rows) >= 3
            and all(row["domain_bpb"]["improvement"] > 1.0 for row in host_rows)
            and all(row["specialist_generation"]["trace"]["execution_path"] == "selected_cake" for row in host_rows)
            and source_lifecycle is not None
            and source_lifecycle["route_after_remove"]["core_fallback"]
            and source_lifecycle["archive_hash_preserved"]
        ) else "FAIL",
        "execution": "real trained checkpoints, signed cake, learned router, CPU-only incremental generation",
        "package": {
            "path": str(cake_path), "archive_sha256": package_hash_before,
            "content_hash": package_probe.manifest.package_hash,
            "payload_hash": package_probe.manifest.tensor_payload_hash,
            "signed": package_probe.signed,
        },
        "hosts": host_rows,
        "source_lifecycle": source_lifecycle,
        "limitations": [
            "the development cake's ordinary syntax-task gate failed",
            "the direct CPU-versus-GPU result is reported separately and is not promoted",
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence
