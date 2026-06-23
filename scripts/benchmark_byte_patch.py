from __future__ import annotations

import argparse
import torch

from _common import emit
from layercake.abi import ABISpec
from layercake.benchmarks import BenchmarkRecord, timed_run
from layercake.byte_patch import ByteCodec, FixedBytePatcher
from layercake.input_interfaces import InputInterfaceSpec
from layercake.model_v2 import BytePatchLayerCake


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()
    ids = torch.tensor([ByteCodec.encode_text("abc def " * 32)], dtype=torch.long)
    interface = InputInterfaceSpec(
        mode="byte_patch", patching="fixed:4", max_patch_size=4
    )
    abi = ABISpec(version="lc-abi/2", d_abi=32, input_interface=interface)
    model = BytePatchLayerCake(
        abi, d_model=32, n_layers=1, n_heads=4, patcher=FixedBytePatcher(4)
    ).eval()
    with torch.no_grad():
        _, _, metadata = model(ids)
        elapsed, throughput = timed_run(
            lambda: [model(ids) for _ in range(args.iterations)],
            ids.numel() * args.iterations,
        )
    record = BenchmarkRecord(
        benchmark="byte_patch_inference",
        model="smoke-32d",
        input_mode="byte_patch",
        wall_seconds=elapsed,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
        units_processed=ids.numel() * args.iterations,
        unit="bytes",
        throughput=throughput,
        patch_compression_ratio=metadata[0].compression_ratio,
    )
    emit(record.to_dict(), args.output)


if __name__ == "__main__":
    main()
