from __future__ import annotations

import argparse
import torch

from _common import emit
from layercake.abi import ABISpec
from layercake.byte_patch import ByteCodec, FixedBytePatcher
from layercake.domain_bricks import LowRankDomainOperator
from layercake.input_interfaces import InputInterfaceSpec
from layercake.model_v2 import BytePatchLayerCake


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.manual_seed(42)
    text = "LayerCake byte-patch smoke: café λ"
    ids = torch.tensor([ByteCodec.encode_text(text)], dtype=torch.long)
    interface = InputInterfaceSpec(
        mode="byte_patch", patching="fixed:4", max_patch_size=4
    )
    abi = ABISpec(version="lc-abi/2", d_abi=32, input_interface=interface)
    brick = LowRankDomainOperator(abi, rank=4, alpha_init=0.0)
    model = BytePatchLayerCake(
        abi, d_model=32, n_layers=1, n_heads=4, patcher=FixedBytePatcher(4), domain_brick=brick
    )
    logits, states, metadata = model(ids)
    emit(
        {
            "status": "PASS",
            "bytes": ids.numel(),
            "patches": metadata[0].patch_count,
            "patch_compression_ratio": metadata[0].compression_ratio,
            "byte_logits_shape": list(logits.shape),
            "abi_shape": list(states.shape),
            "abi_hash": abi.hash(),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
