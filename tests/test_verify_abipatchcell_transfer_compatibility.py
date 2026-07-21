from __future__ import annotations

from scripts.verify_abipatchcell_transfer_compatibility import verify


def test_abipatchcell_transfer_compatibility_passes_exact_source_receiver_copy():
    config = {
        "model": {
            "patch_size": 2,
            "d_byte": 8,
            "d_model": 32,
            "d_abi": 16,
            "layers": 1,
            "heads": 4,
            "max_patches": 8,
            "direct_global_context": True,
            "local_decoder": "abi_patch_cell",
            "local_width": 32,
            "modern_blocks": True,
            "fused_attention": True,
            "dropout": 0.0,
            "qk_norm": False,
            "global_block": "sparse_state_patch",
            "sparse_state_local_window": 4,
            "sparse_state_dilated_offsets": [4, 6],
            "sparse_state_chunk_size": 4,
        },
        "training": {"seq_len": 16},
    }

    result = verify(config, seed=7, device="cpu")

    assert result["status"] == "PASS"
    assert result["metrics"]["transfer_ppl_ratio"] == 1.0
    assert result["metrics"]["transfer_max_logit_diff"] == 0.0
    assert result["gates"]["transfer_generation_exact"] is True
