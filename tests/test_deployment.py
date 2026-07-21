import torch

from layercake.causal_byte_models import CausalBytePatchLM
from layercake.deployment import PatchGenerationDeployment


def _model() -> CausalBytePatchLM:
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=16,
        d_abi=8,
        layers=1,
        heads=4,
        max_patches=16,
        local_decoder="window_transformer",
        local_layers=1,
        local_width=16,
        modern_blocks=True,
        fused_attention=True,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_generation_width=16,
        patch_generation_bytes=6,
        patch_generation_copy_window=8,
        patch_generation_copy_dim=4,
        patch_generation_position_copy=True,
    ).eval()
    return model


def test_patch_deployment_is_byte_exact_and_prunes_unused_local_decoder():
    torch.manual_seed(901)
    model = _model()
    deployment = PatchGenerationDeployment(model).eval()
    prompt = torch.randint(0, 256, (2, 12))

    expected = model.generate_next_patch(prompt)
    actual = deployment.generate_next_patch(prompt)

    assert torch.equal(actual, expected)
    assert sum(p.numel() for p in deployment.parameters()) < sum(
        p.numel() for p in model.parameters()
    )
    assert deployment.deployment_manifest()["scope"] == (
        "global autoregressive patch generation only"
    )


def test_patch_deployment_supports_linear_dynamic_int8():
    torch.manual_seed(902)
    deployment = PatchGenerationDeployment(_model()).eval()
    quantized = torch.ao.quantization.quantize_dynamic(
        deployment,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    generated = quantized.generate_next_patch(
        torch.randint(0, 256, (1, 12))
    )

    assert generated.shape == (1, 6)
    assert generated.dtype == torch.long
