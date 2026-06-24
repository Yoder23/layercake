import torch

from layercake.rolling.registry import ModuleRegistry


def test_registry_hash_freeze_save_restore(tmp_path):
    registry = ModuleRegistry()
    model = torch.nn.Linear(2, 1)
    registry.register("toy", model)
    before = registry.module_hash("toy")
    registry.freeze("toy")
    assert not any(parameter.requires_grad for parameter in model.parameters())
    path = registry.save_module("toy", tmp_path / "toy.pt")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    assert registry.module_hash("toy") != before
    registry.restore_module("toy", path)
    assert registry.module_hash("toy") == before
    registry.unfreeze("toy")
    assert all(parameter.requires_grad for parameter in model.parameters())
