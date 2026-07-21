import pytest
import torch
from pathlib import Path

from layercake.cake.cli import DEFAULT_ABI_HASH, DEFAULT_ABI_VERSION
from layercake.cake.installer import CakeInstaller, HostCapabilities, InstallationError
from layercake.cake.package import build_package
from layercake.cake.registry import CakeRegistry
from layercake.portable_domain import PortableDomainDecoder

from .test_package_security import manifest


def test_atomic_install_verify_remove_reinstall_and_abi_guard(tmp_path):
    model = PortableDomainDecoder(feature_width=8, hidden_width=16)
    package = build_package(tmp_path / "python.cake", manifest(model), model.state_dict())
    registry = CakeRegistry(tmp_path / "registry")
    host = HostCapabilities(DEFAULT_ABI_VERSION, DEFAULT_ABI_HASH)
    installer = CakeInstaller(registry, host)
    installed = installer.install(package, trusted_local=True)
    assert installed["status"] == "INSTALLED"
    assert installer.verify("python")["status"] == "PASS"
    assert installer.remove("python")["status"] == "REMOVED"
    assert registry.get("python") is None
    assert installer.install(package, trusted_local=True)["archive_hash"] == installed["archive_hash"]

    incompatible = CakeInstaller(registry, HostCapabilities("other", "f" * 64))
    with pytest.raises(InstallationError, match="ABI"):
        incompatible.inspect(package, trusted_local=True)


def test_update_and_rollback_restore_the_complete_old_record(tmp_path):
    first_model = PortableDomainDecoder(feature_width=8, hidden_width=16)
    first_manifest = manifest(first_model)
    first_package = build_package(tmp_path / "python-v1.cake", first_manifest, first_model.state_dict())
    registry = CakeRegistry(tmp_path / "registry")
    installer = CakeInstaller(registry, HostCapabilities(DEFAULT_ABI_VERSION, DEFAULT_ABI_HASH))
    first = installer.install(first_package, trusted_local=True)

    torch.manual_seed(99)
    second_model = PortableDomainDecoder(feature_width=8, hidden_width=16)
    second_manifest = type(first_manifest).from_dict({
        **first_manifest.canonical_dict(), "version": "1.3.0", "parent_version": "1.2.3",
    })
    second_package = build_package(tmp_path / "python-v2.cake", second_manifest, second_model.state_dict())
    updated = installer.update(second_package, trusted_local=True)
    assert updated["version"] == "1.3.0"
    assert updated["archive_hash"] != first["archive_hash"]

    rolled_back = installer.rollback("python")
    assert rolled_back["version"] == "1.2.3"
    assert rolled_back["archive_hash"] == first["archive_hash"]
    assert Path(rolled_back["blob"]) == registry.blob_path(first["archive_hash"])
    assert installer.verify("python")["archive_hash"] == first["archive_hash"]
