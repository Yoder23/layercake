"""Fail-closed installation, update, rollback, verification, and removal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

from .package import CakePackage, PackageError, load_package, sha256_bytes
from .registry import CakeRegistry, RegistryError


class InstallationError(ValueError):
    pass


@dataclass(frozen=True)
class HostCapabilities:
    abi_version: str
    abi_hash: str
    precisions: tuple[str, ...] = ("fp32",)
    backends: tuple[str, ...] = ("pytorch",)
    capabilities: frozenset[str] = frozenset({"byte_input", "safe_tensors"})


class CakeInstaller:
    def __init__(
        self,
        registry: CakeRegistry,
        host: HostCapabilities,
        *,
        trust_store: Mapping[str, bytes | str | Path] | None = None,
        strict_signatures: bool = True,
    ):
        self.registry = registry
        self.host = host
        self.trust_store = dict(trust_store or {})
        self.strict_signatures = bool(strict_signatures)

    def _validate_host(self, package: CakePackage) -> None:
        manifest = package.manifest
        if manifest.abi_version != self.host.abi_version or manifest.abi_hash != self.host.abi_hash:
            raise InstallationError("cake ABI version/hash is incompatible with this host")
        if not set(manifest.supported_precisions) & set(self.host.precisions):
            raise InstallationError("cake and host share no supported precision")
        if not set(manifest.supported_backends) & set(self.host.backends):
            raise InstallationError("cake and host share no supported runtime backend")
        required = manifest.minimum_host_capabilities.get("features", [])
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise InstallationError("minimum_host_capabilities.features must be a string array")
        missing = set(required) - set(self.host.capabilities)
        if missing:
            raise InstallationError(f"host lacks required capabilities: {sorted(missing)}")

    def inspect(self, source: str | Path, *, trusted_local: bool = False) -> CakePackage:
        try:
            package = load_package(
                source,
                trust_store=self.trust_store,
                require_signature=self.strict_signatures and not trusted_local,
                allow_local_development=trusted_local,
            )
            self._validate_host(package)
            return package
        except (PackageError, RegistryError) as exc:
            raise InstallationError(str(exc)) from exc

    def install(self, source: str | Path, *, trusted_local: bool = False) -> dict[str, Any]:
        package = self.inspect(source, trusted_local=trusted_local)
        manifest = package.manifest
        for dependency in manifest.dependencies:
            if self.registry.get(dependency) is None:
                raise InstallationError(f"missing cake dependency: {dependency}")
        blob = self.registry.store_blob(package.path, package.archive_hash)
        record = {
            "cake_id": manifest.cake_id,
            "name": manifest.name,
            "description": manifest.description,
            "version": manifest.version,
            "cake_type": manifest.cake_type,
            "abi_version": manifest.abi_version,
            "abi_hash": manifest.abi_hash,
            "archive_hash": package.archive_hash,
            "package_hash": manifest.package_hash,
            "tensor_payload_hash": manifest.tensor_payload_hash,
            "signed": package.signed,
            "trusted_local": bool(trusted_local),
            "publisher": manifest.publisher,
            "domains": list(manifest.domains),
            "keywords": list(manifest.keywords),
            "permissions": list(manifest.permissions),
            "composition": manifest.output_contract.get(
                "composition", manifest.output_contract.get("combination", "none")
            ),
            "installed_at": time.time(),
            "blob": str(blob),
        }
        previous = self.registry.activate(record)
        return {"status": "UPDATED" if previous else "INSTALLED", **record}

    def update(self, source: str | Path, *, trusted_local: bool = False) -> dict[str, Any]:
        package = self.inspect(source, trusted_local=trusted_local)
        existing = self.registry.get(package.manifest.cake_id)
        if existing is None:
            raise InstallationError("update requires an installed cake")
        def semver_core(value: str) -> tuple[int, int, int]:
            return tuple(int(part) for part in value.split("-", 1)[0].split("."))
        if semver_core(package.manifest.version) <= semver_core(existing["version"]):
            raise InstallationError("update version must be newer than the installed version")
        if package.manifest.parent_version != existing["version"]:
            raise InstallationError("update parent_version must equal the installed version")
        return self.install(source, trusted_local=trusted_local)

    def verify(self, cake_id: str) -> dict[str, Any]:
        record = self.registry.get(cake_id)
        if record is None:
            raise InstallationError(f"cake is not installed: {cake_id}")
        blob = self.registry.blob_path(record["archive_hash"])
        if not blob.is_file() or sha256_bytes(blob.read_bytes()) != record["archive_hash"]:
            raise InstallationError("installed content-addressed blob is missing or corrupt")
        package = self.inspect(blob, trusted_local=bool(record.get("trusted_local")))
        if package.manifest.package_hash != record["package_hash"]:
            raise InstallationError("registry/package hash mismatch")
        return {
            "status": "PASS",
            "cake_id": cake_id,
            "version": record["version"],
            "archive_hash": record["archive_hash"],
            "package_hash": record["package_hash"],
            "payload_hash": package.manifest.tensor_payload_hash,
            "signed": package.signed,
        }

    def remove(self, cake_id: str) -> dict[str, Any]:
        previous = self.registry.remove(cake_id)
        return {"status": "REMOVED", "cake_id": cake_id, "version": previous["version"]}

    def rollback(self, cake_id: str) -> dict[str, Any]:
        record = self.registry.rollback(cake_id)
        self.verify(cake_id)
        return {"status": "ROLLED_BACK", **record}
