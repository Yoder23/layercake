"""Safe, declarative LayerCake extension packages."""

from .installer import CakeInstaller, HostCapabilities
from .manifest import CakeManifest, ManifestError
from .package import CakePackage, PackageError, build_package, load_package
from .registry import CakeRegistry

__all__ = [
    "CakeInstaller",
    "CakeManifest",
    "CakePackage",
    "CakeRegistry",
    "HostCapabilities",
    "ManifestError",
    "PackageError",
    "build_package",
    "load_package",
]
