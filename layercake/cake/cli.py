from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .installer import CakeInstaller, HostCapabilities, InstallationError
from .manifest import abi_hash
from .registry import CakeRegistry, RegistryError
from .signing import generate_keypair


DEFAULT_ABI_VERSION = "lc-portable-byte/1"
DEFAULT_ABI_CONTRACT = {
    "input": "causal_utf8_bytes",
    "vocab_size": 256,
    "portable_path": "causal_byte_anchors",
    "anchor_version": "lc-causal-byte-anchor/1",
}
DEFAULT_ABI_HASH = abi_hash(DEFAULT_ABI_VERSION, DEFAULT_ABI_CONTRACT)


def _json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _trust_store(path: str | None) -> dict:
    if path is None:
        return {}
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("trust store must map key ids to public-key paths")
    return {
        str(key): (source.parent / value).resolve() if not Path(value).is_absolute() else Path(value)
        for key, value in data.items()
    }


def _catalog(path: str | None) -> list[dict]:
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError("catalog must be a JSON array of package records")
    return data


def _resolve_source(value: str, catalog: list[dict]) -> Path:
    source = Path(value)
    if source.is_file():
        return source
    matches = [row for row in catalog if row.get("cake_id") == value]
    if not matches:
        raise FileNotFoundError(f"no local file or catalog cake named {value!r}")
    return Path(matches[-1]["path"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="layercake cake", description="Safe LayerCake extension manager")
    parser.add_argument("--registry", default=os.environ.get("LAYERCAKE_REGISTRY"))
    parser.add_argument("--trust-store", default=os.environ.get("LAYERCAKE_TRUST_STORE"))
    parser.add_argument("--catalog", default=os.environ.get("LAYERCAKE_CATALOG"))
    parser.add_argument("--abi-version", default=os.environ.get("LAYERCAKE_ABI_VERSION", DEFAULT_ABI_VERSION))
    parser.add_argument("--abi-hash", default=os.environ.get("LAYERCAKE_ABI_HASH", DEFAULT_ABI_HASH))
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search")
    search.add_argument("query", nargs="?", default="")
    for name in ("install", "update"):
        command = sub.add_parser(name)
        command.add_argument("source")
        command.add_argument("--trusted-local", action="store_true")
    sub.add_parser("list")
    for name in ("verify", "info", "remove", "rollback"):
        command = sub.add_parser(name)
        command.add_argument("cake_id")
    keygen = sub.add_parser("keygen")
    keygen.add_argument("--private", required=True)
    keygen.add_argument("--public", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "keygen":
            private, public, identifier = generate_keypair()
            private_path, public_path = Path(args.private), Path(args.public)
            for path in (private_path, public_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            private_path.write_bytes(private)
            public_path.write_bytes(public)
            _json({"status": "CREATED", "key_id": identifier, "private": str(private_path), "public": str(public_path)})
            return 0
        registry = CakeRegistry(args.registry)
        catalog = _catalog(args.catalog)
        if args.command == "search":
            local = registry.search(args.query)
            terms = args.query.casefold().split()
            available = [row for row in catalog if all(term in json.dumps(row).casefold() for term in terms)]
            _json({"installed": local, "available": available})
            return 0
        if args.command == "list":
            _json(registry.list())
            return 0
        if args.command == "info":
            record = registry.get(args.cake_id)
            if record is None:
                raise RegistryError(f"cake is not installed: {args.cake_id}")
            _json(record)
            return 0
        host = HostCapabilities(
            abi_version=args.abi_version,
            abi_hash=args.abi_hash,
            precisions=("fp32", "fp16", "bf16", "int8"),
            backends=("pytorch", "torchscript", "cuda"),
        )
        installer = CakeInstaller(
            registry, host, trust_store=_trust_store(args.trust_store), strict_signatures=True
        )
        if args.command in {"install", "update"}:
            source = _resolve_source(args.source, catalog)
            result = getattr(installer, args.command)(source, trusted_local=args.trusted_local)
        elif args.command in {"verify", "remove", "rollback"}:
            result = getattr(installer, args.command)(args.cake_id)
        else:
            raise AssertionError("unhandled cake command")
        _json(result)
        return 0
    except (ValueError, OSError, RegistryError, InstallationError) as exc:
        print(f"layercake cake: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
