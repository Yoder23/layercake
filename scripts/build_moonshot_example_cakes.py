"""Build deterministic trusted-local example cakes; no training claim is made."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from _common import ROOT
from layercake.moonshot import _cake_manifest
from layercake.cake.package import build_package
from layercake.portable_domain import PortableDomainDecoder


definitions = {
    "python": ["python", "generator", "iterator", "memory", "csv"],
    "mathematics": ["math", "algebra", "quadratic", "integrate"],
    "biomedical": ["biomedical", "clinical", "cohort", "endpoint"],
    "actions": ["application", "action", "json", "schema", "button"],
    "game": ["game", "archer", "brute", "stamina", "cover"],
}
destination = ROOT / "examples"
destination.mkdir(parents=True, exist_ok=True)
catalog = []
for index, (cake_id, keywords) in enumerate(definitions.items()):
    torch.manual_seed(20260721 + index)
    model = PortableDomainDecoder(feature_width=16, hidden_width=32)
    path = build_package(
        destination / f"{cake_id}.cake",
        _cake_manifest(cake_id, model, keywords),
        model.state_dict(),
    )
    catalog.append({
        "cake_id": cake_id, "version": "0.1.0", "path": f"examples/{path.name}",
        "trust": "trusted_local", "quality_claim": "none_untrained_smoke",
    })
(destination / "catalog.json").write_text(
    json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(catalog, indent=2, sort_keys=True))
