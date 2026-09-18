# Contributing to LayerCake

LayerCake welcomes fixes, documentation, security improvements, runtime work,
new tests, and well-evidenced research. Start with the
[repository map](docs/REPOSITORY_MAP.md) and read [AGENTS.md](AGENTS.md) before
changing governed or evidence-bound surfaces.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m layercake --help
python scripts/check_docs.py
pytest -q
```

Linux and macOS use `source .venv/bin/activate`.

## Find the right surface

| Contribution | Start in | Minimum validation |
| --- | --- | --- |
| Package lifecycle | `layercake/cake/`, `tests/cake/` | Focused tests plus package tamper/compatibility coverage |
| Routing or orchestration | `layercake/routing/`, `tests/routing/` | Focused routing tests plus fail-closed negative cases |
| Runtime or kernels | `layercake/runtime/` | Numerical identity, incremental-state, memory, and device tests appropriate to the change |
| Model or training research | `layercake/models/`, `layercake/training/`, a versioned config | Frozen splits, seeds, commands, raw results, and honest promotion outcome |
| Documentation | Root canonical docs or `docs/` | `python scripts/check_docs.py` |
| New scientific claim | Governed protocol plus `results/moonshot/` | Exact raw rows, derivation, hashes, environment, verifier, and invalidation analysis |

## Pull-request checklist

1. Explain the user-visible or scientific problem.
2. Keep the change scoped to the smallest responsible subsystem.
3. Add a regression test for behavior changes.
4. Run focused tests and the complete suite when practical.
5. Run `python scripts/check_docs.py` when documentation or public commands
   change.
6. Do not modify generated certificates by hand.
7. Do not overwrite or delete negative or superseded evidence.
8. State whether the invalidation matrix affects any sealed claim.

## Code style

- Python 3.10+
- type hints on public function signatures
- `ruff check layercake tests scripts/check_docs.py`
- deterministic serialization for manifests, packages, and evidence
- fail-closed validation for signatures, hashes, package schemas, permissions,
  and scientific inputs

The installable package depends on PyTorch, NumPy, safetensors, cryptography,
psutil, and PyYAML. Add dependencies only when the public runtime or a clearly
declared optional feature requires them.

## Claim standards

A pull request that adds or changes a scientific or performance claim must
include:

1. the exact final artifact and source commit;
2. exact commands, seeds, data and artifact hashes, environment, and hardware;
3. immutable raw observations and a verifier that recomputes aggregates;
4. the comparison contract, including what was held constant;
5. negative controls, failures, and uncertainty required by the protocol; and
6. an invalidation analysis for dependent certificates.

Do not infer semantic portability from byte identity, quality from a construct
test, performance from another checkpoint, or independent reproduction from a
same-machine rerun.

## Evidence and large files

Do not add arbitrary generated output to `results/` or checkpoints to
`artifacts/`. Evidence-bearing files require a locked protocol, content hashes,
and a retention/publication plan. Large immutable payloads should use an
approved hash-addressed distribution path rather than silently growing normal
Git history.

Never commit private signing keys, secrets, restricted training data, or
unreviewed executable package payloads.

## Reporting issues

Open a GitHub issue with:

- operating system and hardware;
- Python and PyTorch versions;
- exact commit or tag;
- exact command;
- full error output or divergent value; and
- whether required model, data, or package assets were available.

Security issues should follow [SECURITY.md](SECURITY.md).

## License

Contributions are licensed under Apache-2.0; see [LICENSE](LICENSE).
