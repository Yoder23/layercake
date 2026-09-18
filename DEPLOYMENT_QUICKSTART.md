# LayerCake deployment quickstart

This guide separates the package-manager demo, artifact-backed inference, and
scientific release verification. Those are different workflows with different
requirements.

## Requirements

- Python 3.10 or newer
- Windows, Linux, or macOS for the Python reference surfaces
- PyTorch 2.0 or newer
- a CUDA-capable PyTorch installation only if you select `--device cuda`

The repository contains substantial historical evidence and artifacts. Use a
partial sparse clone when you only need the library, examples, and docs:

```bash
git clone --filter=blob:none --sparse --depth 1 https://github.com/Yoder23/layercake.git
cd layercake
git sparse-checkout set layercake layercake_extensions docs examples
python -m venv .venv
```

Use a normal full clone when reproducing historical evidence or contributing
across tests, scripts, configs, and campaign contracts.

Activate the environment and install the package:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m layercake --help
```

On Linux or macOS, activation is `source .venv/bin/activate`.

## Five-minute package lifecycle

The bundled cakes are deterministic, untrained fixtures. They are useful for
exercising safe package management without downloading a model:

```powershell
python -m layercake cake --registry .cache/demo-registry --catalog examples/catalog.json search python
python -m layercake cake --registry .cache/demo-registry install examples/python.cake --trusted-local
python -m layercake cake --registry .cache/demo-registry list
python -m layercake cake --registry .cache/demo-registry info python
python -m layercake cake --registry .cache/demo-registry verify python
python -m layercake cake --registry .cache/demo-registry remove python
```

Expected lifecycle statuses are `INSTALLED`, `PASS`, and `REMOVED`. The
registry stores immutable archive bytes under
`.cache/demo-registry/blobs/sha256/` and keeps the active record in
`registry.json`.

`--trusted-local` is deliberately explicit. It allows an unsigned development
fixture and records that trust mode. Production packages should be signed and
validated through a trust store.

## Run inference with your artifacts

Inference requires both:

1. a compatible core directory containing `metadata.json` and
   `model.safetensors`; and
2. an installed cake whose ABI, precision, backend, and required features match
   the host.

The current general CLI loader accepts a `LayerCakeFoundationV2` checkpoint.
The untrained examples prove package lifecycle only and should not be used to
evaluate generation quality.

Install a compatible cake, then select it explicitly:

```powershell
python -m layercake cake --registry <registry-dir> install <package.cake> --trusted-local
python -m layercake run --registry <registry-dir> --core <core-dir> --cake <cake-id> --device cpu --trace "Your prompt"
```

Or use an eligible installed catalog and router:

```powershell
python -m layercake run --registry <registry-dir> --core <core-dir> --auto-route --router <router.safetensors> --device cpu --trace "Your prompt"
```

If CUDA was requested but is unavailable, the CLI falls back to CPU. Inspect
the trace rather than assuming which device or cake executed.

The sealed Moonshot product and newer direct-neural-core constructs have
artifact-specific runners and exact evidence protocols. They are not silently
loaded through an incompatible generic checkpoint interface. Follow the
certificate's exact command when reproducing one of those results.

## Sign a package

Generate an Ed25519 key pair:

```powershell
python -m layercake cake keygen --private publisher-private.pem --public publisher-public.pem
```

Keep the private key out of the repository. A trust-store JSON file maps a
publisher key ID to its public-key path. See the
[authoring guide](docs/CAKE_AUTHORING.md),
[registry specification](docs/CAKE_REGISTRY_SPEC.md), and
[threat model](docs/CAKE_THREAT_MODEL.md) before distributing a package.

## Verify the sealed research release

Scientific verification is not the same as running current development HEAD.
Use a separate detached worktree so your working branch remains untouched:

```powershell
git worktree add ..\layercake-sealed layercake-moonshot-final
Set-Location ..\layercake-sealed
python -m pip install -e .
python -m layercake.moonshot_campaign verify-sealed 8
python -m layercake.moonshot_campaign verify-all
python -m layercake.moonshot_final verify
```

The retained assets required by the sealed protocol must be present at their
declared hashes. `verify-all` must report `completed_phases_valid: true`.
Current post-release HEAD is expected to reject the old seal because registered
components changed; see [Project status](docs/PROJECT_STATUS.md).

## Common failures

| Error | Meaning | Resolution |
| --- | --- | --- |
| `cake ABI version/hash is incompatible` | Package and host contracts differ | Use a package built for the host's exact ABI or a compatible host |
| `signature is required` | An unsigned package reached strict installation | Sign the package, configure the trust store, or use `--trusted-local` only for a local fixture |
| `cake is not installed` | The registry has no active record for that ID | Install it using the same `--registry` path |
| Missing `metadata.json` or `model.safetensors` | `--core` is not a complete compatible core directory | Supply the complete core artifact |
| Sealed verifier fails on current HEAD | Development changes invalidated inherited certificates | Verify from the exact detached tag |

For installation problems, include `python --version`,
`python -c "import torch; print(torch.__version__)"`, the full command, and
complete error output in a GitHub issue.
