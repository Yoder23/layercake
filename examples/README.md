# Example cakes

These deterministic, untrained fixtures demonstrate the package lifecycle.
They do not demonstrate English or specialist quality.

From the repository root, use an isolated ignored registry:

```powershell
python -m layercake cake --registry .cache/demo-registry --catalog examples/catalog.json search python
python -m layercake cake --registry .cache/demo-registry install examples/python.cake --trusted-local
python -m layercake cake --registry .cache/demo-registry list
python -m layercake cake --registry .cache/demo-registry verify python
python -m layercake cake --registry .cache/demo-registry remove python
```

To rebuild the example archives deterministically:

```powershell
python scripts/build_moonshot_example_cakes.py
```

Inference additionally requires a compatible core and an installed package:

```powershell
python -m layercake run --registry <registry-dir> --core <core-dir> --cake <cake-id> "Your prompt"
```

See the [deployment quickstart](../DEPLOYMENT_QUICKSTART.md) for artifact and
trust requirements.
