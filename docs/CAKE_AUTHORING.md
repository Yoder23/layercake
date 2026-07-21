# Cake authoring guide

1. Choose `portable_decoder` only when the complete prediction path excludes host
   representations. Otherwise declare `host_residual`.
2. Freeze training, provenance, split hashes, architecture, ABI, precision, runtime, and
   evaluation evidence before packaging.
3. Construct a `CakeManifest` and call `build_package`. Tensor names/shapes/dtypes must
   exactly match `model.state_dict()`; arbitrary files and code are forbidden.
4. Generate an Ed25519 publisher key with `python -m layercake cake keygen`. Keep the
   private key offline and distribute its public key through a trust store.
5. Test strict installation, tampering rejection, uninstallation, reinstallation, and
   (for portable decoders) three independently trained receiver hosts.

Local experiments may use `signature.algorithm=none` and install with `--trusted-local`.
That trust decision is recorded and cannot be mistaken for a published signature.

```powershell
python -m layercake cake install examples/python.cake --trusted-local
python -m layercake cake verify python
python -m layercake run --cake python "Explain a Python generator"
python -m layercake cake remove python
```

Do not place training code, tokenizers, templates, retrieval corpora, or stored answers in
a cake. Evaluation evidence belongs in the manifest; large reproducibility materials
belong in a separately authenticated research release.
