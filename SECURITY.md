# Security policy

## Project security boundary

LayerCake is an active research project. The sealed Moonshot is an evidence
release, not a blanket production-security certification. Deployments remain
responsible for publisher trust, private-key custody, process and network
isolation, resource controls, monitoring, and incident response.

The supported package path uses schema-closed, non-executable `.cake` archives,
safetensors payloads, cryptographic hashes, Ed25519 signatures, explicit ABI
compatibility, permissions, and content-addressed local storage. Read the
[package threat model](docs/CAKE_THREAT_MODEL.md) and
[registry specification](docs/CAKE_REGISTRY_SPEC.md) before deployment.

## Report a vulnerability

Report security vulnerabilities privately:

- open a [GitHub Security Advisory](https://github.com/Yoder23/layercake/security/advisories/new); or
- contact the maintainer privately through the GitHub profile.

Do not open a public issue containing an unpatched vulnerability, secret,
private key, exploit payload, or restricted data.

Please include the affected commit/tag, operating system, package or artifact
hash, reproduction steps, impact, and any proposed mitigation. We aim to
acknowledge reports within 72 hours and provide an initial remediation plan
within seven days.

## Untrusted inputs

Treat all of the following as untrusted:

- `.cake` archives and manifests;
- catalogs and archive-bound routing profiles;
- public keys and trust-store configuration;
- model checkpoints and tokenizers;
- dataset paths and imported evidence; and
- prompts, field-addressed inputs, and declared destinations.

Never bypass signature, archive-hash, tensor-shape, ABI, permission, or output
validation to make an artifact load.

## PyTorch checkpoints

Legacy research workflows still read `.pt` checkpoint containers. Repository
load sites use `weights_only=True`; this reduces Python object-deserialization
risk but does not make an untrusted model safe. Validate provenance and hashes,
load onto the intended device explicitly, and prefer signed safetensors-based
packages for distributed capability artifacts.

Never commit publisher private keys, credentials, restricted training data, or
unreviewed executable payloads.
