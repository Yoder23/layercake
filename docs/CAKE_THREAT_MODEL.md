# Cake package threat model

## Assets and adversary

The host model, other installed cakes, registry metadata, publisher identity, and user
data are protected assets. A cake source is assumed malicious: it may use malformed ZIP
metadata, duplicate names, traversal paths, compression abuse, pickle payloads, false
tensor metadata, ABI confusion, tampered evidence, forged signatures, or excessive memory
requirements.

## Security boundary

A `.cake` accepts exactly `manifest.json`, `tensors.safetensors`, and optionally
`signature.json`. It never imports or executes package content. The loader:

1. limits archive and expanded size;
2. rejects absolute, nested, traversal, duplicate, case-ambiguous, encrypted, directory,
   symlink, and unknown entries;
3. parses a schema-closed canonical JSON manifest;
4. authenticates the manifest and safetensors bytes with a content hash;
5. verifies an Ed25519 signature against an explicit trust store for published cakes;
6. requires an explicit trusted-local flag for unsigned development cakes;
7. loads safetensors only, then checks every tensor name, dtype, and shape;
8. constructs one allow-listed architecture from complete metadata and loads it strictly;
9. enforces exact ABI hash/version, precision, backend, capabilities, dependencies,
   permissions, and memory policy before activation.

Installation copies the archive into a SHA-256 content-addressed blob and atomically
replaces a registry JSON pointer. Update history supports rollback; removal changes only
that cake’s record. Cakes never receive filesystem or registry mutation callbacks.

## Residual risks

Denial of service inside allowed tensor-size limits, compromised publisher keys, bugs in
Python/ZIP/safetensors/cryptography, and adversarial neural outputs remain possible.
Signature trust does not mean model safety. Runtime permission checks and output
verification remain necessary. Cross-device floating-point bit equality is not promised.
