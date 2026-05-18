# Security Policy

## Supported Versions

This is a research preview (v0.1.x). No production deployments are expected.

## Reporting a Vulnerability

If you discover a security vulnerability in this code, please report it privately:

- Open a GitHub Security Advisory (preferred):
  `https://github.com/Yoder23/layercake/security/advisories/new`
- Or email the maintainer directly (see GitHub profile).

Do not open a public issue for security vulnerabilities.

We will acknowledge your report within 72 hours and provide a fix or mitigation
timeline within 7 days.

## Scope

The primary security concern for this codebase is:

- **Unsafe deserialization**: `torch.load` without `weights_only=True` can execute
  arbitrary code in a malicious checkpoint file. Never load untrusted `.pt` files
  without inspecting them first or using `torch.load(..., weights_only=True)`.

All `torch.load` calls in this repository use `map_location="cpu"`. For production
use, add `weights_only=True` (requires PyTorch >= 2.0).
