# Model commits

A `ModelCommit` records the exact state needed to audit a LayerCake training step:

- `commit_id`;
- `parent_commit_id`;
- branch and status;
- model family;
- ABI hash;
- input-interface hash;
- byte-patch hash;
- module hashes;
- artifact paths;
- rubric hash;
- evaluation/certificate hashes;
- human message and timestamp.

Commit IDs are derived from canonical JSON content. The commit object can be saved,
loaded, verified, marked passed/failed, and compared to a parent.

Generated commit artifacts live under `artifacts/commits/` by default and are ignored by
git because model artifacts can become large. Certificates and curated result JSONs are
the evidence intended for version control.
