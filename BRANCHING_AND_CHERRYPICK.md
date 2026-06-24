# Branching and cherry-pick

`BranchStore` provides the minimal branch/tag/HEAD mechanics needed for rolling model
experiments. It is not a replacement for git; it is a model-artifact index.

Supported operations:

- create branch;
- checkout commit;
- list branches;
- tag commit;
- diff two commits;
- cherry-pick one module from a compatible source commit into a target commit;
- bisect a sequence of commits by gate status.

Cherry-pick currently enforces:

- matching ABI hash;
- matching input-interface hash;
- source module exists.

The intended use is domain-brick or PX-payload transfer after both source and receiver
certificates pass. If ABI or input-interface hashes differ, cherry-pick fails instead of
silently copying incompatible weights.
