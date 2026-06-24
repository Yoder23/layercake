# Rollback

Rollback restores registered module artifacts from a passing parent commit.

Supported smoke behavior:

- failed commit is kept;
- trainable modules are restored from the parent artifact paths;
- rollback reports the restored commit and modules;
- the next rubric continues from the restored parent state.

Rollback is intentionally module-scoped. Optimizer-state rollback is represented as a
stub interface because not all training loops use the same optimizer storage. Production
scale runs should persist optimizer and scheduler state in the commit artifacts whenever
resume-exactness matters.
