# Semantic CI

Semantic CI is the rule that a candidate model is not promoted by training loss alone.
It must pass its declared gates and write a certificate.

Current smoke path:

```powershell
python -m layercake.rolling.cli --help
python scripts/demo_rolling_training.py --smoke
pytest tests/test_rolling_cli.py tests/test_rolling_rubric.py tests/test_rolling_trainer.py tests/test_model_commit.py tests/test_module_registry.py tests/test_dataset_manifest.py tests/test_gates.py tests/test_rollback.py tests/test_branching.py tests/test_cherrypick.py tests/test_bisect.py -q
```

CI now runs the rolling CLI, demo, and rolling unit tests. The generated certificate is
`results/certificates/rolling_demo_certificate.json`.

For real LayerCake scale runs, semantic CI should require both:

1. source/core certificate passes; and
2. receiver-after-transfer certificate passes with exact transfer and no generation
   degradation.
