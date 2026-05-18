# Contributing to LayerCake

Thank you for your interest in contributing. This is an active research project and we
welcome contributions that improve reproducibility, fix bugs, or add well-evidenced results.

---

## Before You Open a PR

1. **Run the paste test first**: `python tests/test_paste_lossless.py` must pass.
2. **Do not modify `results/`** unless you have new locked benchmark data with
   reproducible scripts.
3. **Do not weaken the "Does NOT Claim" section** in README.md or CLAIMS.md.
   Honest scope is non-negotiable.

---

## Types of Contributions We Welcome

| Type | Notes |
|------|-------|
| Bug fixes | Include a test that reproduces the bug. |
| New benchmark results | Must include the script that produced them and a fairness description. |
| New domain examples | Provide domain data, training command, and eval results. |
| Documentation improvements | Clarity and accuracy over length. |
| New model sizes | Must include a config JSON and pass the paste test. |

---

## Code Style

- Python 3.10+
- `ruff` for linting (`pip install ruff; ruff check .`)
- Type hints on public function signatures
- No external dependencies beyond `torch`, `numpy` in core files
  (`model.py`, `data.py`, `baseline_lm.py`, `paste_domain.py`)

---

## Claim Standards

Any PR that adds or modifies a performance claim must:

1. Include the exact script that produced the result
2. Document all hyperparameters (steps, seed, batch size, LR)
3. State what was held constant vs. what was varied
4. Include a fairness note if comparing against a baseline

Overclaiming — even by accident — erodes trust. If you are unsure whether a result is
strong enough to claim, open an issue first and we will discuss it.

---

## Reporting Issues

Use GitHub Issues. For reproducibility failures, include:
- Your Python + PyTorch version
- The exact command you ran
- The full error output or divergent metric value

---

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0
license (see [LICENSE](LICENSE)).
