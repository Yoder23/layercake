# Example cakes

Build or refresh the deterministic, untrained packaging examples:

```powershell
python scripts/build_moonshot_example_cakes.py
python -m layercake cake --catalog examples/catalog.json search python
python -m layercake cake install examples/python.cake --trusted-local
python -m layercake cake verify python
python -m layercake run --cake python "Explain why this Python generator leaks memory"
python -m layercake cake remove python
```

These files demonstrate the safe extension lifecycle only. Their manifests explicitly say
`UNTRAINED_SMOKE`; they are not evidence of specialist quality.
