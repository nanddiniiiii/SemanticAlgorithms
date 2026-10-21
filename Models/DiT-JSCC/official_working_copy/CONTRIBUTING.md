# Contributing

Thank you for helping improve DiT-JSCC.

## Development setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -e ".[dev]"
```

Run the lightweight checks before opening a pull request:

```bash
python -m compileall -q dit_jscc configs app.py infer.py
pytest -q
ruff check dit_jscc app.py infer.py
```

## Pull requests

- Keep the public inference API backward compatible when possible.
- Separate research experiments from the stable `dit_jscc/` package.
- Do not commit datasets, checkpoints, API keys, local absolute paths, or experiment outputs.
- Document any new checkpoint convention, dependency, channel model, or user-facing parameter.
- Include a minimal test for fixes that do not require model weights.

For architecture or metric changes, describe how the result was validated and which checkpoint/configuration was used.
