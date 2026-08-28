# Contributing

## Development Setup

```bash
git clone https://github.com/Arhaan-P/DeepFake-Detection.git
cd DeepFake-Detection

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -e .
pip install -r requirements.txt
```

The editable install (`pip install -e .`) makes the `models` and `utils`
packages importable from any script location. GPU training requires CUDA;
see the comment at the top of `requirements.txt` for the PyTorch CUDA install
command.

## Running the smoke test

Before opening a PR, confirm the pipeline still builds and runs end-to-end on
synthetic data (no dataset or GPU required):

```bash
python tests/smoke_test.py
```

## Code style

This project uses [black](https://github.com/psf/black) for formatting and
[ruff](https://github.com/astral-sh/ruff) for linting:

```bash
pip install black ruff
black models/ utils/ scripts/ ieee_scripts/ tests/
ruff check models/ utils/ scripts/ ieee_scripts/ tests/
```

## Scope of changes

- `models/` and `utils/` implement the exact architecture and feature
  extraction used to produce the results reported in the paper. Changes here
  should not alter model behavior, training dynamics, or evaluation numbers
  without an accompanying re-evaluation — see `AUDIT_FINDINGS.md` for known
  issues that are documented but intentionally not fixed.
- `.github/instructions/rules.instructions.md` has project-specific
  conventions (PowerShell on Windows, MediaPipe as the feature backend, LOOCV
  protocol, etc.) worth reading before making changes to the pipeline.
