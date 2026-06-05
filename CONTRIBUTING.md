# Contributing to agent-tool-result-cache

Thanks for your interest in contributing! This project is a small, zero-dependency
LRU + TTL cache for agent tool call results. Contributions of all kinds are welcome:
bug reports, documentation improvements, tests, and features.

## Getting started

1. Fork the repository and clone your fork.
2. Create a virtual environment (Python 3.10 or newer is required):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. Install the package in editable mode with the development extras:

   ```bash
   pip install -e ".[dev]"
   ```

## Development workflow

Create a feature branch off `main`:

```bash
git checkout -b my-change
```

Make your changes, keeping them focused and well-scoped.

### Linting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting. Run it before
opening a pull request:

```bash
ruff check src/ tests/
```

### Tests

Tests are written with [pytest](https://docs.pytest.org/). Run the suite with:

```bash
pytest -v
```

Please add or update tests for any behavior you change. The runtime cache has no
third-party dependencies, so keep the core (`src/agent_tool_result_cache/`) free of
new dependencies.

## Submitting a pull request

1. Make sure `ruff check src/ tests/` and `pytest` both pass locally. The same
   checks run in CI across Python 3.10–3.13.
2. Write a clear commit message and PR description explaining the motivation.
3. Open the pull request against the `main` branch.

A maintainer will review your change. Thanks for helping improve the project!

## Reporting issues

When filing a bug report, please include:

- The Python version you are using.
- A minimal code snippet that reproduces the problem.
- What you expected to happen and what actually happened.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License, the same license that covers this project.
