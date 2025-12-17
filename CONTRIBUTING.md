# Contributing to CodeGrok MCP

Thank you for your interest in contributing to CodeGrok MCP! Contributions are welcome from everyone. Please follow the guidelines below to make the process smooth.

## How to report issues

- Search existing issues to avoid duplicates.
- Open a new issue with a clear title and a minimal reproducible example when relevant.

## How to submit pull requests

1. Fork the repository and create a branch for your change.
2. Write clear, focused commits. Prefer small, reviewable changes.
3. Run the test suite locally: `pytest` from the project root.
4. Add tests for new features and fixes.
5. Open a PR against `main`, describe the change, and link any related issues.

## Code style and linters

- Formatting: Run `black .` (line length 100).
- Type checking: Use `mypy` and keep strictness reasonable.
- Linting: `flake8` for style complaints.

## Testing

- Tests are located in the `tests/` directory.
- Use `pytest` for running tests; we recommend `pytest -q` for quick feedback.
- Add unit tests for new functionality and regression tests for bug fixes.

## Development setup

1. Create and activate a virtualenv (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```
2. Run tests and linters before submitting a PR.

If you have questions about the process, open an issue or discussion and we'll help out.

Thanks for helping improve CodeGrok MCP! 🚀
