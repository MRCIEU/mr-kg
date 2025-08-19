# Development guidelines

This repository uses Python (>=3.10) with multiple subprojects. Follow these guidelines for agentic coding:

## Build, Lint, and Test Commands
- **Test:** `poetry run pytest -vv` (src/yiutils) or use `just` recipes in other subprojects.
- **Lint:** `python -m flake8` and `python -m mypy` (src/yiutils), or `uv run ty check` (webapp, processing, common_funcs).
- **Format:** Use `autoflake`, `isort`, and `black` (src/yiutils), or `uv run ruff format` and `uv run ruff check --fix` (webapp, processing, common_funcs).
- **Build:** `poetry build` (src/yiutils).
- **Single test:** `pytest path/to/test_file.py::test_function` (standard pytest).

## Code Style Guidelines
- **Imports:** Use `isort` for sorting; remove unused imports with `autoflake`.
- **Formatting:** Use `black` or `ruff format` (line length 79).
- **Linting:** Use `flake8` (ignore E501), `ruff check`, and `ty check` for type safety.
- **Types:** Use type hints; check with `mypy` or `ty`.
- **Naming:** Use descriptive, snake_case for functions/variables, PascalCase for classes.
- **Error Handling:** Prefer explicit exceptions; use logging (e.g., loguru) for errors.
- **Dependencies:** Use `pyproject.toml` and `environment.yml` for dependency management.
- **Environment:** Use `.env` for secrets (e.g., `ACCOUNT_CODE` for HPC).
- **General:** Keep code modular, readable, and well-documented.
