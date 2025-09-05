# MR-KG Backend

FastAPI backend for MR-KG (Mendelian Randomization Knowledge Graph).
Provides RESTful APIs for trait exploration, study analysis, and
similarity computation.

## Features

- FastAPI with automatic OpenAPI documentation
- DuckDB integration for vector similarity search
- Environment-based configuration via `.env`
- Hot reload development server
- Comprehensive test suite with pytest
- Code quality with ruff and type checking (ty)
- Task automation with a curated `justfile`

## Local Development

For initial setup and prerequisites, see @docs/SETTING-UP.md.

### Quick Start

```bash
cd backend

# Install dependencies and setup environment
just install
just env-setup

# Start development server (hot reload)
just dev
```

### Access Points

- Base URL: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health/

## Project Structure

```
backend/
├── app/
│   ├── api/          # API route handlers (v1 endpoints)
│   ├── core/         # Config, database, dependencies
│   ├── models/       # Pydantic data models
│   ├── services/     # Business logic services
│   ├── utils/        # Utilities and helpers
│   └── main.py       # FastAPI application entry point
├── tests/            # Test suite
├── justfile          # Task runner commands
└── pyproject.toml    # Python project configuration
```

## Commands

### Development

```bash
just install           # Install dependencies
just install-dev       # Install with dev dependencies
just dev               # Start server with hot reload
```

### API Tools

```bash
just docs              # Open Swagger UI
just validate-schema   # Validate OpenAPI schema
just test-endpoints    # Simple curl-based checks
```

### Database

```bash
just check-db          # Run DB integration tests
just test-db-health    # Check DB health endpoint
```

### Testing

```bash
just test              # Run tests (pytest -vv)
just test-cov          # Run tests with coverage report
just test-file tests/<path>.py   # Run a specific test file
just test-infra        # Infra-focused tests
just test-health       # Health endpoint tests
```

### Code Quality

```bash
just fmt               # Format and fix with ruff
just lint              # Lint with ruff
just ty                # Type check with ty
just check             # Lint + type check
```

### Performance

```bash
just load-test         # Basic load test (if ab is installed)
just profile           # Quick latency check
```

### Maintenance

```bash
just clean             # Remove caches and build artifacts
just reset             # Clean and remove .env, uv caches
```

## Configuration

Configuration is provided via environment variables in `.env`.
Use `just env-setup` to create `.env` from `.env.example`.
See @docs/ENV.md for complete variable documentation.

## API Documentation

Interactive API documentation is available when the server is running:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

Run the test suite locally:

```bash
just test      # All tests
just test-cov  # With coverage
```

The suite covers health checks, API routes, database integration, and
basic error handling.

## Code Quality

Maintain code quality with provided commands:

```bash
just check     # Run all quality checks (lint + type check)
just fmt       # Format code
```

Tools used:
- ruff for formatting and linting
- ty for type checking
- pytest for tests

## References

- Environment configuration: @docs/ENV.md
- Development workflows: @docs/DEVELOPMENT.md
- Testing guidelines: @docs/TESTING.md
- System architecture: @docs/ARCHITECTURE.md
