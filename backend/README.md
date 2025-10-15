# MR-KG Backend

FastAPI backend for MR-KG (Mendelian Randomization Knowledge Graph).

For comprehensive backend documentation, see @docs/backend/overview.md.

For initial setup and prerequisites, see @docs/setting-up.md.

## Quick Start

```bash
cd backend

# Install dependencies and setup environment
just install
just env-setup

# Start development server (hot reload)
just dev
```

Access points:
- Base URL: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health/

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

## Documentation

- Backend overview: @docs/backend/overview.md
- API design patterns: @docs/backend/api-design.md
- Database layer: @docs/backend/database-layer.md
- Environment configuration: @docs/env.md
- Testing guidelines: @docs/testing.md
