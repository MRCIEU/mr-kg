# Backend overview

FastAPI backend for MR-KG providing RESTful APIs for trait exploration, study analysis, and similarity computation.

## Tech stack

- FastAPI with automatic OpenAPI documentation
- DuckDB for vector similarity search
- Pydantic for data validation
- Connection pooling and repository pattern
- Comprehensive health monitoring

## Quick reference

For commands and development workflows, see @backend/README.md.

## Architecture documentation

- API design patterns: @docs/backend/api-design.md
- Database layer: @docs/backend/database-layer.md

## Key features

- Standardized response models for consistent API contracts
- Custom exception handling with proper HTTP status codes
- Security middleware (CORS, headers, rate limiting)
- Versioned router architecture under /api/v1
- Health monitoring and metrics collection
- Connection pooling for DuckDB databases

## API structure

```
/api/v1/
├── health/          # Health checks and monitoring
├── system/          # System information
├── core/            # Core utilities (ping, version, echo)
├── traits/          # Trait exploration API
├── studies/         # Study analysis API
└── similarities/    # Similarity computation API
```

## Development

See @backend/README.md for:
- Local development setup
- Available commands
- Testing procedures
- Code quality tools
