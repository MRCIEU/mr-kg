# MR-KG Backend

FastAPI backend for MR-KG (Mendelian Randomization Knowledge Graph), providing RESTful APIs for trait exploration, study analysis, and similarity computation.

## Features

- FastAPI framework with automatic OpenAPI documentation
- DuckDB integration for vector similarity search
- Environment-based configuration management
- Hot reload development environment
- Comprehensive test suite with pytest
- Code quality enforcement with ruff and type checking

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- [just](https://github.com/casey/just) task runner

### Quick Start

1. **Install dependencies**:
   ```bash
   just install
   ```

2. **Set up environment**:
   ```bash
   just env-example
   just env-setup
   ```

3. **Start development server**:
   ```bash
   just dev
   ```

4. **Access the API**:
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Available Commands

```bash
# Development
just dev              # Start development server with hot reload
just install          # Install dependencies

# Code Quality
just fmt              # Format code with ruff
just lint             # Lint code with ruff
just ty               # Type check with ty
just check            # Run all quality checks

# Testing
just test             # Run tests with pytest
just test-cov         # Run tests with coverage

# Docker
just docker-build     # Build development Docker image
just docker-run       # Run development Docker container

# Environment
just env-example      # Create example environment file
just env-setup        # Set up .env from example
```

## Project Structure

```
backend/
├── app/
│   ├── api/          # API route handlers
│   ├── core/         # Core configuration and utilities
│   ├── models/       # Pydantic data models
│   ├── services/     # Business logic services
│   ├── utils/        # Utility functions
│   └── main.py       # FastAPI application entry point
├── tests/            # Test suite
├── Dockerfile.dev    # Development Docker configuration
├── justfile          # Task runner configuration
└── pyproject.toml    # Python project configuration
```

## Configuration

Configuration is managed through environment variables. See `.env.example` for available options:

- `DEBUG`: Enable debug mode (default: true)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DB_PROFILE`: Database profile (local/docker)
- `VECTOR_STORE_PATH`: Path to vector store database
- `TRAIT_PROFILE_PATH`: Path to trait profile database

## Database Integration

The backend integrates with existing DuckDB vector stores:

- **Vector Store**: Contains trait embeddings and model results
- **Trait Profile**: Contains precomputed similarity matrices

Database paths are configurable via environment variables and support both local development and Docker deployment profiles.

## Docker Development

Build and run the development environment with Docker:

```bash
# Build development image
just docker-build

# Run with hot reload and volume mounting
just docker-run
```

The Docker setup includes:
- Hot reload for development
- Volume mounting for code changes
- Environment variable injection
- Non-root user execution

## Testing

Run the test suite with:

```bash
# Basic test run
just test

# With coverage reporting
just test-cov
```

Tests include:
- Health check endpoints
- API route validation
- Database connection testing
- Error handling verification

## Code Quality

Code quality is enforced through:

- **ruff**: Code formatting and linting
- **ty**: Type checking
- **pytest**: Testing framework
- **pre-commit**: Git hook validation

Run all quality checks:

```bash
just check
```

## API Documentation

When running in development mode, interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

The API follows RESTful conventions and includes:
- Request/response validation
- Error handling with appropriate HTTP status codes
- OpenAPI schema generation
- Type-safe data models

## Next Steps

This foundation setup provides the scaffolding for:

1. Database integration layer (Task 1-2)
2. Core API infrastructure (Task 2-1)
3. Domain-specific API endpoints (Tasks 2-2, 2-3, 2-4)
4. Integration with frontend application

See the project master plan for detailed implementation phases.