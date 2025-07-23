# MR-KG Backend

A FastAPI backend application for the MR-KG project, built with modern Python tooling using uv.

## Features

- **FastAPI** - Modern, fast web framework for building APIs
- **uv** - Ultra-fast Python package manager and project manager
- **Python 3.12** - Latest stable Python version
- **Docker** - Containerized deployment for development and production
- **Health Checks** - Built-in health monitoring endpoints

## Development Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Local Development

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run the development server:**
   ```bash
   uv run uvicorn app.main:app --reload
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Docker Development

1. **Build and run with Docker Compose:**
   ```bash
   # From project root
   docker-compose --profile dev up backend-dev --build
   ```

2. **Or use the Makefile:**
   ```bash
   make backend
   ```

## API Endpoints

- `GET /` - Root endpoint with welcome message
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## Project Structure

```
backend/
├── app/
│   └── main.py          # FastAPI application
├── Dockerfile           # Production Docker image
├── Dockerfile.dev       # Development Docker image
├── .dockerignore        # Docker ignore file
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock             # Locked dependencies
└── README.md           # This file
```

## Environment Variables

- `ENVIRONMENT` - Set to `development` or `production`

## Docker Commands

### Build Images
```bash
# Production
docker build -t mr-kg-backend .

# Development
docker build -f Dockerfile.dev -t mr-kg-backend-dev .
```

### Run Containers
```bash
# Production
docker run -p 8000:8000 mr-kg-backend

# Development with volume mounting
docker run -p 8000:8000 -v $(pwd):/app mr-kg-backend-dev
```

## Adding Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update dependencies
uv sync
```

## Health Monitoring

The backend includes a health check endpoint at `/health` that returns:
```json
{
  "status": "healthy"
}
```

This endpoint is used by Docker health checks and load balancers.

## Production Deployment

The production Docker image:
- Uses multi-stage builds for optimization
- Runs as a non-root user for security
- Includes health checks
- Optimized for minimal attack surface

Deploy using docker-compose:
```bash
docker-compose up backend --build -d
```
