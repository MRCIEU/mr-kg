# FastAPI Backend for MR-KG

REST API backend for the MR-KG web application, providing endpoints to explore Mendelian Randomization literature data.

## Features

- **RESTful API** with automatic OpenAPI documentation
- **Database integration** with existing DuckDB databases
- **CORS configuration** for frontend integration
- **Profile-based configuration** (local/docker environments)
- **Type safety** with Pydantic models

## Development

### Local Setup

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn app.main:app --reload --port 8000
```

### Docker Development

```bash
# Build development image
docker build -f Dockerfile.dev -t mr-kg-backend:dev .

# Run with hot reloading
docker run -p 8000:8000 -v $(pwd)/app:/app/app mr-kg-backend:dev
```

## API Documentation

When running, visit:
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Environment Variables

- `BACKEND_PROFILE`: "local" or "docker" (default: "local")
- `BACKEND_HOST`: Host to bind to (default: "0.0.0.0")
- `BACKEND_PORT`: Port to bind to (default: 8000)
- `BACKEND_RELOAD`: Enable hot reloading (default: false)
