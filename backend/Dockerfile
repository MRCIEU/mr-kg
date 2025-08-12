# Production Dockerfile for FastAPI Backend
FROM python:3.12-slim as base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY backend/pyproject.toml backend/uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Copy source code
COPY backend/app ./app

# Copy common dependencies from project root
COPY src/ ./lib/

# Set Python path to include local libraries
ENV PYTHONPATH="/app/lib:/app"

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
