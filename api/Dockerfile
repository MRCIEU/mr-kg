FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy common_funcs first (dependency)
COPY src/common_funcs /app/src/common_funcs

# Copy api dependency files and install
COPY api/pyproject.toml api/uv.lock* /app/api/

WORKDIR /app/api

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy api application code
COPY api/ /app/api/

# Expose port
EXPOSE 8000

# Run uvicorn server
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
