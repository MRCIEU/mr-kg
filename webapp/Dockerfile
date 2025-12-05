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

# Copy webapp dependency files and install
COPY webapp/pyproject.toml webapp/uv.lock* /app/webapp/

WORKDIR /app/webapp

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy webapp application code
COPY webapp/ /app/webapp/

# Expose port
EXPOSE 8501

# Set Streamlit environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run streamlit server
CMD ["uv", "run", "streamlit", "run", "app.py"]
