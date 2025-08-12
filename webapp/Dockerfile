# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy src directory (required for local dependencies)
COPY src ./src

# Copy webapp files
COPY webapp ./webapp

# Set working directory to webapp
WORKDIR /app/webapp

# Install Python dependencies using uv
RUN uv sync --frozen

# Create data directory structure for mounting
RUN mkdir -p /app/data/db

# Expose Streamlit default port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Run the application with docker profile
CMD ["uv", "run", "streamlit", "run", "app.py", "--", "--profile", "docker"]
