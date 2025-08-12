# Docker Setup for Webapp

This Docker setup allows you to run the Streamlit webapp in a containerized environment.

## Prerequisites

- Docker and Docker Compose installed
- The `data/` directory with required databases at the project root

## Usage

### Building and Running

From the project root directory:

```bash
# Build the webapp image
docker compose build webapp

# Run the webapp service (default port 8501)
docker compose up webapp

# Run with custom port
WEBAPP_PORT=8502 docker compose up webapp

# Run in detached mode
docker compose up -d webapp
```

The webapp will be available at <http://localhost:8501> (or your specified port)

### Configuration

The webapp runs with the "docker" profile, which expects:

- Database files mounted from `./data/db/` on the host to `/app/data/db/` in the container
- The container runs on port 8501 (mapped to host port 8501 by default)

### Environment Variables

- `WEBAPP_PORT` - Host port to expose the webapp on (defaults to 8501)

Example:
```bash
# Run on port 8502 instead of 8501
WEBAPP_PORT=8502 docker compose up webapp
```

### Development

For local development, use the justfile commands:
```bash
# Local development (uses "local" profile)
just local-run

# Docker development (uses "docker" profile, but runs outside container)
just docker-run
```

### Volume Mounts

- `./data:/app/data:ro` - Read-only mount of the data directory containing the DuckDB databases
