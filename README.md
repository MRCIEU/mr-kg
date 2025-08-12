# MR KG

---

# setting up

## models

cd models

wget https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

extract

## data

wget https://github.com/EBISPOT/efo/releases/download/v3.80.0/efo.json

## processing

cd processing

micromamba env create -f environment.yml

uv sync

---

# Deployment

## Docker Setup

The project includes a Docker setup for the Streamlit webapp that allows easy deployment and development.

### Prerequisites

- Docker and Docker Compose installed
- Built databases in the `data/db/` directory

### Quick Start

```bash
# Build the webapp image
docker compose build webapp

# Run the webapp (default port 8501)
docker compose up webapp

# Run with custom port
WEBAPP_PORT=8502 docker compose up webapp

# Run in detached mode
docker compose up -d webapp
```

The webapp will be available at http://localhost:8501 (or your specified port).

### Configuration

- **Port**: Use `WEBAPP_PORT` environment variable to customize the host port (defaults to 8501)
- **Data**: The `./data` directory is mounted read-only into the container
- **Profile**: Runs with the "docker" profile for containerized database paths

For more details, see [webapp/DOCKER.md](webapp/DOCKER.md).

---

# Other information

- For details about data, refer to ./DATA.md
- For details about processing, refer to ./processing/README.md
