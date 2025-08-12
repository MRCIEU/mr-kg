# MR KG

A full-stack application for exploring Mendelian Randomization knowledge graphs, consisting of a FastAPI backend, Vue.js frontend, and legacy Streamlit webapp.

---

## Architecture

### Components

- **Backend** (`./backend/`): FastAPI REST API with DuckDB integration
- **Frontend** (`./frontend/`): Vue 3 + TypeScript single-page application  
- **Webapp** (`./webapp/`): Legacy Streamlit application
- **Processing** (`./processing/`): Data processing pipeline

---

## Setting Up

### Prerequisites

- Docker and Docker Compose
- Python 3.12+ with uv package manager
- Node.js 18+ with npm
- Built databases in the `data/db/` directory

### Models

```bash
cd models
wget https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
# Extract the model
```

### Data

```bash
wget https://github.com/EBISPOT/efo/releases/download/v3.80.0/efo.json
```

### Processing

```bash
cd processing
micromamba env create -f environment.yml
uv sync
```

---

## Development

### Backend Development

```bash
cd backend
uv sync                    # Install dependencies
uv run uvicorn app.main:app --reload  # Start development server
```

The API will be available at <http://localhost:8000> with automatic reloading.

### Frontend Development

```bash
cd frontend
npm install                # Install dependencies
npm run dev               # Start development server
```

The frontend will be available at <http://localhost:3000> with hot reloading.

### Development with Docker

Run all services in development mode with hot reloading:

```bash
# Start all services
docker compose -f docker-compose.dev.yml up

# Start specific services
docker compose -f docker-compose.dev.yml up backend frontend

# Build and start
docker compose -f docker-compose.dev.yml up --build
```

Services will be available at:

- Frontend: <http://localhost:3000>
- Backend API: <http://localhost:8000>
- Streamlit Webapp: <http://localhost:8501>

---

## Production Deployment

### Docker Setup

The project includes production-optimized Docker containers for all services.

#### Quick Start

```bash
# Build and start all services
docker compose up --build

# Start in detached mode
docker compose up -d

# Start specific services
docker compose up backend frontend
```

#### Service URLs

- **Frontend**: <http://localhost:3000> (or `FRONTEND_PORT`)
- **Backend API**: <http://localhost:8000> (or `BACKEND_PORT`)  
- **Streamlit Webapp**: <http://localhost:8501> (or `WEBAPP_PORT`)

#### Configuration

Environment variables for customization:

- `BACKEND_PORT`: Backend API port (default: 8000)
- `FRONTEND_PORT`: Frontend port (default: 3000)
- `WEBAPP_PORT`: Streamlit webapp port (default: 8501)

Example:

```bash
FRONTEND_PORT=3001 BACKEND_PORT=8001 docker compose up
```

#### Production Optimizations

- **Backend**: Multi-stage build with minimal Python image
- **Frontend**: Static build served by nginx with API proxying
- **Webapp**: Streamlit optimized for containerized deployment

---

## Additional Resources

- For details about data, refer to ./DATA.md
- For details about processing, refer to ./processing/README.md

---

# Other information

- For details about data, refer to ./DATA.md
- For details about processing, refer to ./processing/README.md
