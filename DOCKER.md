# MR-KG Docker Setup

This project includes Docker configuration for both frontend (Vue.js) and backend (FastAPI) services in development and production environments.

## Quick Start

### Production Build
```bash
# Build and run all services (frontend + backend)
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

The frontend will be available at <http://localhost:3000>
The backend API will be available at <http://localhost:8000>

### Development Build
```bash
# Run the development environment with hot reload
docker-compose --profile dev up --build

# Or run in detached mode
docker-compose --profile dev up --build -d
```

The development frontend will be available at <http://localhost:5173>
The development backend will be available at <http://localhost:8001>

## Services

### Frontend (Production)

- **Port**: 3000
- **Image**: Built from `frontend/Dockerfile`
- **Features**:
  - Multi-stage build with Node.js and Nginx
  - Optimized for production with gzip compression
  - Security headers configured
  - Health check endpoint at `/health`

### Frontend-Dev (Development)

- **Port**: 5173
- **Image**: Built from `frontend/Dockerfile.dev`
- **Features**:
  - Hot module replacement (HMR)
  - Volume mounting for live code changes
  - Development optimizations

### Backend (Production)

- **Port**: 8000
- **Image**: Built from `backend/Dockerfile`
- **Features**:
  - FastAPI with uv package manager
  - Non-root user for security
  - Health check endpoint at `/health`
  - Optimized for production

### Backend-Dev (Development)

- **Port**: 8001
- **Image**: Built from `backend/Dockerfile.dev`
- **Features**:
  - FastAPI with hot reload
  - Volume mounting for live code changes
  - uv for fast dependency management

## Docker Commands

### Build specific service
```bash
# Build only the frontend
docker-compose build frontend

# Build development frontend
docker-compose build frontend-dev
```

### Run specific service
```bash
# Run only frontend in production mode
docker-compose up frontend

# Run only frontend in development mode
docker-compose --profile dev up frontend-dev
```

### View logs
```bash
# View all logs
docker-compose logs

# View frontend logs
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f frontend
```

### Stop services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Health Checks

The production frontend includes a health check endpoint:
- **URL**: http://localhost:3000/health
- **Response**: "healthy" with 200 status code

## File Structure

```
.
├── docker-compose.yml          # Main compose file
└── frontend/
    ├── Dockerfile              # Production build
    ├── Dockerfile.dev          # Development build
    ├── .dockerignore          # Files to exclude from build
    └── nginx.conf             # Nginx configuration
```

## Environment Variables

### Frontend
- `NODE_ENV`: Set to `production` or `development`

## Future Services

The docker-compose.yml file includes commented sections for:
- API backend service
- PostgreSQL database
- Additional microservices

Uncomment and configure these sections as needed when adding backend services.

## Troubleshooting

### Port conflicts
If ports 3000 or 5173 are already in use, modify the port mappings in `docker-compose.yml`:
```yaml
ports:
  - "8080:80"  # Change 3000 to 8080
```

### File watching issues in development
The development Dockerfile includes polling for file changes. If hot reload isn't working:
1. Ensure the volume mounting is correct
2. Check that `usePolling: true` is set in `vite.config.ts`

### Permission issues
On Linux/macOS, you might need to adjust file permissions:
```bash
sudo chown -R $USER:$USER frontend/node_modules
```
