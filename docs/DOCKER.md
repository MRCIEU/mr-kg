# Docker Deployment Guide

This guide covers Docker deployment options for the MR-KG application, supporting both development and production environments.

## Quick Start

### Development
```bash
# Setup and start development environment
just start

# Or manually:
just setup-dev
just dev
```

### Production
```bash
# Setup and deploy production environment
just setup-prod
just prod
```

## Architecture Overview

The application consists of three main services:

- **Frontend**: Vue.js application with nginx (production) or Vite dev server (development)
- **Backend**: FastAPI application with Python/uv
- **Legacy Webapp**: Streamlit application (for compatibility)

## Environment Configurations

### Development Environment

- **Docker Compose**: `docker-compose.yml`
- **Environment**: `.env.development`
- **Features**:
  - Hot reload for both frontend and backend
  - Debug logging enabled
  - Volume mounting for live code changes
  - Exposed ports for direct access

#### Development Ports
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Legacy Webapp: http://localhost:8501

### Production Environment

- **Docker Compose**: `docker-compose.prod.yml`
- **Environment**: `.env.production`
- **Features**:
  - Multi-stage optimized builds
  - Resource limits and health checks
  - Security hardening
  - Logging configuration

#### Production Ports
- Frontend: http://localhost (port 80)
- Backend API: http://localhost:8000
- Legacy Webapp: http://localhost:8501

## Docker Images

### Backend (`backend/`)

#### Development Image (`Dockerfile.dev`)
- Base: `python:3.12-slim`
- Features: Hot reload, debug mode, volume mounting
- Entry: `uvicorn app.main:app --reload`

#### Production Image (`Dockerfile`)
- Multi-stage build with optimized dependencies
- Security: Non-root user, minimal attack surface
- Health checks: Built-in API health endpoint
- Size: <500MB target

### Frontend (`frontend/`)

#### Development Image (`Dockerfile.dev`)
- Base: `node:18-alpine`
- Features: Vite dev server with hot reload
- Entry: `npm run dev`

#### Production Image (`Dockerfile`)
- Multi-stage build: Node.js build + nginx serve
- Features: Static asset optimization, gzip compression
- Security: Security headers, non-root user
- Size: <200MB target

## Environment Variables

### Backend Configuration
```bash
# Core settings
DEBUG=false                    # Production: false, Dev: true
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO                 # Production: INFO, Dev: DEBUG

# Database paths
DB_PROFILE=docker              # or 'local' for development
VECTOR_STORE_PATH=/app/data/db/vector_store.db
TRAIT_PROFILE_PATH=/app/data/db/trait_profile_db.db

# Security
ALLOWED_ORIGINS=http://localhost:3000,https://your-domain.com
SECRET_KEY=your-super-secret-key-change-this-in-production
```

### Frontend Configuration
```bash
# Core settings
NODE_ENV=production            # or 'development'
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_APP_TITLE=MR-KG Explorer
VITE_APP_DESCRIPTION=Mendelian Randomization Knowledge Graph
```

## Data Management

### Volume Mounting

#### Development
- Code directories mounted for hot reload
- Database files mounted read-only from host
- Logs directory for debugging

#### Production
- Database files mounted read-only from host
- Logs directory for monitoring
- No code mounting (security)

### Database Files
The application requires two DuckDB files:
- `data/db/vector_store.db` - Main vector database
- `data/db/trait_profile_db.db` - Trait similarity database

These must be present before starting containers.

## Deployment Commands

### Using Top-Level Justfile

```bash
# Quick start development
just start                    # Setup and start dev environment
just dev                      # Start development stack
just dev-down                 # Stop development stack

# Production deployment
just prod                     # Deploy production stack
just prod-down                # Stop production stack
just prod-update              # Rolling update

# Build management
just build-dev                # Build development images
just build-prod               # Build production images

# Testing
just test-backend             # Run backend tests in container
just test-prod-local          # Test production build locally

# Health monitoring
just health                   # Check service health
just status                   # Docker container status

# Maintenance
just clean                    # Clean up Docker resources
just backup                   # Create database backup
```

### Using Service-Specific Justfiles

#### Backend (`cd backend/`)
```bash
just docker-build-dev         # Build development image
just docker-build-prod        # Build production image
just docker-run-dev           # Run development container
just docker-run-prod          # Run production container
```

#### Frontend (`cd frontend/`)
```bash
just docker-build-dev         # Build development image
just docker-build-prod        # Build production image
just docker-run-dev           # Run development container
just docker-run-prod          # Run production container
just test-prod                # Test production build
```

## Health Checks

### Built-in Health Endpoints

#### Backend
- **Endpoint**: `GET /api/v1/health/`
- **Checks**: Database connectivity, API status
- **Docker**: Used for container health checks

#### Frontend
- **Endpoint**: `GET /health` (nginx only)
- **Response**: Simple "healthy" text response

#### Legacy Webapp
- **Endpoint**: `GET /_stcore/health`
- **Streamlit**: Built-in health check

### Monitoring Commands
```bash
# Check all services
just health

# View container status
just status

# View logs
just dev-logs [service]
just prod-logs [service]
```

## Security Considerations

### Production Security

#### Docker Security
- Non-root users in all containers
- Minimal base images (Alpine Linux where possible)
- No development tools in production images
- Read-only database mounts

#### Network Security
- Container network isolation
- Configurable CORS origins
- Security headers in nginx
- No exposed internal ports

#### Secrets Management
- Environment variables for configuration
- No secrets in Docker images
- Separate production environment files

### Development Security
- Debug mode enabled
- Broader CORS settings for development
- Volume mounting for code changes
- Extended logging

## Performance Optimization

### Production Optimizations

#### Backend
- Multi-stage Docker build
- uv for fast dependency resolution
- Single worker configuration (adjust as needed)
- Health check optimizations

#### Frontend
- Multi-stage build with nginx
- Static asset compression
- Cache headers for assets
- Optimized nginx configuration

#### Resource Limits
```yaml
deploy:
  resources:
    limits:
      memory: 1G      # Backend
      cpus: '0.5'
    reservations:
      memory: 512M    # Frontend
      cpus: '0.25'
```

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
just dev-logs [service]
just prod-logs [service]

# Check container status
just status

# Rebuild images
just build-dev
just build-prod
```

#### Database Connection Issues
```bash
# Verify database files exist
ls -la data/db/

# Check database permissions
# Ensure read access for Docker containers

# Test database health
curl http://localhost:8000/api/v1/health/
```

#### Frontend Not Loading
```bash
# Check if backend is accessible
curl http://localhost:8000/api/v1/health/

# Verify environment variables
just dev-logs frontend

# Test production build locally
just test-prod-local
```

### Performance Issues

#### Memory Usage
```bash
# Monitor resource usage
just usage

# Adjust resource limits in docker-compose files
# Scale workers based on available resources
```

#### Build Times
```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Use Docker layer caching
docker build --cache-from mr-kg-backend:latest
```

## Maintenance

### Regular Maintenance

#### Backup Database
```bash
just backup                   # Create timestamped backup
just list-backups             # List available backups
```

#### Update Images
```bash
just pull                     # Pull latest base images
just prod-update              # Rolling production update
```

#### Clean Up Resources
```bash
just clean                    # Clean unused resources
just clean-all                # Clean everything (with confirmation)
```

### Monitoring

#### Log Management
- JSON structured logging in production
- Log rotation configured in docker-compose
- Centralized logging via Docker driver

#### Resource Monitoring
```bash
just usage                    # Docker resource usage
just status                   # Container status
```

## Scaling Considerations

### Horizontal Scaling
- Backend: Increase worker count via environment variables
- Frontend: Use load balancer with multiple nginx instances
- Database: Read-only DuckDB files support multiple readers

### Vertical Scaling
- Adjust resource limits in docker-compose files
- Monitor memory usage and adjust accordingly
- Consider SSD for database file storage

## Integration with CI/CD

### Build Pipeline
```bash
# Build and test
just build-prod
just test-backend
just test-prod-local

# Deploy
just prod-update
```

### Environment Management
- Use different .env files for different environments
- Secure secrets management in CI/CD platform
- Automated health checks after deployment