# MR-KG Deployment Guide

This comprehensive guide covers deployment strategies for the MR-KG fullstack application, including development environments, production deployment, and operational considerations.

## Overview

The MR-KG system consists of multiple components that can be deployed independently or as a coordinated stack:

- **Frontend**: Vue.js application (development server or static files with nginx)
- **Backend**: FastAPI application with Python runtime
- **Legacy Webapp**: Streamlit application for compatibility
- **Data Layer**: DuckDB databases (read-only files)

## Quick Start

### Development Deployment

```bash
# Setup environment and start all services
just start

# Or step by step:
just setup-dev    # Create environment files
just dev          # Start development stack
```

Access points:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000 (docs at /docs)
- Legacy Webapp: http://localhost:8501

### Production Deployment

```bash
# Setup and deploy production stack
just setup-prod
just prod
```

Access points:
- Frontend: http://localhost (port 80)
- Backend API: http://localhost:8000
- Legacy Webapp: http://localhost:8501

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 2 cores (4 recommended)
- **Memory**: 4GB RAM (8GB recommended)
- **Storage**: 10GB available space
- **Network**: Internet access for initial setup

#### Production Requirements
- **CPU**: 4+ cores for concurrent operations
- **Memory**: 8GB+ RAM for optimal performance
- **Storage**: 50GB+ for databases and logs
- **Network**: Stable connection for user access

### Required Software

#### Development Environment
```bash
# Core requirements
- Docker 20.10+
- Docker Compose 2.0+
- Git
- just (task runner)

# Optional for local development
- Python 3.12+
- Node.js 18+
- uv (Python package manager)
```

#### Production Environment
```bash
# Minimal production requirements
- Docker 20.10+
- Docker Compose 2.0+
- Git (for deployment updates)
- just (task runner)

# Recommended additions
- nginx (reverse proxy)
- systemd (service management)
- logrotate (log management)
```

### Database Prerequisites

The application requires processed DuckDB databases:

```bash
# Required database files
data/db/vector_store.db      # Main vector database
data/db/trait_profile_db.db  # Similarity analysis database

# Check database files exist
ls -la data/db/
```

If databases don't exist, run the processing pipeline:
```bash
cd processing
just pipeline-full  # See processing/README.md for details
```

## Environment Configuration

### Environment Files

The system uses environment-specific configuration files:

```bash
# Development environment
.env.development         # Development-specific settings
.env                     # Local overrides (git-ignored)

# Production environment  
.env.production          # Production-specific settings
.env.prod.local          # Production overrides (git-ignored)
```

### Creating Environment Files

```bash
# Create initial environment files
just setup-dev           # Development setup
just setup-prod          # Production setup

# Or manually copy examples
cp .env.development.example .env.development
cp .env.production.example .env.production
```

### Key Configuration Variables

#### Backend Configuration
```bash
# Server settings
DEBUG=false                    # Enable debug mode (dev: true, prod: false)
HOST=0.0.0.0                  # Server bind address
PORT=8000                     # Server port
LOG_LEVEL=INFO                # Logging level (DEBUG/INFO/WARNING/ERROR)

# Database configuration
DB_PROFILE=docker             # Database profile (local/docker)
VECTOR_STORE_PATH=/app/data/db/vector_store.db
TRAIT_PROFILE_PATH=/app/data/db/trait_profile_db.db

# Security settings
ALLOWED_ORIGINS=http://localhost:3000,https://your-domain.com
SECRET_KEY=your-super-secret-key-change-this-in-production
CORS_ALLOW_CREDENTIALS=true

# Performance settings
CONNECTION_POOL_SIZE=10       # Database connection pool size
REQUEST_TIMEOUT=30            # Request timeout in seconds
RATE_LIMIT_PER_MINUTE=100    # Rate limiting (requests per minute)
```

#### Frontend Configuration
```bash
# Application settings
NODE_ENV=production           # Environment mode
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_APP_TITLE=MR-KG Explorer
VITE_APP_DESCRIPTION=Mendelian Randomization Knowledge Graph

# Build settings (production)
VITE_BUILD_TARGET=es2020      # JavaScript target
VITE_MINIFY=true             # Enable minification
VITE_SOURCEMAP=false         # Generate source maps
```

#### Legacy Webapp Configuration
```bash
# Streamlit settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## Development Deployment

### Docker Compose Development

The development environment uses `docker-compose.yml` with:

```yaml
services:
  frontend:
    # Vue.js development server with hot reload
    ports: ["3000:3000"]
    volumes: ["./frontend:/app"]  # Live code mounting
    
  backend: 
    # FastAPI with auto-reload
    ports: ["8000:8000"]
    volumes: ["./backend:/app"]   # Live code mounting
    
  webapp:
    # Streamlit with Docker profile
    ports: ["8501:8501"]
    volumes: ["./data:/app/data:ro"]  # Read-only data access
```

### Development Workflow

```bash
# Start development environment
just dev                     # All services
just backend-dev            # Backend only
just frontend-dev           # Frontend only
just webapp-dev             # Legacy webapp only

# Development utilities
just dev-logs               # View all service logs
just dev-logs backend       # View specific service logs
just dev-down               # Stop development stack
just dev-restart            # Restart development stack

# Code quality checks
just test-backend           # Run backend tests
just check-frontend         # Frontend linting and type checking
just health                 # Check service health
```

### Hot Reload Configuration

#### Backend Hot Reload
- **uvicorn --reload**: Automatic restart on code changes
- **Volume mounting**: Live code updates without rebuilds
- **Debug mode**: Enhanced error reporting and logging

#### Frontend Hot Reload
- **Vite HMR**: Hot module replacement for instant updates
- **TypeScript compilation**: Real-time type checking
- **Style updates**: Instant CSS/Tailwind changes

### Development Debugging

```bash
# Debug individual services
docker exec -it mr-kg-backend-1 /bin/bash    # Backend shell
docker exec -it mr-kg-frontend-1 /bin/sh     # Frontend shell

# View service logs
just dev-logs backend        # Backend application logs
just dev-logs frontend       # Frontend build logs
just dev-logs webapp         # Streamlit logs

# Health checks
curl http://localhost:8000/api/v1/health/     # Backend health
curl http://localhost:3000/                   # Frontend health
curl http://localhost:8501/_stcore/health     # Webapp health
```

## Production Deployment

### Docker Compose Production

The production environment uses `docker-compose.prod.yml` with:

```yaml
services:
  frontend:
    # Multi-stage build: Node.js build + nginx serve
    # Optimized static assets, gzip compression
    
  backend:
    # Multi-stage build: Python dependencies + runtime
    # Security hardening, resource limits
    
  webapp:
    # Streamlit with production configuration
    # Health checks and restart policies
```

### Production Build Process

```bash
# Build production images
just build-prod             # Build all production images
just build-prod-frontend    # Frontend only
just build-prod-backend     # Backend only

# Deploy production stack
just prod                   # Start production stack
just prod-update            # Rolling update deployment
just prod-down              # Stop production stack
```

### Multi-Stage Docker Builds

#### Frontend Production Build
```dockerfile
# Stage 1: Node.js build environment
FROM node:18-alpine AS builder
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# Stage 2: nginx serving
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
```

#### Backend Production Build
```dockerfile
# Stage 1: Python build environment
FROM python:3.12-slim AS builder
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 2: Runtime environment
FROM python:3.12-slim
COPY --from=builder /app/.venv /app/.venv
COPY app/ /app/app/
USER nonroot
```

### Production Optimizations

#### Performance Optimizations
- **Multi-stage builds**: Minimal image sizes
- **Static asset optimization**: Gzip compression, cache headers
- **Connection pooling**: Efficient database connections
- **Resource limits**: CPU and memory constraints

#### Security Hardening
- **Non-root users**: All containers run as non-root
- **Read-only filesystems**: Where applicable
- **Security headers**: Comprehensive HTTP security headers
- **Secrets management**: Environment-based secrets

#### Health Checks
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health/"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

## Reverse Proxy Configuration

### nginx Configuration

For production deployments, use nginx as a reverse proxy:

```nginx
# /etc/nginx/sites-available/mr-kg
server {
    listen 80;
    server_name your-domain.com;
    
    # Frontend static files
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    # Legacy webapp
    location /webapp/ {
        proxy_pass http://localhost:8501/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

### SSL/TLS Configuration

```nginx
# HTTPS configuration
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    # SSL security settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;
    
    # Proxy configuration (same as HTTP)
    # ...
}

# HTTP redirect
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

## Monitoring and Health Checks

### Built-in Health Endpoints

#### Backend Health Checks
```bash
# Basic health check
curl http://localhost:8000/api/v1/health/

# Detailed health with database status
curl http://localhost:8000/api/v1/health/detailed

# Database connectivity
curl http://localhost:8000/api/v1/health/database

# Kubernetes-style probes
curl http://localhost:8000/api/v1/health/ready    # Readiness probe
curl http://localhost:8000/api/v1/health/live     # Liveness probe
```

#### Frontend Health Checks
```bash
# Frontend availability
curl http://localhost:3000/

# nginx health (production)
curl http://localhost/health
```

#### Legacy Webapp Health
```bash
# Streamlit health check
curl http://localhost:8501/_stcore/health
```

### Monitoring Commands

```bash
# Service status
just status                  # Docker container status
just health                  # All service health checks

# Resource monitoring
just usage                   # Resource usage statistics
just logs                    # View recent logs

# Performance monitoring
just metrics                 # Application metrics
just benchmark               # Performance benchmarks
```

### Log Management

#### Log Collection
```bash
# View logs
just prod-logs               # All production logs
just prod-logs backend       # Backend logs only
just dev-logs frontend       # Frontend development logs

# Follow logs
just prod-logs -f            # Follow all logs
just prod-logs backend -f    # Follow backend logs
```

#### Log Rotation
```bash
# Configure logrotate for Docker logs
/etc/logrotate.d/docker-mr-kg:
/var/lib/docker/containers/*/*-json.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
}
```

## Backup and Disaster Recovery

### Database Backup

```bash
# Create database backup
just backup                  # Timestamped backup
just backup-with-name "pre-deployment"

# List backups
just list-backups

# Restore from backup
just restore-backup "2024-01-01-backup"
```

### Configuration Backup

```bash
# Backup configuration
tar -czf mr-kg-config-$(date +%Y%m%d).tar.gz \
    .env.production \
    docker-compose.prod.yml \
    nginx.conf

# Backup application state
docker run --rm -v mr-kg_data:/data -v $(pwd):/backup \
    alpine tar czf /backup/mr-kg-data-$(date +%Y%m%d).tar.gz /data
```

### Disaster Recovery Procedure

1. **Service Recovery**:
   ```bash
   # Stop damaged services
   just prod-down
   
   # Restore from backup
   just restore-backup "latest"
   
   # Rebuild and restart
   just build-prod
   just prod
   ```

2. **Data Recovery**:
   ```bash
   # Restore database files
   cp backup/vector_store.db data/db/
   cp backup/trait_profile_db.db data/db/
   
   # Verify database integrity
   just health
   ```

## Performance Tuning

### Backend Performance

#### Connection Pool Tuning
```bash
# Environment variables
CONNECTION_POOL_SIZE=20      # Increase for high concurrency
CONNECTION_TIMEOUT=60        # Connection timeout
QUERY_TIMEOUT=30            # Query timeout
```

#### Resource Limits
```yaml
# docker-compose.prod.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Frontend Performance

#### nginx Optimization
```nginx
# Compression
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_types text/plain text/css application/json application/javascript;

# Caching
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

#### Build Optimization
```bash
# Environment variables
VITE_BUILD_CHUNK_SIZE_WARNING_LIMIT=1000  # Chunk size warning
VITE_BUILD_ROLLUP_OPTIONS='{"output":{"manualChunks":{"vendor":["vue","pinia"]}}}'
```

### Database Performance

#### Query Optimization
```bash
# Monitor query performance
just describe-db             # Database schema and indexes
just analyze-queries         # Query performance analysis
```

#### Index Optimization
```sql
-- Ensure proper indexes exist
CREATE INDEX IF NOT EXISTS idx_trait_embeddings_trait_index 
ON trait_embeddings(trait_index);

CREATE INDEX IF NOT EXISTS idx_model_results_pmid 
ON model_results(pmid);
```

## Scaling Considerations

### Horizontal Scaling

#### Load Balancing
```nginx
# nginx load balancing
upstream mr-kg-backend {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    location /api/ {
        proxy_pass http://mr-kg-backend;
    }
}
```

#### Multiple Backend Instances
```yaml
# docker-compose.scale.yml
services:
  backend:
    deploy:
      replicas: 3
    ports:
      - "8000-8002:8000"
```

### Vertical Scaling

#### Resource Allocation
```bash
# Increase container resources
docker update --memory=4g --cpus=2 mr-kg-backend-1
docker update --memory=2g --cpus=1 mr-kg-frontend-1
```

#### Database Optimization
```bash
# DuckDB memory settings
PRAGMA memory_limit='4GB';
PRAGMA threads=4;
```

## Security Considerations

### Production Security Checklist

#### Container Security
- [ ] Non-root users in all containers
- [ ] Read-only root filesystems where possible
- [ ] No secrets in Docker images
- [ ] Regular base image updates
- [ ] Security scanning of images

#### Network Security
- [ ] HTTPS/TLS encryption
- [ ] Proper CORS configuration
- [ ] Rate limiting enabled
- [ ] Firewall rules configured
- [ ] VPN access for admin functions

#### Application Security
- [ ] Environment-based configuration
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention
- [ ] XSS protection headers
- [ ] CSRF protection (future)

#### Data Security
- [ ] Database file permissions (read-only for web services)
- [ ] Encrypted backups
- [ ] Access logging
- [ ] Data retention policies

### Security Updates

```bash
# Regular security updates
just security-update         # Update base images
just vulnerability-scan      # Scan for vulnerabilities
just security-audit         # Comprehensive security audit
```

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Diagnose container issues
docker logs mr-kg-backend-1  # Check container logs
docker inspect mr-kg-backend-1  # Inspect container config
just health                   # Run health checks

# Common fixes
just clean                    # Clean Docker resources
just build-prod              # Rebuild images
just setup-prod              # Recreate environment
```

#### Database Connection Issues
```bash
# Check database files
ls -la data/db/
file data/db/vector_store.db  # Verify file type

# Check permissions
chmod 644 data/db/*.db        # Ensure read permissions

# Test database connectivity
curl http://localhost:8000/api/v1/health/database
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats                 # Real-time resource usage
just usage                   # Resource summary

# Check for memory leaks
just monitor                 # Long-term monitoring
just benchmark               # Performance benchmarks
```

#### Network Issues
```bash
# Check service connectivity
curl -I http://localhost:8000/api/v1/health/
curl -I http://localhost:3000/
curl -I http://localhost:8501/

# Check port conflicts
netstat -tulpn | grep :8000  # Check port usage
lsof -i :3000                # Check port conflicts
```

### Debug Mode

#### Enable Debug Logging
```bash
# Backend debug mode
DEBUG=true just backend-dev

# Frontend debug mode
NODE_ENV=development just frontend-dev

# View debug logs
just dev-logs -f
```

#### Debug Commands
```bash
# Interactive debugging
just debug-backend           # Backend debug shell
just debug-frontend          # Frontend debug shell
just debug-database          # Database inspection
```

## Maintenance

### Regular Maintenance Tasks

#### Daily
- [ ] Check service health status
- [ ] Monitor resource usage
- [ ] Review error logs

#### Weekly
- [ ] Update container images
- [ ] Backup databases
- [ ] Review performance metrics

#### Monthly
- [ ] Security updates
- [ ] Log rotation
- [ ] Performance optimization review

### Maintenance Commands

```bash
# System maintenance
just maintenance             # Run all maintenance tasks
just cleanup                # Clean unused resources
just update                  # Update all components

# Health monitoring
just health-report          # Generate health report
just performance-report     # Generate performance report
```

This deployment guide provides comprehensive coverage of all deployment scenarios for the MR-KG fullstack application, from development environments to production deployments with monitoring and maintenance considerations.