# MR-KG Docker Management Commands

default:
    @just --list --unsorted

# ==== Development Commands ====

# Start development environment with Docker Compose
[group('development')]
dev:
    @echo "Starting development environment..."
    docker compose --env-file .env.development up --build

# Start development environment in background
[group('development')]
dev-detached:
    @echo "Starting development environment in background..."
    docker compose --env-file .env.development up --build -d

# Start webapp development server locally without Docker
[group('development')]
webapp-dev:
    @echo "Starting webapp development server..."
    cd webapp && just dev

# Start API development server locally without Docker
[group('development')]
api-dev:
    @echo "Starting API development server..."
    cd api && just dev

# Stop development environment
[group('development')]
dev-down:
    @echo "Stopping development environment..."
    docker compose down

# View development logs (optionally for specific service)
[group('development')]
dev-logs service="":
    @if [ "{{service}}" = "" ]; then \
        docker compose logs -f; \
    else \
        docker compose logs -f {{service}}; \
    fi

# Restart development services (optionally specific service)
[group('development')]
dev-restart service="":
    @if [ "{{service}}" = "" ]; then \
        docker compose restart; \
    else \
        docker compose restart {{service}}; \
    fi

# ==== Production Commands ====

# Deploy production environment with Docker Compose
[group('production')]
prod:
    @echo "Deploying production environment..."
    docker compose -f docker-compose.prod.yml --env-file .env.production up --build -d

# Stop production environment
[group('production')]
prod-down:
    @echo "Stopping production environment..."
    docker compose -f docker-compose.prod.yml down

# View production logs (optionally for specific service)
[group('production')]
prod-logs service="":
    @if [ "{{service}}" = "" ]; then \
        docker compose -f docker-compose.prod.yml logs -f; \
    else \
        docker compose -f docker-compose.prod.yml logs -f {{service}}; \
    fi

# Update production deployment with latest images
[group('production')]
prod-update:
    @echo "Updating production deployment..."
    docker compose -f docker-compose.prod.yml --env-file .env.production pull
    docker compose -f docker-compose.prod.yml --env-file .env.production up --build -d

# Restart production services (optionally specific service)
[group('production')]
prod-restart service="":
    @if [ "{{service}}" = "" ]; then \
        docker compose -f docker-compose.prod.yml restart; \
    else \
        docker compose -f docker-compose.prod.yml restart {{service}}; \
    fi

# ==== Build Commands ====

# Build development Docker images
[group('build')]
build-dev:
    @echo "Building development images..."
    docker compose build

# Build production Docker images
[group('build')]
build-prod:
    @echo "Building production images..."
    docker compose -f docker-compose.prod.yml build

# Build specific Docker service
[group('build')]
build service:
    @echo "Building {{service}}..."
    docker compose build {{service}}

# ==== Health Checks ====

# Check health of running services
[group('health')]
health:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Checking service health..."
    echo ""
    echo "=== API Health ==="
    api_response=$(curl -sf http://localhost:8000/api/health 2>/dev/null) && {
        echo "API is healthy"
        echo "Response: $api_response"
    } || echo "API not accessible"
    echo ""
    echo "=== Webapp Health ==="
    curl -sf http://localhost:8501/_stcore/health > /dev/null 2>&1 && echo "Webapp is healthy" || echo "Webapp not accessible"

# View status of Docker containers
[group('health')]
status:
    @echo "Docker container status:"
    docker compose ps
    @echo ""
    docker compose -f docker-compose.prod.yml ps 2>/dev/null || true

# ==== Maintenance Commands ====

# Pull latest Docker images
[group('maintenance')]
pull:
    @echo "Pulling latest images..."
    docker compose pull
    docker compose -f docker-compose.prod.yml pull

# Clean up unused Docker resources
[group('maintenance')]
clean:
    @echo "Cleaning up Docker resources..."
    docker system prune -f
    docker image prune -f
    docker volume prune -f

# Remove all Docker resources including volumes (with confirmation)
[group('maintenance')]
clean-all:
    @echo "WARNING: This will remove all Docker resources including volumes!"
    @read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
    docker compose down -v
    docker compose -f docker-compose.prod.yml down -v
    docker system prune -a -f
    docker volume prune -f

# Display Docker resource usage statistics
[group('maintenance')]
usage:
    @echo "Docker resource usage:"
    docker system df

# ==== Database Commands ====

# Create timestamped database backup
[group('database')]
backup:
    @echo "Creating database backup..."
    docker run --rm -v $(pwd)/data:/data alpine tar czf /data/backup-$(date +%Y%m%d-%H%M%S).tar.gz -C /data db/

# List available database backups
[group('database')]
list-backups:
    @echo "Available backups:"
    @ls -la data/backup-*.tar.gz 2>/dev/null || echo "No backups found"

# ==== Environment Management ====

# Set up development environment files
[group('environment')]
setup-dev:
    @echo "Setting up development environment..."
    @if [ ! -f .env ]; then echo "Creating .env from .env.development..."; cp .env.development .env; fi
    @if [ ! -f api/.env ]; then \
        echo "Creating api/.env with database paths..."; \
        echo "# API Development Environment" > api/.env; \
        echo "# Database paths relative to project root (one level up from api/)" >> api/.env; \
        echo "VECTOR_STORE_PATH=../data/db/vector_store.db" >> api/.env; \
        echo "TRAIT_PROFILE_PATH=../data/db/trait_profile_db.db" >> api/.env; \
        echo "EVIDENCE_PROFILE_PATH=../data/db/evidence_profile_db.db" >> api/.env; \
        echo "DEFAULT_MODEL=gpt-5" >> api/.env; \
        echo "LOG_LEVEL=DEBUG" >> api/.env; \
    fi
    @if [ ! -f webapp/.env ]; then \
        echo "Creating webapp/.env with database paths..."; \
        echo "# Webapp Development Environment" > webapp/.env; \
        echo "# Database paths relative to project root (one level up from webapp/)" >> webapp/.env; \
        echo "VECTOR_STORE_PATH=../data/db/vector_store.db" >> webapp/.env; \
        echo "TRAIT_PROFILE_PATH=../data/db/trait_profile_db.db" >> webapp/.env; \
        echo "EVIDENCE_PROFILE_PATH=../data/db/evidence_profile_db.db" >> webapp/.env; \
        echo "DEFAULT_MODEL=gpt-5" >> webapp/.env; \
    fi
    @echo "Development environment configured!"

# Set up production environment files
[group('environment')]
setup-prod:
    @echo "Setting up production environment..."
    @if [ ! -f .env.production ]; then echo "Please create .env.production with your production settings"; exit 1; fi
    @echo "Production environment configured!"

# ==== Quick Start Commands ====

# Quick start: setup and run development environment
[group('quickstart')]
start: setup-dev dev

# Stop all Docker Compose environments
[group('quickstart')]
stop:
    docker compose down
    docker compose -f docker-compose.prod.yml down

# Reset: stop, clean, and restart development environment
[group('quickstart')]
reset: stop clean setup-dev dev

# ==== Verification Commands ====

# Verify deployment is working correctly
[group('verification')]
verify:
    @./scripts/verify-deployment.sh
