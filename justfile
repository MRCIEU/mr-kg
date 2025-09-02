# MR-KG Docker Management Commands

# Default recipe
default:
    @just --list --unsorted

# ==== Development Commands ====

# Start development environment with hot reload
[group('development')]
dev:
    @echo "Starting development environment..."
    docker-compose up --build

# Start development environment in background
[group('development')]
dev-detached:
    @echo "Starting development environment in background..."
    docker-compose up --build -d

# Stop development environment
[group('development')]
dev-down:
    @echo "Stopping development environment..."
    docker-compose down

# View development logs
[group('development')]
dev-logs service="":
    @if [ "{{service}}" = "" ]; then \
        docker-compose logs -f; \
    else \
        docker-compose logs -f {{service}}; \
    fi

# Restart development services
[group('development')]
dev-restart service="":
    @if [ "{{service}}" = "" ]; then \
        docker-compose restart; \
    else \
        docker-compose restart {{service}}; \
    fi

# ==== Production Commands ====

# Deploy production environment
[group('production')]
prod:
    @echo "Deploying production environment..."
    docker-compose -f docker-compose.prod.yml up --build -d

# Stop production environment
[group('production')]
prod-down:
    @echo "Stopping production environment..."
    docker-compose -f docker-compose.prod.yml down

# View production logs
[group('production')]
prod-logs service="":
    @if [ "{{service}}" = "" ]; then \
        docker-compose -f docker-compose.prod.yml logs -f; \
    else \
        docker-compose -f docker-compose.prod.yml logs -f {{service}}; \
    fi

# Update production deployment (rolling update)
[group('production')]
prod-update:
    @echo "Updating production deployment..."
    docker-compose -f docker-compose.prod.yml pull
    docker-compose -f docker-compose.prod.yml up --build -d

# ==== Build Commands ====

# Build all development images
[group('build')]
build-dev:
    @echo "Building development images..."
    docker-compose build

# Build all production images
[group('build')]
build-prod:
    @echo "Building production images..."
    docker-compose -f docker-compose.prod.yml build

# Build specific service
[group('build')]
build service:
    @echo "Building {{service}}..."
    docker-compose build {{service}}

# ==== Testing Commands ====

# Run backend tests in Docker
[group('testing')]
test-backend:
    @echo "Running backend tests in Docker..."
    docker-compose exec backend uv run pytest -v

# Run backend tests with coverage
[group('testing')]
test-backend-cov:
    @echo "Running backend tests with coverage..."
    docker-compose exec backend uv run pytest --cov=app --cov-report=term-missing

# Run frontend tests in Docker
[group('testing')]
test-frontend:
    @echo "Running frontend tests in Docker..."
    docker-compose exec frontend npm test

# Test production build locally
[group('testing')]
test-prod-local:
    @echo "Testing production build locally..."
    docker-compose -f docker-compose.prod.yml up --build -d
    @echo "Production stack is running on:"
    @echo "  Frontend: http://localhost"
    @echo "  Backend API: http://localhost:8000"
    @echo "  Legacy Webapp: http://localhost:8501"
    @echo ""
    @echo "Press Ctrl+C to stop when testing is complete..."
    @read -p ""
    docker-compose -f docker-compose.prod.yml down

# ==== Health Checks ====

# Check health of all services
[group('health')]
health:
    @echo "Checking service health..."
    @echo "\n=== Backend Health ==="
    @curl -s http://localhost:8000/api/v1/health/ | python -m json.tool || echo "Backend not accessible"
    @echo "\n=== Frontend Health ==="
    @curl -s http://localhost:3000 > /dev/null && echo "Frontend is healthy" || echo "Frontend not accessible"
    @echo "\n=== Legacy Webapp Health ==="
    @curl -s http://localhost:8501/_stcore/health > /dev/null && echo "Webapp is healthy" || echo "Webapp not accessible"

# Check Docker container status
[group('health')]
status:
    @echo "Docker container status:"
    docker-compose ps

# ==== Maintenance Commands ====

# Pull latest images
[group('maintenance')]
pull:
    @echo "Pulling latest images..."
    docker-compose pull
    docker-compose -f docker-compose.prod.yml pull

# Clean up Docker resources
[group('maintenance')]
clean:
    @echo "Cleaning up Docker resources..."
    docker system prune -f
    docker image prune -f
    docker volume prune -f

# Clean up all Docker resources (including volumes)
[group('maintenance')]
clean-all:
    @echo "WARNING: This will remove all Docker resources including volumes!"
    @read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
    docker-compose down -v
    docker-compose -f docker-compose.prod.yml down -v
    docker system prune -a -f
    docker volume prune -f

# View Docker resource usage
[group('maintenance')]
usage:
    @echo "Docker resource usage:"
    docker system df

# ==== Database Commands ====

# Create database backup
[group('database')]
backup:
    @echo "Creating database backup..."
    docker run --rm -v $(pwd)/data:/data alpine tar czf /data/backup-$(date +%Y%m%d-%H%M%S).tar.gz -C /data db/

# List database backups
[group('database')]
list-backups:
    @echo "Available backups:"
    @ls -la data/backup-*.tar.gz 2>/dev/null || echo "No backups found"

# ==== Environment Management ====

# Setup development environment
[group('environment')]
setup-dev:
    @echo "Setting up development environment..."
    @if [ ! -f .env.development ]; then echo "Creating .env.development..."; cp .env.development .env.development; fi
    @if [ ! -f backend/.env ]; then echo "Creating backend/.env..."; cp .env.development backend/.env; fi
    @if [ ! -f frontend/.env ]; then echo "Creating frontend/.env..."; cp .env.development frontend/.env; fi
    @echo "Development environment configured!"

# Setup production environment
[group('environment')]
setup-prod:
    @echo "Setting up production environment..."
    @if [ ! -f .env.production ]; then echo "Please create .env.production with your production settings"; exit 1; fi
    @echo "Production environment configured!"

# ==== Quick Start Commands ====

# Complete setup and start development
[group('quickstart')]
start: setup-dev dev

# Stop everything
[group('quickstart')]
stop:
    docker-compose down
    docker-compose -f docker-compose.prod.yml down

# Reset everything (development)
[group('quickstart')]
reset: stop clean setup-dev dev