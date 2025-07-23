# Makefile for MR-KG Docker operations

.PHONY: help build build-dev up up-dev down logs clean backend frontend

# Default target
help:
	@echo "Available commands:"
	@echo "  build        - Build all production images"
	@echo "  build-dev    - Build all development images"
	@echo "  up           - Start all production services"
	@echo "  up-dev       - Start all development services"
	@echo "  down         - Stop all services"
	@echo "  logs         - View logs from all services"
	@echo "  clean        - Remove all containers and images"
	@echo "  backend      - Start only backend services"
	@echo "  frontend     - Start only frontend services"

# Build all production images
build:
	docker-compose build

# Build all development images
build-dev:
	docker-compose build frontend-dev backend-dev

# Start all production services
up:
	docker-compose up --build -d

# Start all development services
up-dev:
	docker-compose --profile dev up --build -d

# Start only backend services
backend:
	docker-compose up backend --build -d

# Start only frontend services
frontend:
	docker-compose up frontend --build -d

# Stop all services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Clean up everything
clean:
	docker-compose down -v --rmi all --remove-orphans
	docker system prune -f
