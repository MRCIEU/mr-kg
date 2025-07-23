# Makefile for MR-KG Docker operations

.PHONY: help build build-dev up up-dev down logs clean

# Default target
help:
	@echo "Available commands:"
	@echo "  build     - Build production images"
	@echo "  build-dev - Build development images"
	@echo "  up        - Start production services"
	@echo "  up-dev    - Start development services"
	@echo "  down      - Stop all services"
	@echo "  logs      - View logs from all services"
	@echo "  clean     - Remove all containers and images"

# Build production images
build:
	docker-compose build

# Build development images
build-dev:
	docker-compose build frontend-dev

# Start production services
up:
	docker-compose up --build -d

# Start development services
up-dev:
	docker-compose --profile dev up --build -d

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
