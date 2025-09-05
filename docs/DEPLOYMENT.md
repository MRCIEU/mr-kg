# Production Deployment (Docker)

Scope and references

- This guide focuses on production deployment only
- Development-in-Docker is covered in @docs/DEVELOPMENT.md
- Environment variables and configuration are documented in @docs/ENV.md

## Overview

The production stack runs as containers with a read-only data layer:

- Frontend served by nginx (Vue static assets)
- Backend served by uvicorn (FastAPI)
- Legacy Streamlit webapp (optional)
- DuckDB files mounted read-only from the host

All examples assume the provided docker-compose.prod.yml at the repo root.

## Prerequisites

- Docker Engine and Docker Compose plugin installed
- A production environment file created (see @docs/ENV.md)
- Required databases present on the host under data/db/
  - data/db/vector_store.db
  - data/db/trait_profile_db.db
- Adequate CPU, RAM, storage, and network access for your workload

## Production Compose

Core usage

```bash
# Build images and start the stack
just prod

# Or with Docker Compose directly
docker-compose -f docker-compose.prod.yml up --build -d

# View container status
docker-compose -f docker-compose.prod.yml ps

# Tail production logs
just prod-logs
just prod-logs backend
```

Notes

- The backend mounts ./data and ./src read-only into the container
- Databases under data/db are read-only in the backend container
- Health checks are defined for all services
- Resource limits are set via deploy.resources in compose
- Port bindings can be customized with environment values (see @docs/ENV.md)

## Build Process

Recommended flows

- Local build and run

  ```bash
  just build-prod
  just prod
  ```

- Update existing deployment with minimal downtime

  ```bash
  just prod-update
  # Equivalent to pull + up --build -d
  ```

- Rebuild a single service only

  ```bash
  docker-compose -f docker-compose.prod.yml build backend
  docker-compose -f docker-compose.prod.yml up -d backend
  ```

Image considerations

- Multi-stage Dockerfiles are used to minimize image size
- Non-root users and health checks are configured in the images
- Pin image tags in production CI/CD to avoid accidental upgrades

## Reverse Proxy and TLS

You can deploy behind a reverse proxy on the host or add a proxy
container. Both approaches are supported.

Host-managed nginx example

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        proxy_pass http://127.0.0.1:8080; # frontend container exposed on host 8080
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Legacy webapp
    location /webapp/ {
        proxy_pass http://127.0.0.1:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

TLS with nginx

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/certs/your.crt;
    ssl_certificate_key /etc/ssl/private/your.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;

    add_header Strict-Transport-Security "max-age=63072000" always;

    # proxy_pass locations as above
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

Compose-managed proxy (optional)

- A commented proxy service exists in docker-compose.prod.yml
- Provide an nginx.conf and TLS assets via bind mounts
- Use this if you prefer to keep everything inside Compose

## Monitoring and Logs

Health and status

```bash
# Built-in checks
curl http://localhost:8000/api/v1/health/
curl http://localhost/health
curl http://localhost:8501/_stcore/health

# Compose status and logs
docker-compose -f docker-compose.prod.yml ps
just prod-logs
just prod-logs backend -f
```

Logging

- Containers use the json-file logging driver with rotation
- Adjust max-size and max-file in docker-compose.prod.yml if needed
- Consider system-wide logrotate for Docker engine logs

Observability options

- cAdvisor for container metrics
- Prometheus and Grafana for time-series monitoring
- Loki or a centralized log solution for log aggregation

## Backup and Restore

Targets

- Database files under data/db (read-only at runtime)
- Configuration files, Compose files, and proxy config
- TLS certificates if managed on the host

Commands

```bash
# Quick database backup (timestamped tarball)
just backup

# List available backups
just list-backups
```

Manual backup

```bash
# Databases
mkdir -p backups
cp -a data/db backups/db-$(date +%Y%m%d-%H%M%S)

# Compose and config
tar -czf backups/mr-kg-config-$(date +%Y%m%d).tar.gz \
  docker-compose.prod.yml \
  .env.production \
  nginx.conf 2>/dev/null || true
```

Restore guidance

- Stop the stack before replacing database files
- Restore the desired backup into data/db
- Start the stack and verify health endpoints

## Performance Tuning

Backend

- Scale out with multiple backend containers or increase workers
- To change workers, override the command in Compose

  ```yaml
  services:
    backend:
      command: [
        "python", "-m", "uvicorn", "app.main:app",
        "--host", "0.0.0.0", "--port", "8000",
        "--workers", "2"
      ]
  ```

- Configure timeouts, request sizes, and CORS via environment values
  documented in @docs/ENV.md
- Ensure data and log volumes use fast storage

Frontend

- Static assets are served by nginx with gzip and long-lived cache
- Place the proxy and frontend on the same host to minimize latency

Databases

- DuckDB reads benefit from more RAM and CPU
- Keep databases on local SSDs for best throughput

Container resources

- Update deploy.resources limits and reservations in Compose to match
  your workload and host capacity

## Scaling

Horizontal scaling

- Run multiple backend containers

  ```bash
  docker-compose -f docker-compose.prod.yml up -d --scale backend=3
  ```

- Place a reverse proxy in front of scaled backends
- Prefer a reverse proxy that supports service discovery
  (Traefik or nginx with Docker DNS resolver)

nginx upstream example

```nginx
# Use Docker DNS resolver inside the proxy container
resolver 127.0.0.11 valid=30s;

upstream mrkg_backend {
    server backend:8000 resolve;
}

server {
    listen 80;
    location /api/ {
        proxy_pass http://mrkg_backend;
    }
}
```

Vertical scaling

- Increase container CPU and memory limits in Compose
- Use a bigger VM with SSD storage and more RAM

## Security

Container hardening

- Non-root users inside images
- Read-only bind mounts for data
- Minimal base images with regular updates

Network and transport

- Enforce TLS for all public traffic
- Restrict CORS origins to trusted domains
- Place services on a private Docker network

Secrets and config

- Do not bake secrets into images
- Provide secrets via environment or Docker secrets
- See @docs/ENV.md for configuration guidance

Headers and policies

- Add standard security headers in nginx
- Consider a Content Security Policy for the frontend

## Troubleshooting

Startup issues

```bash
# View logs and inspect containers
just prod-logs backend
just prod-logs frontend

docker ps

docker inspect mr-kg-backend-1 | head -n 50
```

Health and connectivity

```bash
curl -i http://localhost:8000/api/v1/health/
curl -I http://localhost/health
```

Data access

- Verify data/db files exist and are readable by the Docker engine
- Ensure mount paths match those in docker-compose.prod.yml

Port conflicts

- Adjust host ports via environment values (see @docs/ENV.md)
- Confirm nothing else is bound to the same ports on the host

## Maintenance

Regular tasks

- Update images and restart with minimal downtime using just prod-update
- Prune unused images and volumes periodically with just clean
- Rotate logs at the engine level if needed
- Renew TLS certificates if managed on the host

Zero-downtime tips

- Use a reverse proxy and scale backends to more than one replica
- Perform rolling updates by updating one replica at a time

## CI/CD Integration

Recommended approach

- Build and push versioned images from CI to a registry
- Deploy by pulling new images on the server and running just prod-update

GitHub Actions example

```yaml
name: build-and-deploy

on:
  push:
    branches: [ main ]
    tags: [ "v*" ]

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push backend
        uses: docker/build-push-action@v6
        with:
          context: ./backend
          file: ./backend/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/mr-kg-backend:latest
            ghcr.io/${{ github.repository_owner }}/mr-kg-backend:${{ github.sha }}

      - name: Build and push frontend
        uses: docker/build-push-action@v6
        with:
          context: ./frontend
          file: ./frontend/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/mr-kg-frontend:latest
            ghcr.io/${{ github.repository_owner }}/mr-kg-frontend:${{ github.sha }}

      - name: Build and push webapp
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./webapp/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/mr-kg-webapp:latest
            ghcr.io/${{ github.repository_owner }}/mr-kg-webapp:${{ github.sha }}

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Remote deploy
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.DEPLOY_HOST }}
          username: ${{ secrets.DEPLOY_USER }}
          key: ${{ secrets.DEPLOY_SSH_KEY }}
          script: |
            cd /opt/mr-kg
            docker-compose -f docker-compose.prod.yml pull
            docker-compose -f docker-compose.prod.yml up -d
```

Tagging strategy

- Use immutable image tags per commit (SHA) and a moving tag like latest
- Prefer rolling updates with pull + up -d on the server

That is all you need to operate MR-KG in production with Docker. For
any configuration values not covered here, see @docs/ENV.md.
