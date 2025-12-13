## Deployment

### Docker deployment

The production stack uses Docker Compose with the docker-compose.prod.yml file.

Prerequisites:

- Docker Engine and Docker Compose installed
- Production environment file created (.env.production)
- Required databases present under data/db/

Deploy:

```bash
# Build and start production stack
just prod

# View logs
just prod-logs

# Update deployment
just prod-update

# Stop stack
just prod-down
```

Access the services:

- Webapp: http://localhost:8501/mr-kg
- API: http://localhost:8000/mr-kg/api

### Resource limits

Production containers have resource limits configured in docker-compose.prod.yml:

API:

- Memory: 512M limit, 256M reservation
- CPU: 0.5 limit, 0.25 reservation

Webapp:

- Memory: 1G limit, 512M reservation
- CPU: 0.5 limit, 0.25 reservation

Adjust these as needed for your workload.

### Health checks

Both services include health check endpoints:

- API: http://localhost:8000/mr-kg/api/health
- Webapp: http://localhost:8501/mr-kg/_stcore/health

Check service health:

```bash
just health
```


