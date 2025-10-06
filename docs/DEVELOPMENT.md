# Docker-based development

This guide covers local development using Docker and Docker Compose.
It focuses on the development stack only and does not cover production.
For production, see @docs/DEPLOYMENT.md.

See @docs/ENV.md for environment variables and configuration. Avoid
editing container commands directly; prefer adjusting your .env files.


## Scope and overview (Docker dev only)

- Runs backend (FastAPI), frontend (Vue), and the legacy Streamlit
  webapp in Docker.
- Hot reload is enabled for code changes via the dev servers and
  docker-compose develop.watch.
- Data under ./data is mounted read-only into containers where needed.
- Uses top-level just recipes to start, stop, and inspect the stack.


## Quick start

Ensure your environment files are set up as described in @docs/ENV.md.
Then bring up the whole development stack and follow logs as needed.

```sh
# Start stack in the foreground (rebuilds if needed)
just dev

# Stop and remove containers
just dev-down

# Tail logs for all services or a specific one
just dev-logs              # all services
just dev-logs backend      # backend only
just dev-logs frontend     # frontend only
just dev-logs webapp       # webapp only
```

Access the apps after startup:

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000 (Swagger UI at /docs)
- Legacy webapp: http://localhost:8501


## Services in development

- backend on 8000
  - FastAPI with automatic reload
  - Health endpoint at /api/v1/health/
- frontend on 3000
  - Vite dev server with HMR
- webapp on 8501
  - Streamlit legacy interface

Port mappings can be customized via environment variables documented in
@docs/ENV.md.


## Hot reload and file watching

The stack uses two mechanisms to reflect changes quickly:

- Dev servers inside containers
  - Backend starts uvicorn with --reload, reloading on code changes.
  - Frontend runs Vite dev server with hot module reload.
- docker-compose develop.watch (compose v2.22+)
  - backend
    - sync: ./backend/app -> /app/app
    - rebuild on changes to ./backend/pyproject.toml
  - frontend
    - sync: ./frontend/src -> /app/src
    - rebuild on changes to ./frontend/package.json

Notes and tips:

- If changes are not detected, try restarting the affected service:
  ```sh
  docker compose restart backend
  docker compose restart frontend
  ```
- Dependency file changes (pyproject.toml, package.json) trigger an
  image rebuild. The containers will restart automatically once the
  rebuild completes.
- The frontend container keeps /app/node_modules as an internal volume,
  avoiding host OS differences. Run installs through the container.


## Logs and shell access

Follow logs with just recipes or with docker compose directly:

```sh
# All services
just dev-logs
# Specific service
just dev-logs backend
```

Open an interactive shell inside a running container:

```sh
# Backend shell
docker compose exec backend sh

# Frontend shell
docker compose exec frontend sh

# Webapp shell
docker compose exec webapp sh
```

Example one-off commands inside containers:

```sh
# Run backend tests
docker compose exec backend uv run pytest -v

# Check Node and Vite versions
docker compose exec frontend node -v
docker compose exec frontend npm -v
```


## Health checks (dev)

Quick curl checks from your host machine:

```sh
# Backend health
curl -s http://localhost:8000/api/v1/health/

# Frontend (HTTP 200 with HTML)
curl -I http://localhost:3000

# Legacy webapp health
curl -s http://localhost:8501/_stcore/health
```

Useful backend URLs:

- OpenAPI JSON: http://localhost:8000/openapi.json
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc


## Troubleshooting common dev issues

- Rebuild images when changes are not picked up
  ```sh
  docker compose build
  docker compose up --build
  # or
  just dev
  ```

- Clean up Docker artifacts if containers behave oddly
  ```sh
  # Remove stopped containers, unused networks, dangling images
  just clean

  # Remove everything including volumes (destructive)
  just clean-all
  ```

- Port conflicts on 8000, 3000, or 8501
  - Stop any processes using those ports.
  - Adjust BACKEND_PORT, FRONTEND_PORT, WEBAPP_PORT in your .env as
    described in @docs/ENV.md.
  - Restart the stack:
    ```sh
    just dev-down && just dev
    ```

- File changes not reloading
  - Confirm your Docker Compose version supports develop.watch.
  - Ensure edits occur under the watched paths listed above.
  - Restart the affected service with docker compose restart.

- Missing data files
  - The backend and webapp mount ./data read-only. Ensure required
    database files exist under ./data/db as documented elsewhere in
    this repository.


## Related docs

- Environment configuration: @docs/ENV.md
- Backend guide: @backend/README.md
- Frontend guide: @frontend/README.md
- Legacy webapp guide: @webapp/README.md
- Production deployment: @docs/DEPLOYMENT.md
