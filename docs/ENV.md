# Environment Configuration (ENV)

Single source of truth for all environment variables used across this
repository. This file documents purpose, defaults, and examples for each
variable, grouped by component.

## How environment values are loaded and which wins

From highest to lowest precedence:

- Docker Compose service environment entries (services.environment) inside
  docker-compose*.yml. These are injected into containers and override the
  app-level .env files.
- Shell environment when running without Docker, or values exported before
  `docker compose up`. For Compose variable interpolation, shell variables
  override the project .env file.
- Compose project .env file at repository root (used for variable
  interpolation like ${VAR} in docker-compose*.yml). If a value is not in
  the shell, Compose reads from this .env file. Fallbacks in YAML like
  `${VAR:-default}` apply last.
- Backend only: backend/.env, read by Pydantic Settings inside the app.
  OS environment seen by the process always overrides values in this file.
- Backend only: defaults defined in code at
  backend/app/core/config.py (authoritative for backend defaults).
- Container images may set ENV defaults in Dockerfiles. These apply only if
  nothing above sets a value.

Notes:

- ALLOWED_ORIGINS is a comma-separated list without spaces when provided
  via environment.
- Paths differ between local host and containers. Use DB_PROFILE to choose
  the right defaults.

## Global (stack-wide)

Used by multiple components for logging and stack behavior.

- LOG_LEVEL
  - component: global
  - required: optional
  - default: DEBUG (dev), INFO (prod)
  - description: Log verbosity for services
  - example: LOG_LEVEL=INFO

- LOG_FORMAT
  - component: global
  - required: optional
  - default: text (dev), json (prod)
  - description: Log output format (text or json)
  - example: LOG_FORMAT=json

- PYTHON_ENV
  - component: global
  - required: optional
  - default: development (dev), production (prod)
  - description: Environment hint for Python tooling
  - example: PYTHON_ENV=development

- HOT_RELOAD
  - component: global
  - required: optional
  - default: true (dev only)
  - description: Hint to enable hot reload in dev workflows
  - example: HOT_RELOAD=true

## Backend (FastAPI)

Defaults come from backend/app/core/config.py unless overridden by env.

- DEBUG
  - component: backend
  - required: optional
  - default: true
  - description: Enable debug mode (affects dev behavior)
  - example: DEBUG=false

- HOST
  - component: backend
  - required: optional
  - default: 0.0.0.0
  - description: Bind address for the API server
  - example: HOST=127.0.0.1

- PORT
  - component: backend
  - required: optional
  - default: 8000
  - description: Listen port for the API server
  - example: PORT=8000

- ALLOWED_ORIGINS
  - component: backend
  - required: optional
  - default: http://localhost:3000,http://localhost:5173
  - description: Comma-separated CORS origins allowed by the API
  - example: ALLOWED_ORIGINS=https://your-ui.example.com

- DB_PROFILE
  - component: backend
  - required: optional
  - default: local
  - description: Database path profile. Allowed: local, docker
  - example: DB_PROFILE=docker

- VECTOR_STORE_PATH
  - component: backend
  - required: optional
  - default: ../data/db/vector_store.db (local)
  - description: Path to vector store database file
  - example: VECTOR_STORE_PATH=/app/data/db/vector_store.db

- TRAIT_PROFILE_PATH
  - component: backend
  - required: optional
  - default: ../data/db/trait_profile_db.db (local)
  - description: Path to trait profile database file
  - example: TRAIT_PROFILE_PATH=/app/data/db/trait_profile_db.db

- API_V1_PREFIX
  - component: backend
  - required: optional
  - default: /api/v1
  - description: URL prefix for version 1 API routes
  - example: API_V1_PREFIX=/api/v1

- MAX_REQUEST_SIZE
  - component: backend
  - required: optional
  - default: 10MB
  - description: Max request size hint for server or proxy
  - example: MAX_REQUEST_SIZE=10MB

- REQUEST_TIMEOUT
  - component: backend
  - required: optional
  - default: 30
  - description: Request timeout seconds hint for server or proxy
  - example: REQUEST_TIMEOUT=30

- SECRET_KEY
  - component: backend
  - required: optional
  - default: not set
  - description: Reserved for future auth/signing. Not used by code yet
  - example: SECRET_KEY=replace-with-a-strong-secret

- UVICORN_WORKERS
  - component: backend
  - required: optional
  - default: 1
  - description: Worker processes for Uvicorn in production images
  - example: UVICORN_WORKERS=2

- MAX_CONNECTIONS
  - component: backend
  - required: optional
  - default: 100
  - description: Max concurrent connections hint for server tuning
  - example: MAX_CONNECTIONS=200

- KEEPALIVE_TIMEOUT
  - component: backend
  - required: optional
  - default: 5
  - description: Keep-alive timeout seconds hint for server tuning
  - example: KEEPALIVE_TIMEOUT=10

- PYTHONPATH
  - component: backend
  - required: optional
  - default: /app (in containers)
  - description: Module search path for the app
  - example: PYTHONPATH=/app

Notes:

- The backend reads OS env first, then backend/.env, then uses code defaults.
- Paths shown for container examples assume DB_PROFILE=docker.

## Frontend (Vue.js, Vite)

Vite only exposes variables prefixed with VITE_ to client code.

- VITE_API_BASE_URL
  - component: frontend
  - required: required
  - default: http://localhost:8000/api/v1 (dev)
  - description: Base URL the frontend calls for API requests
  - example: VITE_API_BASE_URL=https://api.example.com/api/v1

- VITE_APP_TITLE
  - component: frontend
  - required: optional
  - default: MR-KG Explorer
  - description: Application title shown in the UI
  - example: VITE_APP_TITLE=MR-KG Explorer

- VITE_APP_DESCRIPTION
  - component: frontend
  - required: optional
  - default: Mendelian Randomization Knowledge Graph
  - description: Short application description
  - example: VITE_APP_DESCRIPTION=MR-KG Explorer for MR studies

- VITE_DEV_MODE
  - component: frontend
  - required: optional
  - default: true (dev)
  - description: Development flag used by the dev image
  - example: VITE_DEV_MODE=false

- NODE_ENV
  - component: frontend
  - required: optional
  - default: development (dev), production (prod)
  - description: Node environment hint for tooling and builds
  - example: NODE_ENV=production

- API_BASE_URL
  - component: frontend
  - required: optional
  - default: not set
  - description: Provided by Compose in production to mirror backend URL.
    Not read by the app directly; VITE_API_BASE_URL is authoritative
  - example: API_BASE_URL=http://backend:8000/api/v1

Notes:

- In production images, startup scripts substitute VITE_ values into built
  assets. Set VITE_API_BASE_URL, title, and description via Compose.

## Processing (ETL and HPC)

- ACCOUNT_CODE
  - component: processing
  - required: required for HPC submissions
  - default: not set
  - description: SLURM account code used by batch jobs
  - example: ACCOUNT_CODE=your-hpc-account

Notes:

- Export in your shell or store in a local .env not committed to git.

## Webapp (legacy Streamlit)

- STREAMLIT_SERVER_PORT
  - component: webapp
  - required: optional
  - default: 8501
  - description: Streamlit server port inside the container
  - example: STREAMLIT_SERVER_PORT=8501

- STREAMLIT_SERVER_ADDRESS
  - component: webapp
  - required: optional
  - default: 0.0.0.0
  - description: Bind address for Streamlit server
  - example: STREAMLIT_SERVER_ADDRESS=0.0.0.0

- STREAMLIT_SERVER_HEADLESS
  - component: webapp
  - required: optional
  - default: true
  - description: Run Streamlit in headless mode
  - example: STREAMLIT_SERVER_HEADLESS=true

- STREAMLIT_SERVER_ENABLE_CORS
  - component: webapp
  - required: optional
  - default: false (image default)
  - description: Whether Streamlit enables CORS
  - example: STREAMLIT_SERVER_ENABLE_CORS=false

- STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION
  - component: webapp
  - required: optional
  - default: false (image default)
  - description: Whether Streamlit enables XSRF protection
  - example: STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

- STREAMLIT_BROWSER_GATHER_USAGE_STATS
  - component: webapp
  - required: optional
  - default: false
  - description: Disable Streamlit usage stats collection
  - example: STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

## Compose and deployment

Variables used by docker-compose.yml and docker-compose.prod.yml.

- BACKEND_PORT
  - component: compose
  - required: optional
  - default: 8000
  - description: Host port mapped to backend container port 8000
  - example: BACKEND_PORT=8000

- FRONTEND_PORT
  - component: compose
  - required: optional
  - default: 3000 (dev), 80 (prod)
  - description: Host port mapped to frontend container port 3000/80
  - example: FRONTEND_PORT=3000

- WEBAPP_PORT
  - component: compose
  - required: optional
  - default: 8501
  - description: Host port mapped to Streamlit container port 8501
  - example: WEBAPP_PORT=8501

- BACKEND_API_URL
  - component: compose
  - required: optional
  - default: http://backend:8000/api/v1
  - description: Internal backend URL for other services in Docker network
  - example: BACKEND_API_URL=http://backend:8000/api/v1

- ALLOWED_ORIGINS
  - component: compose
  - required: optional
  - default: http://localhost,http://localhost:3000
  - description: Passed through to backend for CORS
  - example: ALLOWED_ORIGINS=https://ui.example.com

Optional database settings (commented in production compose):

- POSTGRES_DB
  - component: compose
  - required: optional
  - default: mrkg (if enabled)
  - description: PostgreSQL database name for optional services
  - example: POSTGRES_DB=mrkg

- POSTGRES_USER
  - component: compose
  - required: optional
  - default: mrkg_user (if enabled)
  - description: PostgreSQL username for optional services
  - example: POSTGRES_USER=mrkg_user

- POSTGRES_PASSWORD
  - component: compose
  - required: required if database is enabled
  - default: not set
  - description: PostgreSQL user password
  - example: POSTGRES_PASSWORD=change-this-password

Observability (optional, commented in .env.production):

- SENTRY_DSN
  - component: global/compose
  - required: optional
  - default: not set
  - description: Error monitoring DSN if Sentry is configured
  - example: SENTRY_DSN=https://key@o0.ingest.sentry.io/123

- METRICS_ENABLED
  - component: global/compose
  - required: optional
  - default: not set
  - description: Toggle metrics collection if implemented
  - example: METRICS_ENABLED=true

- HEALTH_CHECK_INTERVAL
  - component: global/compose
  - required: optional
  - default: not set
  - description: Health check interval for monitoring stacks
  - example: HEALTH_CHECK_INTERVAL=30

Notes:

- Compose loads values for ${VAR} from the shell first, then the project
  .env file, then applies YAML defaults with :- if present.

## Recommended local files and secrets guidance

Guidelines for safe and convenient local configuration.

- Recommended files:
  - backend/.env
    - copy from backend/.env.example and adjust paths for local dev
    - used only when running the backend outside Docker
  - .env.development
    - repository root file with dev defaults for the stack
    - copy to .env for Compose dev or export variables in your shell
  - .env.production
    - repository root file with prod defaults for the stack
    - copy to .env on the deployment host and adjust values
  - frontend/.env.local
    - optional overrides for Vite variables when running `npm run dev`
  - processing/.env.local or shell exports
    - store ACCOUNT_CODE here for HPC jobs
