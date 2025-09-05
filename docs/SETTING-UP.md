# Setting up MR-KG

This guide walks you through one-time project setup and a simple quickstart
so you can run MR-KG locally. It focuses on initial setup only. For day-to-day
Docker development workflows, see @docs/DEVELOPMENT.md.

## Overview and scope

- Audience: new contributors bringing the project up on their machine
- Scope: clone the repo, prepare environments, ensure data prerequisites, and
  run a first local instance
- Out of scope: ongoing Docker usage, container profiles, deployment details
  (see @docs/DEVELOPMENT.md)

## Prerequisites

- Python 3.12+ and the uv package manager
- Node.js 18+ (npm or yarn available)
- just task runner
- Docker and Docker Compose

Helpful checks:

- python --version (should be 3.12+)
- uv --version
- node --version (should be 18+)
- npm --version
- just --version
- docker --version and docker compose version

## Get the code and initial setup

- Clone the repository and run the top-level setup recipe:

```bash
git clone https://github.com/MRCIEU/mr-kg
cd mr-kg
just setup-dev
```

The setup-dev recipe prepares local environment files from examples and ensures
basic tooling is ready. You can rerun it safely if needed.

## Environment setup examples

Use the provided examples to create environment files per component. For the
complete list of variables and explanations, see @docs/ENV.md.

- Backend (FastAPI): create backend/.env from backend/.env.example, or run
  backend-specific setup recipes (see @backend/README.md). Minimal example:

```env
# backend/.env
DEBUG=true
VECTOR_STORE_PATH=./data/db/vector_store.db
TRAIT_PROFILE_PATH=./data/db/trait_profile_db.db
```

- Frontend (Vue): create frontend/.env.local from frontend/.env.example, or run
  frontend-specific setup recipes (see @frontend/README.md). Minimal example:

```env
# frontend/.env.local
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_APP_TITLE=MR-KG Explorer
```

- Legacy webapp (Streamlit): see @webapp/README.md for its environment options.

Refer to @docs/ENV.md for all supported variables and guidance on profiles.

## Data prerequisites overview

MR-KG services expect local DuckDB databases. At minimum:

- data/db/vector_store.db
- data/db/trait_profile_db.db

You can produce these via the processing pipeline or use prepared datasets if
available. Start here:

- Processing pipeline quickstart: @processing/README.md
- Data structures and schema details: @docs/DATA.md

Ensure the VECTOR_STORE_PATH and TRAIT_PROFILE_PATH in backend/.env point to
your local database files.

## First run options

Local component quickstart (see component READMEs for exact commands):

- Backend API: @backend/README.md
- Frontend app: @frontend/README.md
- Legacy Streamlit app: @webapp/README.md

Docker development quickstart:

- See @docs/DEVELOPMENT.md for Docker-based dev workflows and helpful commands

## Verify everything works

After starting the components locally, verify basic health.

- Backend
  - Open http://localhost:8000/docs to view the API docs
  - Health check should return JSON with a healthy status:

```bash
curl -s http://localhost:8000/api/v1/health | jq .
```

- Frontend
  - Open http://localhost:3000 and confirm the app loads

- Legacy webapp
  - Open http://localhost:8501 and confirm pages render

If the backend shows database connectivity warnings on startup, confirm the
DuckDB files exist and the paths in backend/.env are correct.

## Next steps

- Full environment reference: @docs/ENV.md
- Docker development workflows: @docs/DEVELOPMENT.md
- Architecture overview: @docs/ARCHITECTURE.md
- Data model and schema: @docs/DATA.md
- Testing guides: @docs/TESTING.md
