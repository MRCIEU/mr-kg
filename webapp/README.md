# MR-KG Web Application

Streamlit app for exploring structural PubMed literature data extracted by
large language models (LLMs). It reads prebuilt DuckDB databases and
provides interactive pages for trait browsing, study details, and similarity
exploration.

Features

- Trait Explorer: browse and filter trait labels, pick a model, list studies,
  and drill into results.
- Explore Results: view study metadata, extracted traits, model metadata and
  results JSON, plus a PubMed link and related studies.
- Trait Profile Similarities: filter by similarity threshold, view pairs,
  and see summary statistics.
- Fast, read-only DuckDB access with cached queries.

Tech

- Streamlit 1.40+
- DuckDB 1.3+
- Pandas
- common_funcs (shared utilities) for locating database files
- uv as the Python runner and just for task automation

Requirements

- Python 3.12+ with uv installed
- Databases built and available at:
  - data/db/vector_store.db
  - data/db/trait_profile_db.db
  Build these via the processing pipeline (see project docs).
- Optional: Docker and Docker Compose for container workflows

Local development

Install dependencies:

```bash
cd webapp
uv sync
```

Run the app (local profile):

```bash
# from webapp/
just local-run
```

Run with docker profile (use inside the container or when project root is
"/app"):

```bash
# from webapp/
just docker-run
```

Access URL

- http://localhost:8501

Configuration

- Server address and port are set in webapp/.streamlit/config.toml.
- Profiles:
  - local: databases under <project_root>/data/db
  - docker: databases under /app/data/db
- On startup the app logs the resolved database paths. If a path does not
  exist you will see a FileNotFoundError; ensure the DuckDB files are present
  at the expected locations.

Troubleshooting

- Port in use: adjust server.port in webapp/.streamlit/config.toml or run
  Streamlit with a different port, for example:
  `uv run streamlit run app.py -- --profile local --server.port 8502`.
- Missing databases: build the DuckDB files via the processing pipeline and
  place them under data/db/.

See also

- @docs/ENV.md for environment configuration and profiles
- @docs/DEVELOPMENT.md for project-wide development workflow
- webapp/DOCKER.md for container usage
