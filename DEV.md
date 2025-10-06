# MR-KG Development Portal

## Infrastructure documentation

### Entrypoints

- Setup environment: @docs/SETTING-UP.md
- Development workflow and conventions: @docs/DEVELOPMENT.md
- Environment variables and profiles: @docs/ENV.md

### Architecture and Data

- System architecture: @docs/ARCHITECTURE.md
- Data model and schema: @docs/DATA.md

### Testing and Quality

- Testing strategy and commands: @docs/TESTING.md

### Deployment

- Deployments and operations: @docs/DEPLOYMENT.md

### Component Portals

- Backend API (FastAPI): @backend/README.md
- Frontend (Vue 3 + TS): @frontend/README.md
- Processing pipeline (ETL): @processing/README.md
- common_funcs: @src/common_funcs/README.md
- Legacy webapp (Streamlit): @webapp/README.md

### Documentation Structure

```text
README.md                         # Brief project overview + citation
├── DEV.md                        # This file: minimal index (links only)
├── docs/
│   ├── SETTING-UP.md             # SINGLE SOURCE: All setup instructions
│   ├── DEVELOPMENT.md            # SINGLE SOURCE: Docker dev workflows
│   ├── ARCHITECTURE.md           # SINGLE SOURCE: System architecture
│   ├── ENV.md                    # SINGLE SOURCE: All environment vars
│   ├── TESTING.md                # SINGLE SOURCE: Testing strategy
│   └── DEPLOYMENT.md             # SINGLE SOURCE: Production deployment
├── backend/README.md             # Backend-specific commands & patterns
├── frontend/README.md            # Frontend-specific commands & patterns
├── processing/README.md          # ENHANCED: Complete processing pipeline docs
└── webapp/README.md              # Webapp-specific usage
└── src/common_funcs/README.md    # Webapp-specific usage
```

## Other docs

- Details on the vector stores @processing/docs/databases.md
- Details on the trait similarity profiles @processing/docs/trait-similarity.md
