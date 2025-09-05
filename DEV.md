# MR-KG Development Portal

A minimal index to all developer documentation.

## SSOT Policy

- Each topic has one canonical document
- This page is an index, not a copy of content
- If information conflicts, the canonical doc is the source of truth
- Contribute updates directly to the canonical doc
- Use @-style links across docs to avoid duplication

## Start Here

- Setup your environment: @docs/SETTING-UP.md
- Development workflow and conventions: @docs/DEVELOPMENT.md
- Environment variables and profiles: @docs/ENV.md

## Architecture and Data

- System architecture: @docs/ARCHITECTURE.md
- Data model and schema: @docs/DATA.md

## Testing and Quality

- Testing strategy and commands: @docs/TESTING.md

## Deployment

- Deployments and operations: @docs/DEPLOYMENT.md

## Roadmap

- Project milestones and plans: @docs/ROADMAP.md

## Component Portals

- Backend API (FastAPI): @backend/README.md
- Frontend (Vue 3 + TS): @frontend/README.md
- Processing pipeline (ETL): @processing/README.md
- Legacy webapp (Streamlit): @webapp/README.md

## Documentation Structure

```text
README.md                  # Brief project overview + citation
├── DEV.md                 # This file: minimal index (links only)
├── docs/
│   ├── SETTING-UP.md      # SINGLE SOURCE: All setup instructions
│   ├── DEVELOPMENT.md     # SINGLE SOURCE: Docker dev workflows
│   ├── ARCHITECTURE.md    # SINGLE SOURCE: System architecture
│   ├── ENV.md             # SINGLE SOURCE: All environment vars
│   ├── TESTING.md         # SINGLE SOURCE: Testing strategy
│   ├── DEPLOYMENT.md      # SINGLE SOURCE: Production deployment
│   └── ROADMAP.md         # SINGLE SOURCE: Future plans
├── backend/README.md      # Backend-specific commands & patterns
├── frontend/README.md     # Frontend-specific commands & patterns
├── processing/README.md   # ENHANCED: Complete processing pipeline docs
└── webapp/README.md       # Webapp-specific usage
```

## Notes

- Prefer linking to the canonical doc for each topic
- Keep instructions in their canonical location, not here
