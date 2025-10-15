# MR-KG Development Portal

## Documentation Structure

```text
README.md                      # Brief project overview + citation
├── DEV.md                     # This file: minimal index (links only)
├── docs/
│   ├── setting-up.md          # Setup instructions
│   ├── development.md         # Development workflows
│   ├── architecture.md        # System architecture
│   ├── env.md                 # Environment variables
│   ├── testing.md             # Testing strategy
│   ├── deployment.md          # Deployment guide
│   ├── data.md                # Data structure
│   ├── backend/
│   │   ├── overview.md        # Backend overview
│   │   ├── api-design.md      # API design patterns
│   │   └── database-layer.md  # Database layer
│   ├── frontend/
│   │   ├── overview.md        # Frontend overview
│   │   └── testing.md         # Frontend testing
│   └── processing/
│       ├── overview.md        # Processing overview
│       ├── databases.md       # Vector stores
│       ├── db-schema.md       # Database schema
│       └── trait-similarity.md # Trait similarity
├── backend/README.md          # Backend commands
├── frontend/README.md         # Frontend commands
├── processing/README.md       # Processing pipeline
├── webapp/README.md           # Streamlit webapp
└── src/common_funcs/README.md # Common utilities
```

## Infrastructure documentation

### Entrypoints

- Setup environment: @docs/setting-up.md
- Development workflow: @docs/development.md
- Environment variables: @docs/env.md

### Architecture and Data

- System architecture: @docs/architecture.md
- Data structure: @docs/data.md

### Testing and Quality

- Testing strategy: @docs/testing.md

### Deployment

- Deployment guide: @docs/deployment.md

### Component Documentation

#### Backend

- Backend portal: @docs/backend/overview.md
- API design patterns: @docs/backend/api-design.md
- Database layer: @docs/backend/database-layer.md
- Commands & quick reference: @backend/README.md

#### Frontend

- Frontend portal: @docs/frontend/overview.md
- Frontend testing: @docs/frontend/testing.md
- Commands & quick reference: @frontend/README.md

#### Processing

- Processing portal: @docs/processing/overview.md
- Vector stores: @docs/processing/databases.md
- Database schema: @docs/processing/db-schema.md
- Trait similarity: @docs/processing/trait-similarity.md
- Commands & pipeline: @processing/README.md

#### Common funcs

- Common utilities: @src/common_funcs/README.md

#### Legacy info

- Streamlit webapp: @webapp/README.md

## Auto-generated docs

- @docs/processing/db-schema.md
