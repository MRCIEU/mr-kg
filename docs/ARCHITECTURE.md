# System architecture

MR-KG is a fullstack system for exploring Mendelian Randomization studies using LLM-extracted traits and vector similarity search.

For project overview and navigation, see @DEV.md.

## System overview

```mermaid
flowchart TB
    subgraph User Interfaces
        Frontend[Frontend<br/>Vue 3 + TypeScript<br/>Pinia + Router]
        Streamlit[Legacy Streamlit<br/>Read-only access]
    end

    subgraph Backend Services
        API[Backend API<br/>FastAPI + Pydantic<br/>REST endpoints]
    end

    subgraph Data Layer
        VectorDB[(vector_store.db<br/>Traits + Embeddings)]
        TraitDB[(trait_profile_db.db<br/>Similarities + Profiles)]
    end

    subgraph Processing Pipeline
        Raw[Raw Inputs<br/>LLM extractions<br/>EFO ontology]
        Preprocess[Preprocessing<br/>Normalization<br/>Deduplication]
        Embed[Embedding<br/>spaCy vectors<br/>HPC jobs]
        Build[Database Build<br/>DuckDB creation<br/>View materialization]
    end

    Frontend -->|HTTP/JSON| API
    API -->|SQL queries| VectorDB
    API -->|SQL queries| TraitDB
    Streamlit -->|Direct read| VectorDB
    Streamlit -->|Direct read| TraitDB

    Raw --> Preprocess
    Preprocess --> Embed
    Embed --> Build
    Build -->|Creates| VectorDB
    Build -->|Creates| TraitDB
```

## Backend architecture

### Overview

The backend follows a layered architecture with clear separation of concerns:

```mermaid
graph TB
    subgraph "API Layer"
        A[FastAPI Routers]
    end

    subgraph "Service Layer"
        B[Service Classes]
    end

    subgraph "Repository Layer"
        C[Repository Classes]
    end

    subgraph "Data Layer"
        D[(DuckDB Databases)]
    end

    A -->|Depends on| B
    B -->|Uses| C
    C -->|Queries| D

    style A fill:#e1f5ff
    style B fill:#fff4e6
    style C fill:#f3e5f5
    style D fill:#e8f5e9
```

### API structure

Backend API endpoints organized by domain under `backend/app/api/v1/`:

- `/health` - Health checks and monitoring
- `/system` - System information and diagnostics
- `/traits` - Trait exploration and search
- `/studies` - Study analysis and retrieval
- `/similarities` - Similarity computation and ranking

For detailed endpoint documentation, see @docs/backend/api-design.md.

### Service layer

```mermaid
graph TB
    subgraph "Service Classes"
        A[TraitService]
        B[StudyService]
        C[SimilarityService]
        D[EFOService]
        E[AnalyticsService]
        F[DatabaseService]
    end

    subgraph "Repository Classes"
        G[TraitRepository]
        H[StudyRepository]
        I[SimilarityRepository]
        J[EFORepository]
    end

    subgraph "Database Connections"
        K[(Vector Store DB)]
        L[(Trait Profile DB)]
    end

    A -->|Uses| G
    B -->|Uses| H
    C -->|Uses| I
    D -->|Uses| J
    E -->|Uses| I
    F -->|Manages| K
    F -->|Manages| L

    G -->|Queries| K
    H -->|Queries| K
    I -->|Queries| L
    J -->|Queries| K

    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#e1f5ff
    style D fill:#e1f5ff
    style E fill:#e1f5ff
    style F fill:#ffebee
    style G fill:#f3e5f5
    style H fill:#f3e5f5
    style I fill:#f3e5f5
    style J fill:#f3e5f5
    style K fill:#e8f5e9
    style L fill:#e8f5e9
```

For database layer details, see @docs/backend/database-layer.md.

## Frontend architecture

### Overview

Vue 3 single-page application with component-based architecture:

```mermaid
graph TB
    subgraph "Views"
        A[HomeView]
        B[TraitsView]
        C[StudiesView]
        D[SimilaritiesView]
        E[AboutView]
    end

    subgraph "Components"
        F[TraitCard]
        G[PaginationControls]
        H[LoadingSpinner]
        I[ErrorAlert]
        J[SearchInput]
    end

    subgraph "Services"
        K[ApiService]
    end

    subgraph "Stores"
        L[Studies Store]
        M[Traits Store]
    end

    A -->|Uses| K
    B -->|Uses| F
    B -->|Uses| G
    B -->|Uses| J
    C -->|Uses| G
    C -->|Uses| J
    D -->|Uses| G

    B -->|Uses| M
    C -->|Uses| L

    K -->|HTTP| N[Backend API]

    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#e1f5ff
    style D fill:#e1f5ff
    style E fill:#e1f5ff
    style F fill:#fff4e6
    style G fill:#fff4e6
    style H fill:#fff4e6
    style I fill:#fff4e6
    style J fill:#fff4e6
    style K fill:#f3e5f5
    style L fill:#ffebee
    style M fill:#ffebee
    style N fill:#e8f5e9
```

For detailed frontend architecture, see @docs/frontend/overview.md.

### Backend integration flow

```mermaid
sequenceDiagram
    participant U as User
    participant V as Vue Component
    participant S as Pinia Store
    participant A as ApiService
    participant B as Backend API

    U->>V: Interacts with UI
    V->>S: Dispatch action
    S->>A: Call API method
    A->>B: HTTP request
    B->>A: JSON response
    A->>A: Validate with Zod
    A->>S: Return typed data
    S->>V: Update reactive state
    V->>U: Re-render UI
```

## Data layer

File-backed DuckDB databases optimized for vector search and analytical queries:

- **Vector store database** (`vector_store.db`): Trait embeddings, EFO embeddings, model results
- **Trait profile database** (`trait_profile_db.db`): Precomputed trait similarities

Access patterns:
- Repository-style data access from backend services
- Vector similarity queries over embedding tables
- Read-optimized schemas and views for common aggregations

For complete database schema, see @docs/processing/db-schema.md.

## Processing pipeline

Converts raw inputs into vectorized databases via HPC-enabled ETL:

1. **Preprocessing**: Normalize traits and EFO terms, create indices
2. **Embedding**: Generate 200-dim vectors using SciSpaCy (HPC)
3. **Database build**: Create vector_store.db with embeddings and results
4. **Similarity**: Compute trait profile similarities (HPC)
5. **Profile database**: Create trait_profile_db.db for network analysis

For complete pipeline documentation, see @processing/README.md.

## Integration patterns

### Frontend to backend

HTTP(S) with JSON payloads via centralized API service:

```
Vue component -> Pinia action -> API service -> FastAPI endpoint
UI update    <- state update  <- response    <- DB query
```

### Backend to data layer

- Centralized database service managing connections
- Repository methods encapsulate SQL and vector operations
- Read-mostly workload with batched queries and pagination

### Processing to data layer

- Deterministic schema materialization with validation
- Versioned outputs enabling reproducibility
- Separation of write-time processing from read-time serving

## Deployment topology

```
Internet -> Reverse proxy / CDN
  - Serves SPA static assets
  - Proxies /api/v1 -> FastAPI -> DuckDB files
```

For deployment details, see @docs/deployment.md.
