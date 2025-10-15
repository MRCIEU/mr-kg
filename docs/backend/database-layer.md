# Database layer

Database integration architecture for the FastAPI backend.

## Architecture

The database layer implements modern patterns for maintainable data access:

```
app/
├── core/
│   ├── database.py              # Connection management and pooling
│   ├── schema_validation.py     # Schema validation
│   └── dependencies.py          # Dependency injection
├── models/
│   └── database.py              # Pydantic models
└── services/
    ├── repositories.py          # Repository pattern
    └── database_service.py      # Business logic services
```

## Connection management

### DatabaseConnectionPool

Manages connections to both DuckDB databases:

- Vector Store Database: Trait embeddings, EFO terms, model results
- Trait Profile Database: Similarity analysis and trait profiles

Key features:
- Configurable pool size (default: 10 connections per database)
- Automatic connection health checks
- Connection retry and recovery
- Context manager support for cleanup
- Thread-safe operations with asyncio locks

Usage:
```python
async with pool.get_vector_store_connection() as conn:
    result = conn.execute("SELECT * FROM trait_embeddings").fetchall()
```

### Configuration

Supports both local development and Docker deployment:

```python
# Local development
config = create_database_config(profile="local")

# Docker deployment
config = create_database_config(profile="docker")
```

Configuration includes:
- Database file paths (profile-specific)
- Connection pool settings
- Timeout configurations
- Retry mechanisms

## Schema integration

### Schema validation

The SchemaValidator validates database structure:

- Imports schema definitions from src/common_funcs/
- Validates table existence and structure
- Checks view and index availability
- Reports row counts and metadata

### Health monitoring

Comprehensive health checks include:
- Database accessibility verification
- Table structure validation
- Performance metric collection
- Connection pool status monitoring

## Repository pattern

### Base repository

BaseRepository provides common database operations:
- Query execution with parameter binding
- Error handling and logging
- Result formatting and type conversion
- Connection management

### Specialized repositories

#### TraitRepository
- Trait search and retrieval
- Vector similarity operations
- Trait statistics and metadata
- Pagination support

#### StudyRepository
- Study search and filtering
- PubMed metadata integration
- Study-trait relationship queries
- Cross-reference resolution

#### EFORepository
- EFO term search and retrieval
- Trait-to-EFO mapping operations
- Ontology navigation
- Term validation

#### SimilarityRepository
- Trait profile similarity analysis
- Query combination management
- Similarity ranking and filtering
- Performance-optimized computation

## Service layer

High-level services provide business logic:

### TraitService
- Advanced trait search with filters
- Trait detail aggregation from multiple sources
- Similar trait discovery
- Caching strategies

### StudyService
- Complex study search operations
- Study detail compilation
- Cross-reference resolution
- Study categorization and filtering

### SimilarityService
- Similarity analysis workflows
- Profile comparison operations
- Ranking and recommendation logic
- Real-time similarity computation

### AnalyticsService
- Database summary statistics
- Performance analytics
- Usage metrics collection
- Trend analysis

## Pydantic models

Type-safe API responses:

**Core models:**
- TraitEmbedding: Trait with vector representation
- EFOEmbedding: EFO term with semantic vector
- ModelResult: LLM analysis result
- QueryCombination: PMID-model combination metadata

**Response models:**
- TraitSearchResponse: Paginated trait search results
- StudyDetailResponse: Comprehensive study information
- HealthCheckResponse: System health status
- SimilarityAnalysisResponse: Similarity analysis results

**Filter models:**
- TraitSearchFilters: Trait search parameters
- StudySearchFilters: Study search parameters
- SimilaritySearchFilters: Similarity search parameters

## Dependency injection

FastAPI dependencies for clean service injection:

```python
@app.get("/traits/{trait_index}")
async def get_trait_details(
    trait_index: int,
    trait_service: TraitService = Depends(get_trait_service)
) -> TraitDetailResponse:
    return await trait_service.get_trait_details(trait_index)
```

Validation dependencies:
- validate_trait_index: Ensure valid trait indices
- validate_pmid: Validate PubMed ID format
- validate_similarity_threshold: Range validation

## Error handling

Database errors convert to appropriate HTTP exceptions:
- 503 Service Unavailable: Database connection issues
- 404 Not Found: Resource not found
- 408 Request Timeout: Database timeout
- 500 Internal Server Error: Unexpected errors

Comprehensive logging throughout:
- Connection lifecycle events
- Query execution details
- Error context information
- Performance metrics

## Performance optimization

### Strategies

1. Connection pooling: Reduces connection overhead
2. Prepared statements: Optimizes repeated queries
3. Index utilization: Leverages database indexes
4. Result caching: Caches frequently accessed data
5. Pagination: Efficient handling of large result sets

### Metrics

- Connection establishment: < 100ms
- Query response times: < 500ms for 95% of requests
- Pool overhead: < 10% of application memory
- Concurrent connections: 50+ simultaneous operations

## Deployment considerations

### Environment configuration

**Local development:**
```bash
DB_PROFILE=local
VECTOR_STORE_PATH=./data/db/vector_store.db
TRAIT_PROFILE_PATH=./data/db/trait_profile_db.db
```

**Docker production:**
```bash
DB_PROFILE=docker
# Paths automatically resolved to /app/data/db/
```

### Resource requirements

- Memory: 2GB+ for connection pools and caching
- Storage: Direct access to DuckDB files required
- CPU: Moderate usage for vector similarity operations
- Network: Local file system access (no network database)

### Monitoring

Key metrics to monitor:
- Connection pool utilization
- Query execution times
- Error rates and types
- Memory usage patterns
- Database file sizes
