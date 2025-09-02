# Database Integration Layer Implementation

This document describes the comprehensive database integration layer implemented for the FastAPI backend of the MR-KG (Mendelian Randomization Knowledge Graph) fullstack system.

## Overview

The database integration layer provides a robust, scalable, and maintainable interface between the FastAPI backend and the existing DuckDB vector stores, serving the Vue.js frontend with efficient data access. It implements modern patterns including connection pooling, repository pattern, dependency injection, and comprehensive health monitoring.

## Architecture

### Components

```text
app/
├── core/
│   ├── database.py              # Connection management and pooling
│   ├── schema_validation.py     # Schema validation and health checks
│   ├── dependencies.py          # Dependency injection and health endpoints
│   └── config.py               # Configuration settings
├── models/
│   └── database.py             # Pydantic models for API responses
├── services/
│   ├── repositories.py         # Repository pattern implementation
│   └── database_service.py     # High-level business logic services
└── tests/
    └── test_database_integration.py  # Comprehensive test suite
```

### Key Features

1. **Connection Pool Management**: Efficient DuckDB connection pooling with automatic resource cleanup
2. **Schema Integration**: Seamless integration with existing database schemas from `src/common_funcs/`
3. **Repository Pattern**: Clean separation between data access and business logic
4. **Service Layer**: High-level abstractions for complex database operations
5. **Health Monitoring**: Comprehensive health checks and performance metrics
6. **Error Handling**: Structured error handling with appropriate HTTP status codes
7. **Testing**: Extensive test suite with both unit and integration tests
8. **Frontend Integration**: Optimized for Vue.js frontend consumption

## Connection Management

### DatabaseConnectionPool

The `DatabaseConnectionPool` class manages connections to both DuckDB databases for the fullstack application:

- **Vector Store Database**: Contains trait embeddings, EFO terms, model results, and PubMed data
- **Trait Profile Database**: Contains similarity analysis and trait profiles

**Key Features:**
- Configurable pool size (default: 10 connections per database)
- Automatic connection health checks for fullstack reliability
- Connection retry and recovery mechanisms
- Context manager support for automatic cleanup
- Thread-safe operations with asyncio locks for concurrent frontend requests

**Usage:**
```python
async with pool.get_vector_store_connection() as conn:
    result = conn.execute("SELECT * FROM trait_embeddings").fetchall()
```

### Configuration

Database configuration supports both local development and Docker deployment for the fullstack system:

```python
# Local development
config = create_database_config(profile="local")

# Docker deployment  
config = create_database_config(profile="docker")
```

Configuration includes:
- Database file paths (profile-specific for different deployment environments)
- Connection pool settings optimized for API usage
- Timeout configurations suitable for web requests
- Retry mechanisms for robust frontend experience

## Schema Integration

### Schema Validation

The `SchemaValidator` class validates database structure against expected schemas for fullstack reliability:

- Imports existing schema definitions from `src/common_funcs/`
- Validates table existence and structure for API operations
- Checks view and index availability for optimal query performance
- Reports row counts and metadata for monitoring dashboards

### Health Monitoring

Comprehensive health checks for fullstack monitoring include:
- Database accessibility verification for frontend requests
- Table structure validation to ensure API consistency
- Performance metric collection for optimization
- Connection pool status monitoring for scaling decisions

## Repository Pattern

### Base Repository

The `BaseRepository` class provides common database operations optimized for API usage:
- Query execution with parameter binding for security
- Error handling and logging for debugging
- Result formatting and type conversion for frontend consumption
- Connection management optimized for web requests

### Specialized Repositories

#### TraitRepository
- Trait search and retrieval for frontend exploration
- Vector similarity operations for trait recommendations
- Trait statistics and metadata for frontend displays
- Pagination support for large result sets

#### StudyRepository  
- Study search and filtering for frontend analysis
- PubMed metadata integration for detailed views
- Study-trait relationship queries for network visualization
- Cross-reference resolution for comprehensive data

#### EFORepository
- EFO term search and retrieval for semantic mapping
- Trait-to-EFO mapping operations for categorization
- Ontology navigation for hierarchical displays
- Term validation for user input

#### SimilarityRepository
- Trait profile similarity analysis for recommendations
- Query combination management for complex searches
- Similarity ranking and filtering for frontend display
- Performance-optimized similarity computation

## Service Layer

### High-Level Services

#### TraitService
- Advanced trait search with filters for frontend interfaces
- Trait detail aggregation from multiple data sources
- Similar trait discovery for recommendation features
- Caching strategies for frequently accessed data

#### StudyService
- Complex study search operations for frontend analysis
- Study detail compilation with related information
- Cross-reference resolution for comprehensive views
- Study categorization and filtering

#### SimilarityService
- Similarity analysis workflows for frontend visualization
- Profile comparison operations for trait analysis
- Ranking and recommendation logic for user experience
- Real-time similarity computation

#### AnalyticsService
- Database summary statistics for frontend dashboards
- Performance analytics for system monitoring
- Usage metrics collection for optimization
- Trend analysis for research insights

## API Models

### Pydantic Models

Comprehensive Pydantic models for type-safe API responses optimized for frontend consumption:

**Core Models:**
- `TraitEmbedding`: Trait with vector representation for frontend display
- `EFOEmbedding`: EFO term with semantic vector for categorization
- `ModelResult`: LLM analysis result for study details
- `QueryCombination`: PMID-model combination metadata

**Response Models:**
- `TraitSearchResponse`: Paginated trait search results for frontend tables
- `StudyDetailResponse`: Comprehensive study information for detail views
- `HealthCheckResponse`: System health status for monitoring
- `SimilarityAnalysisResponse`: Similarity analysis results for visualization

**Filter Models:**
- `TraitSearchFilters`: Trait search parameters from frontend forms
- `StudySearchFilters`: Study search parameters for analysis
- `SimilaritySearchFilters`: Similarity search parameters for recommendations

## Dependency Injection

### FastAPI Dependencies

Clean dependency injection for database services optimized for API endpoints:

```python
@app.get("/traits/{trait_index}")
async def get_trait_details(
    trait_index: int,
    trait_service: TraitService = Depends(get_trait_service)
) -> TraitDetailResponse:
    return await trait_service.get_trait_details(trait_index)
```

### Validation Dependencies

Input validation and error handling for frontend requests:
- `validate_trait_index`: Ensure valid trait indices from frontend
- `validate_pmid`: Validate PubMed ID format for frontend inputs
- `validate_similarity_threshold`: Range validation for similarity scores

## Health Check Endpoints

### Available Endpoints

1. **Basic Health Check** (`/health`)
   - Simple service availability check for load balancers
   - Quick response for frontend health monitoring

2. **Database Health Check** (`/health/database`)
   - Comprehensive database validation for system monitoring
   - Schema verification for API reliability
   - Performance metrics for optimization

3. **Connectivity Check** (`/health/connectivity`)
   - Quick database connectivity test for frontend status
   - Connection pool status for scaling decisions

4. **Pool Status** (`/health/pool`)
   - Connection pool utilization for monitoring
   - Resource monitoring for performance optimization

### Health Check Response

```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T00:00:00Z",
  "vector_store": {
    "database_path": "/path/to/vector_store.db",
    "accessible": true,
    "table_count": 5,
    "view_count": 3,
    "index_count": 10
  },
  "trait_profile": {
    "database_path": "/path/to/trait_profile_db.db", 
    "accessible": true,
    "table_count": 2,
    "view_count": 3,
    "index_count": 7
  },
  "performance_metrics": {
    "vector_store_query_time": 0.05,
    "trait_profile_query_time": 0.03,
    "complex_query_time": 0.12
  }
}
```

## Error Handling

### Structured Error Handling

Database errors are converted to appropriate HTTP exceptions for frontend consumption:
- `503 Service Unavailable`: Database connection issues
- `404 Not Found`: Resource not found for frontend requests
- `408 Request Timeout`: Database timeout for user feedback
- `500 Internal Server Error`: Unexpected database errors

### Logging

Comprehensive logging throughout the stack for fullstack debugging:
- Connection lifecycle events for infrastructure monitoring
- Query execution details for performance optimization
- Error context information for debugging
- Performance metrics for system optimization

## Testing

### Test Suite Coverage

**Unit Tests:**
- Configuration validation for different deployment profiles
- Connection pool management for reliable API operations
- Repository operations for data integrity
- Model validation for frontend compatibility
- Error handling for robust user experience

**Integration Tests:**
- End-to-end database operations for API workflows
- Health check workflows for monitoring systems
- Service layer functionality for business logic
- Real database interactions for data validation

**Performance Tests:**
- Connection pool stress testing for scalability
- Query performance validation for user experience
- Memory usage monitoring for resource optimization
- Concurrent operation testing for API load

### Running Tests

```bash
# Run all database integration tests
pytest tests/test_database_integration.py -v

# Run integration tests only
pytest tests/test_database_integration.py::TestRealDatabaseIntegration -v

# Run with coverage
pytest tests/test_database_integration.py --cov=app --cov-report=html
```

## Performance Considerations

### Optimization Strategies

1. **Connection Pooling**: Reduces connection overhead for concurrent frontend requests
2. **Prepared Statements**: Optimizes repeated query execution for API endpoints
3. **Index Utilization**: Leverages existing database indexes for fast queries
4. **Result Caching**: Caches frequently accessed data for frontend performance
5. **Pagination**: Efficient handling of large result sets for frontend display

### Performance Metrics

- Connection establishment: < 100ms for responsive API
- Query response times: < 500ms for 95% of frontend requests
- Pool overhead: < 10% of application memory
- Concurrent connections: 50+ simultaneous frontend operations

## Deployment Considerations

### Environment Configuration

**Local Development:**
```bash
DB_PROFILE=local
VECTOR_STORE_PATH=./data/db/vector_store.db
TRAIT_PROFILE_PATH=./data/db/trait_profile_db.db
```

**Docker Production:**
```bash
DB_PROFILE=docker
# Paths automatically resolved to /app/data/db/
```

### Resource Requirements

- **Memory**: 2GB+ for connection pools and caching
- **Storage**: Direct access to DuckDB files required
- **CPU**: Moderate usage for vector similarity operations
- **Network**: Local file system access (no network database)

### Monitoring

Key metrics to monitor for fullstack operations:
- Connection pool utilization for scaling decisions
- Query execution times for performance optimization
- Error rates and types for reliability monitoring
- Memory usage patterns for resource planning
- Database file sizes for storage management

## Integration with Existing Systems

### Compatibility

- **Vue.js Frontend**: Optimized API responses for frontend consumption
- **Legacy Streamlit Application**: Maintains compatibility with existing webapp
- **Processing Pipeline**: No impact on ETL workflows
- **Database Schema**: Uses existing schemas without modification
- **File Structure**: Works with current data organization

### Migration Strategy

1. **Phase 1**: Deploy FastAPI backend alongside existing Streamlit app
2. **Phase 2**: Implement Vue.js frontend with API integration
3. **Phase 3**: Use FastAPI as primary interface, Streamlit for specialized views

## Frontend Integration Patterns

### API Response Optimization

Database services are optimized for frontend consumption:

```python
# Optimized for frontend pagination
async def search_traits_paginated(
    filters: TraitSearchFilters,
    page: int = 1,
    page_size: int = 20
) -> TraitSearchResponse:
    """Search traits with pagination optimized for frontend tables."""
    offset = (page - 1) * page_size
    traits = await self.trait_repository.search_traits(
        filters, 
        limit=page_size, 
        offset=offset
    )
    
    total_count = await self.trait_repository.count_traits(filters)
    
    return TraitSearchResponse(
        traits=traits,
        pagination=PaginationInfo(
            page=page,
            page_size=page_size,
            total_count=total_count,
            total_pages=math.ceil(total_count / page_size)
        )
    )
```

### Real-time Data Features

Prepared for real-time frontend features:
- Connection pooling for efficient concurrent requests
- Caching strategies for frequently accessed data
- Optimized queries for interactive exploration
- Performance metrics for user experience optimization

## Future Enhancements

### Planned Improvements

1. **Caching Layer**: Redis integration for frequently accessed data
2. **Query Optimization**: Advanced query planning and optimization
3. **Monitoring Integration**: Prometheus metrics and Grafana dashboards
4. **API Versioning**: Support for multiple API versions
5. **Authentication**: JWT-based authentication for protected endpoints

### Scalability Considerations

- **Read Replicas**: Support for read-only database replicas
- **Load Balancing**: Multiple FastAPI instances with shared connection pools
- **Horizontal Scaling**: Microservice decomposition for specialized operations
- **Async Processing**: Background job processing for expensive operations

## Troubleshooting

### Common Issues

1. **Database Files Not Found**
   - Verify `DB_PROFILE` environment variable
   - Check file paths in configuration
   - Ensure proper file permissions for Docker containers

2. **Connection Pool Exhaustion**
   - Monitor pool utilization via `/health/pool`
   - Increase `max_connections` if needed
   - Check for connection leaks in application code

3. **Query Performance Issues**
   - Use `/health/database` to check performance metrics
   - Verify database indexes are properly created
   - Review query patterns and optimization opportunities

4. **Schema Validation Failures**
   - Check database schema against expected definitions
   - Verify all required tables and views exist
   - Review migration scripts and data integrity

### Debug Mode

Enable debug logging for detailed troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This provides detailed information about:
- Connection lifecycle for infrastructure debugging
- Query execution plans for performance optimization
- Error stack traces for issue resolution
- Performance timings for optimization

### Frontend Integration Debugging

Monitor frontend-backend interactions:
- Request correlation IDs for tracing frontend requests
- API response timing for user experience optimization
- Error tracking across the fullstack
- Database query performance for frontend responsiveness

This comprehensive database integration layer ensures reliable, performant, and maintainable data access for the MR-KG fullstack application while maintaining compatibility with existing systems and providing excellent developer experience.