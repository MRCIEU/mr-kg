# MR-KG System Architecture

This document provides a comprehensive overview of the MR-KG (Mendelian Randomization Knowledge Graph) system architecture, covering the fullstack implementation with FastAPI backend, Vue.js frontend, and supporting infrastructure.

## System Overview

MR-KG is a fullstack web application designed to explore Mendelian Randomization studies through large language model-extracted trait information and vector similarity search. The system transforms raw PubMed literature data into an interactive knowledge graph platform.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MR-KG System                            │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Vue.js)    │   Backend (FastAPI)   │   Data Layer   │
│  ┌─────────────────┐  │  ┌─────────────────┐  │  ┌───────────┐ │
│  │ Vue.js 3 + TS   │  │  │ FastAPI + Python │  │  │ DuckDB    │ │
│  │ Pinia Store     │◄─┼─►│ REST API        │◄─┼─►│ Vector    │ │
│  │ Tailwind CSS    │  │  │ Vector Search   │  │  │ Databases │ │
│  │ Vue Router      │  │  │ Business Logic  │  │  │           │ │
│  └─────────────────┘  │  └─────────────────┘  │  └───────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                   Processing Pipeline (ETL)                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │ Raw LLM     │ │ Trait       │ │ Vector      │ │ Database  │ │
│  │ Results     │►│ Processing  │►│ Embeddings  │►│ Building  │ │
│  │ + EFO Data  │ │ + Linking   │ │ (HPC)       │ │ (DuckDB)  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                Legacy Interface (Streamlit)                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Streamlit App (Maintained for Compatibility)              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Frontend Architecture (Vue.js)

The frontend is a modern single-page application built with Vue.js 3 and TypeScript.

#### Technology Stack
- **Framework**: Vue.js 3 with Composition API
- **Language**: TypeScript for type safety
- **State Management**: Pinia for centralized state
- **Routing**: Vue Router for navigation
- **Styling**: Tailwind CSS utility framework
- **Build Tool**: Vite for fast development and optimized builds
- **HTTP Client**: Axios for API communication

#### Component Structure
```
frontend/src/
├── components/          # Reusable Vue components
│   ├── TraitExplorer/   # Trait browsing components
│   └── StudyAnalyzer/   # Study analysis components
├── views/               # Page-level components
│   ├── HomeView.vue     # Landing page
│   ├── TraitsView.vue   # Trait exploration interface
│   ├── StudiesView.vue  # Study analysis interface
│   ├── SimilaritiesView.vue # Similarity analysis
│   └── AboutView.vue    # Project documentation
├── stores/              # Pinia state management
│   ├── app.ts           # Global application state
│   ├── traits.ts        # Trait data and search state
│   ├── studies.ts       # Study data management
│   └── similarities.ts  # Similarity analysis state
├── services/            # API service layer
│   ├── api.ts           # HTTP client configuration
│   ├── traits.ts        # Trait API methods
│   ├── studies.ts       # Study API methods
│   └── similarities.ts  # Similarity API methods
└── types/               # TypeScript type definitions
    ├── api.ts           # API response types
    ├── traits.ts        # Trait data types
    └── studies.ts       # Study data types
```

#### State Management Architecture
- **Global State**: Application configuration, user preferences
- **Feature States**: Domain-specific state (traits, studies, similarities)
- **Computed Properties**: Derived state and filtered data
- **Actions**: API calls and state mutations
- **Reactive Data Flow**: Vue's reactivity system for UI updates

### 2. Backend Architecture (FastAPI)

The backend provides a RESTful API with comprehensive data access and business logic.

#### Technology Stack
- **Framework**: FastAPI for high-performance APIs
- **Language**: Python 3.12+ with type hints
- **Data Validation**: Pydantic for request/response models
- **Database**: DuckDB for vector similarity search
- **Package Manager**: uv for fast dependency management
- **ASGI Server**: Uvicorn for production deployment

#### Application Structure
```
backend/app/
├── api/                 # API route handlers
│   └── v1/              # API version 1
│       ├── health.py    # Health check endpoints
│       ├── system.py    # System information
│       ├── traits.py    # Trait exploration API
│       ├── studies.py   # Study analysis API
│       └── similarities.py # Similarity computation API
├── core/                # Core application components
│   ├── config.py        # Application configuration
│   ├── database.py      # Database connection management
│   ├── dependencies.py  # Dependency injection
│   ├── middleware.py    # Request/response middleware
│   ├── exceptions.py    # Custom exception classes
│   └── error_handlers.py # Global error handling
├── models/              # Pydantic data models
│   ├── database.py      # Database entity models
│   ├── responses.py     # API response models
│   └── requests.py      # API request models
├── services/            # Business logic layer
│   ├── repositories.py  # Data access layer
│   ├── database_service.py # High-level database operations
│   └── similarity_service.py # Similarity computation logic
└── utils/               # Utility functions
    └── helpers.py       # Common helper functions
```

#### API Architecture
- **RESTful Design**: Resource-based URLs with HTTP methods
- **Versioning**: URL-based versioning (/api/v1/)
- **Documentation**: Automatic OpenAPI/Swagger documentation
- **Validation**: Comprehensive request/response validation
- **Error Handling**: Structured error responses with proper HTTP codes
- **Middleware Stack**: Security, logging, rate limiting, CORS

### 3. Data Layer Architecture

The data layer consists of DuckDB vector databases optimized for similarity search.

#### Database Structure
```
data/db/
├── vector_store.db      # Main vector database
│   ├── trait_embeddings # Trait vectors with spaCy embeddings
│   ├── efo_embeddings   # EFO ontology term vectors
│   ├── model_results    # Raw LLM extraction results
│   ├── model_result_traits # Study-trait relationship links
│   └── query_combinations # PMID-model metadata
└── trait_profile_db.db  # Similarity analysis database
    ├── trait_similarities # Precomputed pairwise similarities
    ├── trait_profiles     # Aggregated trait profiles
    └── similarity_views   # Optimized similarity queries
```

#### Data Access Patterns
- **Connection Pooling**: Efficient connection management
- **Repository Pattern**: Clean separation of data access logic
- **Vector Search**: Optimized similarity queries using embeddings
- **Caching**: Strategic caching for frequently accessed data
- **Transactions**: Atomic operations for data consistency

### 4. Processing Pipeline Architecture

The ETL pipeline transforms raw data into structured vector databases.

#### Pipeline Stages
```
Raw Data → Preprocessing → Embeddings → Database Building
    ↓           ↓            ↓              ↓
 EFO JSON   Trait Links   spaCy Vectors   DuckDB Files
 LLM Results  Dedup      HPC Batches     Optimized Schema
 PubMed Data  Indexing   Aggregation     Vector Indexes
```

#### Key Components
- **Trait Preprocessing**: Extract and deduplicate traits across models
- **EFO Integration**: Process ontology terms for semantic mapping
- **Embedding Generation**: Create vector representations using spaCy
- **Database Construction**: Build optimized DuckDB schemas
- **Similarity Computation**: Precompute trait-trait similarities
- **HPC Integration**: Leverage SLURM for computationally intensive tasks

## Integration Patterns

### 1. Frontend-Backend Integration

#### API Communication
- **HTTP Protocol**: RESTful API over HTTP/HTTPS
- **Data Format**: JSON for all request/response payloads
- **Authentication**: JWT tokens (planned for future implementation)
- **Error Handling**: Structured error responses with user-friendly messages
- **Loading States**: Progress indicators for long-running operations

#### Request Flow
```
Vue Component → Pinia Action → API Service → FastAPI Endpoint
     ↓              ↓            ↓              ↓
 UI Update ← State Update ← Response ← Database Query
```

### 2. Backend-Database Integration

#### Connection Management
- **Pool Strategy**: Connection pooling for performance
- **Health Checks**: Continuous database connectivity monitoring
- **Transaction Management**: Proper transaction boundaries
- **Error Recovery**: Automatic retry and failover mechanisms

#### Query Patterns
- **Vector Similarity**: Efficient cosine similarity searches
- **Filtering**: Multi-dimensional filtering with indexes
- **Pagination**: Cursor-based pagination for large result sets
- **Aggregation**: Summary statistics and analytical queries

### 3. Processing-Database Integration

#### Schema Management
- **Schema Validation**: Comprehensive validation against expected structures
- **Version Control**: Schema versioning for migration support
- **Integrity Checks**: Data consistency validation
- **Performance Optimization**: Index creation and query optimization

## Security Architecture

### 1. Frontend Security
- **Content Security Policy**: Prevent XSS attacks
- **Input Validation**: Client-side validation (with server-side verification)
- **Secure Storage**: Secure handling of authentication tokens
- **HTTPS Enforcement**: Secure communication channels

### 2. Backend Security
- **CORS Configuration**: Controlled cross-origin resource sharing
- **Rate Limiting**: Protection against DoS attacks
- **Input Sanitization**: Comprehensive input validation and sanitization
- **Security Headers**: Standard security headers (HSTS, X-Frame-Options, etc.)
- **Authentication**: JWT-based authentication (planned)
- **Authorization**: Role-based access control (planned)

### 3. Data Security
- **Database Isolation**: Read-only database access for web interfaces
- **Connection Security**: Encrypted database connections
- **Access Control**: Restricted file system permissions
- **Audit Logging**: Comprehensive access logging

## Performance Architecture

### 1. Frontend Performance
- **Code Splitting**: Lazy loading of application modules
- **Asset Optimization**: Minification and compression
- **Caching Strategy**: Browser caching for static assets
- **Progressive Loading**: Incremental data loading
- **Virtual Scrolling**: Efficient handling of large lists

### 2. Backend Performance
- **Async Operations**: Non-blocking I/O with asyncio
- **Connection Pooling**: Efficient database connection reuse
- **Caching Layer**: In-memory caching for frequently accessed data
- **Query Optimization**: Optimized database queries with proper indexing
- **Response Compression**: Gzip compression for API responses

### 3. Database Performance
- **Vector Indexes**: Optimized indexes for similarity search
- **Query Planning**: Efficient query execution plans
- **Memory Management**: Optimal memory allocation for large datasets
- **Concurrent Access**: Thread-safe operations for multiple readers

## Deployment Architecture

### 1. Development Environment
- **Docker Compose**: Multi-container development setup
- **Hot Reload**: Live code reloading for rapid development
- **Volume Mounting**: Live code updates without container rebuilds
- **Service Isolation**: Separate containers for each service
- **Shared Databases**: Common database access across services

### 2. Production Environment
- **Container Orchestration**: Docker containers with health checks
- **Load Balancing**: Multiple backend instances (planned)
- **Reverse Proxy**: Nginx for static asset serving and proxying
- **Resource Limits**: Memory and CPU constraints for containers
- **Health Monitoring**: Continuous health checks and alerting

### 3. Infrastructure Components
```
Internet → Load Balancer → Frontend (nginx) → Backend (FastAPI) → Database (DuckDB)
                    ↓
              Static Assets
```

## Monitoring and Observability

### 1. Application Monitoring
- **Health Checks**: Multi-level health endpoints
- **Performance Metrics**: Request timing and throughput
- **Error Tracking**: Structured error logging and reporting
- **User Analytics**: Usage patterns and feature adoption

### 2. Infrastructure Monitoring
- **Container Health**: Docker container status monitoring
- **Resource Usage**: CPU, memory, and disk utilization
- **Network Performance**: Request latency and bandwidth
- **Database Performance**: Query performance and connection status

### 3. Logging Strategy
- **Structured Logging**: JSON-formatted logs for analysis
- **Log Levels**: Configurable verbosity (DEBUG, INFO, WARN, ERROR)
- **Request Correlation**: Unique request IDs for tracing
- **Centralized Logging**: Log aggregation (planned)

## Scalability Considerations

### 1. Horizontal Scaling
- **Stateless Design**: No server-side session state
- **Load Distribution**: Multiple backend instances
- **Database Scaling**: Read replicas for query distribution
- **Microservice Architecture**: Service decomposition (future)

### 2. Vertical Scaling
- **Resource Optimization**: Efficient memory and CPU usage
- **Connection Pooling**: Optimized database connections
- **Caching Strategies**: Multi-level caching implementation
- **Query Optimization**: Database query performance tuning

### 3. Data Scaling
- **Pagination**: Efficient handling of large result sets
- **Index Optimization**: Strategic indexing for query performance
- **Data Partitioning**: Logical data separation (future)
- **Archive Strategy**: Historical data management (future)

## Migration Strategy

### 1. Legacy Integration
- **Gradual Migration**: Phased transition from Streamlit to fullstack
- **Compatibility Layer**: Maintain existing functionality during transition
- **Data Consistency**: Shared database access between old and new systems
- **Feature Parity**: Ensure all existing features are available

### 2. Development Workflow
- **Parallel Development**: New features in fullstack, maintenance in legacy
- **Testing Strategy**: Comprehensive testing for both systems
- **Deployment Coordination**: Coordinated releases for compatibility
- **Documentation Sync**: Keep documentation current for both systems

## Future Enhancements

### 1. Planned Features
- **Authentication System**: User accounts and personalization
- **Advanced Analytics**: Machine learning-powered insights
- **Data Export**: Comprehensive data export capabilities
- **API Extensions**: Additional domain-specific endpoints
- **Real-time Updates**: WebSocket connections for live data

### 2. Technical Improvements
- **Microservice Architecture**: Service decomposition for scalability
- **Event-Driven Architecture**: Asynchronous processing with message queues
- **Advanced Caching**: Distributed caching with Redis
- **Monitoring Integration**: Prometheus/Grafana monitoring stack
- **CI/CD Pipeline**: Automated testing and deployment

### 3. User Experience
- **Progressive Web App**: Offline capabilities and mobile optimization
- **Advanced Visualization**: Interactive charts and network diagrams
- **Collaboration Features**: Shared workspaces and annotations
- **Accessibility**: WCAG compliance for inclusive design
- **Internationalization**: Multi-language support

This architecture provides a solid foundation for the MR-KG system while maintaining flexibility for future enhancements and scaling requirements.