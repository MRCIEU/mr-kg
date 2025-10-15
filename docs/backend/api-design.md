# API design patterns

Core infrastructure patterns for the FastAPI backend.

## Response models

Standardized response formats ensure consistency across all endpoints.

### Base response structure

```python
class BaseResponse(BaseModel):
    success: bool = True
    timestamp: datetime
    request_id: Optional[str] = None

class DataResponse(BaseResponse, Generic[T]):
    data: T

class ErrorResponse(BaseResponse):
    success: bool = False
    error: ErrorDetail
```

Key features:
- Generic type support for type-safe responses
- Request correlation with request IDs
- Pagination support for list responses
- Consistent error formatting

## Exception handling

Custom exception classes map to appropriate HTTP status codes:

- `ValidationError`: Input validation failures (422)
- `DatabaseError`: Database operation failures (503)
- `NotFoundError`: Resource not found (404)
- `BusinessLogicError`: Business rule violations (400)
- `RateLimitError`: Rate limiting violations (429)

Exception handlers provide:
- Structured error responses
- Context preservation for debugging
- Security-aware messages (no sensitive data leakage)
- Request correlation for tracking

## Middleware stack

Production-ready middleware ordered for optimal performance:

1. CORS (outermost) - Handle preflight requests first
2. Request Logging - Track all requests with correlation IDs
3. Security Headers - Apply security early
4. Rate Limiting - Protect resources
5. Metrics Collection (innermost) - Capture performance data

### Request logging

- Request/response logging with correlation IDs
- Performance timing
- Configurable verbosity levels

### Security headers

- Content Security Policy (CSP)
- X-Frame-Options, X-Content-Type-Options
- Referrer-Policy
- Optional HSTS

### Rate limiting

- Per-IP rate limiting
- Configurable limits (requests per minute/hour)
- Graceful degradation with Retry-After headers

### Metrics collection

- Request counting and timing
- Error rate tracking
- Memory-efficient storage

## Router architecture

Organized endpoint structure with versioning:

```
/api/v1/
├── health/          # Health checks for monitoring
├── system/          # System information and capabilities
├── core/            # Core utilities (ping, version, echo)
├── traits/          # Trait exploration endpoints
├── studies/         # Study analysis endpoints
└── similarities/    # Similarity computation endpoints
```

Each router inherits:
- Standardized response formats
- Comprehensive error handling
- Security protections
- Monitoring capabilities

## Configuration management

Environment-based configuration using Pydantic Settings:

```python
class Settings(BaseSettings):
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str]
    DB_PROFILE: str = "local"
```

Configuration supports:
- Environment-specific settings
- Type validation
- Default values
- Secret management

## Security features

### CORS configuration

- Frontend-specific allowed origins
- Credential support for authentication
- Method and header restrictions

### Input validation

- Pydantic model validation for all endpoints
- Request size limits
- Type checking for parameters
- Sanitization of user inputs

## Monitoring and observability

### Health checks

- Basic health: Simple liveness check
- Detailed health: Comprehensive system status
- Database health: Connection and schema validation
- Pool status: Connection pool utilization

### Metrics

- Request counting and timing
- Error rate tracking
- Database query metrics
- Response time statistics

### Logging

- Structured logging with JSON format
- Request correlation IDs
- Performance timing
- Error context preservation

## Performance considerations

### Caching strategy

- In-memory metrics storage
- Efficient rate limiting counters
- Connection pooling for databases
- Response caching for frequently accessed data

### Error handling

- Minimal performance impact
- Structured error responses
- Context preservation without overhead

## API documentation

- Automatic OpenAPI schema generation
- Interactive documentation at /docs
- ReDoc documentation at /redoc
- JSON schema at /openapi.json
