# API Integration Tests

This directory contains integration tests that verify the frontend works correctly with the actual backend API.

## Overview

Unlike unit tests that use MSW (Mock Service Worker) to mock API responses, these integration tests connect to a real backend instance to verify:

- API contract compliance (response schemas match frontend types)
- Correct HTTP status codes
- Pagination behavior
- Error handling
- Query parameter handling

## Prerequisites

Before running integration tests, ensure:

1. Backend is running on `http://localhost:8000` (or set `VITE_API_BASE_URL`)
2. Database is populated with test data
3. All backend services are healthy

## Running Integration Tests

```bash
# Run integration tests (requires backend running)
npm run test:integration

# Run with watch mode
npm run test:integration:watch

# Run specific test file
VITE_API_BASE_URL=http://localhost:8000 npx vitest tests/integration/api/api-integration.spec.ts
```

## Configuration

Set the backend URL via environment variable:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run test:integration
```

## Test Categories

### Health Endpoints
- `/ping` - Basic connectivity test
- `/health` - Health status check
- `/version` - Version information

### Traits API
- `GET /traits` - List traits with pagination
- `GET /traits/search` - Search traits
- `GET /traits/:id` - Get trait details
- `GET /traits/:id/similar` - Get similar traits
- `GET /traits/stats/overview` - Traits overview statistics

### Studies API
- `GET /studies` - List studies with pagination and filters
- `GET /studies/search` - Search studies
- Filter by model (GWAS, MR)

### Similarities API
- `GET /similarities` - List similarities with pagination

### Error Handling
- 404 responses for missing resources
- 422 responses for validation errors
- Invalid parameter handling

## Test Behavior

Tests are designed to:

1. **Check backend availability** - Tests skip gracefully if backend is not running
2. **Validate schemas** - Ensure response structures match TypeScript types
3. **Test with real data** - Use actual data from the database when available
4. **Handle empty states** - Work correctly when database has no data

## Continuous Integration

In CI environments:

```yaml
# Example GitHub Actions workflow
- name: Start backend
  run: docker-compose up -d backend

- name: Wait for backend
  run: npx wait-on http://localhost:8000/api/v1/ping

- name: Run integration tests
  run: npm run test:integration
```

## Writing New Integration Tests

When adding new integration tests:

1. Check backend availability in `beforeAll`
2. Skip tests gracefully if backend unavailable
3. Use actual TypeScript types from `@/types/api`
4. Test both success and error cases
5. Validate response schemas thoroughly
6. Handle empty data states

Example:

```typescript
import { describe, it, expect, beforeAll } from 'vitest'
import axios from 'axios'
import type { DataResponse, MyType } from '@/types/api'

const API_URL = `${process.env.VITE_API_BASE_URL}/api/v1`
let backendAvailable = false

describe('My Feature Integration', () => {
  beforeAll(async () => {
    try {
      await axios.get(`${API_URL}/ping`, { timeout: 5000 })
      backendAvailable = true
    } catch {
      backendAvailable = false
    }
  })

  it('should test my feature', async () => {
    if (!backendAvailable) return

    const response = await axios.get(`${API_URL}/my-endpoint`)
    const data = response.data as DataResponse<MyType>

    expect(response.status).toBe(200)
    expect(data.data).toBeDefined()
  })
})
```

## Troubleshooting

### Tests Skip with "Backend not available"

Check:
- Backend is running: `curl http://localhost:8000/api/v1/ping`
- Database connections are healthy
- No firewall blocking localhost:8000

### Tests Fail with Schema Errors

This indicates a mismatch between frontend types and backend responses:
1. Check backend API documentation
2. Update TypeScript types in `src/types/api.ts`
3. Update integration tests to match new schema

### Tests Timeout

Increase timeout in axios requests:
```typescript
await axios.get(url, { timeout: 10000 })
```
