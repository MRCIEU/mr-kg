# Frontend testing

Testing strategy for Vue 3 frontend covering unit, integration, and E2E tests.

For backend testing and general testing strategy, see @docs/testing.md.

## Test structure

```
frontend/tests/
├── unit/              # Component and utility tests (Vitest + MSW)
├── integration/       # API contract validation (real backend)
├── e2e/              # End-to-end workflows (Playwright)
├── mocks/            # MSW handlers
└── setup.ts          # Test configuration
```

## Unit tests

Unit tests use MSW (Mock Service Worker) to mock API responses.

### Coverage

- Component rendering and behavior
- User interactions
- State management
- Utility functions
- Error handling

### Running unit tests

```bash
cd frontend
just test              # Run all unit tests
just test-watch        # Watch mode
just test-cov          # With coverage report
```



## Integration tests

Integration tests validate the frontend works correctly with the actual backend API.

### Coverage

- API contract compliance (response schemas match frontend types)
- Correct HTTP status codes
- Pagination behavior
- Error handling
- Query parameter handling

### Prerequisites

Before running integration tests:
1. Backend running on http://localhost:8000
2. Database populated with test data
3. All backend services healthy

### Running integration tests

```bash
cd frontend
just test-integration              # Run integration tests
just test-integration-watch        # Watch mode
```

### Configuration

Set backend URL via environment variable:

```bash
VITE_API_BASE_URL=http://localhost:8000 just test-integration
```

### Test categories

Validates API contracts for health, traits, studies, and similarities endpoints including pagination, error handling, and schema compliance.

### Test behavior

Tests are designed to:
1. Check backend availability - Skip gracefully if backend not running
2. Validate schemas - Ensure response structures match TypeScript types
3. Test with real data - Use actual data from database when available
4. Handle empty states - Work correctly when database has no data

## E2E tests

E2E tests use Playwright to test complete user workflows.

### Coverage

**Traits workflow:**
- Navigation to traits page
- Trait search functionality
- Trait detail views
- Similar traits display
- Pagination navigation
- Loading states
- Empty state handling

**Studies view:**
- Navigation to studies page
- Page rendering
- Placeholder state validation
- Cross-page navigation

**Similarities view:**
- Navigation to similarities page
- Page rendering
- Placeholder state validation
- Cross-page navigation

**Cross-page navigation:**
- Full navigation flow through all pages
- Browser back/forward functionality
- Active state maintenance
- Page refresh route preservation
- Filter state preservation
- 404 error handling

### Prerequisites

E2E tests require both servers running:

1. Backend API server at http://localhost:8000
   - Start with: cd backend && just dev

2. Frontend dev server at http://localhost:3000
   - Start with: cd frontend && just dev

### Running E2E tests

```bash
cd frontend
just test-e2e              # Run E2E tests headless
just test-e2e-headed       # Run with browser UI (for debugging)
```

### Environment configuration

E2E tests use correct API base URL:
- VITE_API_BASE_URL=http://localhost:8000 (without duplicated /api/v1)



## Troubleshooting

### Integration tests skip with "Backend not available"

Check:
- Backend is running: curl http://localhost:8000/api/v1/ping
- Database connections are healthy
- No firewall blocking localhost:8000

### Integration tests fail with schema errors

This indicates mismatch between frontend types and backend responses:
1. Check backend API documentation
2. Update TypeScript types in src/types/api.ts
3. Update integration tests to match new schema



### E2E tests fail

Ensure both servers are running:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000



## Continuous integration

In CI environments:

```yaml
# Example GitHub Actions workflow
- name: Start backend
  run: docker-compose up -d backend

- name: Wait for backend
  run: npx wait-on http://localhost:8000/api/v1/ping

- name: Run integration tests
  run: npm run test:integration

- name: Start frontend
  run: npm run dev &

- name: Wait for frontend
  run: npx wait-on http://localhost:3000

- name: Run E2E tests
  run: npm run test:e2e
```
