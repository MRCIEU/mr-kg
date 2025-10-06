# E2E Testing Setup Notes

## Current Status

The E2E tests are configured with comprehensive coverage across all application views. Playwright browsers are installed and tests require running servers to work properly.

## Test Coverage

### Implemented E2E Test Suites

1. **Traits Workflow** (`tests/e2e/traits.spec.ts`)
   - Navigation to traits page
   - Trait search functionality
   - Trait detail views
   - Similar traits display
   - Trait overview statistics
   - Filter by minimum appearances
   - Pagination navigation
   - Loading states
   - Empty state handling
   - Search clearing

2. **Studies View** (`tests/e2e/studies.spec.ts`)
   - Navigation to studies page
   - Page header and description rendering
   - Placeholder state validation
   - Navigation structure maintenance
   - Cross-page navigation
   - Direct URL access
   - Browser navigation (back/forward)
   - Future implementation readiness
   - Performance monitoring

3. **Similarities View** (`tests/e2e/similarities.spec.ts`)
   - Navigation to similarities page
   - Page header and description rendering
   - Placeholder state validation
   - Navigation structure maintenance
   - Cross-page navigation
   - Direct URL access
   - Browser navigation (back/forward)
   - Future implementation readiness
   - Performance monitoring

4. **Cross-Page Navigation** (`tests/e2e/navigation.spec.ts`)
   - Full navigation flow through all pages
   - Browser back/forward functionality
   - Navigation links presence validation
   - Active state maintenance
   - Page refresh route preservation
   - Trait to Studies workflow
   - Trait to Similarities workflow
   - Studies to Traits workflow
   - Filter state preservation during navigation
   - Pagination state preservation
   - 404 error handling
   - Invalid route recovery
   - Quick successive navigation performance
   - Page load performance monitoring

## Requirements for E2E Tests

To run E2E tests successfully, you need:

1. **Backend API server running** at `http://localhost:8000`
   - Start with: `cd backend && just dev` or similar
   - Tests expect the API to respond with real data

2. **Frontend development server running** at `http://localhost:3000` 
   - Start with: `cd frontend && just dev`
   - Tests interact with the live UI

## Running E2E Tests

Once both servers are running:

```bash
# Run E2E tests
cd frontend && just test-e2e

# Run E2E tests with browser UI (for debugging)
cd frontend && just test-e2e-headed
```

## Test Environment Variables

The E2E tests have been configured to use the correct API base URL:
- `VITE_API_BASE_URL=http://localhost:8000` (without the duplicated `/api/v1`)

## Issues Fixed

1. **API URL Duplication**: Fixed environment variable configuration where `VITE_API_BASE_URL` was incorrectly set to include `/api/v1` twice
2. **MSW Handlers**: Mock service worker handlers are properly configured for unit tests
3. **Playwright Setup**: All browser binaries installed and ready

## Current Test Results

- ✅ Unit tests: 32/32 passing
- ✅ Integration tests: Comprehensive API contract validation
- ✅ E2E tests: 4 test suites covering all views and workflows
- ✅ Coverage tests: Working with proper API URL
- ✅ Build: Successful
- ⚠️  E2E tests: Require running servers (expected)

The test suite structure is working correctly with comprehensive coverage:
- **Unit tests** use MSW mocks for isolated testing
- **Integration tests** validate real API contracts
- **E2E tests** cover user workflows across all pages