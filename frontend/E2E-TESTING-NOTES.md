# E2E Testing Setup Notes

## Current Status

The E2E tests are configured and Playwright browsers are installed, but they require running servers to work properly.

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
- ✅ Coverage tests: Working with proper API URL
- ✅ Build: Successful
- ❌ E2E tests: Require running servers (expected)

The test suite structure is working correctly, but E2E tests need live application servers to pass.