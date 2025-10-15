# Frontend overview

Vue 3 + TypeScript application for exploring MR-KG traits, studies, and similarities.

## Tech stack

- Vue 3 with Composition API
- TypeScript with strict mode
- Tailwind CSS for styling
- Pinia for state management
- Vue Router for navigation
- Axios for API communication
- Vitest for unit testing
- Playwright for E2E testing

## Quick reference

For commands and development workflows, see @frontend/README.md.

## Architecture documentation

- Testing strategy: @docs/frontend/testing.md

## Key features

- Component-based architecture with reusable UI components
- Centralized API service with Zod schema validation
- Type-safe state management with Pinia
- Responsive design with Tailwind CSS
- Comprehensive test coverage (unit, integration, E2E)

## Project structure

```
frontend/src/
├── assets/            # Styles and static assets
├── components/        # Reusable components
│   ├── common/        # Shared UI components
│   └── trait/         # Trait-specific components
├── router/            # Vue Router configuration
├── services/          # API client and service layer
├── stores/            # Pinia stores
├── types/             # TypeScript types
├── views/             # Page-level components
├── App.vue            # Root component
└── main.ts            # App entry
```

## Main views

- HomeView: Landing page with navigation
- TraitsView: Browse and search traits
- StudiesView: Browse and search studies
- SimilaritiesView: Explore trait and study similarities
- AboutView: Project information

## Reusable components

- TraitCard: Display trait information
- PaginationControls: Navigate paginated data
- LoadingSpinner: Loading indicator
- ErrorAlert: Error message display
- SearchInput: Search with debounce

## API integration

Centralized API service in src/services/api.ts:
- Singleton pattern for consistent client
- Axios-based HTTP client with interceptors
- Zod schema validation for type safety
- Centralized error handling
- Request/response logging

## State management

Pinia stores for reactive state:
- Studies store: Studies list, pagination, filters
- Traits store: Traits list, pagination, filters

## Development

See @frontend/README.md for:
- Local development setup
- Available commands
- Testing procedures
- Code quality tools
- Troubleshooting guide
