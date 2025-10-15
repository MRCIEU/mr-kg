# MR-KG Frontend

Vue 3 + TypeScript app for exploring MR-KG traits, studies, and similarities.

For comprehensive frontend documentation, see @docs/frontend/overview.md.

For initial setup and prerequisites, see @docs/setting-up.md.

## Quick Start

```bash
cd frontend

# Install dependencies and setup environment
just install
just env-setup

# Start development server
just dev
```

Open http://localhost:3000

## Commands

### Development

```bash
just dev              # Start Vite dev server with HMR
just install          # Install dependencies
```

### Build

```bash
just build            # Production build
just preview          # Preview production build locally
```

### Code Quality

```bash
just format           # Prettier format
just lint             # ESLint linting (auto-fix where possible)
just type-check       # Type-check with vue-tsc
just check            # Lint + type-check
```

## Documentation

- Frontend overview: @docs/frontend/overview.md
- Testing strategy: @docs/frontend/testing.md
- Environment configuration: @docs/env.md
- Development workflows: @docs/development.md
