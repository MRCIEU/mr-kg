# MR-KG Frontend

Vue 3 + TypeScript app that powers the MR-KG user interface for
exploring traits, studies, and similarities.

## Local Development

For initial setup and prerequisites, see @docs/SETTING-UP.md.

### Quick Start

```bash
cd frontend

# Install dependencies and setup environment
just install
just env-setup

# Start development server
just dev
```

Open http://localhost:3000

### NPM/Yarn Alternatives

If not using just:
- Install: `npm install`
- Dev server: `npm run dev`
- Build: `npm run build`
- Preview: `npm run preview`

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

## Configuration

Copy `.env.example` to `.env` with `just env-setup`, then adjust as needed.
See @docs/ENV.md for complete environment variable documentation.

**Note**: Set `VITE_API_BASE_URL` to the backend origin only (e.g., `http://localhost:8000`). 
The API service automatically appends `/api/v1`. See @frontend/src/services/api.ts.

## Project Structure

```
frontend/
├── src/
│   ├── assets/            # Styles and static assets
│   ├── components/        # Reusable components
│   ├── router/            # Vue Router configuration
│   ├── services/          # API client and service layer
│   ├── stores/            # Pinia stores
│   ├── types/             # TypeScript types
│   ├── views/             # Page-level components
│   ├── App.vue            # Root component
│   └── main.ts            # App entry
├── .env.example           # Example env file (copy to .env)
├── .eslintrc.cjs          # ESLint config
├── .prettierrc.json       # Prettier config
├── env.d.ts               # Vite env types
├── index.html             # Vite HTML entry
├── justfile               # Task runner commands
├── package.json           # Scripts and dependencies
├── tsconfig.json          # TS base config
├── tsconfig.app.json      # TS app config
└── vite.config.ts         # Vite config
```

## Tooling and Code Quality

- ESLint for linting (`just lint`)
- Prettier for formatting (`just format`)
- Type-checking with vue-tsc (`just type-check`)
- TypeScript strict mode throughout the app
- Recommended workflow: run `just check` before committing

## API Client

- Centralized in `src/services/api.ts`
- Reads `VITE_API_BASE_URL` from `.env`
- Adds `/api/v1` automatically and handles errors/logging

## Troubleshooting

### Port Conflicts (3000)
- Stop other dev servers or change the Vite port via CLI env/config

### Backend Not Reachable
- Ensure the FastAPI backend is running and the `.env` URL is correct

### API Path Issues
- Do not include `/api/v1` in `VITE_API_BASE_URL`; it is appended by the client

### Node Version Mismatch
- Use Node 18+ (try nvm to switch versions)

### Stale Vite Cache
- Restart dev server; if needed, clear `node_modules/.vite`

## References

- Environment configuration: @docs/ENV.md
- Development guide and Docker usage: @docs/DEVELOPMENT.md
- Testing strategy: @docs/TESTING.md
- System architecture: @docs/ARCHITECTURE.md
