# MR-KG Frontend

Vue.js TypeScript frontend for MR-KG (Mendelian Randomization Knowledge Graph), providing an interactive web interface for exploring trait relationships, study analysis, and similarity computation.

## Features

- Vue.js 3 with TypeScript and Composition API
- Modern build tooling with Vite and hot module replacement
- State management with Pinia for centralized application state
- Routing with Vue Router for single-page application navigation
- Responsive design with Tailwind CSS utility framework
- API integration with axios for backend communication

## Development Setup

### Prerequisites

- Node.js 18+ with npm/yarn
- [just](https://github.com/casey/just) task runner

### Quick Start

1. **Install dependencies**:
   ```bash
   just install
   ```

2. **Set up environment**:
   ```bash
   just env-example
   just env-setup
   ```

3. **Start development server**:
   ```bash
   just dev
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - Hot reload enabled for development

### Available Commands

```bash
# Development
just dev              # Start development server with hot reload
just install          # Install dependencies

# Build
just build            # Build for production
just preview          # Preview production build

# Code Quality
just format           # Format code with prettier
just lint             # Lint code with eslint
just type-check       # Type check with vue-tsc
just check            # Run all quality checks

# Docker
just docker-build     # Build development Docker image
just docker-run       # Run development Docker container

# Environment
just env-example      # Create example environment file
just env-setup        # Set up .env from example
```

## Project Structure

```
frontend/
├── src/
│   ├── components/   # Reusable Vue components
│   ├── views/        # Page-level Vue components
│   ├── stores/       # Pinia state management stores
│   ├── services/     # API service layer
│   ├── types/        # TypeScript type definitions
│   ├── utils/        # Utility functions
│   ├── router/       # Vue Router configuration
│   ├── assets/       # Static assets and styles
│   ├── App.vue       # Root Vue component
│   └── main.ts       # Application entry point
├── public/           # Static public assets
├── Dockerfile.dev    # Development Docker configuration
├── justfile          # Task runner configuration
└── package.json      # Node.js project configuration
```

## Configuration

Configuration is managed through environment variables. See `.env.example` for available options:

- `VITE_API_BASE_URL`: Backend API base URL (default: http://localhost:8000/api/v1)
- `VITE_APP_TITLE`: Application title
- `VITE_APP_DESCRIPTION`: Application description

## Pages and Features

### Home Page
- Overview of the MR-KG system with navigation to main features
- Feature highlights and system information
- Quick access cards to different exploration tools

### Traits Exploration (Task 3-2)
- Browse and search trait labels from MR studies
- Filter traits by appearance frequency and model
- View trait statistics and related studies
- Interactive trait selection and comparison

### Studies Analysis (Task 3-3)
- View detailed study metadata and results
- Explore study-trait associations
- Find similar studies using vector similarity
- Study filtering and sorting capabilities

### Similarities Analysis (Task 3-4)
- Analyze trait and study similarity relationships
- Interactive similarity threshold controls
- Visualization of similarity networks
- Export similarity data and insights

### About Page
- Project information and methodology overview
- Technology stack documentation
- Data sources and processing pipeline details

## API Integration

The frontend communicates with the FastAPI backend through:

- Axios HTTP client for API requests
- Environment-based API URL configuration
- Type-safe request/response interfaces
- Error handling and loading states
- Request interceptors for authentication (future)

## Docker Development

Build and run the development environment with Docker:

```bash
# Build development image
just docker-build

# Run with hot reload and volume mounting
just docker-run
```

The Docker setup includes:
- Hot reload for rapid development
- Volume mounting for code changes
- Environment variable injection
- Non-root user execution for security

## Code Quality

Code quality is enforced through:

- **ESLint**: JavaScript/TypeScript linting
- **Prettier**: Code formatting
- **Vue TSC**: TypeScript type checking
- **Vue ESLint**: Vue.js specific linting rules

Run all quality checks:

```bash
just check
```

## State Management

The application uses Pinia for state management with stores for:

- **Application State**: Global UI state and configuration
- **Traits Store**: Trait data, filtering, and search state
- **Studies Store**: Study data and metadata management
- **Similarities Store**: Similarity analysis and results

## Routing

Vue Router handles navigation with routes for:

- `/` - Home page with overview and navigation
- `/traits` - Traits exploration interface
- `/studies` - Studies analysis interface
- `/similarities` - Similarities analysis interface
- `/about` - Project information and documentation

## Development Patterns

- **Composition API**: Using Vue 3's modern composition API
- **TypeScript Strict Mode**: Full type safety throughout the application
- **Component-based Architecture**: Modular, reusable component design
- **Reactive Data**: Vue's reactivity system for dynamic interfaces
- **Single File Components**: Vue SFC format for component organization

## Next Steps

This foundation setup provides the scaffolding for:

1. Traits interface components (Task 3-2)
2. Studies interface components (Task 3-3)
3. Similarities interface components (Task 3-4)
4. Integration with backend APIs

See the project master plan for detailed implementation phases.