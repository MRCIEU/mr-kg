# MR-KG Roadmap

This is a living roadmap for MR-KG. It is non-binding and subject to
change as priorities and constraints evolve.

Sources of planned work include the Future Enhancements in
@docs/ARCHITECTURE.md and tasks captured in @TODO.org. This document
summarizes those items and groups them by theme.

## Platform

- Authentication and authorization
  - Add login, roles, and permissions across API and UI
  - Support token-based auth and optional SSO
  - Reference design in @docs/ARCHITECTURE.md

- Real-time updates
  - Deliver live progress and notifications via SSE or WebSockets
  - Use background jobs for long-running tasks
  - Coordinated backend and UI changes

- Architecture evolution
  - Evaluate microservice boundaries and service contracts
  - Explore event-driven workflows with a message bus
  - Phase changes safely with clear interfaces

- Advanced caching
  - Layered caching for API reads and heavy queries
  - Tune invalidation strategies tied to data refresh
  - Document cache behavior and limits

- Collaboration features
  - Saved views, shareable links, and lightweight comments
  - Respect permissions and privacy by design

- Architecture validation
  - Verify feasibility of planned items in
    @docs/ARCHITECTURE.md before build
  - Capture decisions as short RFCs in repo

## Backend

- API extensions
  - Richer filters, sorting, and pagination on key endpoints
  - Similarity and exploration routes aligned to user flows
  - Keep OpenAPI docs clean and consistent

- Data export services
  - Export to CSV, JSON, and Parquet for key entities
  - Async export with job status and streaming where possible
  - Traceability of export parameters and versions

- Real-time channels
  - SSE or WebSocket endpoints for job and data updates
  - Secure subscriptions with auth scopes

- Monitoring hooks
  - Metrics, tracing, and structured logs
  - Health endpoints and readiness probes
  - Tie into dashboards during deploys

## Frontend

- Progressive web app
  - Installable app, offline basics, and caching strategy
  - Respect auth and cache invalidation rules

- Advanced visualization
  - Similarity heatmaps, network graphs, and drill-downs
  - Keep interactions accessible and performant

- Accessibility
  - Aim for WCAG 2.1 AA across pages
  - Keyboard navigation and focus management
  - Color contrast and semantic structure

- Internationalization
  - i18n infrastructure and locale switcher
  - Externalize strings and formats

- Frontend testing
  - Unit tests with Vitest and Vue Test Utils
  - E2E tests with Playwright or Cypress
  - See @docs/TESTING.md and @frontend/README.md

- Real-time UI
  - Live status and notifications for long jobs
  - Optimistic updates with server reconciliation

- Data export UI
  - Download buttons and job status modals
  - Clear format and column descriptions

## Data

- Advanced analytics
  - Clustering, ranking, and facet summaries on traits and studies
  - Reusable primitives for similarity exploration
  - Align with needs captured in @docs/ARCHITECTURE.md

- Data export
  - Reproducible extracts from DuckDB sources
  - Data dictionary and version tags for each export
  - Cross-reference with @docs/DATA.md

- Upgrade common utilities
  - Upgrade to the new version of yiutils
  - Audit breaking changes and adapt callers
  - Validate with processing and backend tests
  - See @src/yiutils/README.md

## DevOps

- CI/CD pipeline
  - Build, lint, type-check, and test gates on pull requests
  - Automated image builds and environment promotion
  - Release notes and artifact retention

- Monitoring integration
  - Application metrics, logs, traces, and uptime checks
  - Alerting on SLOs with actionable runbooks

- Documentation hygiene
  - Simplify docs and remove duplication
  - Keep a single source of truth and link out
  - Consolidate in @DEV.md, @README.md, and @docs/*

- Containers and environments
  - Harden images and slim layers
  - Clear dev, docker, and prod profiles
  - See @docs/DOCKER.md and docker-compose files

- Planning discipline
  - Verify how planned items will be implemented before commit
  - Track open items in @TODO.org and link to issues

## Notes

- Items listed here are directional. Sequencing and scope can change.
- Use small, incremental pull requests. Keep tests and docs current.
- Propose updates to this roadmap via PR, referencing relevant docs.