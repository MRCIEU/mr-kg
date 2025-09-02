# Contributing to MR-KG

Welcome to the MR-KG (Mendelian Randomization Knowledge Graph) project! This guide will help you contribute effectively to our fullstack application for exploring trait relationships in Mendelian Randomization studies.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Standards](#code-standards)
- [Contributing Workflow](#contributing-workflow)
- [Component Guidelines](#component-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Code Review Process](#code-review-process)
- [Issue Guidelines](#issue-guidelines)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Git**: Version control system
- **Docker & Docker Compose**: For containerized development
- **just**: Task runner for project commands
- **Python 3.12+**: For backend development (optional for Docker-only workflow)
- **Node.js 18+**: For frontend development (optional for Docker-only workflow)
- **uv**: Python package manager (if working locally)

### Repository Structure

```
mr-kg/
├── backend/          # FastAPI REST API server
├── frontend/         # Vue.js TypeScript interface  
├── webapp/           # Legacy Streamlit application
├── processing/       # ETL pipeline and data processing
├── data/            # Vector databases and processed data
├── src/common_funcs/ # Shared schemas and utilities
└── docs/            # Project documentation
```

### First-Time Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mr-kg.git
   cd mr-kg
   ```

2. **Set up development environment**:
   ```bash
   # Setup environment files
   just setup-dev
   
   # Start development stack
   just dev
   ```

3. **Verify setup**:
   ```bash
   # Check service health
   just health
   
   # Access applications:
   # Frontend: http://localhost:3000
   # Backend API: http://localhost:8000/docs
   # Legacy Webapp: http://localhost:8501
   ```

## Development Environment

### Docker-based Development (Recommended)

The recommended approach uses Docker for consistent development environments:

```bash
# Start all services
just dev

# Start individual services
just backend-dev      # Backend only
just frontend-dev     # Frontend only
just webapp-dev       # Legacy webapp only

# View logs
just dev-logs         # All services
just dev-logs backend # Specific service

# Stop services
just dev-down
```

### Local Development

For local development without Docker:

#### Backend
```bash
cd backend
just install      # Install dependencies
just env-setup    # Setup environment
just dev          # Start development server
```

#### Frontend
```bash
cd frontend
just install      # Install dependencies
just env-setup    # Setup environment
just dev          # Start development server
```

### Development Commands

Each component has a `justfile` with standardized commands:

```bash
# Development
just dev              # Start development server
just install          # Install dependencies

# Code quality
just fmt              # Format code
just lint             # Lint code
just check            # Run all quality checks

# Testing
just test             # Run tests
just test-cov         # Run tests with coverage

# Docker
just docker-build     # Build Docker image
just docker-run       # Run Docker container
```

## Code Standards

### General Principles

- **Clarity over cleverness**: Write readable, maintainable code
- **Type safety**: Use TypeScript for frontend, type hints for Python
- **Documentation**: Document complex logic and API interfaces
- **Testing**: Write tests for new features and bug fixes
- **Security**: Follow security best practices
- **Performance**: Consider performance implications of changes

### Python (Backend) Standards

#### Code Style
- **Formatter**: `ruff format` for consistent formatting
- **Linter**: `ruff check` for code quality
- **Type checker**: `ty` for type validation
- **Line length**: 80 characters where practical
- **Import sorting**: Organized import statements

#### Code Organization
```python
# Standard library imports
import asyncio
from datetime import datetime
from typing import List, Optional

# Third-party imports
from fastapi import FastAPI, Depends
from pydantic import BaseModel

# Local imports
from app.core.config import settings
from app.models.database import TraitEmbedding
```

#### Function Documentation
```python
async def search_traits(
    filters: TraitSearchFilters,
    database_service: DatabaseService = Depends(get_database_service)
) -> TraitSearchResponse:
    """Search traits based on provided filters.
    
    Args:
        filters: Search criteria including term, limits, and thresholds
        database_service: Injected database service dependency
        
    Returns:
        TraitSearchResponse with matching traits and pagination info
        
    Raises:
        DatabaseError: If database query fails
        ValidationError: If search filters are invalid
    """
    # Implementation here
```

#### Error Handling
```python
# Use custom exceptions
from app.core.exceptions import NotFoundError, ValidationError

# Proper error context
try:
    result = await database_service.get_trait(trait_index)
except Exception as e:
    logger.error(f"Failed to retrieve trait {trait_index}: {e}")
    raise DatabaseError(f"Trait retrieval failed") from e
```

### TypeScript (Frontend) Standards

#### Code Style
- **Formatter**: Prettier for consistent formatting
- **Linter**: ESLint with Vue.js rules
- **Type checker**: vue-tsc for TypeScript validation
- **Naming**: camelCase for variables, PascalCase for components

#### Vue.js Component Structure
```vue
<template>
  <div class="trait-explorer">
    <SearchInput
      v-model="searchTerm"
      :loading="isLoading"
      @search="handleSearch"
    />
    <TraitResults 
      :traits="filteredTraits"
      :pagination="pagination"
      @trait-select="handleTraitSelect"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useTraitsStore } from '@/stores/traits'
import type { Trait, TraitSearchFilters } from '@/types/traits'

// Component logic here
</script>

<style scoped>
.trait-explorer {
  @apply container mx-auto p-4;
}
</style>
```

#### Pinia Store Structure
```typescript
// stores/traits.ts
export const useTraitsStore = defineStore('traits', () => {
  // State
  const traits = ref<Trait[]>([])
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  
  // Getters
  const filteredTraits = computed(() => {
    // Computed logic here
  })
  
  // Actions
  const searchTraits = async (filters: TraitSearchFilters) => {
    isLoading.value = true
    try {
      const response = await api.searchTraits(filters)
      traits.value = response.data.traits
    } catch (err) {
      error.value = handleApiError(err)
    } finally {
      isLoading.value = false
    }
  }
  
  return {
    // State
    traits: readonly(traits),
    isLoading: readonly(isLoading),
    error: readonly(error),
    
    // Getters
    filteredTraits,
    
    // Actions
    searchTraits
  }
})
```

### Database and Data Processing

#### SQL Style
```sql
-- Use clear, readable SQL
SELECT 
    te.trait_index,
    te.trait_label,
    COUNT(mrt.pmid) as study_count
FROM trait_embeddings te
LEFT JOIN model_result_traits mrt 
    ON te.trait_index = mrt.trait_index
WHERE te.trait_label ILIKE '%height%'
GROUP BY te.trait_index, te.trait_label
ORDER BY study_count DESC
LIMIT 50;
```

#### Data Processing
```python
# Use type hints and clear variable names
def process_trait_embeddings(
    raw_embeddings: List[Dict[str, Any]],
    embedding_model: str = "en_core_sci_lg"
) -> List[TraitEmbedding]:
    """Process raw trait embeddings into structured format."""
    processed_embeddings = []
    
    for raw_embedding in raw_embeddings:
        # Validation and processing logic
        processed = TraitEmbedding(
            trait_index=raw_embedding["trait_index"],
            trait_label=raw_embedding["trait_label"],
            embedding=normalize_vector(raw_embedding["embedding"])
        )
        processed_embeddings.append(processed)
    
    return processed_embeddings
```

## Contributing Workflow

### Branch Naming

Use descriptive branch names with prefixes:

```bash
# Feature branches
feature/trait-similarity-visualization
feature/advanced-search-filters

# Bug fixes
fix/database-connection-timeout
fix/frontend-pagination-bug

# Documentation
docs/api-documentation-update
docs/deployment-guide-revision

# Refactoring
refactor/database-service-cleanup
refactor/component-structure-improvement
```

### Commit Messages

Follow conventional commit format:

```bash
# Format: <type>(<scope>): <description>

# Examples:
feat(backend): add trait similarity search endpoint
fix(frontend): resolve pagination component state issue
docs(readme): update development setup instructions
test(backend): add integration tests for health endpoints
refactor(frontend): extract search logic into composable
perf(database): optimize trait similarity query performance
```

### Pull Request Process

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and commit**:
   ```bash
   # Make your changes
   git add .
   git commit -m "feat(component): add new feature"
   ```

3. **Run quality checks**:
   ```bash
   # Backend checks
   cd backend && just check
   
   # Frontend checks  
   cd frontend && just check
   
   # Run tests
   just test-backend
   ```

4. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

5. **PR Requirements**:
   - [ ] All tests pass
   - [ ] Code quality checks pass
   - [ ] Documentation updated (if needed)
   - [ ] Changelog updated (for significant changes)
   - [ ] Breaking changes noted

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] All existing tests pass

## Documentation
- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] User documentation updated
- [ ] Changelog updated

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Changes are backward compatible (or breaking changes documented)
- [ ] Related issues linked
```

## Component Guidelines

### Backend Development

#### API Endpoint Development
```python
# Use consistent endpoint patterns
@router.get("/traits/search", response_model=DataResponse[TraitSearchResponse])
async def search_traits(
    search_term: Optional[str] = Query(None, description="Search term for trait labels"),
    min_appearance_count: int = Query(5, ge=1, description="Minimum appearance count"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Results offset for pagination"),
    database_service: DatabaseService = Depends(get_database_service)
) -> DataResponse[TraitSearchResponse]:
    """Search traits based on filters."""
    # Implementation
```

#### Database Service Patterns
```python
class TraitService:
    """High-level service for trait operations."""
    
    def __init__(self, database_pool: DatabaseConnectionPool):
        self.trait_repository = TraitRepository(database_pool)
        self.similarity_repository = SimilarityRepository(database_pool)
    
    async def search_traits_with_similarities(
        self, 
        filters: TraitSearchFilters
    ) -> TraitSearchWithSimilarities:
        """Search traits and include similarity information."""
        # Coordinate multiple repository calls
        traits = await self.trait_repository.search_traits(filters)
        
        # Add similarity data if requested
        if filters.include_similarities:
            for trait in traits:
                trait.similar_traits = await self.similarity_repository.get_similar_traits(
                    trait.trait_index, 
                    threshold=0.7, 
                    limit=5
                )
        
        return TraitSearchWithSimilarities(traits=traits)
```

### Frontend Development

#### Component Development
```vue
<!-- Use composition API and TypeScript -->
<template>
  <div class="search-component">
    <form @submit.prevent="handleSubmit">
      <SearchInput
        v-model="searchForm.term"
        :disabled="isLoading"
        placeholder="Search traits..."
      />
      <FilterControls
        v-model="searchForm.filters"
        :options="filterOptions"
      />
      <SubmitButton
        :loading="isLoading"
        :disabled="!isFormValid"
      />
    </form>
  </div>
</template>

<script setup lang="ts">
interface Props {
  initialFilters?: TraitSearchFilters
}

interface Emits {
  (e: 'search', filters: TraitSearchFilters): void
  (e: 'clear'): void
}

const props = withDefaults(defineProps<Props>(), {
  initialFilters: () => ({})
})

const emit = defineEmits<Emits>()

// Component logic with proper typing
const searchForm = reactive<{
  term: string
  filters: TraitSearchFilters
}>({
  term: '',
  filters: { ...props.initialFilters }
})
</script>
```

#### State Management Patterns
```typescript
// Composable for reusable logic
export function useTraitSearch() {
  const traitsStore = useTraitsStore()
  
  const searchTraits = async (filters: TraitSearchFilters) => {
    try {
      await traitsStore.searchTraits(filters)
    } catch (error) {
      console.error('Search failed:', error)
      // Handle error appropriately
    }
  }
  
  return {
    searchTraits,
    traits: computed(() => traitsStore.traits),
    isLoading: computed(() => traitsStore.isLoading),
    error: computed(() => traitsStore.error)
  }
}
```

### Data Processing

#### Processing Script Patterns
```python
#!/usr/bin/env python3
"""
Trait preprocessing script for MR-KG pipeline.

This script processes raw LLM results to extract unique traits
and create canonical trait indices.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

from common_funcs.schema.raw_data_schema import RawLLMResult
from common_funcs.schema.processed_data_schema import ProcessedTrait

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # ---- Input/Output ----
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw LLM results"
    )
    
    parser.add_argument(
        "--output-file", 
        type=Path,
        required=True,
        help="Output file for processed traits"
    )
    
    # ---- Processing Options ----
    parser.add_argument(
        "--min-appearances",
        type=int,
        default=5,
        help="Minimum trait appearances to include"
    )
    
    # ---- Execution Options ----
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    return parser.parse_args()

def main() -> None:
    """Main processing function."""
    args = parse_args()
    
    logger.info(f"Processing traits from {args.input_dir}")
    
    if args.dry_run:
        logger.info("DRY RUN - no files will be modified")
        return
    
    # Processing implementation
    process_traits(args.input_dir, args.output_file, args.min_appearances)

if __name__ == "__main__":
    main()
```

## Testing Requirements

### Test Coverage Goals
- **Backend**: Minimum 80% code coverage
- **Frontend**: Minimum 75% code coverage  
- **Critical paths**: 100% coverage for core business logic
- **API endpoints**: 100% coverage for all endpoints

### Backend Testing

#### Unit Tests
```python
# tests/unit/test_trait_service.py
import pytest
from unittest.mock import AsyncMock, Mock

from app.services.trait_service import TraitService
from app.models.database import TraitSearchFilters

@pytest.fixture
def mock_trait_repository():
    """Mock trait repository for testing."""
    return AsyncMock()

@pytest.fixture 
def trait_service(mock_trait_repository):
    """Create trait service with mocked dependencies."""
    return TraitService(mock_trait_repository)

@pytest.mark.asyncio
async def test_search_traits_success(trait_service, mock_trait_repository):
    """Test successful trait search."""
    # Arrange
    filters = TraitSearchFilters(search_term="height", limit=10)
    expected_traits = [Mock(trait_index=1, trait_label="height")]
    mock_trait_repository.search_traits.return_value = expected_traits
    
    # Act
    result = await trait_service.search_traits(filters)
    
    # Assert
    assert len(result) == 1
    assert result[0].trait_label == "height"
    mock_trait_repository.search_traits.assert_called_once_with(filters)
```

#### Integration Tests
```python
# tests/integration/test_api_integration.py
@pytest.mark.integration
def test_trait_search_api_integration(client, test_database):
    """Test complete trait search API workflow."""
    # Test data setup
    setup_test_traits(test_database)
    
    # API call
    response = client.get("/api/v1/traits/search?search_term=height&limit=5")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]["traits"]) <= 5
    assert all("height" in trait["trait_label"].lower() 
              for trait in data["data"]["traits"])
```

### Frontend Testing (Planned)

#### Component Tests
```typescript
// tests/unit/components/TraitSearch.spec.ts
import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import TraitSearch from '@/components/TraitSearch.vue'

describe('TraitSearch', () => {
  it('emits search event with correct filters', async () => {
    const wrapper = mount(TraitSearch, {
      global: {
        plugins: [createTestingPinia()]
      }
    })
    
    await wrapper.find('[data-testid="search-input"]').setValue('height')
    await wrapper.find('form').trigger('submit')
    
    expect(wrapper.emitted('search')).toBeTruthy()
    expect(wrapper.emitted('search')?.[0]).toEqual([
      { search_term: 'height' }
    ])
  })
})
```

### Test Commands

```bash
# Backend testing
cd backend
just test                # Run all tests
just test-cov            # Run with coverage
just test tests/unit/    # Run specific test category

# Frontend testing (planned)
cd frontend  
just test                # Run all tests
just test:coverage       # Run with coverage
just test:unit           # Run unit tests only
```

## Documentation

### Code Documentation

#### API Documentation
- All endpoints must have comprehensive docstrings
- Include parameter descriptions, response formats, and error conditions
- Use OpenAPI/Swagger annotations for automatic documentation

#### Component Documentation
- Document complex Vue.js components with JSDoc comments
- Include prop interfaces and event emissions
- Document component usage examples

#### Processing Documentation
- All processing scripts must have module docstrings
- Document input/output formats and processing logic
- Include usage examples and parameter descriptions

### User Documentation

When adding new features, update relevant documentation:

- **README.md**: Update if setup process changes
- **API documentation**: Update for new endpoints
- **User guides**: Update for new UI features
- **Deployment guides**: Update for infrastructure changes

### Documentation Standards

```python
# Python docstring format
def compute_trait_similarity(
    trait1_embedding: List[float],
    trait2_embedding: List[float],
    method: str = "cosine"
) -> float:
    """Compute similarity between two trait embeddings.
    
    Args:
        trait1_embedding: Vector representation of first trait
        trait2_embedding: Vector representation of second trait  
        method: Similarity computation method ("cosine", "euclidean")
        
    Returns:
        Similarity score between 0 and 1, where 1 is identical
        
    Raises:
        ValueError: If embeddings have different dimensions
        NotImplementedError: If method is not supported
        
    Example:
        >>> embedding1 = [0.1, 0.2, 0.3]
        >>> embedding2 = [0.2, 0.3, 0.4]  
        >>> similarity = compute_trait_similarity(embedding1, embedding2)
        >>> print(f"Similarity: {similarity:.3f}")
        Similarity: 0.993
    """
```

## Code Review Process

### Review Checklist

#### For Authors
- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] Breaking changes documented
- [ ] Security implications considered
- [ ] Performance impact assessed

#### For Reviewers
- [ ] Code is readable and maintainable
- [ ] Logic is correct and efficient
- [ ] Error handling is appropriate
- [ ] Tests are comprehensive
- [ ] Documentation is clear
- [ ] Security best practices followed

### Review Guidelines

#### Constructive Feedback
```markdown
# Good feedback examples:

## Suggestion for improvement:
Consider extracting this logic into a separate function for better testability:
```python
# Current
if trait.appearance_count > threshold and trait.similarity > min_similarity:
    filtered_traits.append(trait)

# Suggested
def should_include_trait(trait: Trait, threshold: int, min_similarity: float) -> bool:
    return trait.appearance_count > threshold and trait.similarity > min_similarity
```

## Question for clarification:
What happens if the database connection fails during this operation? Should we implement retry logic?

## Security concern:
This endpoint appears to be missing input validation. Consider adding validation for the trait_index parameter to prevent SQL injection.
```

#### Review Response
- Respond to all feedback
- Explain design decisions when questioned
- Make requested changes or provide justification
- Thank reviewers for their time and feedback

## Issue Guidelines

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
Clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., macOS, Ubuntu]
- Browser: [e.g., Chrome, Firefox]
- Version: [e.g., 1.0.0]

**Additional Context**
Screenshots, logs, or other relevant information.
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
Clear description of the requested feature.

**Use Case**
Describe the problem this feature would solve.

**Proposed Solution**
Describe how you envision this feature working.

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Any other relevant information.
```

### Enhancement Proposals

For significant changes, create enhancement proposals:

```markdown
**Enhancement Title**
Brief title describing the enhancement.

**Motivation**
Why is this enhancement needed?

**Design Overview**
High-level description of the proposed changes.

**Implementation Plan**
Detailed implementation steps.

**Testing Strategy**
How will this be tested?

**Documentation Impact**
What documentation needs to be updated?

**Breaking Changes**
Any breaking changes this introduces.
```

## Release Process

### Version Management

We follow semantic versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

#### Pre-release
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version numbers updated
- [ ] Security scan completed
- [ ] Performance benchmarks run

#### Release
- [ ] Create release branch
- [ ] Tag release in Git
- [ ] Build and test release artifacts
- [ ] Deploy to staging environment
- [ ] Run integration tests
- [ ] Deploy to production

#### Post-release
- [ ] Monitor for issues
- [ ] Update documentation site
- [ ] Communicate changes to users
- [ ] Plan next release cycle

### Changelog Format

```markdown
# Changelog

## [1.2.0] - 2024-01-15

### Added
- New trait similarity visualization component
- Advanced search filters for trait exploration
- API endpoint for batch trait processing

### Changed
- Improved database query performance for large datasets
- Updated Vue.js to version 3.4
- Enhanced error handling in API responses

### Fixed
- Fixed pagination bug in trait search results
- Resolved memory leak in vector similarity computation
- Corrected database connection timeout issues

### Security
- Updated dependencies to address security vulnerabilities
- Added rate limiting to API endpoints

### Breaking Changes
- Changed API response format for trait search endpoint
- Removed deprecated /v0/ API endpoints
```

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Documentation**: Comprehensive guides and API reference
- **Code Comments**: Inline documentation for complex logic

### Development Support

- **Setup Issues**: Check README.md and DEV.md first
- **API Questions**: Refer to interactive documentation at `/docs`
- **Architecture Questions**: See ARCHITECTURE.md
- **Deployment Issues**: Check DEPLOYMENT.md

### Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Share knowledge and experience
- Follow the code of conduct

Thank you for contributing to MR-KG! Your contributions help advance research in Mendelian Randomization and make complex genetic data more accessible to researchers worldwide.