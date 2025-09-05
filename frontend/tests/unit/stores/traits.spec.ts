import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useTraitsStore } from '@/stores/traits'

describe('Traits Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('initializes with empty state', () => {
    const store = useTraitsStore()
    
    expect(store.traits).toEqual([])
    expect(store.selectedTrait).toBe(null)
    expect(store.traitsOverview).toBe(null)
    expect(store.similarTraits).toEqual([])
    expect(store.isLoading).toBe(false)
    expect(store.error).toBeUndefined()
    expect(store.currentPage).toBe(1)
    expect(store.pageSize).toBe(50)
    expect(store.totalCount).toBe(0)
    expect(store.hasTraits).toBe(false)
    expect(store.hasSelectedTrait).toBe(false)
  })

  it('fetches traits successfully', async () => {
    const store = useTraitsStore()
    
    await store.fetchTraits()
    
    expect(store.isLoading).toBe(false)
    expect(store.traits.length).toBeGreaterThan(0)
    expect(store.totalCount).toBeGreaterThan(0)
    expect(store.hasTraits).toBe(true)
    
    // Check that we got the expected mock data
    const heightTrait = store.traits.find(t => t.trait_label === 'height')
    expect(heightTrait).toBeDefined()
    expect(heightTrait?.appearance_count).toBe(150)
  })

  it('searches traits with query', async () => {
    const store = useTraitsStore()
    
    await store.searchTraits('height')
    
    expect(store.isLoading).toBe(false)
    expect(store.traits.length).toBeGreaterThan(0)
    expect(store.filters.query).toBe('height')
    
    // All returned traits should match the search query
    store.traits.forEach(trait => {
      expect(trait.trait_label.toLowerCase()).toContain('height')
    })
  })

  it('fetches trait details', async () => {
    const store = useTraitsStore()
    
    await store.fetchTraitDetails(1)
    
    expect(store.detailLoading.isLoading).toBe(false)
    expect(store.selectedTrait).not.toBe(null)
    expect(store.selectedTrait?.trait.trait_index).toBe(1)
    expect(store.selectedTrait?.trait.trait_label).toBe('height')
    expect(store.hasSelectedTrait).toBe(true)
  })

  it('fetches similar traits', async () => {
    const store = useTraitsStore()
    
    await store.fetchSimilarTraits(1)
    
    expect(store.similarTraits.length).toBeGreaterThan(0)
    expect(store.similarTraits[0].similarity_score).toBeGreaterThan(0)
  })

  it('fetches traits overview', async () => {
    const store = useTraitsStore()
    
    await store.fetchTraitsOverview()
    
    expect(store.overviewLoading.isLoading).toBe(false)
    expect(store.traitsOverview).not.toBe(null)
    expect(store.traitsOverview?.total_traits).toBeGreaterThan(0)
  })

  it('handles pagination correctly', async () => {
    const store = useTraitsStore()
    
    await store.fetchTraits()
    const initialPage = store.currentPage
    
    if (store.hasNext) {
      await store.nextPage()
      expect(store.currentPage).toBe(initialPage + 1)
    }
    
    if (store.hasPrevious) {
      await store.previousPage()
      expect(store.currentPage).toBe(initialPage)
    }
  })

  it('updates filters correctly', () => {
    const store = useTraitsStore()
    
    store.updateFilters({ min_appearances: 10 })
    
    expect(store.filters.min_appearances).toBe(10)
  })

  it('clears filters', () => {
    const store = useTraitsStore()
    
    // Set some filters first
    store.updateFilters({ 
      query: 'test',
      min_appearances: 10,
      model: 'GWAS'
    })
    
    store.clearFilters()
    
    expect(store.filters.query).toBe('')
    expect(store.filters.min_appearances).toBeUndefined()
    expect(store.filters.model).toBeUndefined()
    expect(store.currentPage).toBe(1)
  })

  it('clears selected trait', () => {
    const store = useTraitsStore()
    
    // Set selected trait first
    store.selectedTrait = {
      trait: { trait_index: 1, trait_label: 'test' },
      statistics: {} as any,
      studies: [],
      similar_traits: [],
      efo_mappings: []
    }
    store.similarTraits = [{ index: 1, label: 'similar', similarity_score: 0.8 }]
    
    store.clearSelectedTrait()
    
    expect(store.selectedTrait).toBe(null)
    expect(store.similarTraits).toEqual([])
    expect(store.hasSelectedTrait).toBe(false)
  })

  it('resets all state', () => {
    const store = useTraitsStore()
    
    // Set some state first
    store.traits = [{ trait_index: 1, trait_label: 'test', appearance_count: 1 }]
    store.currentPage = 5
    store.totalCount = 100
    
    store.reset()
    
    expect(store.traits).toEqual([])
    expect(store.selectedTrait).toBe(null)
    expect(store.traitsOverview).toBe(null)
    expect(store.similarTraits).toEqual([])
    expect(store.currentPage).toBe(1)
    expect(store.totalCount).toBe(0)
    expect(store.isLoading).toBe(false)
  })

  it('handles empty search by falling back to fetchTraits', async () => {
    const store = useTraitsStore()
    
    await store.searchTraits('')
    
    // Should have fetched all traits instead of searching
    expect(store.isLoading).toBe(false)
    expect(store.traits.length).toBeGreaterThan(0)
    expect(store.filters.query).toBe('')
  })

  it('computes pagination info correctly', async () => {
    const store = useTraitsStore()
    
    await store.fetchTraits()
    
    const paginationInfo = store.paginationInfo
    expect(paginationInfo.currentPage).toBe(store.currentPage)
    expect(paginationInfo.pageSize).toBe(store.pageSize)
    expect(paginationInfo.totalCount).toBe(store.totalCount)
    expect(paginationInfo.totalPages).toBe(store.totalPages)
    expect(paginationInfo.hasNext).toBe(store.hasNext)
    expect(paginationInfo.hasPrevious).toBe(store.hasPrevious)
  })
})