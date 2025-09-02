import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  TraitListItem,
  TraitDetailExtended,
  TraitsOverview,
  PaginatedDataResponse,
  SimilaritySearchResult,
  LoadingState,
  TraitSearchFilters,
} from '@/types/api'
import { apiService } from '@/services/api'

export const useTraitsStore = defineStore('traits', () => {
  // ==== State ====

  const traits = ref<TraitListItem[]>([])
  const selectedTrait = ref<TraitDetailExtended | null>(null)
  const traitsOverview = ref<TraitsOverview | null>(null)
  const similarTraits = ref<SimilaritySearchResult[]>([])

  // Pagination state
  const currentPage = ref(1)
  const pageSize = ref(50)
  const totalCount = ref(0)
  const totalPages = ref(0)
  const hasNext = ref(false)
  const hasPrevious = ref(false)

  // Filter state
  const filters = ref<TraitSearchFilters>({
    query: '',
    min_appearances: undefined,
    model: undefined,
  })

  // Loading states
  const loading = ref<LoadingState>({ isLoading: false })
  const overviewLoading = ref<LoadingState>({ isLoading: false })
  const detailLoading = ref<LoadingState>({ isLoading: false })

  // ==== Computed ====

  const isLoading = computed(() => loading.value.isLoading)
  const error = computed(() => loading.value.error)

  const hasTraits = computed(() => traits.value.length > 0)
  const hasSelectedTrait = computed(() => selectedTrait.value !== null)

  const paginationInfo = computed(() => ({
    currentPage: currentPage.value,
    pageSize: pageSize.value,
    totalCount: totalCount.value,
    totalPages: totalPages.value,
    hasNext: hasNext.value,
    hasPrevious: hasPrevious.value,
  }))

  // ==== Actions ====

  async function fetchTraits(
    params: {
      page?: number
      page_size?: number
      order_by?: string
      order_desc?: boolean
      min_appearances?: number
    } = {}
  ) {
    loading.value = { isLoading: true, error: undefined }

    try {
      const response: PaginatedDataResponse<TraitListItem[]> =
        await apiService.getTraits({
          page: params.page || currentPage.value,
          page_size: params.page_size || pageSize.value,
          order_by: params.order_by || 'appearance_count',
          order_desc:
            params.order_desc !== undefined ? params.order_desc : true,
          min_appearances:
            params.min_appearances || filters.value.min_appearances,
        })

      traits.value = response.data
      currentPage.value = response.page
      pageSize.value = response.page_size
      totalCount.value = response.total_count
      totalPages.value = response.total_pages
      hasNext.value = response.has_next
      hasPrevious.value = response.has_previous

      loading.value = { isLoading: false }
    } catch (error) {
      console.error('Failed to fetch traits:', error)
      loading.value = {
        isLoading: false,
        error:
          error instanceof Error ? error.message : 'Failed to fetch traits',
      }
    }
  }

  async function searchTraits(
    query: string,
    params: {
      page?: number
      page_size?: number
      min_appearances?: number
    } = {}
  ) {
    if (!query.trim()) {
      return fetchTraits(params)
    }

    loading.value = { isLoading: true, error: undefined }

    try {
      const response = await apiService.searchTraits({
        q: query,
        page: params.page || 1,
        page_size: params.page_size || pageSize.value,
        min_appearances:
          params.min_appearances || filters.value.min_appearances,
      })

      traits.value = response.data
      currentPage.value = response.page
      pageSize.value = response.page_size
      totalCount.value = response.total_count
      totalPages.value = response.total_pages
      hasNext.value = response.has_next
      hasPrevious.value = response.has_previous

      filters.value.query = query
      loading.value = { isLoading: false }
    } catch (error) {
      console.error('Failed to search traits:', error)
      loading.value = {
        isLoading: false,
        error:
          error instanceof Error ? error.message : 'Failed to search traits',
      }
    }
  }

  async function fetchTraitDetails(
    traitIndex: number,
    params: {
      include_studies?: boolean
      include_similar?: boolean
      include_efo?: boolean
      max_studies?: number
      max_similar?: number
      similarity_threshold?: number
    } = {}
  ) {
    detailLoading.value = { isLoading: true, error: undefined }

    try {
      const response = await apiService.getTraitDetails(traitIndex, {
        include_studies: params.include_studies !== false,
        include_similar: params.include_similar !== false,
        include_efo: params.include_efo !== false,
        max_studies: params.max_studies || 50,
        max_similar: params.max_similar || 10,
        similarity_threshold: params.similarity_threshold || 0.3,
      })

      selectedTrait.value = response.data
      detailLoading.value = { isLoading: false }
    } catch (error) {
      console.error('Failed to fetch trait details:', error)
      detailLoading.value = {
        isLoading: false,
        error:
          error instanceof Error
            ? error.message
            : 'Failed to fetch trait details',
      }
    }
  }

  async function fetchSimilarTraits(
    traitIndex: number,
    params: {
      max_results?: number
      similarity_threshold?: number
    } = {}
  ) {
    try {
      const response = await apiService.getSimilarTraits(traitIndex, {
        max_results: params.max_results || 10,
        similarity_threshold: params.similarity_threshold || 0.3,
      })

      similarTraits.value = response.data
    } catch (error) {
      console.error('Failed to fetch similar traits:', error)
    }
  }

  async function fetchTraitsOverview() {
    overviewLoading.value = { isLoading: true, error: undefined }

    try {
      const response = await apiService.getTraitsOverview()
      traitsOverview.value = response.data
      overviewLoading.value = { isLoading: false }
    } catch (error) {
      console.error('Failed to fetch traits overview:', error)
      overviewLoading.value = {
        isLoading: false,
        error:
          error instanceof Error
            ? error.message
            : 'Failed to fetch traits overview',
      }
    }
  }

  // Navigation helpers
  async function goToPage(page: number) {
    if (page >= 1 && page <= totalPages.value) {
      currentPage.value = page
      if (filters.value.query) {
        await searchTraits(filters.value.query, { page })
      } else {
        await fetchTraits({ page })
      }
    }
  }

  async function nextPage() {
    if (hasNext.value) {
      await goToPage(currentPage.value + 1)
    }
  }

  async function previousPage() {
    if (hasPrevious.value) {
      await goToPage(currentPage.value - 1)
    }
  }

  // Filter management
  function updateFilters(newFilters: Partial<TraitSearchFilters>) {
    filters.value = { ...filters.value, ...newFilters }
  }

  function clearFilters() {
    filters.value = {
      query: '',
      min_appearances: undefined,
      model: undefined,
    }
    currentPage.value = 1
  }

  // Reset state
  function clearSelectedTrait() {
    selectedTrait.value = null
    similarTraits.value = []
  }

  function reset() {
    traits.value = []
    selectedTrait.value = null
    traitsOverview.value = null
    similarTraits.value = []
    currentPage.value = 1
    totalCount.value = 0
    totalPages.value = 0
    hasNext.value = false
    hasPrevious.value = false
    clearFilters()
    loading.value = { isLoading: false }
    overviewLoading.value = { isLoading: false }
    detailLoading.value = { isLoading: false }
  }

  return {
    // State
    traits,
    selectedTrait,
    traitsOverview,
    similarTraits,
    currentPage,
    pageSize,
    totalCount,
    totalPages,
    hasNext,
    hasPrevious,
    filters,
    loading,
    overviewLoading,
    detailLoading,

    // Computed
    isLoading,
    error,
    hasTraits,
    hasSelectedTrait,
    paginationInfo,

    // Actions
    fetchTraits,
    searchTraits,
    fetchTraitDetails,
    fetchSimilarTraits,
    fetchTraitsOverview,
    goToPage,
    nextPage,
    previousPage,
    updateFilters,
    clearFilters,
    clearSelectedTrait,
    reset,
  }
})
