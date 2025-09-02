import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  StudyListItem,
  StudyDetailExtended,
  StudyAnalytics,
  PaginatedDataResponse,
  SimilaritySearchResult,
  LoadingState,
  StudySearchFilters,
} from '@/types/api'
import { apiService } from '@/services/api'

export const useStudiesStore = defineStore('studies', () => {
  // ==== State ====

  const studies = ref<StudyListItem[]>([])
  const selectedStudy = ref<StudyDetailExtended | null>(null)
  const studiesAnalytics = ref<StudyAnalytics | null>(null)
  const similarStudies = ref<SimilaritySearchResult[]>([])

  // Pagination state
  const currentPage = ref(1)
  const pageSize = ref(50)
  const totalCount = ref(0)
  const totalPages = ref(0)
  const hasNext = ref(false)
  const hasPrevious = ref(false)

  // Filter state
  const filters = ref<StudySearchFilters>({
    model: undefined,
    journal: undefined,
    date_from: undefined,
    date_to: undefined,
    min_trait_count: undefined,
    trait_index: undefined,
  })

  // Loading states
  const loading = ref<LoadingState>({ isLoading: false })
  const analyticsLoading = ref<LoadingState>({ isLoading: false })
  const detailLoading = ref<LoadingState>({ isLoading: false })

  // ==== Computed ====

  const isLoading = computed(() => loading.value.isLoading)
  const error = computed(() => loading.value.error)

  const hasStudies = computed(() => studies.value.length > 0)
  const hasSelectedStudy = computed(() => selectedStudy.value !== null)

  const paginationInfo = computed(() => ({
    currentPage: currentPage.value,
    pageSize: pageSize.value,
    totalCount: totalCount.value,
    totalPages: totalPages.value,
    hasNext: hasNext.value,
    hasPrevious: hasPrevious.value,
  }))

  // ==== Actions ====

  async function fetchStudies(
    params: {
      page?: number
      page_size?: number
      model?: string
      journal?: string
      date_from?: string
      date_to?: string
      min_trait_count?: number
      order_by?: string
      order_desc?: boolean
    } = {}
  ) {
    loading.value = { isLoading: true, error: undefined }

    try {
      const response: PaginatedDataResponse<StudyListItem[]> =
        await apiService.getStudies({
          page: params.page || currentPage.value,
          page_size: params.page_size || pageSize.value,
          model: params.model || filters.value.model,
          journal: params.journal || filters.value.journal,
          date_from: params.date_from || filters.value.date_from,
          date_to: params.date_to || filters.value.date_to,
          min_trait_count:
            params.min_trait_count || filters.value.min_trait_count,
          order_by: params.order_by || 'pub_date',
          order_desc:
            params.order_desc !== undefined ? params.order_desc : true,
        })

      studies.value = response.data
      currentPage.value = response.page
      pageSize.value = response.page_size
      totalCount.value = response.total_count
      totalPages.value = response.total_pages
      hasNext.value = response.has_next
      hasPrevious.value = response.has_previous

      loading.value = { isLoading: false }
    } catch (error) {
      console.error('Failed to fetch studies:', error)
      loading.value = {
        isLoading: false,
        error:
          error instanceof Error ? error.message : 'Failed to fetch studies',
      }
    }
  }

  async function searchStudies(
    query: string,
    params: {
      page?: number
      page_size?: number
      model?: string
      journal?: string
      date_from?: string
      date_to?: string
    } = {}
  ) {
    if (!query.trim()) {
      return fetchStudies(params)
    }

    loading.value = { isLoading: true, error: undefined }

    try {
      const response = await apiService.searchStudies({
        q: query,
        page: params.page || 1,
        page_size: params.page_size || pageSize.value,
        model: params.model || filters.value.model,
        journal: params.journal || filters.value.journal,
        date_from: params.date_from || filters.value.date_from,
        date_to: params.date_to || filters.value.date_to,
      })

      studies.value = response.data
      currentPage.value = response.page
      pageSize.value = response.page_size
      totalCount.value = response.total_count
      totalPages.value = response.total_pages
      hasNext.value = response.has_next
      hasPrevious.value = response.has_previous

      loading.value = { isLoading: false }
    } catch (error) {
      console.error('Failed to search studies:', error)
      loading.value = {
        isLoading: false,
        error:
          error instanceof Error ? error.message : 'Failed to search studies',
      }
    }
  }

  async function fetchStudyDetails(
    studyId: number,
    params: {
      include_similar?: boolean
      include_traits?: boolean
      max_similar?: number
      similarity_threshold?: number
    } = {}
  ) {
    detailLoading.value = { isLoading: true, error: undefined }

    try {
      const response = await apiService.getStudyDetails(studyId, {
        include_similar: params.include_similar !== false,
        include_traits: params.include_traits !== false,
        max_similar: params.max_similar || 10,
        similarity_threshold: params.similarity_threshold || 0.3,
      })

      selectedStudy.value = response.data
      detailLoading.value = { isLoading: false }
    } catch (error) {
      console.error('Failed to fetch study details:', error)
      detailLoading.value = {
        isLoading: false,
        error:
          error instanceof Error
            ? error.message
            : 'Failed to fetch study details',
      }
    }
  }

  async function fetchStudyByPmid(
    pmid: string,
    model: string,
    params: {
      include_similar?: boolean
      include_traits?: boolean
      max_similar?: number
      similarity_threshold?: number
    } = {}
  ) {
    detailLoading.value = { isLoading: true, error: undefined }

    try {
      const response = await apiService.getStudyByPmid(pmid, model, {
        include_similar: params.include_similar !== false,
        include_traits: params.include_traits !== false,
        max_similar: params.max_similar || 10,
        similarity_threshold: params.similarity_threshold || 0.3,
      })

      selectedStudy.value = response.data
      detailLoading.value = { isLoading: false }
    } catch (error) {
      console.error('Failed to fetch study details by PMID:', error)
      detailLoading.value = {
        isLoading: false,
        error:
          error instanceof Error
            ? error.message
            : 'Failed to fetch study details',
      }
    }
  }

  async function fetchSimilarStudies(
    studyId: number,
    params: {
      max_results?: number
      similarity_threshold?: number
    } = {}
  ) {
    try {
      const response = await apiService.getSimilarStudies(studyId, {
        max_results: params.max_results || 10,
        similarity_threshold: params.similarity_threshold || 0.3,
      })

      similarStudies.value = response.data
    } catch (error) {
      console.error('Failed to fetch similar studies:', error)
    }
  }

  async function fetchStudiesAnalytics() {
    analyticsLoading.value = { isLoading: true, error: undefined }

    try {
      const response = await apiService.getStudiesAnalytics()
      studiesAnalytics.value = response.data
      analyticsLoading.value = { isLoading: false }
    } catch (error) {
      console.error('Failed to fetch studies analytics:', error)
      analyticsLoading.value = {
        isLoading: false,
        error:
          error instanceof Error
            ? error.message
            : 'Failed to fetch studies analytics',
      }
    }
  }

  // Navigation helpers
  async function goToPage(page: number) {
    if (page >= 1 && page <= totalPages.value) {
      currentPage.value = page
      await fetchStudies({ page })
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
  function updateFilters(newFilters: Partial<StudySearchFilters>) {
    filters.value = { ...filters.value, ...newFilters }
  }

  function clearFilters() {
    filters.value = {
      model: undefined,
      journal: undefined,
      date_from: undefined,
      date_to: undefined,
      min_trait_count: undefined,
      trait_index: undefined,
    }
    currentPage.value = 1
  }

  // Reset state
  function clearSelectedStudy() {
    selectedStudy.value = null
    similarStudies.value = []
  }

  function reset() {
    studies.value = []
    selectedStudy.value = null
    studiesAnalytics.value = null
    similarStudies.value = []
    currentPage.value = 1
    totalCount.value = 0
    totalPages.value = 0
    hasNext.value = false
    hasPrevious.value = false
    clearFilters()
    loading.value = { isLoading: false }
    analyticsLoading.value = { isLoading: false }
    detailLoading.value = { isLoading: false }
  }

  return {
    // State
    studies,
    selectedStudy,
    studiesAnalytics,
    similarStudies,
    currentPage,
    pageSize,
    totalCount,
    totalPages,
    hasNext,
    hasPrevious,
    filters,
    loading,
    analyticsLoading,
    detailLoading,

    // Computed
    isLoading,
    error,
    hasStudies,
    hasSelectedStudy,
    paginationInfo,

    // Actions
    fetchStudies,
    searchStudies,
    fetchStudyDetails,
    fetchStudyByPmid,
    fetchSimilarStudies,
    fetchStudiesAnalytics,
    goToPage,
    nextPage,
    previousPage,
    updateFilters,
    clearFilters,
    clearSelectedStudy,
    reset,
  }
})
