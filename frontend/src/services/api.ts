import axios, { AxiosInstance, AxiosResponse } from 'axios'
import type {
  DataResponse,
  PaginatedDataResponse,
  TraitListItem,
  TraitDetailExtended,
  StudyListItem,
  StudyDetailExtended,
  SimilaritySearchResult,
  TraitsOverview,
  StudyAnalytics,
  SimilarityPair,
  PaginationParams,
  TraitSearchFilters,
  StudySearchFilters,
  SimilaritySearchFilters,
} from '@/types/api'

// ==== API Configuration ====

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

class ApiService {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}/api/v1`,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Request interceptor for logging
    this.client.interceptors.request.use(
      (config) => {
        console.log(
          `API Request: ${config.method?.toUpperCase()} ${config.url}`
        )
        return config
      },
      (error) => {
        return Promise.reject(error)
      }
    )

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message)
        return Promise.reject(error)
      }
    )
  }

  // ==== Generic Request Methods ====

  private async get<T>(url: string, params?: Record<string, any>): Promise<T> {
    const response: AxiosResponse<T> = await this.client.get(url, { params })
    return response.data
  }

  private async post<T>(url: string, data?: any): Promise<T> {
    const response: AxiosResponse<T> = await this.client.post(url, data)
    return response.data
  }

  // ==== Core API Methods ====

  async ping(): Promise<DataResponse<{ message: string; timestamp: string }>> {
    return this.get('/ping')
  }

  async getVersion(): Promise<DataResponse<Record<string, string>>> {
    return this.get('/version')
  }

  async getHealth(): Promise<DataResponse<Record<string, any>>> {
    return this.get('/health')
  }

  // ==== Traits API ====

  async getTraits(
    params: {
      page?: number
      page_size?: number
      order_by?: string
      order_desc?: boolean
      min_appearances?: number
    } = {}
  ): Promise<PaginatedDataResponse<TraitListItem[]>> {
    return this.get('/traits', params)
  }

  async searchTraits(params: {
    q: string
    page?: number
    page_size?: number
    min_appearances?: number
  }): Promise<PaginatedDataResponse<TraitListItem[]>> {
    return this.get('/traits/search', params)
  }

  async getTraitDetails(
    traitIndex: number,
    params: {
      include_studies?: boolean
      include_similar?: boolean
      include_efo?: boolean
      max_studies?: number
      max_similar?: number
      similarity_threshold?: number
    } = {}
  ): Promise<DataResponse<TraitDetailExtended>> {
    return this.get(`/traits/${traitIndex}`, params)
  }

  async getTraitStudies(
    traitIndex: number,
    params: {
      page?: number
      page_size?: number
      model?: string
      journal?: string
      date_from?: string
      date_to?: string
    } = {}
  ): Promise<PaginatedDataResponse<Record<string, any>[]>> {
    return this.get(`/traits/${traitIndex}/studies`, params)
  }

  async getSimilarTraits(
    traitIndex: number,
    params: {
      max_results?: number
      similarity_threshold?: number
    } = {}
  ): Promise<DataResponse<SimilaritySearchResult[]>> {
    return this.get(`/traits/${traitIndex}/similar`, params)
  }

  async getTraitEfoMappings(
    traitIndex: number,
    params: {
      max_results?: number
      similarity_threshold?: number
    } = {}
  ): Promise<DataResponse<SimilaritySearchResult[]>> {
    return this.get(`/traits/${traitIndex}/efo-mappings`, params)
  }

  async getTraitsOverview(): Promise<DataResponse<TraitsOverview>> {
    return this.get('/traits/stats/overview')
  }

  async getTraitsBulk(traitIndices: number[]): Promise<DataResponse<any[]>> {
    return this.post('/traits/bulk', traitIndices)
  }

  // ==== Studies API ====

  async getStudies(
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
  ): Promise<PaginatedDataResponse<StudyListItem[]>> {
    return this.get('/studies', params)
  }

  async searchStudies(params: {
    q: string
    page?: number
    page_size?: number
    model?: string
    journal?: string
    date_from?: string
    date_to?: string
  }): Promise<PaginatedDataResponse<StudyListItem[]>> {
    return this.get('/studies/search', params)
  }

  async getStudyDetails(
    studyId: number,
    params: {
      include_similar?: boolean
      include_traits?: boolean
      max_similar?: number
      similarity_threshold?: number
    } = {}
  ): Promise<DataResponse<StudyDetailExtended>> {
    return this.get(`/studies/${studyId}`, params)
  }

  async getStudyByPmid(
    pmid: string,
    model: string,
    params: {
      include_similar?: boolean
      include_traits?: boolean
      max_similar?: number
      similarity_threshold?: number
    } = {}
  ): Promise<DataResponse<StudyDetailExtended>> {
    return this.get(`/studies/pmid/${pmid}`, { model, ...params })
  }

  async getSimilarStudies(
    studyId: number,
    params: {
      max_results?: number
      similarity_threshold?: number
    } = {}
  ): Promise<DataResponse<SimilaritySearchResult[]>> {
    return this.get(`/studies/${studyId}/similar`, params)
  }

  async getStudiesAnalytics(): Promise<DataResponse<StudyAnalytics>> {
    return this.get('/studies/stats/overview')
  }

  // ==== Similarities API ====

  async getTopSimilarities(
    params: {
      page?: number
      page_size?: number
      model?: string
      min_similarity?: number
      order_by?: string
      order_desc?: boolean
    } = {}
  ): Promise<PaginatedDataResponse<SimilarityPair[]>> {
    return this.get('/similarities', params)
  }

  async searchSimilarities(params: {
    q: string
    page?: number
    page_size?: number
    model?: string
    min_similarity?: number
  }): Promise<PaginatedDataResponse<SimilarityPair[]>> {
    return this.get('/similarities/search', params)
  }

  async getTraitSimilarities(
    traitIndex: number,
    params: {
      page?: number
      page_size?: number
      min_similarity?: number
      max_results?: number
    } = {}
  ): Promise<PaginatedDataResponse<SimilaritySearchResult[]>> {
    return this.get(`/similarities/trait/${traitIndex}`, params)
  }

  async getStudySimilarities(
    pmid: string,
    model: string,
    params: {
      page?: number
      page_size?: number
      min_similarity?: number
      max_results?: number
    } = {}
  ): Promise<PaginatedDataResponse<SimilarityPair[]>> {
    return this.get(`/similarities/study/${pmid}`, { model, ...params })
  }

  async getSimilaritiesOverview(): Promise<DataResponse<Record<string, any>>> {
    return this.get('/similarities/stats/overview')
  }

  // ==== System Information API ====

  async getSystemInfo(): Promise<DataResponse<Record<string, any>>> {
    return this.get('/system/info')
  }

  async getDatabaseSchema(): Promise<DataResponse<Record<string, any>>> {
    return this.get('/system/schema')
  }

  async getDatabaseStats(): Promise<DataResponse<Record<string, any>>> {
    return this.get('/system/stats')
  }
}

// Export singleton instance
export const apiService = new ApiService()
export default apiService
