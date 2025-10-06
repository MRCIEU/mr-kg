import axios from 'axios'
import type { AxiosInstance, AxiosResponse } from 'axios'
import { z } from 'zod'
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
} from '@/types/api'
import {
  TraitsListResponseSchema,
  TraitDetailResponseSchema,
  StudiesListResponseSchema,
  StudyDetailResponseSchema,
  SimilaritiesListResponseSchema,
  TraitsOverviewResponseSchema,
  StudyAnalyticsResponseSchema,
  DataResponseSchema,
  PaginatedDataResponseSchema,
  SimilaritySearchResultSchema,
} from '@/types/schemas'

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

  private async get<T>(
    url: string,
    params?: Record<string, any>,
    schema?: z.ZodSchema<T>
  ): Promise<T> {
    const response: AxiosResponse<T> = await this.client.get(url, { params })
    if (schema) {
      return schema.parse(response.data)
    }
    return response.data
  }

  private async post<T>(
    url: string,
    data?: any,
    schema?: z.ZodSchema<T>
  ): Promise<T> {
    const response: AxiosResponse<T> = await this.client.post(url, data)
    if (schema) {
      return schema.parse(response.data)
    }
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
    return this.get('/traits', params, TraitsListResponseSchema)
  }

  async searchTraits(params: {
    q: string
    page?: number
    page_size?: number
    min_appearances?: number
  }): Promise<PaginatedDataResponse<TraitListItem[]>> {
    return this.get('/traits/search', params, TraitsListResponseSchema)
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
    return this.get(`/traits/${traitIndex}`, params, TraitDetailResponseSchema)
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
    return this.get(
      `/traits/${traitIndex}/similar`,
      params,
      DataResponseSchema(z.array(SimilaritySearchResultSchema))
    )
  }

  async getTraitEfoMappings(
    traitIndex: number,
    params: {
      max_results?: number
      similarity_threshold?: number
    } = {}
  ): Promise<DataResponse<SimilaritySearchResult[]>> {
    return this.get(
      `/traits/${traitIndex}/efo-mappings`,
      params,
      DataResponseSchema(z.array(SimilaritySearchResultSchema))
    )
  }

  async getTraitsOverview(): Promise<DataResponse<TraitsOverview>> {
    return this.get('/traits/stats/overview', {}, TraitsOverviewResponseSchema)
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
    return this.get('/studies', params, StudiesListResponseSchema)
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
    return this.get('/studies/search', params, StudiesListResponseSchema)
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
    return this.get(`/studies/${studyId}`, params, StudyDetailResponseSchema)
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
    return this.get(
      `/studies/pmid/${pmid}`,
      { model, ...params },
      StudyDetailResponseSchema
    )
  }

  async getSimilarStudies(
    studyId: number,
    params: {
      max_results?: number
      similarity_threshold?: number
    } = {}
  ): Promise<DataResponse<SimilaritySearchResult[]>> {
    return this.get(
      `/studies/${studyId}/similar`,
      params,
      DataResponseSchema(z.array(SimilaritySearchResultSchema))
    )
  }

  async getStudiesAnalytics(): Promise<DataResponse<StudyAnalytics>> {
    return this.get('/studies/stats/overview', {}, StudyAnalyticsResponseSchema)
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
    return this.get('/similarities', params, SimilaritiesListResponseSchema)
  }

  async searchSimilarities(params: {
    q: string
    page?: number
    page_size?: number
    model?: string
    min_similarity?: number
  }): Promise<PaginatedDataResponse<SimilarityPair[]>> {
    return this.get(
      '/similarities/search',
      params,
      SimilaritiesListResponseSchema
    )
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
    return this.get(
      `/similarities/trait/${traitIndex}`,
      params,
      PaginatedDataResponseSchema(z.array(SimilaritySearchResultSchema))
    )
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
    return this.get(
      `/similarities/study/${pmid}`,
      { model, ...params },
      SimilaritiesListResponseSchema
    )
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
