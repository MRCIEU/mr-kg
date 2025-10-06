import { describe, it, expect, beforeAll } from 'vitest'
import axios from 'axios'
import type {
  DataResponse,
  PaginatedDataResponse,
  TraitListItem,
  TraitDetailExtended,
  StudyListItem,
  SimilaritySearchResult,
  TraitsOverview,
} from '@/types/api'

const API_BASE_URL = process.env.VITE_API_BASE_URL || 'http://localhost:8000'
const API_URL = `${API_BASE_URL}/api/v1`

let backendAvailable = false

describe('API Integration Tests', () => {
  beforeAll(async () => {
    try {
      const response = await axios.get(`${API_URL}/ping`, { timeout: 5000 })
      backendAvailable = response.status === 200
    } catch (error) {
      console.warn('Backend not available, skipping integration tests')
      backendAvailable = false
    }
  })

  describe('Health Endpoints', () => {
    it('should ping backend successfully', async () => {
      if (!backendAvailable) return

      const response = await axios.get(`${API_URL}/ping`)

      expect(response.status).toBe(200)
      const data = response.data as DataResponse<{
        message: string
        timestamp: string
      }>
      expect(data.data.message).toBeDefined()
      expect(data.data.timestamp).toBeDefined()
    })

    it('should get health status', async () => {
      if (!backendAvailable) return

      const response = await axios.get(`${API_URL}/health`)

      expect(response.status).toBe(200)
      const data = response.data as DataResponse<Record<string, any>>
      expect(data.data).toBeDefined()
      expect(data.data.status).toBeDefined()
    })

    it('should get version information', async () => {
      if (!backendAvailable) return

      const response = await axios.get(`${API_URL}/version`)

      expect(response.status).toBe(200)
      const data = response.data as DataResponse<Record<string, string>>
      expect(data.data).toBeDefined()
      expect(data.data.version).toBeDefined()
    })
  })

  describe('Traits API Contract', () => {
    it('should get traits list with correct schema', async () => {
      if (!backendAvailable) return

      const response = await axios.get(`${API_URL}/traits`, {
        params: { page: 1, page_size: 10 },
      })

      expect(response.status).toBe(200)
      const data = response.data as PaginatedDataResponse<TraitListItem[]>

      expect(data.data).toBeInstanceOf(Array)
      expect(data.total_count).toBeGreaterThanOrEqual(0)
      expect(data.page).toBe(1)
      expect(data.page_size).toBe(10)
      expect(data.total_pages).toBeGreaterThanOrEqual(0)
      expect(typeof data.has_next).toBe('boolean')
      expect(typeof data.has_previous).toBe('boolean')

      if (data.data.length > 0) {
        const trait = data.data[0]
        expect(trait).toHaveProperty('trait_index')
        expect(trait).toHaveProperty('trait_label')
        expect(trait).toHaveProperty('appearance_count')
        expect(typeof trait.trait_index).toBe('number')
        expect(typeof trait.trait_label).toBe('string')
        expect(typeof trait.appearance_count).toBe('number')
      }
    })

    it('should search traits with query parameter', async () => {
      if (!backendAvailable) return

      const response = await axios.get(`${API_URL}/traits/search`, {
        params: { q: 'height', page: 1, page_size: 5 },
      })

      expect(response.status).toBe(200)
      const data = response.data as PaginatedDataResponse<TraitListItem[]>

      expect(data.data).toBeInstanceOf(Array)
      expect(data.data.length).toBeLessThanOrEqual(5)

      if (data.data.length > 0) {
        const trait = data.data[0]
        expect(trait.trait_label.toLowerCase()).toContain('height')
      }
    })

    it('should get trait details by index', async () => {
      if (!backendAvailable) return

      const listResponse = await axios.get(`${API_URL}/traits`, {
        params: { page: 1, page_size: 1 },
      })
      const listData = listResponse.data as PaginatedDataResponse<
        TraitListItem[]
      >

      if (listData.data.length === 0) {
        console.warn('No traits available for detail test')
        return
      }

      const traitIndex = listData.data[0].trait_index
      const response = await axios.get(`${API_URL}/traits/${traitIndex}`)

      expect(response.status).toBe(200)
      const data = response.data as DataResponse<TraitDetailExtended>

      expect(data.data).toBeDefined()
      expect(data.data.trait).toBeDefined()
      expect(data.data.trait.trait_index).toBe(traitIndex)
      expect(data.data.trait.trait_label).toBeDefined()
      expect(data.data.statistics).toBeDefined()
    })

    it('should handle trait not found error', async () => {
      if (!backendAvailable) return

      try {
        await axios.get(`${API_URL}/traits/999999`)
        expect.fail('Should have thrown 404 error')
      } catch (error: any) {
        expect(error.response.status).toBe(404)
        expect(error.response.data).toBeDefined()
      }
    })

    it('should get similar traits', async () => {
      if (!backendAvailable) return

      const listResponse = await axios.get(`${API_URL}/traits`, {
        params: { page: 1, page_size: 1 },
      })
      const listData = listResponse.data as PaginatedDataResponse<
        TraitListItem[]
      >

      if (listData.data.length === 0) {
        console.warn('No traits available for similarity test')
        return
      }

      const traitIndex = listData.data[0].trait_index
      const response = await axios.get(
        `${API_URL}/traits/${traitIndex}/similar`,
        {
          params: { max_results: 5 },
        }
      )

      expect(response.status).toBe(200)
      const data = response.data as DataResponse<SimilaritySearchResult[]>

      expect(data.data).toBeInstanceOf(Array)
      if (data.data.length > 0) {
        const similar = data.data[0]
        expect(similar).toHaveProperty('index')
        expect(similar).toHaveProperty('label')
        expect(similar).toHaveProperty('similarity_score')
        expect(typeof similar.similarity_score).toBe('number')
        expect(similar.similarity_score).toBeGreaterThan(0)
        expect(similar.similarity_score).toBeLessThanOrEqual(1)
      }
    })

    it('should get traits overview statistics', async () => {
      if (!backendAvailable) return

      const response = await axios.get(`${API_URL}/traits/stats/overview`)

      expect(response.status).toBe(200)
      const data = response.data as DataResponse<TraitsOverview>

      expect(data.data).toBeDefined()
      expect(data.data.total_traits).toBeGreaterThanOrEqual(0)
      expect(data.data.total_appearances).toBeGreaterThanOrEqual(0)
      expect(data.data.average_appearances).toBeGreaterThanOrEqual(0)
      expect(data.data.top_traits).toBeInstanceOf(Array)
    })
  })

  describe('Studies API Contract', () => {
    it('should get studies list with correct schema', async () => {
      if (!backendAvailable) return

      const response = await axios.get(`${API_URL}/studies`, {
        params: { page: 1, page_size: 10 },
      })

      expect(response.status).toBe(200)
      const data = response.data as PaginatedDataResponse<StudyListItem[]>

      expect(data.data).toBeInstanceOf(Array)
      expect(data.total_count).toBeGreaterThanOrEqual(0)
      expect(data.page).toBe(1)
      expect(data.page_size).toBe(10)

      if (data.data.length > 0) {
        const study = data.data[0]
        expect(study).toHaveProperty('id')
        expect(study).toHaveProperty('pmid')
        expect(study).toHaveProperty('model')
        expect(typeof study.id).toBe('number')
        expect(typeof study.pmid).toBe('string')
        expect(typeof study.model).toBe('string')
      }
    })

    it('should search studies with query parameter', async () => {
      if (!backendAvailable) return

      const response = await axios.get(`${API_URL}/studies/search`, {
        params: { q: 'genetic', page: 1, page_size: 5 },
      })

      expect(response.status).toBe(200)
      const data = response.data as PaginatedDataResponse<StudyListItem[]>

      expect(data.data).toBeInstanceOf(Array)
      expect(data.data.length).toBeLessThanOrEqual(5)
    })

    it('should filter studies by model parameter', async () => {
      if (!backendAvailable) return

      const response = await axios.get(`${API_URL}/studies`, {
        params: { page: 1, page_size: 10, model: 'GWAS' },
      })

      expect(response.status).toBe(200)
      const data = response.data as PaginatedDataResponse<StudyListItem[]>

      expect(data.data).toBeInstanceOf(Array)
      if (data.data.length > 0) {
        data.data.forEach((study) => {
          expect(study.model).toBe('GWAS')
        })
      }
    })
  })

  describe('Pagination Tests', () => {
    it('should handle pagination correctly for traits', async () => {
      if (!backendAvailable) return

      const response1 = await axios.get(`${API_URL}/traits`, {
        params: { page: 1, page_size: 5 },
      })

      const data1 = response1.data as PaginatedDataResponse<TraitListItem[]>

      expect(data1.page).toBe(1)
      expect(data1.page_size).toBe(5)

      if (data1.has_next) {
        const response2 = await axios.get(`${API_URL}/traits`, {
          params: { page: 2, page_size: 5 },
        })

        const data2 = response2.data as PaginatedDataResponse<TraitListItem[]>

        expect(data2.page).toBe(2)
        expect(data2.has_previous).toBe(true)

        if (data1.data.length > 0 && data2.data.length > 0) {
          expect(data1.data[0].trait_index).not.toBe(
            data2.data[0].trait_index
          )
        }
      }
    })

    it('should handle pagination correctly for studies', async () => {
      if (!backendAvailable) return

      const response1 = await axios.get(`${API_URL}/studies`, {
        params: { page: 1, page_size: 5 },
      })

      const data1 = response1.data as PaginatedDataResponse<StudyListItem[]>

      expect(data1.page).toBe(1)
      expect(data1.page_size).toBe(5)

      if (data1.has_next) {
        const response2 = await axios.get(`${API_URL}/studies`, {
          params: { page: 2, page_size: 5 },
        })

        const data2 = response2.data as PaginatedDataResponse<StudyListItem[]>

        expect(data2.page).toBe(2)
        expect(data2.has_previous).toBe(true)
      }
    })
  })

  describe('Error Handling', () => {
    it('should handle 404 errors with correct schema', async () => {
      if (!backendAvailable) return

      try {
        await axios.get(`${API_URL}/traits/999999`)
        expect.fail('Should have thrown error')
      } catch (error: any) {
        expect(error.response.status).toBe(404)
        expect(error.response.data).toBeDefined()
      }
    })

    it('should handle invalid query parameters', async () => {
      if (!backendAvailable) return

      try {
        await axios.get(`${API_URL}/traits/invalid-index`)
        expect.fail('Should have thrown error')
      } catch (error: any) {
        expect([400, 404, 422]).toContain(error.response.status)
      }
    })

    it('should handle invalid page numbers gracefully', async () => {
      if (!backendAvailable) return

      const response = await axios.get(`${API_URL}/traits`, {
        params: { page: -1, page_size: 10 },
      })

      expect([200, 400, 422]).toContain(response.status)
    })
  })

  describe('Similarities API Contract', () => {
    it('should get similarities with correct schema', async () => {
      if (!backendAvailable) return

      const response = await axios.get(`${API_URL}/similarities`, {
        params: { page: 1, page_size: 10 },
      })

      expect(response.status).toBe(200)
      const data = response.data as PaginatedDataResponse<any[]>

      expect(data.data).toBeInstanceOf(Array)
      expect(data.total_count).toBeGreaterThanOrEqual(0)
      expect(data.page).toBe(1)
    })
  })
})
