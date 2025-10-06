import { describe, it, expect } from 'vitest'
import { z } from 'zod'
import apiService from '@/services/api'
import {
  TraitsListResponseSchema,
  TraitDetailResponseSchema,
  StudiesListResponseSchema,
  StudyDetailResponseSchema,
  SimilaritiesListResponseSchema,
  TraitsOverviewResponseSchema,
  StudyAnalyticsResponseSchema,
  TraitListItemSchema,
  TraitDetailExtendedSchema,
  StudyListItemSchema,
  StudyDetailExtendedSchema,
  SimilarityPairSchema,
  PaginatedDataResponseSchema,
  DataResponseSchema,
} from '@/types/schemas'

describe('API Contract Validation', () => {
  describe('Traits API Contracts', () => {
    it('should validate traits list response schema', async () => {
      const response = await apiService.getTraits({ page: 1, page_size: 10 })

      expect(() => TraitsListResponseSchema.parse(response)).not.toThrow()

      expect(response).toHaveProperty('data')
      expect(Array.isArray(response.data)).toBe(true)
      expect(response).toHaveProperty('total_count')
      expect(response).toHaveProperty('page')
      expect(response).toHaveProperty('page_size')
      expect(response).toHaveProperty('total_pages')
      expect(response).toHaveProperty('has_next')
      expect(response).toHaveProperty('has_previous')

      if (response.data.length > 0) {
        const firstTrait = response.data[0]
        expect(() => TraitListItemSchema.parse(firstTrait)).not.toThrow()
        expect(firstTrait).toHaveProperty('trait_index')
        expect(firstTrait).toHaveProperty('trait_label')
        expect(firstTrait).toHaveProperty('appearance_count')
        expect(typeof firstTrait.trait_index).toBe('number')
        expect(typeof firstTrait.trait_label).toBe('string')
        expect(typeof firstTrait.appearance_count).toBe('number')
      }
    })

    it('should validate trait search response schema', async () => {
      const response = await apiService.searchTraits({
        q: 'diabetes',
        page: 1,
        page_size: 10,
      })

      expect(() => TraitsListResponseSchema.parse(response)).not.toThrow()
      expect(response).toHaveProperty('data')
      expect(Array.isArray(response.data)).toBe(true)
    })

    it('should validate trait detail response schema', async () => {
      const listResponse = await apiService.getTraits({
        page: 1,
        page_size: 1,
      })
      expect(listResponse.data.length).toBeGreaterThan(0)

      const traitIndex = listResponse.data[0].trait_index
      const response = await apiService.getTraitDetails(traitIndex, {
        include_studies: true,
        include_similar: true,
        include_efo: true,
      })

      expect(() => TraitDetailResponseSchema.parse(response)).not.toThrow()
      expect(() =>
        TraitDetailExtendedSchema.parse(response.data)
      ).not.toThrow()

      expect(response.data).toHaveProperty('trait')
      expect(response.data).toHaveProperty('statistics')
      expect(response.data).toHaveProperty('studies')
      expect(response.data).toHaveProperty('similar_traits')
      expect(response.data).toHaveProperty('efo_mappings')

      expect(response.data.trait).toHaveProperty('trait_index')
      expect(response.data.trait).toHaveProperty('trait_label')
      expect(response.data.statistics).toHaveProperty('appearance_count')
      expect(response.data.statistics).toHaveProperty('study_count')
      expect(Array.isArray(response.data.studies)).toBe(true)
      expect(Array.isArray(response.data.similar_traits)).toBe(true)
      expect(Array.isArray(response.data.efo_mappings)).toBe(true)
    })

    it('should validate traits overview response schema', async () => {
      const response = await apiService.getTraitsOverview()

      expect(() => TraitsOverviewResponseSchema.parse(response)).not.toThrow()

      expect(response.data).toHaveProperty('total_traits')
      expect(response.data).toHaveProperty('total_appearances')
      expect(response.data).toHaveProperty('average_appearances')
      expect(response.data).toHaveProperty('top_traits')
      expect(response.data).toHaveProperty('appearance_distribution')
      expect(response.data).toHaveProperty('model_coverage')

      expect(typeof response.data.total_traits).toBe('number')
      expect(typeof response.data.total_appearances).toBe('number')
      expect(typeof response.data.average_appearances).toBe('number')
      expect(Array.isArray(response.data.top_traits)).toBe(true)
      expect(Array.isArray(response.data.model_coverage)).toBe(true)
    })
  })

  describe('Studies API Contracts', () => {
    it('should validate studies list response schema', async () => {
      const response = await apiService.getStudies({ page: 1, page_size: 10 })

      expect(() => StudiesListResponseSchema.parse(response)).not.toThrow()

      expect(response).toHaveProperty('data')
      expect(Array.isArray(response.data)).toBe(true)
      expect(response).toHaveProperty('total_count')
      expect(response).toHaveProperty('page')
      expect(response).toHaveProperty('page_size')

      if (response.data.length > 0) {
        const firstStudy = response.data[0]
        expect(() => StudyListItemSchema.parse(firstStudy)).not.toThrow()
        expect(firstStudy).toHaveProperty('id')
        expect(firstStudy).toHaveProperty('model')
        expect(firstStudy).toHaveProperty('pmid')
        expect(firstStudy).toHaveProperty('trait_count')
        expect(typeof firstStudy.id).toBe('number')
        expect(typeof firstStudy.model).toBe('string')
        expect(typeof firstStudy.pmid).toBe('string')
        expect(typeof firstStudy.trait_count).toBe('number')
      }
    })

    it('should validate study search response schema', async () => {
      const response = await apiService.searchStudies({
        q: 'cardiovascular',
        page: 1,
        page_size: 10,
      })

      expect(() => StudiesListResponseSchema.parse(response)).not.toThrow()
      expect(response).toHaveProperty('data')
      expect(Array.isArray(response.data)).toBe(true)
    })

    it('should validate study detail response schema', async () => {
      const listResponse = await apiService.getStudies({
        page: 1,
        page_size: 1,
      })
      expect(listResponse.data.length).toBeGreaterThan(0)

      const studyId = listResponse.data[0].id
      const response = await apiService.getStudyDetails(studyId, {
        include_similar: true,
        include_traits: true,
      })

      expect(() => StudyDetailResponseSchema.parse(response)).not.toThrow()
      expect(() =>
        StudyDetailExtendedSchema.parse(response.data)
      ).not.toThrow()

      expect(response.data).toHaveProperty('study')
      expect(response.data).toHaveProperty('traits')
      expect(response.data).toHaveProperty('similar_studies')
      expect(response.data).toHaveProperty('statistics')

      expect(response.data.study).toHaveProperty('id')
      expect(response.data.study).toHaveProperty('model')
      expect(response.data.study).toHaveProperty('pmid')
      expect(Array.isArray(response.data.traits)).toBe(true)
      expect(Array.isArray(response.data.similar_studies)).toBe(true)
    })

    it('should validate studies analytics response schema', async () => {
      const response = await apiService.getStudiesAnalytics()

      expect(() =>
        StudyAnalyticsResponseSchema.parse(response)
      ).not.toThrow()

      expect(response.data).toHaveProperty('total_studies')
      expect(response.data).toHaveProperty('total_pmids')
      expect(response.data).toHaveProperty('model_distribution')
      expect(response.data).toHaveProperty('journal_distribution')
      expect(response.data).toHaveProperty('year_distribution')
      expect(response.data).toHaveProperty('trait_count_distribution')

      expect(typeof response.data.total_studies).toBe('number')
      expect(typeof response.data.total_pmids).toBe('number')
    })
  })

  describe('Similarities API Contracts', () => {
    it('should validate similarities list response schema', async () => {
      const response = await apiService.getTopSimilarities({
        page: 1,
        page_size: 10,
      })

      expect(() =>
        SimilaritiesListResponseSchema.parse(response)
      ).not.toThrow()

      expect(response).toHaveProperty('data')
      expect(Array.isArray(response.data)).toBe(true)
      expect(response).toHaveProperty('total_count')
      expect(response).toHaveProperty('page')
      expect(response).toHaveProperty('page_size')

      if (response.data.length > 0) {
        const firstPair = response.data[0]
        expect(() => SimilarityPairSchema.parse(firstPair)).not.toThrow()
        expect(firstPair).toHaveProperty('query_pmid')
        expect(firstPair).toHaveProperty('query_title')
        expect(firstPair).toHaveProperty('similar_pmid')
        expect(firstPair).toHaveProperty('similar_title')
        expect(firstPair).toHaveProperty('trait_profile_similarity')
        expect(firstPair).toHaveProperty('trait_jaccard_similarity')
        expect(typeof firstPair.query_pmid).toBe('string')
        expect(typeof firstPair.similar_pmid).toBe('string')
        expect(typeof firstPair.trait_profile_similarity).toBe('number')
        expect(typeof firstPair.trait_jaccard_similarity).toBe('number')
      }
    })

    it('should validate similarity search response schema', async () => {
      const response = await apiService.searchSimilarities({
        q: 'hypertension',
        page: 1,
        page_size: 10,
      })

      expect(() =>
        SimilaritiesListResponseSchema.parse(response)
      ).not.toThrow()
      expect(response).toHaveProperty('data')
      expect(Array.isArray(response.data)).toBe(true)
    })
  })

  describe('Schema Validation Robustness', () => {
    it('should reject invalid paginated response structure', () => {
      const invalidResponse = {
        data: [],
        total_count: -1,
        page: 0,
        page_size: 0,
      }

      expect(() =>
        PaginatedDataResponseSchema(z.array(z.any())).parse(invalidResponse)
      ).toThrow()
    })

    it('should reject invalid trait list item', () => {
      const invalidTrait = {
        trait_index: -1,
        trait_label: '',
        appearance_count: -5,
      }

      expect(() => TraitListItemSchema.parse(invalidTrait)).toThrow()
    })

    it('should reject invalid similarity scores', () => {
      const invalidPair = {
        query_pmid: '12345',
        query_title: 'Test',
        similar_pmid: '67890',
        similar_title: 'Test 2',
        trait_profile_similarity: 1.5,
        trait_jaccard_similarity: -0.5,
        query_trait_count: 10,
        similar_trait_count: 8,
        model: 'test',
      }

      expect(() => SimilarityPairSchema.parse(invalidPair)).toThrow()
    })

    it('should reject study list item with missing required fields', () => {
      const invalidStudy = {
        id: 1,
        model: 'test',
      }

      expect(() => StudyListItemSchema.parse(invalidStudy)).toThrow()
    })

    it('should accept valid data response with optional message', () => {
      const validResponse = {
        data: { test: 'value' },
        message: 'Success',
      }

      expect(() =>
        DataResponseSchema(z.object({ test: z.string() })).parse(
          validResponse
        )
      ).not.toThrow()
    })

    it('should accept valid data response without message', () => {
      const validResponse = {
        data: { test: 'value' },
      }

      expect(() =>
        DataResponseSchema(z.object({ test: z.string() })).parse(
          validResponse
        )
      ).not.toThrow()
    })
  })

  describe('Runtime Validation Error Handling', () => {
    it('should throw ZodError on schema mismatch from API', async () => {
      const mockInvalidEndpoint = async () => {
        const response = {
          data: [
            {
              trait_index: 'invalid',
              trait_label: 123,
              appearance_count: 'not a number',
            },
          ],
          total_count: 1,
          page: 1,
          page_size: 10,
          total_pages: 1,
          has_next: false,
          has_previous: false,
        }
        return TraitsListResponseSchema.parse(response)
      }

      await expect(mockInvalidEndpoint()).rejects.toThrow(z.ZodError)
    })
  })
})
