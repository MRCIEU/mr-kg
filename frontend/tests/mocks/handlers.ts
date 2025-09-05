import { http, HttpResponse } from 'msw'
import type {
  TraitListItem,
  TraitDetailExtended,
  TraitsOverview,
  StudyListItem,
  StudyDetailExtended,
  SimilaritySearchResult,
  PaginatedDataResponse,
  DataResponse,
} from '@/types/api'

const API_BASE_URL = 'http://localhost:8000/api/v1'

// ==== Mock Data ====

const mockTraits: TraitListItem[] = [
  { trait_index: 1, trait_label: 'height', appearance_count: 150 },
  { trait_index: 2, trait_label: 'weight', appearance_count: 120 },
  { trait_index: 3, trait_label: 'BMI', appearance_count: 95 },
  { trait_index: 4, trait_label: 'blood pressure', appearance_count: 80 },
  { trait_index: 5, trait_label: 'cholesterol', appearance_count: 75 },
]

const mockTraitDetail: TraitDetailExtended = {
  trait: {
    trait_index: 1,
    trait_label: 'height',
    vector: [0.1, 0.2, 0.3],
  },
  statistics: {
    trait_index: 1,
    trait_label: 'height',
    appearance_count: 150,
    study_count: 45,
    model_distribution: { GWAS: 30, MR: 15 },
    publication_years: [2020, 2021, 2022, 2023],
  },
  studies: [],
  similar_traits: [
    { index: 2, label: 'body height', similarity_score: 0.95 },
    { index: 3, label: 'stature', similarity_score: 0.89 },
  ],
  efo_mappings: [],
}

const mockTraitsOverview: TraitsOverview = {
  total_traits: 1250,
  total_appearances: 15000,
  average_appearances: 12,
  top_traits: mockTraits.slice(0, 3),
  appearance_distribution: { '1-5': 500, '6-10': 300, '11+': 450 },
  model_coverage: [
    { model: 'GWAS', unique_traits: 800, total_mentions: 9000 },
    { model: 'MR', unique_traits: 450, total_mentions: 6000 },
  ],
}

const mockStudies: StudyListItem[] = [
  {
    id: 1,
    model: 'GWAS',
    pmid: '12345678',
    title: 'Genome-wide association study of height',
    journal: 'Nature Genetics',
    pub_date: '2023-01-15',
    trait_count: 5,
  },
  {
    id: 2,
    model: 'MR',
    pmid: '87654321',
    title: 'Mendelian randomization analysis of BMI',
    journal: 'Nature Medicine',
    pub_date: '2023-02-20',
    trait_count: 3,
  },
]

// ==== Request Handlers ====

export const handlers = [
  // ==== Health endpoints ====
  http.get(`${API_BASE_URL}/ping`, () => {
    return HttpResponse.json({
      data: {
        message: 'pong',
        timestamp: new Date().toISOString(),
      },
    } as DataResponse<{ message: string; timestamp: string }>)
  }),

  http.get(`${API_BASE_URL}/health`, () => {
    return HttpResponse.json({
      data: {
        status: 'healthy',
        database: 'connected',
        version: '1.0.0',
      },
    } as DataResponse<Record<string, any>>)
  }),

  // ==== Traits endpoints ====
  http.get(`${API_BASE_URL}/traits`, ({ request }) => {
    const url = new URL(request.url)
    const page = parseInt(url.searchParams.get('page') || '1')
    const pageSize = parseInt(url.searchParams.get('page_size') || '50')
    const minAppearances = url.searchParams.get('min_appearances')

    let filteredTraits = [...mockTraits]
    if (minAppearances) {
      const min = parseInt(minAppearances)
      filteredTraits = filteredTraits.filter((t) => t.appearance_count >= min)
    }

    const start = (page - 1) * pageSize
    const end = start + pageSize
    const paginatedTraits = filteredTraits.slice(start, end)

    return HttpResponse.json({
      data: paginatedTraits,
      total_count: filteredTraits.length,
      page,
      page_size: pageSize,
      total_pages: Math.ceil(filteredTraits.length / pageSize),
      has_next: end < filteredTraits.length,
      has_previous: page > 1,
    } as PaginatedDataResponse<TraitListItem[]>)
  }),

  http.get(`${API_BASE_URL}/traits/search`, ({ request }) => {
    const url = new URL(request.url)
    const query = url.searchParams.get('q') || ''
    const page = parseInt(url.searchParams.get('page') || '1')
    const pageSize = parseInt(url.searchParams.get('page_size') || '50')

    const filteredTraits = mockTraits.filter((trait) =>
      trait.trait_label.toLowerCase().includes(query.toLowerCase())
    )

    const start = (page - 1) * pageSize
    const end = start + pageSize
    const paginatedTraits = filteredTraits.slice(start, end)

    return HttpResponse.json({
      data: paginatedTraits,
      total_count: filteredTraits.length,
      page,
      page_size: pageSize,
      total_pages: Math.ceil(filteredTraits.length / pageSize),
      has_next: end < filteredTraits.length,
      has_previous: page > 1,
    } as PaginatedDataResponse<TraitListItem[]>)
  }),

  http.get(`${API_BASE_URL}/traits/:traitIndex`, ({ params }) => {
    const traitIndex = parseInt(params.traitIndex as string)

    if (traitIndex === 1) {
      return HttpResponse.json({
        data: mockTraitDetail,
      } as DataResponse<TraitDetailExtended>)
    }

    return HttpResponse.json({ error: 'Trait not found' }, { status: 404 })
  }),

  http.get(`${API_BASE_URL}/traits/:traitIndex/similar`, ({ params }) => {
    const traitIndex = parseInt(params.traitIndex as string)

    const similarTraits: SimilaritySearchResult[] = [
      { index: 2, label: 'body height', similarity_score: 0.95 },
      { index: 3, label: 'stature', similarity_score: 0.89 },
      { index: 4, label: 'tall stature', similarity_score: 0.85 },
    ]

    return HttpResponse.json({
      data: similarTraits,
    } as DataResponse<SimilaritySearchResult[]>)
  }),

  http.get(`${API_BASE_URL}/traits/stats/overview`, () => {
    return HttpResponse.json({
      data: mockTraitsOverview,
    } as DataResponse<TraitsOverview>)
  }),

  // ==== Studies endpoints ====
  http.get(`${API_BASE_URL}/studies`, ({ request }) => {
    const url = new URL(request.url)
    const page = parseInt(url.searchParams.get('page') || '1')
    const pageSize = parseInt(url.searchParams.get('page_size') || '50')

    const start = (page - 1) * pageSize
    const end = start + pageSize
    const paginatedStudies = mockStudies.slice(start, end)

    return HttpResponse.json({
      data: paginatedStudies,
      total_count: mockStudies.length,
      page,
      page_size: pageSize,
      total_pages: Math.ceil(mockStudies.length / pageSize),
      has_next: end < mockStudies.length,
      has_previous: page > 1,
    } as PaginatedDataResponse<StudyListItem[]>)
  }),

  http.get(`${API_BASE_URL}/studies/search`, ({ request }) => {
    const url = new URL(request.url)
    const query = url.searchParams.get('q') || ''
    const page = parseInt(url.searchParams.get('page') || '1')
    const pageSize = parseInt(url.searchParams.get('page_size') || '50')

    const filteredStudies = mockStudies.filter(
      (study) =>
        study.title?.toLowerCase().includes(query.toLowerCase()) ||
        study.pmid.includes(query)
    )

    const start = (page - 1) * pageSize
    const end = start + pageSize
    const paginatedStudies = filteredStudies.slice(start, end)

    return HttpResponse.json({
      data: paginatedStudies,
      total_count: filteredStudies.length,
      page,
      page_size: pageSize,
      total_pages: Math.ceil(filteredStudies.length / pageSize),
      has_next: end < filteredStudies.length,
      has_previous: page > 1,
    } as PaginatedDataResponse<StudyListItem[]>)
  }),

  // ==== Similarities endpoints ====
  http.get(`${API_BASE_URL}/similarities`, ({ request }) => {
    return HttpResponse.json({
      data: [],
      total_count: 0,
      page: 1,
      page_size: 50,
      total_pages: 0,
      has_next: false,
      has_previous: false,
    })
  }),

  // ==== System endpoints ====
  http.get(`${API_BASE_URL}/version`, () => {
    return HttpResponse.json({
      data: { version: '1.0.0', build: 'test' },
    } as DataResponse<Record<string, string>>)
  }),
]
