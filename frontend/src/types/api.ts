/**
 * API response and data types for MR-KG application
 */

// ==== Base API Response Types ====

export interface DataResponse<T> {
  data: T
  message?: string
}

export interface PaginatedDataResponse<T> extends DataResponse<T> {
  total_count: number
  page: number
  page_size: number
  total_pages: number
  has_next: boolean
  has_previous: boolean
}

// ==== Trait Types ====

export interface TraitEmbedding {
  trait_index: number
  trait_label: string
  vector?: number[]
}

export interface TraitListItem {
  trait_index: number
  trait_label: string
  appearance_count: number
}

export interface TraitStats {
  trait_index: number
  trait_label: string
  appearance_count: number
  study_count: number
  model_distribution: Record<string, number>
  publication_years: number[]
}

export interface TraitDetailExtended {
  trait: TraitEmbedding
  statistics: TraitStats
  studies: StudyItem[]
  similar_traits: SimilaritySearchResult[]
  efo_mappings: SimilaritySearchResult[]
}

// ==== Study Types ====

export interface ModelResult {
  id: number
  model: string
  pmid: string
  metadata: Record<string, any>
  results: Record<string, any>
  created_at?: string
}

export interface MRPubmedData {
  pmid: string
  title: string
  abstract?: string
  pub_date?: string
  journal?: string
  journal_issn?: string
  author_affil?: string
}

export interface ModelResultTrait {
  id: number
  model_result_id: number
  trait_index: number
  trait_label: string
  trait_id_in_result: string
}

export interface StudyListItem {
  id: number
  model: string
  pmid: string
  title?: string
  journal?: string
  pub_date?: string
  trait_count: number
}

export interface StudyItem {
  id: number
  model: string
  pmid: string
  metadata: Record<string, any>
  results: Record<string, any>
  title?: string
  abstract?: string
  journal?: string
  pub_date?: string
  author_affil?: string
}

export interface StudyDetailExtended {
  study: ModelResult
  pubmed_data?: MRPubmedData
  traits: ModelResultTrait[]
  similar_studies: SimilaritySearchResult[]
  statistics: Record<string, any>
}

// ==== Similarity Types ====

export interface SimilaritySearchResult {
  index: number
  label: string
  similarity_score: number
  metadata?: Record<string, any>
}

export interface SimilarityPair {
  query_pmid: string
  query_title: string
  similar_pmid: string
  similar_title: string
  trait_profile_similarity: number
  trait_jaccard_similarity: number
  query_trait_count: number
  similar_trait_count: number
  model: string
}

// ==== Filter and Search Types ====

export interface PaginationParams {
  page: number
  page_size: number
}

export interface TraitSearchFilters {
  query?: string
  min_appearances?: number
  model?: string
}

export interface StudySearchFilters {
  model?: string
  journal?: string
  date_from?: string
  date_to?: string
  min_trait_count?: number
  trait_index?: number
}

export interface SimilaritySearchFilters {
  max_results: number
  min_similarity: number
}

// ==== Analytics Types ====

export interface TraitsOverview {
  total_traits: number
  total_appearances: number
  average_appearances: number
  top_traits: TraitListItem[]
  appearance_distribution: Record<string, number>
  model_coverage: Array<{
    model: string
    unique_traits: number
    total_mentions: number
  }>
}

export interface StudyAnalytics {
  total_studies: number
  total_pmids: number
  model_distribution: Record<string, number>
  journal_distribution: Record<string, number>
  year_distribution: Record<string, number>
  trait_count_distribution: Record<string, number>
}

// ==== UI State Types ====

export interface LoadingState {
  isLoading: boolean
  error?: string
}

export interface TableSortState {
  field: string
  direction: 'asc' | 'desc'
}

export interface FilterState {
  traits: TraitSearchFilters
  studies: StudySearchFilters
  similarities: SimilaritySearchFilters
}
