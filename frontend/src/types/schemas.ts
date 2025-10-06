import { z } from 'zod'

export const DataResponseSchema = <T extends z.ZodTypeAny>(dataSchema: T) =>
  z.object({
    data: dataSchema,
    message: z.string().optional(),
  })

export const PaginatedDataResponseSchema = <T extends z.ZodTypeAny>(
  dataSchema: T
) =>
  z.object({
    data: dataSchema,
    message: z.string().optional(),
    total_count: z.number().int().nonnegative(),
    page: z.number().int().positive(),
    page_size: z.number().int().positive(),
    total_pages: z.number().int().nonnegative(),
    has_next: z.boolean(),
    has_previous: z.boolean(),
  })

export const TraitEmbeddingSchema = z.object({
  trait_index: z.number().int().nonnegative(),
  trait_label: z.string().min(1),
  vector: z.array(z.number()).optional(),
})

export const TraitListItemSchema = z.object({
  trait_index: z.number().int().nonnegative(),
  trait_label: z.string().min(1),
  appearance_count: z.number().int().nonnegative(),
})

export const TraitStatsSchema = z.object({
  trait_index: z.number().int().nonnegative(),
  trait_label: z.string().min(1),
  appearance_count: z.number().int().nonnegative(),
  study_count: z.number().int().nonnegative(),
  model_distribution: z.record(z.string(), z.number().int().nonnegative()),
  publication_years: z.array(z.number().int()),
})

export const ModelResultSchema = z.object({
  id: z.number().int().positive(),
  model: z.string().min(1),
  pmid: z.string().min(1),
  metadata: z.record(z.string(), z.any()),
  results: z.record(z.string(), z.any()),
  created_at: z.string().optional(),
})

export const MRPubmedDataSchema = z.object({
  pmid: z.string().min(1),
  title: z.string().min(1),
  abstract: z.string().optional(),
  pub_date: z.string().optional(),
  journal: z.string().optional(),
  journal_issn: z.string().optional(),
  author_affil: z.string().optional(),
})

export const ModelResultTraitSchema = z.object({
  id: z.number().int().positive(),
  model_result_id: z.number().int().positive(),
  trait_index: z.number().int().nonnegative(),
  trait_label: z.string().min(1),
  trait_id_in_result: z.string().min(1),
})

export const StudyListItemSchema = z.object({
  id: z.number().int().positive(),
  model: z.string().min(1),
  pmid: z.string().min(1),
  title: z.string().optional(),
  journal: z.string().optional(),
  pub_date: z.string().optional(),
  trait_count: z.number().int().nonnegative(),
})

export const StudyItemSchema = z.object({
  id: z.number().int().positive(),
  model: z.string().min(1),
  pmid: z.string().min(1),
  metadata: z.record(z.string(), z.any()),
  results: z.record(z.string(), z.any()),
  title: z.string().optional(),
  abstract: z.string().optional(),
  journal: z.string().optional(),
  pub_date: z.string().optional(),
  author_affil: z.string().optional(),
})

export const SimilaritySearchResultSchema = z.object({
  index: z.number().int().nonnegative(),
  label: z.string().min(1),
  similarity_score: z.number().min(0).max(1),
  metadata: z.record(z.string(), z.any()).optional(),
})

export const TraitDetailExtendedSchema = z.object({
  trait: TraitEmbeddingSchema,
  statistics: TraitStatsSchema,
  studies: z.array(StudyItemSchema),
  similar_traits: z.array(SimilaritySearchResultSchema),
  efo_mappings: z.array(SimilaritySearchResultSchema),
})

export const StudyDetailExtendedSchema = z.object({
  study: ModelResultSchema,
  pubmed_data: MRPubmedDataSchema.optional(),
  traits: z.array(ModelResultTraitSchema),
  similar_studies: z.array(SimilaritySearchResultSchema),
  statistics: z.record(z.string(), z.any()),
})

export const SimilarityPairSchema = z.object({
  query_pmid: z.string().min(1),
  query_title: z.string().min(1),
  similar_pmid: z.string().min(1),
  similar_title: z.string().min(1),
  trait_profile_similarity: z.number().min(0).max(1),
  trait_jaccard_similarity: z.number().min(0).max(1),
  query_trait_count: z.number().int().nonnegative(),
  similar_trait_count: z.number().int().nonnegative(),
  model: z.string().min(1),
})

export const TraitsOverviewSchema = z.object({
  total_traits: z.number().int().nonnegative(),
  total_appearances: z.number().int().nonnegative(),
  average_appearances: z.number().nonnegative(),
  top_traits: z.array(TraitListItemSchema),
  appearance_distribution: z.record(z.string(), z.number().int().nonnegative()),
  model_coverage: z.array(
    z.object({
      model: z.string().min(1),
      unique_traits: z.number().int().nonnegative(),
      total_mentions: z.number().int().nonnegative(),
    })
  ),
})

export const StudyAnalyticsSchema = z.object({
  total_studies: z.number().int().nonnegative(),
  total_pmids: z.number().int().nonnegative(),
  model_distribution: z.record(z.string(), z.number().int().nonnegative()),
  journal_distribution: z.record(z.string(), z.number().int().nonnegative()),
  year_distribution: z.record(z.string(), z.number().int().nonnegative()),
  trait_count_distribution: z.record(
    z.string(),
    z.number().int().nonnegative()
  ),
})

export const TraitsListResponseSchema = PaginatedDataResponseSchema(
  z.array(TraitListItemSchema)
)
export const TraitDetailResponseSchema = DataResponseSchema(
  TraitDetailExtendedSchema
)
export const StudiesListResponseSchema = PaginatedDataResponseSchema(
  z.array(StudyListItemSchema)
)
export const StudyDetailResponseSchema = DataResponseSchema(
  StudyDetailExtendedSchema
)
export const SimilaritiesListResponseSchema = PaginatedDataResponseSchema(
  z.array(SimilarityPairSchema)
)
export const TraitsOverviewResponseSchema =
  DataResponseSchema(TraitsOverviewSchema)
export const StudyAnalyticsResponseSchema =
  DataResponseSchema(StudyAnalyticsSchema)

export type TraitsListResponse = z.infer<typeof TraitsListResponseSchema>
export type TraitDetailResponse = z.infer<typeof TraitDetailResponseSchema>
export type StudiesListResponse = z.infer<typeof StudiesListResponseSchema>
export type StudyDetailResponse = z.infer<typeof StudyDetailResponseSchema>
export type SimilaritiesListResponse = z.infer<
  typeof SimilaritiesListResponseSchema
>
export type TraitsOverviewResponse = z.infer<
  typeof TraitsOverviewResponseSchema
>
export type StudyAnalyticsResponse = z.infer<
  typeof StudyAnalyticsResponseSchema
>
