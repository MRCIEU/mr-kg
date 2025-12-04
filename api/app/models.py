"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel


# ==== Study models ====


class StudyBrief(BaseModel):
    """Brief study information for list views."""

    pmid: str
    title: str
    pub_date: str | None
    journal: str | None
    model: str


class StudiesResponse(BaseModel):
    """Response for study list endpoint."""

    total: int
    limit: int
    offset: int
    studies: list[StudyBrief]


# ==== Extraction models ====


class TraitInfo(BaseModel):
    """Trait information from extraction."""

    trait_index: int
    trait_label: str
    trait_id_in_result: str | None


class ResultInfo(BaseModel):
    """Individual result from extraction."""

    exposure: str
    outcome: str
    beta: float | None
    odds_ratio: float | None
    hazard_ratio: float | None
    ci_lower: float | None
    ci_upper: float | None
    p_value: float | None
    direction: str | None


class ExtractionResponse(BaseModel):
    """Response for extraction endpoint."""

    pmid: str
    model: str
    title: str
    pub_date: str | None
    journal: str | None
    abstract: str | None
    traits: list[TraitInfo]
    results: list[ResultInfo]
    metadata: dict


# ==== Trait similarity models ====


class SimilarStudyTrait(BaseModel):
    """Similar study by trait profile."""

    pmid: str
    title: str
    trait_profile_similarity: float
    trait_jaccard_similarity: float
    trait_count: int


class TraitSimilarityResponse(BaseModel):
    """Response for trait similarity endpoint."""

    query_pmid: str
    query_model: str
    query_title: str
    query_trait_count: int
    similar_studies: list[SimilarStudyTrait]


# ==== Evidence similarity models ====


class SimilarStudyEvidence(BaseModel):
    """Similar study by evidence profile."""

    pmid: str
    title: str
    direction_concordance: float
    matched_pairs: int
    match_type_exact: bool
    match_type_fuzzy: bool
    match_type_efo: bool


class EvidenceSimilarityResponse(BaseModel):
    """Response for evidence similarity endpoint."""

    query_pmid: str
    query_model: str
    query_title: str
    query_result_count: int
    similar_studies: list[SimilarStudyEvidence]


# ==== Health and statistics models ====


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str
    databases: dict[str, bool]


class StatisticsResponse(BaseModel):
    """Response for statistics endpoint."""

    overall: dict
    model_similarity_stats: list[dict]
    model_evidence_stats: list[dict]


# ==== Autocomplete models ====


class TraitAutocompleteResponse(BaseModel):
    """Response for trait autocomplete."""

    traits: list[str]


class StudyAutocompleteResponse(BaseModel):
    """Response for study autocomplete."""

    studies: list[dict]
