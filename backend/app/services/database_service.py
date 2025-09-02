"""High-level database services that abstract repository operations."""

import logging
from typing import List, Optional, Tuple

import duckdb

from app.models.database import (
    TraitEmbedding,
    EFOEmbedding,
    ModelResult,
    StudyDetailResponse,
    TraitDetailResponse,
    SimilarityAnalysisResponse,
    TraitSearchResponse,
    StudySearchResponse,
    SimilaritySearchResult,
    TraitSearchFilters,
    StudySearchFilters,
    SimilaritySearchFilters,
    PaginationParams,
)
from app.services.repositories import (
    TraitRepository,
    StudyRepository,
    EFORepository,
    SimilarityRepository,
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """High-level database service providing business logic operations."""

    def __init__(
        self,
        vector_store_conn: duckdb.DuckDBPyConnection,
        trait_profile_conn: duckdb.DuckDBPyConnection,
    ):
        self.vector_store_conn = vector_store_conn
        self.trait_profile_conn = trait_profile_conn

        # Initialize repositories
        self.trait_repo = TraitRepository(vector_store_conn)
        self.study_repo = StudyRepository(vector_store_conn)
        self.efo_repo = EFORepository(vector_store_conn)
        self.similarity_repo = SimilarityRepository(trait_profile_conn)

    # Add missing convenience methods
    async def get_trait_count(self) -> int:
        """Get total number of traits in the database."""
        try:
            return self.trait_repo.get_count("trait_embeddings")
        except Exception as e:
            logger.error(f"Error getting trait count: {e}")
            raise

    async def get_study_count(self) -> int:
        """Get total number of studies in the database."""
        try:
            return self.study_repo.get_count("model_results")
        except Exception as e:
            logger.error(f"Error getting study count: {e}")
            raise


class TraitService(DatabaseService):
    """Service for trait-related operations."""

    async def search_traits(
        self, filters: TraitSearchFilters, pagination: PaginationParams
    ) -> TraitSearchResponse:
        """Search traits with filters and pagination.

        Args:
            filters: Search filters
            pagination: Pagination parameters

        Returns:
            TraitSearchResponse with results and metadata
        """
        try:
            # Get total count for pagination
            if filters.query:
                total_count = self.trait_repo.get_count(
                    "trait_embeddings",
                    "trait_label ILIKE ?",
                    (f"%{filters.query}%",),
                )
            else:
                total_count = self.trait_repo.get_count("trait_embeddings")

            # Get traits
            if filters.query:
                traits = self.trait_repo.search_traits(
                    filters.query,
                    limit=pagination.page_size,
                    offset=pagination.offset,
                )
            else:
                # Get all traits with pagination
                query = """
                SELECT trait_index, trait_label, vector
                FROM trait_embeddings
                ORDER BY trait_label
                LIMIT ? OFFSET ?
                """
                results = self.trait_repo.execute_query(
                    query, (pagination.page_size, pagination.offset)
                )
                traits = [
                    TraitEmbedding(
                        trait_index=row[0],
                        trait_label=row[1],
                        vector=row[2] if row[2] else None,
                    )
                    for row in results
                ]

            return TraitSearchResponse(
                traits=traits,
                total_count=total_count,
                page=pagination.page,
                page_size=pagination.page_size,
            )

        except Exception as e:
            logger.error(f"Error searching traits: {e}")
            raise

    async def get_trait_details(
        self, trait_index: int
    ) -> Optional[TraitDetailResponse]:
        """Get detailed information about a trait.

        Args:
            trait_index: Trait index

        Returns:
            TraitDetailResponse with comprehensive trait information
        """
        try:
            # Get trait basic info
            trait = self.trait_repo.get_trait_by_index(trait_index)
            if not trait:
                return None

            # Get studies mentioning this trait
            study_query = """
            SELECT DISTINCT mr.id, mr.model, mr.pmid, mr.metadata, mr.results
            FROM model_results mr
            JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
            WHERE mrt.trait_index = ?
            ORDER BY mr.pmid, mr.model
            LIMIT 50
            """
            study_results = self.study_repo.execute_query(
                study_query, (trait_index,)
            )
            studies = [
                ModelResult(
                    id=row[0],
                    model=row[1],
                    pmid=row[2],
                    metadata=row[3],
                    results=row[4],
                )
                for row in study_results
            ]

            # Get similar traits
            similar_traits = self.trait_repo.find_similar_traits(
                trait_index, top_k=10, threshold=0.3
            )

            # Get EFO mappings
            efo_mappings = self.efo_repo.find_trait_efo_mappings(
                trait_index, top_k=5, threshold=0.3
            )

            return TraitDetailResponse(
                trait=trait,
                studies=studies,
                similar_traits=similar_traits,
                efo_mappings=efo_mappings,
            )

        except Exception as e:
            logger.error(f"Error getting trait details for {trait_index}: {e}")
            raise

    async def find_similar_traits(
        self, trait_index: int, filters: SimilaritySearchFilters
    ) -> List[SimilaritySearchResult]:
        """Find traits similar to the given trait.

        Args:
            trait_index: Query trait index
            filters: Similarity search filters

        Returns:
            List of similar traits
        """
        try:
            return self.trait_repo.find_similar_traits(
                trait_index,
                top_k=filters.max_results or 10,
                threshold=filters.min_similarity or 0.3,
            )
        except Exception as e:
            logger.error(f"Error finding similar traits for {trait_index}: {e}")
            raise


class StudyService(DatabaseService):
    """Service for study-related operations."""

    async def search_studies(
        self, filters: StudySearchFilters, pagination: PaginationParams
    ) -> StudySearchResponse:
        """Search studies with filters and pagination.

        Args:
            filters: Search filters
            pagination: Pagination parameters

        Returns:
            StudySearchResponse with results and metadata
        """
        try:
            # Build total count query
            count_where_conditions = []
            count_params = []

            if filters.query:
                count_where_conditions.append(
                    "(pmid ILIKE ? OR metadata::TEXT ILIKE ?)"
                )
                search_pattern = f"%{filters.query}%"
                count_params.extend([search_pattern, search_pattern])

            if filters.models:
                model_placeholders = ",".join("?" * len(filters.models))
                count_where_conditions.append(
                    f"model IN ({model_placeholders})"
                )
                count_params.extend(filters.models)

            # Add trait filter if specified
            if filters.trait_indices:
                count_where_conditions.append(
                    """
                id IN (
                    SELECT DISTINCT model_result_id 
                    FROM model_result_traits 
                    WHERE trait_index IN ({})
                )
                """.format(",".join("?" * len(filters.trait_indices)))
                )
                count_params.extend(filters.trait_indices)

            count_where_clause = (
                " AND ".join(count_where_conditions)
                if count_where_conditions
                else "1=1"
            )
            total_count = self.study_repo.get_count(
                "model_results", count_where_clause, tuple(count_params)
            )

            # Get studies with same filters
            studies = self.study_repo.search_studies(
                query=filters.query or "",
                models=filters.models,
                limit=pagination.page_size,
                offset=pagination.offset,
            )

            # Apply trait filter if needed (this is a simplified approach)
            if filters.trait_indices:
                # Filter studies that have the specified traits
                filtered_studies = []
                for study in studies:
                    study_traits = self.study_repo.get_study_traits(study.id)
                    study_trait_indices = {
                        trait.trait_index for trait in study_traits
                    }
                    if any(
                        trait_idx in study_trait_indices
                        for trait_idx in filters.trait_indices
                    ):
                        filtered_studies.append(study)
                studies = filtered_studies

            return StudySearchResponse(
                studies=studies,
                total_count=total_count,
                page=pagination.page,
                page_size=pagination.page_size,
            )

        except Exception as e:
            logger.error(f"Error searching studies: {e}")
            raise

    async def get_study_details(
        self, study_id: int
    ) -> Optional[StudyDetailResponse]:
        """Get detailed information about a study.

        Args:
            study_id: Model result ID

        Returns:
            StudyDetailResponse with comprehensive study information
        """
        try:
            # Get study basic info
            study = self.study_repo.get_study_by_id(study_id)
            if not study:
                return None

            # Get PubMed metadata
            pubmed_data = self.study_repo.get_pubmed_data(study.pmid)

            # Get associated traits
            traits = self.study_repo.get_study_traits(study_id)

            # Get similar studies from trait profile database
            combination = self.similarity_repo.find_combination(
                study.pmid, study.model
            )
            similar_studies = []
            if combination:
                similar_studies = self.similarity_repo.get_similarities(
                    combination.id, top_k=10, min_similarity=0.3
                )

            return StudyDetailResponse(
                study=study,
                pubmed_data=pubmed_data,
                traits=traits,
                similar_studies=similar_studies,
            )

        except Exception as e:
            logger.error(f"Error getting study details for {study_id}: {e}")
            raise

    async def get_studies_by_pmid(self, pmid: str) -> List[ModelResult]:
        """Get all studies for a specific PMID.

        Args:
            pmid: PubMed ID

        Returns:
            List of studies from different models
        """
        try:
            return self.study_repo.get_studies_by_pmid(pmid)
        except Exception as e:
            logger.error(f"Error getting studies for PMID {pmid}: {e}")
            raise


class SimilarityService(DatabaseService):
    """Service for similarity analysis operations."""

    async def analyze_similarity(
        self, pmid: str, model: str
    ) -> Optional[SimilarityAnalysisResponse]:
        """Analyze similarity for a PMID-model combination.

        Args:
            pmid: PubMed ID
            model: Model name

        Returns:
            SimilarityAnalysisResponse with similarity analysis
        """
        try:
            # Find the query combination
            combination = self.similarity_repo.find_combination(pmid, model)
            if not combination:
                return None

            # Get similarities
            similarities = self.similarity_repo.get_similarities(
                combination.id, top_k=10, min_similarity=0.0
            )

            return SimilarityAnalysisResponse(
                query_combination=combination,
                similarities=similarities,
                total_count=len(similarities),
            )

        except Exception as e:
            logger.error(f"Error analyzing similarity for {pmid}, {model}: {e}")
            raise

    async def search_combinations(
        self, filters: SimilaritySearchFilters, pagination: PaginationParams
    ) -> List:
        """Search query combinations with filters.

        Args:
            filters: Search filters
            pagination: Pagination parameters

        Returns:
            List of query combinations
        """
        try:
            return self.similarity_repo.search_combinations(
                model=filters.model,
                limit=pagination.page_size,
                offset=pagination.offset,
            )
        except Exception as e:
            logger.error(f"Error searching combinations: {e}")
            raise


class EFOService(DatabaseService):
    """Service for EFO (Experimental Factor Ontology) operations."""

    async def search_efo_terms(
        self, query: str, pagination: PaginationParams
    ) -> List[EFOEmbedding]:
        """Search EFO terms by label.

        Args:
            query: Search query
            pagination: Pagination parameters

        Returns:
            List of matching EFO terms
        """
        try:
            return self.efo_repo.search_efo_terms(
                query, limit=pagination.page_size, offset=pagination.offset
            )
        except Exception as e:
            logger.error(f"Error searching EFO terms: {e}")
            raise

    async def get_trait_efo_mappings(
        self, trait_index: int, top_k: int = 5
    ) -> List[SimilaritySearchResult]:
        """Get EFO mappings for a trait.

        Args:
            trait_index: Trait index
            top_k: Number of top mappings

        Returns:
            List of EFO mappings
        """
        try:
            return self.efo_repo.find_trait_efo_mappings(
                trait_index, top_k=top_k, threshold=0.3
            )
        except Exception as e:
            logger.error(
                f"Error getting EFO mappings for trait {trait_index}: {e}"
            )
            raise


class AnalyticsService(DatabaseService):
    """Service for analytics and summary operations."""

    async def get_database_summary(self) -> dict:
        """Get summary statistics about the database.

        Returns:
            Dictionary with database statistics
        """
        try:
            summary = {}

            # Vector store statistics
            summary["trait_count"] = self.trait_repo.get_count(
                "trait_embeddings"
            )
            summary["efo_count"] = self.efo_repo.get_count("efo_embeddings")
            summary["study_count"] = self.study_repo.get_count("model_results")
            summary["pubmed_count"] = self.study_repo.get_count(
                "mr_pubmed_data"
            )

            # Model statistics
            model_query = """
            SELECT model, COUNT(*) as count
            FROM model_results
            GROUP BY model
            ORDER BY count DESC
            """
            model_results = self.study_repo.execute_query(model_query)
            summary["models"] = {row[0]: row[1] for row in model_results}

            # Trait profile statistics
            summary["combination_count"] = self.similarity_repo.get_count(
                "query_combinations"
            )
            summary["similarity_count"] = self.similarity_repo.get_count(
                "trait_similarities"
            )

            # Top journals
            journal_query = """
            SELECT journal, COUNT(*) as count
            FROM mr_pubmed_data
            GROUP BY journal
            ORDER BY count DESC
            LIMIT 10
            """
            journal_results = self.study_repo.execute_query(journal_query)
            summary["top_journals"] = {
                row[0]: row[1] for row in journal_results
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting database summary: {e}")
            raise

    async def get_trait_statistics(self, trait_index: int) -> dict:
        """Get statistics for a specific trait.

        Args:
            trait_index: Trait index

        Returns:
            Dictionary with trait statistics
        """
        try:
            stats = {}

            # Basic counts
            stats["study_count"] = self.trait_repo.get_trait_study_count(
                trait_index
            )

            # Model distribution
            model_query = """
            SELECT mr.model, COUNT(*) as count
            FROM model_results mr
            JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
            WHERE mrt.trait_index = ?
            GROUP BY mr.model
            ORDER BY count DESC
            """
            model_results = self.trait_repo.execute_query(
                model_query, (trait_index,)
            )
            stats["model_distribution"] = {
                row[0]: row[1] for row in model_results
            }

            # Publication years
            year_query = """
            SELECT SUBSTR(mpd.pub_date, 1, 4) as year, COUNT(*) as count
            FROM mr_pubmed_data mpd
            JOIN model_results mr ON mpd.pmid = mr.pmid
            JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
            WHERE mrt.trait_index = ? AND LENGTH(mpd.pub_date) >= 4
            GROUP BY year
            ORDER BY year DESC
            """
            year_results = self.trait_repo.execute_query(
                year_query, (trait_index,)
            )
            stats["publication_years"] = {
                row[0]: row[1] for row in year_results
            }

            return stats

        except Exception as e:
            logger.error(
                f"Error getting trait statistics for {trait_index}: {e}"
            )
            raise
