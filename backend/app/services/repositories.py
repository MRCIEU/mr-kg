"""Repository pattern implementation for database access."""

import logging
from abc import ABC

import duckdb

from app.models.database import (
    EFOEmbedding,
    ModelResult,
    ModelResultTrait,
    MRPubmedData,
    QueryCombination,
    SimilaritySearchResult,
    TraitEmbedding,
    TraitSimilarity,
)

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """Base repository class with common database operations."""

    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def execute_query(
        self, query: str, params: tuple | None = None
    ) -> list[tuple]:
        """Execute a query and return all results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of tuples containing query results
        """
        try:
            if params:
                result = self.connection.execute(query, params)
            else:
                result = self.connection.execute(query)
            return result.fetchall()
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            raise

    def execute_one(
        self, query: str, params: tuple | None = None
    ) -> tuple | None:
        """Execute a query and return one result.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Single tuple result or None
        """
        try:
            if params:
                result = self.connection.execute(query, params)
            else:
                result = self.connection.execute(query)
            return result.fetchone()
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            raise

    def get_count(
        self, table: str, where_clause: str = "", params: tuple | None = None
    ) -> int:
        """Get count of rows in a table with optional where clause.

        Args:
            table: Table name
            where_clause: Optional WHERE clause
            params: Query parameters

        Returns:
            Number of rows
        """
        query = f"SELECT COUNT(*) FROM {table}"
        if where_clause:
            query += f" WHERE {where_clause}"

        result = self.execute_one(query, params)
        return result[0] if result else 0


class TraitRepository(BaseRepository):
    """Repository for trait-related database operations."""

    def get_trait_by_index(self, trait_index: int) -> TraitEmbedding | None:
        """Get trait by its index.

        Args:
            trait_index: Canonical trait index

        Returns:
            TraitEmbedding instance or None
        """
        query = """
        SELECT trait_index, trait_label, vector
        FROM trait_embeddings
        WHERE trait_index = ?
        """

        result = self.execute_one(query, (trait_index,))
        if not result:
            return None

        return TraitEmbedding(
            trait_index=result[0],
            trait_label=result[1],
            vector=result[2] if result[2] else None,
        )

    def search_traits(
        self, query: str, limit: int = 50, offset: int = 0
    ) -> list[TraitEmbedding]:
        """Search traits by label text.

        Args:
            query: Search query string
            limit: Maximum number of results
            offset: Results offset for pagination

        Returns:
            List of matching TraitEmbedding instances
        """
        sql_query = """
        SELECT trait_index, trait_label, vector
        FROM trait_embeddings
        WHERE trait_label ILIKE ?
        ORDER BY trait_label
        LIMIT ? OFFSET ?
        """

        search_pattern = f"%{query}%"
        results = self.execute_query(sql_query, (search_pattern, limit, offset))

        return [
            TraitEmbedding(
                trait_index=row[0],
                trait_label=row[1],
                vector=row[2] if row[2] else None,
            )
            for row in results
        ]

    def get_traits_by_indices(
        self, trait_indices: list[int]
    ) -> list[TraitEmbedding]:
        """Get multiple traits by their indices.

        Args:
            trait_indices: List of trait indices

        Returns:
            List of TraitEmbedding instances
        """
        if not trait_indices:
            return []

        placeholders = ",".join("?" * len(trait_indices))
        query = f"""
        SELECT trait_index, trait_label, vector
        FROM trait_embeddings
        WHERE trait_index IN ({placeholders})
        ORDER BY trait_label
        """

        results = self.execute_query(query, tuple(trait_indices))

        return [
            TraitEmbedding(
                trait_index=row[0],
                trait_label=row[1],
                vector=row[2] if row[2] else None,
            )
            for row in results
        ]

    def find_similar_traits(
        self, trait_index: int, top_k: int = 10, threshold: float = 0.5
    ) -> list[SimilaritySearchResult]:
        """Find similar traits using vector similarity.

        Args:
            trait_index: Query trait index
            top_k: Number of top results
            threshold: Minimum similarity threshold

        Returns:
            List of SimilaritySearchResult instances
        """
        query = """
        SELECT
            query_id,
            query_label,
            result_id,
            result_label,
            similarity
        FROM trait_similarity_search
        WHERE query_id = ? AND similarity >= ?
        ORDER BY similarity DESC
        LIMIT ?
        """

        results = self.execute_query(query, (trait_index, threshold, top_k))

        return [
            SimilaritySearchResult(
                query_id=str(row[0]),
                query_label=row[1],
                result_id=str(row[2]),
                result_label=row[3],
                similarity=row[4],
            )
            for row in results
        ]

    def get_trait_study_count(self, trait_index: int) -> int:
        """Get number of studies mentioning a trait.

        Args:
            trait_index: Trait index

        Returns:
            Number of studies
        """
        return self.get_count(
            "model_result_traits", "trait_index = ?", (trait_index,)
        )


class StudyRepository(BaseRepository):
    """Repository for study-related database operations."""

    def get_study_by_id(self, study_id: int) -> ModelResult | None:
        """Get study by its ID.

        Args:
            study_id: Model result ID

        Returns:
            ModelResult instance or None
        """
        query = """
        SELECT id, model, pmid, metadata, results
        FROM model_results
        WHERE id = ?
        """

        result = self.execute_one(query, (study_id,))
        if not result:
            return None

        return ModelResult(
            id=result[0],
            model=result[1],
            pmid=result[2],
            metadata=result[3],
            results=result[4],
        )

    def get_studies_by_pmid(self, pmid: str) -> list[ModelResult]:
        """Get all studies for a specific PMID.

        Args:
            pmid: PubMed ID

        Returns:
            List of ModelResult instances
        """
        query = """
        SELECT id, model, pmid, metadata, results
        FROM model_results
        WHERE pmid = ?
        ORDER BY model
        """

        results = self.execute_query(query, (pmid,))

        return [
            ModelResult(
                id=row[0],
                model=row[1],
                pmid=row[2],
                metadata=row[3],
                results=row[4],
            )
            for row in results
        ]

    def search_studies(
        self,
        query: str = "",
        models: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ModelResult]:
        """Search studies with optional filters.

        Args:
            query: Text search query
            models: Filter by model names
            limit: Maximum number of results
            offset: Results offset for pagination

        Returns:
            List of ModelResult instances
        """
        where_conditions = []
        params = []

        if query:
            where_conditions.append("(pmid ILIKE ? OR metadata::TEXT ILIKE ?)")
            search_pattern = f"%{query}%"
            params.extend([search_pattern, search_pattern])

        if models:
            model_placeholders = ",".join("?" * len(models))
            where_conditions.append(f"model IN ({model_placeholders})")
            params.extend(models)

        where_clause = (
            " AND ".join(where_conditions) if where_conditions else "1=1"
        )

        sql_query = f"""
        SELECT id, model, pmid, metadata, results
        FROM model_results
        WHERE {where_clause}
        ORDER BY pmid, model
        LIMIT ? OFFSET ?
        """

        params.extend([limit, offset])
        results = self.execute_query(sql_query, tuple(params))

        return [
            ModelResult(
                id=row[0],
                model=row[1],
                pmid=row[2],
                metadata=row[3],
                results=row[4],
            )
            for row in results
        ]

    def get_study_traits(self, study_id: int) -> list[ModelResultTrait]:
        """Get traits associated with a study.

        Args:
            study_id: Model result ID

        Returns:
            List of ModelResultTrait instances
        """
        query = """
        SELECT id, model_result_id, trait_index, trait_label, trait_id_in_result
        FROM model_result_traits
        WHERE model_result_id = ?
        ORDER BY trait_label
        """

        results = self.execute_query(query, (study_id,))

        return [
            ModelResultTrait(
                id=row[0],
                model_result_id=row[1],
                trait_index=row[2],
                trait_label=row[3],
                trait_id_in_result=row[4],
            )
            for row in results
        ]

    def get_pubmed_data(self, pmid: str) -> MRPubmedData | None:
        """Get PubMed metadata for a paper.

        Args:
            pmid: PubMed ID

        Returns:
            MRPubmedData instance or None
        """
        query = """
        SELECT pmid, title, abstract, pub_date, journal, journal_issn, author_affil
        FROM mr_pubmed_data
        WHERE pmid = ?
        """

        result = self.execute_one(query, (pmid,))
        if not result:
            return None

        return MRPubmedData(
            pmid=result[0],
            title=result[1],
            abstract=result[2],
            pub_date=result[3],
            journal=result[4],
            journal_issn=result[5],
            author_affil=result[6],
        )


class EFORepository(BaseRepository):
    """Repository for EFO (Experimental Factor Ontology) operations."""

    def get_efo_by_id(self, efo_id: str) -> EFOEmbedding | None:
        """Get EFO term by its ID.

        Args:
            efo_id: EFO ontology identifier

        Returns:
            EFOEmbedding instance or None
        """
        query = """
        SELECT id, label, vector
        FROM efo_embeddings
        WHERE id = ?
        """

        result = self.execute_one(query, (efo_id,))
        if not result:
            return None

        return EFOEmbedding(
            id=result[0],
            label=result[1],
            vector=result[2] if result[2] else None,
        )

    def search_efo_terms(
        self, query: str, limit: int = 50, offset: int = 0
    ) -> list[EFOEmbedding]:
        """Search EFO terms by label text.

        Args:
            query: Search query string
            limit: Maximum number of results
            offset: Results offset for pagination

        Returns:
            List of matching EFOEmbedding instances
        """
        sql_query = """
        SELECT id, label, vector
        FROM efo_embeddings
        WHERE label ILIKE ?
        ORDER BY label
        LIMIT ? OFFSET ?
        """

        search_pattern = f"%{query}%"
        results = self.execute_query(sql_query, (search_pattern, limit, offset))

        return [
            EFOEmbedding(
                id=row[0], label=row[1], vector=row[2] if row[2] else None
            )
            for row in results
        ]

    def find_trait_efo_mappings(
        self, trait_index: int, top_k: int = 5, threshold: float = 0.3
    ) -> list[SimilaritySearchResult]:
        """Find EFO mappings for a trait.

        Args:
            trait_index: Trait index
            top_k: Number of top results
            threshold: Minimum similarity threshold

        Returns:
            List of SimilaritySearchResult instances
        """
        query = """
        SELECT
            trait_index,
            trait_label,
            efo_id,
            efo_label,
            similarity
        FROM trait_efo_similarity_search
        WHERE trait_index = ? AND similarity >= ?
        ORDER BY similarity DESC
        LIMIT ?
        """

        results = self.execute_query(query, (trait_index, threshold, top_k))

        return [
            SimilaritySearchResult(
                query_id=str(row[0]),
                query_label=row[1],
                result_id=row[2],
                result_label=row[3],
                similarity=row[4],
            )
            for row in results
        ]


class SimilarityRepository(BaseRepository):
    """Repository for trait profile similarity operations."""

    def get_combination_by_id(
        self, combination_id: int
    ) -> QueryCombination | None:
        """Get query combination by ID.

        Args:
            combination_id: Combination ID

        Returns:
            QueryCombination instance or None
        """
        query = """
        SELECT id, pmid, model, title, trait_count
        FROM query_combinations
        WHERE id = ?
        """

        result = self.execute_one(query, (combination_id,))
        if not result:
            return None

        return QueryCombination(
            id=result[0],
            pmid=result[1],
            model=result[2],
            title=result[3],
            trait_count=result[4],
        )

    def find_combination(
        self, pmid: str, model: str
    ) -> QueryCombination | None:
        """Find query combination by PMID and model.

        Args:
            pmid: PubMed ID
            model: Model name

        Returns:
            QueryCombination instance or None
        """
        query = """
        SELECT id, pmid, model, title, trait_count
        FROM query_combinations
        WHERE pmid = ? AND model = ?
        """

        result = self.execute_one(query, (pmid, model))
        if not result:
            return None

        return QueryCombination(
            id=result[0],
            pmid=result[1],
            model=result[2],
            title=result[3],
            trait_count=result[4],
        )

    def get_similarities(
        self,
        combination_id: int,
        top_k: int = 10,
        min_similarity: float = 0.0,
        similarity_type: str = "trait_profile",
    ) -> list[TraitSimilarity]:
        """Get similarities for a query combination.

        Args:
            combination_id: Query combination ID
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold
            similarity_type: Type of similarity to use ("trait_profile" or "jaccard")

        Returns:
            List of TraitSimilarity instances
        """
        # Choose the similarity column based on type
        similarity_column = (
            "trait_profile_similarity"
            if similarity_type == "trait_profile"
            else "trait_jaccard_similarity"
        )

        query = f"""
        SELECT
            id, query_combination_id, similar_pmid, similar_model, similar_title,
            trait_profile_similarity, trait_jaccard_similarity,
            query_trait_count, similar_trait_count
        FROM trait_similarities
        WHERE query_combination_id = ? AND {similarity_column} >= ?
        ORDER BY {similarity_column} DESC
        LIMIT ?
        """

        results = self.execute_query(
            query, (combination_id, min_similarity, top_k)
        )

        return [
            TraitSimilarity(
                id=row[0],
                query_combination_id=row[1],
                similar_pmid=row[2],
                similar_model=row[3],
                similar_title=row[4],
                trait_profile_similarity=row[5],
                trait_jaccard_similarity=row[6],
                query_trait_count=row[7],
                similar_trait_count=row[8],
            )
            for row in results
        ]

    def search_combinations(
        self,
        model: str | None = None,
        min_trait_count: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[QueryCombination]:
        """Search query combinations with filters.

        Args:
            model: Filter by model name
            min_trait_count: Filter by minimum trait count
            limit: Maximum number of results
            offset: Results offset for pagination

        Returns:
            List of QueryCombination instances
        """
        where_conditions = []
        params = []

        if model:
            where_conditions.append("model = ?")
            params.append(model)

        if min_trait_count is not None:
            where_conditions.append("trait_count >= ?")
            params.append(min_trait_count)

        where_clause = (
            " AND ".join(where_conditions) if where_conditions else "1=1"
        )

        sql_query = f"""
        SELECT id, pmid, model, title, trait_count
        FROM query_combinations
        WHERE {where_clause}
        ORDER BY pmid, model
        LIMIT ? OFFSET ?
        """

        params.extend([limit, offset])
        results = self.execute_query(sql_query, tuple(params))

        return [
            QueryCombination(
                id=row[0],
                pmid=row[1],
                model=row[2],
                title=row[3],
                trait_count=row[4],
            )
            for row in results
        ]
