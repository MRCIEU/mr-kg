"""Database schema definitions for the trait profile similarity database.

This module defines the expected database schema for trait profile similarities.
Validation utilities are provided using the database_schema_utils classes.
"""

from .database_schema_utils import (
    ColumnType,
    ColumnDef,
    ForeignKeyDef,
    IndexDef,
    ViewDef,
    TableDef,
)


# Define the trait profile similarity database schema
#
# DATA FLOW OVERVIEW:
# 1. trait-profile-similarities.json contains similarity computations from compute-trait-similarity.py
# 2. query_combinations: Stores PMID-model pairs with their trait profiles
# 3. trait_similarities: Stores top-10 similarity relationships between combinations within same model
# 4. Views provide analysis capabilities for similarity patterns and model comparisons
#
# The build process:
# - Loads trait-profile-similarities.json containing query combinations and their top similarities
# - Creates query_combinations table with unique PMID-model pairs
# - Creates trait_similarities table linking combinations with similarity scores
# - Both semantic (embedding-based) and Jaccard (set-based) similarities are stored
# - Only intra-model comparisons are included (gpt-4 vs gpt-4, etc.)
TRAIT_PROFILE_SCHEMA = {
    # ==== query_combinations ====
    # Build process (create_query_combinations_table):
    # 1. Load trait-profile-similarities.json containing query records
    # 2. Create table with auto-incrementing id as PRIMARY KEY
    # 3. Extract: query_pmid, query_model, query_title, query_trait_count from each record
    # 4. Create UNIQUE constraint on (pmid, model) to prevent duplicates
    # 5. Insert all unique PMID-model combinations with their metadata
    # Result: Canonical reference table for all analyzed PMID-model combinations
    "query_combinations": TableDef(
        name="query_combinations",
        description="PMID-model combinations with trait profile metadata",
        columns=[
            ColumnDef(
                "id",
                ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
                # Auto-incrementing primary key. Uniquely identifies each PMID-model combination
                # and serves as foreign key reference in trait_similarities table.
            ),
            ColumnDef(
                "pmid",
                ColumnType.VARCHAR,
                nullable=False,
                # PubMed ID of the research paper. Links to the original publication
                # that was analyzed by the model for trait extraction.
            ),
            ColumnDef(
                "model",
                ColumnType.VARCHAR,
                nullable=False,
                # Name of the LLM model that analyzed this paper (e.g., "gpt-4", "deepseek-r1").
                # Used to ensure similarity comparisons stay within same model.
            ),
            ColumnDef(
                "title",
                ColumnType.VARCHAR,
                nullable=False,
                # Title of the research paper from PubMed. Used for display and
                # human-readable identification of paper content.
            ),
            ColumnDef(
                "trait_count",
                ColumnType.INTEGER,
                nullable=False,
                # Number of traits identified in this paper by this model.
                # Used for filtering and understanding model extraction patterns.
            ),
        ],
    ),
    # ==== trait_similarities ====
    # Build process (create_trait_similarities_table):
    # 1. For each query record, extract top_similarities list (max 10 entries)
    # 2. Create table with auto-incrementing id as PRIMARY KEY
    # 3. For each similarity in top_similarities:
    #    - Link to query_combination_id (foreign key to query_combinations.id)
    #    - Store similar_pmid, similar_model, similar_title from similarity record
    #    - Store trait_profile_similarity (semantic cosine similarity score)
    #    - Store trait_jaccard_similarity (set intersection/union similarity)
    #    - Store trait counts for both query and similar combinations
    # 4. All similarities are intra-model (same model comparisons only)
    # Result: Top-10 most similar combinations for each query, with dual similarity metrics
    "trait_similarities": TableDef(
        name="trait_similarities",
        description="Similarity relationships between PMID-model combinations within same model",
        columns=[
            ColumnDef(
                "id",
                ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
                # Auto-incrementing primary key. Uniquely identifies each similarity relationship
                # between two PMID-model combinations within the same model.
            ),
            ColumnDef(
                "query_combination_id",
                ColumnType.INTEGER,
                nullable=False,
                # Foreign key to query_combinations.id. References the combination that was
                # used as the query for similarity search. Links similarity back to source.
            ),
            ColumnDef(
                "similar_pmid",
                ColumnType.VARCHAR,
                nullable=False,
                # PubMed ID of the similar paper. Combined with similar_model, identifies
                # the target combination that was found to be similar to the query.
            ),
            ColumnDef(
                "similar_model",
                ColumnType.VARCHAR,
                nullable=False,
                # Model name for the similar combination. Always matches the query model
                # since similarities are computed only within the same model type.
            ),
            ColumnDef(
                "similar_title",
                ColumnType.VARCHAR,
                nullable=False,
                # Title of the similar paper. Stored for display and human-readable
                # identification without requiring joins to external data sources.
            ),
            ColumnDef(
                "trait_profile_similarity",
                ColumnType.DOUBLE,
                nullable=False,
                # Semantic similarity score (0.0 to 1.0) based on cosine similarity of trait embeddings.
                # Computed as average of maximum similarities between trait vectors from both combinations.
                # Higher values indicate more semantically related trait profiles.
            ),
            ColumnDef(
                "trait_jaccard_similarity",
                ColumnType.DOUBLE,
                nullable=False,
                # Jaccard similarity coefficient (0.0 to 1.0) based on trait index sets.
                # Computed as intersection over union of trait indices from both combinations.
                # Higher values indicate more overlapping trait sets regardless of semantics.
            ),
            ColumnDef(
                "query_trait_count",
                ColumnType.INTEGER,
                nullable=False,
                # Number of traits in the query combination. Stored for analysis of how
                # trait count affects similarity patterns and filtering by profile size.
            ),
            ColumnDef(
                "similar_trait_count",
                ColumnType.INTEGER,
                nullable=False,
                # Number of traits in the similar combination. Used to understand relationship
                # between trait count differences and similarity scores.
            ),
        ],
        foreign_keys=[
            ForeignKeyDef("query_combination_id", "query_combinations", "id"),
        ],
    ),
}

# Define expected indexes for query performance optimization
# Build process (create_indexes):
# Creates all indexes after table population for optimal performance
# Focuses on commonly queried columns for similarity analysis and model filtering
TRAIT_PROFILE_INDEXES = [
    # Query combinations indexes
    IndexDef(
        "idx_query_combinations_pmid",
        "query_combinations",
        ["pmid"],
        # Enables fast lookup by PubMed ID for finding specific paper analyses
        # Built after query_combinations table population with PMID extraction
    ),
    IndexDef(
        "idx_query_combinations_model",
        "query_combinations",
        ["model"],
        # Optimizes filtering by model type for model-specific analysis
        # Built after query_combinations table with model name extraction
    ),
    IndexDef(
        "idx_query_combinations_pmid_model",
        "query_combinations",
        ["pmid", "model"],
        # Compound index for unique constraint enforcement and fast lookup
        # Built after query_combinations table to support UNIQUE constraint
    ),
    # Trait similarities indexes
    IndexDef(
        "idx_trait_similarities_query_id",
        "trait_similarities",
        ["query_combination_id"],
        # Optimizes foreign key lookups and queries for all similarities of a specific combination
        # Built after trait_similarities table population with foreign key relationships
    ),
    IndexDef(
        "idx_trait_similarities_similar_pmid",
        "trait_similarities",
        ["similar_pmid"],
        # Enables reverse lookup to find what combinations are similar to a specific paper
        # Built after trait_similarities table with similar_pmid extraction
    ),
    IndexDef(
        "idx_trait_similarities_similar_model",
        "trait_similarities",
        ["similar_model"],
        # Supports model-specific similarity analysis and filtering
        # Built after trait_similarities table with model filtering validation
    ),
    IndexDef(
        "idx_trait_similarities_trait_profile_sim",
        "trait_similarities",
        ["trait_profile_similarity"],
        # Optimizes queries filtering by semantic similarity thresholds
        # Built after trait_similarities table with similarity score computation
    ),
    IndexDef(
        "idx_trait_similarities_jaccard_sim",
        "trait_similarities",
        ["trait_jaccard_similarity"],
        # Speeds up queries filtering by Jaccard similarity thresholds
        # Built after trait_similarities table with Jaccard score computation
    ),
]

# Define expected views for common similarity analysis operations
# Build process (create_views):
# Creates analytical views using JOINs and window functions for similarity analysis
# Views are built after all tables and indexes are created for optimal performance
TRAIT_PROFILE_VIEWS = [
    ViewDef(
        name="trait_similarity_analysis",
        # Build process: JOINs query_combinations with trait_similarities
        # Adds RANK() window function for similarity ranking within each query
        # Built after both tables are populated and foreign keys are established
        # Comprehensive analysis view combining query and similarity data.
        # Ranks similarities within each query combination for easy top-N retrieval.
        # Includes all metadata for both query and similar combinations.
        # Useful for detailed similarity analysis and comparison workflows.
        sql="""
        SELECT
            qc.pmid as query_pmid,
            qc.model as query_model,
            qc.title as query_title,
            qc.trait_count as query_trait_count,
            ts.similar_pmid,
            ts.similar_model,
            ts.similar_title,
            ts.similar_trait_count,
            ts.trait_profile_similarity,
            ts.trait_jaccard_similarity,
            RANK() OVER (
                PARTITION BY qc.id 
                ORDER BY ts.trait_profile_similarity DESC
            ) as similarity_rank
        FROM query_combinations qc
        JOIN trait_similarities ts ON qc.id = ts.query_combination_id
        ORDER BY qc.pmid, qc.model, ts.trait_profile_similarity DESC
        """,
    ),
    ViewDef(
        name="model_similarity_stats",
        # Build process: Aggregates query_combinations by model
        # Computes statistics about trait extraction patterns per model
        # Built after query_combinations table population with complete model data
        # Statistical summary of each model's performance and characteristics.
        # Shows trait extraction patterns, combination counts, and similarity pair totals.
        # Useful for comparing model capabilities and understanding data distribution.
        # Assumes each query has exactly 10 similarity pairs for total calculation.
        sql="""
        SELECT
            model,
            COUNT(*) as total_combinations,
            AVG(trait_count) as avg_trait_count,
            MIN(trait_count) as min_trait_count,
            MAX(trait_count) as max_trait_count,
            COUNT(*) * 10 as total_similarity_pairs
        FROM query_combinations
        GROUP BY model
        ORDER BY model
        """,
    ),
    ViewDef(
        name="top_similarity_pairs",
        # Build process: JOINs tables and filters by high semantic similarity (≥0.8)
        # Orders by model and similarity score for finding best matches
        # Built after trait_similarities table with computed similarity scores
        # High-quality similarity pairs filtered by semantic similarity threshold.
        # Focuses on combinations with strong trait profile relationships (≥0.8 similarity).
        # Includes both semantic and Jaccard similarities for dual-metric analysis.
        # Useful for finding the most related papers and validating similarity quality.
        sql="""
        SELECT
            ts.similar_model as model,
            qc.pmid as query_pmid,
            ts.similar_pmid,
            qc.title as query_title,
            ts.similar_title,
            ts.trait_profile_similarity,
            ts.trait_jaccard_similarity,
            qc.trait_count as query_trait_count,
            ts.similar_trait_count
        FROM trait_similarities ts
        JOIN query_combinations qc ON ts.query_combination_id = qc.id
        WHERE ts.trait_profile_similarity >= 0.8
        ORDER BY ts.similar_model, ts.trait_profile_similarity DESC
        """,
    ),
]
