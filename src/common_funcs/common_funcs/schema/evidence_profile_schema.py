"""Database schema definitions for the evidence profile similarity database.

This module defines the expected database schema for evidence profile similarities.
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


# Define the evidence profile similarity database schema
#
# DATA FLOW OVERVIEW:
# 1. evidence-similarities.json contains similarity computations from compute-evidence-similarity.py
# 2. query_combinations: Stores PMID-model pairs with their evidence profiles
# 3. evidence_similarities: Stores top-10 similarity relationships between combinations within same model
# 4. Views provide analysis capabilities for evidence concordance and comparison patterns
#
# The build process:
# - Loads evidence-similarities.json containing query combinations and their top similarities
# - Creates query_combinations table with unique PMID-model pairs
# - Creates evidence_similarities table linking combinations with similarity scores
# - Multiple similarity metrics stored: effect size, direction concordance, statistical consistency, evidence overlap
# - Two composite scores computed: equal-weighted and direction-prioritized
# - Only intra-model comparisons are included (gpt-4 vs gpt-4, etc.)
EVIDENCE_PROFILE_SCHEMA = {
    # ==== query_combinations ====
    # Build process (create_query_combinations_table):
    # 1. Load evidence-similarities.json containing query records
    # 2. Create table with auto-incrementing id as PRIMARY KEY
    # 3. Extract: query_pmid, query_model, query_title, query_result_count, complete_result_count, data_completeness from each record
    # 4. Create UNIQUE constraint on (pmid, model) to prevent duplicates
    # 5. Insert all unique PMID-model combinations with their metadata
    # Result: Canonical reference table for all analyzed PMID-model combinations with data quality metrics
    "query_combinations": TableDef(
        name="query_combinations",
        description="PMID-model combinations with evidence profile metadata and data quality metrics",
        columns=[
            ColumnDef(
                "id",
                ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
                # Auto-incrementing primary key. Uniquely identifies each PMID-model combination
                # and serves as foreign key reference in evidence_similarities table.
            ),
            ColumnDef(
                "pmid",
                ColumnType.VARCHAR,
                nullable=False,
                # PubMed ID of the research paper. Links to the original publication
                # that was analyzed by the model for result extraction.
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
                "result_count",
                ColumnType.INTEGER,
                nullable=False,
                # Total number of exposure-outcome results identified in this paper by this model.
                # Used for understanding model extraction patterns and filtering by evidence quantity.
            ),
            ColumnDef(
                "complete_result_count",
                ColumnType.INTEGER,
                nullable=False,
                # Number of results with complete effect size data (beta OR OR OR HR plus direction).
                # Used for assessing data quality and filtering studies suitable for similarity analysis.
            ),
            ColumnDef(
                "data_completeness",
                ColumnType.DOUBLE,
                nullable=False,
                # Proportion of results with complete data (complete_result_count / result_count).
                # Range 0.0-1.0. Higher values indicate better data quality. Used to filter
                # studies with insufficient quantitative information.
            ),
            ColumnDef(
                "publication_year",
                ColumnType.INTEGER,
                nullable=True,
                # Year of publication from PubMed metadata. Used for temporal stratification
                # analysis and flagging study pairs with large temporal gaps.
            ),
        ],
    ),
    # ==== evidence_similarities ====
    # Build process (create_evidence_similarities_table):
    # 1. For each query record, extract top_similarities list (max 10 entries)
    # 2. Create table with auto-incrementing id as PRIMARY KEY
    # 3. For each similarity in top_similarities:
    #    - Link to query_combination_id (foreign key to query_combinations.id)
    #    - Store similar_pmid, similar_model, similar_title from similarity record
    #    - Store matched_pairs (number of matched exposure-outcome pairs)
    #    - Store effect_size_similarity (Pearson correlation, nullable if <3 pairs)
    #    - Store direction_concordance (proportion of concordant directions, -1 to 1)
    #    - Store statistical_consistency (Cohen's kappa for significance, nullable)
    #    - Store evidence_overlap (Jaccard similarity of significant findings)
    #    - Store composite_similarity_equal (equal-weighted composite score)
    #    - Store composite_similarity_direction (direction-prioritized composite score)
    #    - Store result counts for both query and similar combinations
    # 4. All similarities are intra-model (same model comparisons only)
    # Result: Top-10 most similar combinations for each query, with multiple similarity metrics
    "evidence_similarities": TableDef(
        name="evidence_similarities",
        description="Similarity relationships between PMID-model combinations within same model based on quantitative causal evidence",
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
                # the target combination that was found to have similar evidence patterns.
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
                "matched_pairs",
                ColumnType.INTEGER,
                nullable=False,
                # Number of matched exposure-outcome pairs between the two studies.
                # Pairs matched by exact trait index correspondence. Minimum of 3 required
                # for meaningful similarity metrics. Higher values increase reliability.
            ),
            ColumnDef(
                "effect_size_similarity",
                ColumnType.DOUBLE,
                nullable=True,
                # Pearson correlation coefficient of harmonized effect sizes across matched pairs.
                # Range -1.0 to 1.0. Measures consistency of effect magnitudes. Nullable when
                # insufficient pairs (<3) for correlation computation. Effect sizes harmonized
                # to common log scale for OR and HR, beta kept as-is.
            ),
            ColumnDef(
                "direction_concordance",
                ColumnType.DOUBLE,
                nullable=False,
                # Proportion of matched pairs with concordant effect directions.
                # Range -1.0 (all discordant) to 1.0 (all concordant). Primary interpretable
                # metric for evidence consistency. Positive values indicate similar causal
                # directions, negative indicate contradictory findings.
            ),
            ColumnDef(
                "statistical_consistency",
                ColumnType.DOUBLE,
                nullable=True,
                # Cohen's kappa measuring agreement in statistical significance patterns.
                # Range -1.0 to 1.0. Assesses whether studies agree on which relationships
                # are statistically significant (p<0.05). Nullable when insufficient data
                # for kappa computation. Values >0.4 indicate moderate agreement.
            ),
            ColumnDef(
                "evidence_overlap",
                ColumnType.DOUBLE,
                nullable=False,
                # Jaccard similarity coefficient of statistically significant findings.
                # Range 0.0 to 1.0. Computed as intersection over union of significant
                # exposure-outcome pairs. Measures proportion of shared significant results.
                # Returns 0.0 when both studies have zero significant findings (not 1.0)
                # to avoid inflating similarity for underpowered studies.
            ),
            ColumnDef(
                "null_concordance",
                ColumnType.DOUBLE,
                nullable=False,
                # Proportion of matched pairs where both results are non-significant.
                # Range 0.0 to 1.0. Measures concordance in null findings. High values
                # may indicate shared power limitations or truly null relationships.
            ),
            ColumnDef(
                "effect_size_within_type",
                ColumnType.DOUBLE,
                nullable=True,
                # Pearson correlation for matched pairs with same effect type (beta-beta,
                # OR-OR, HR-HR). Range -1.0 to 1.0. More reliable than cross-type comparison.
                # Nullable when insufficient within-type pairs (<3).
            ),
            ColumnDef(
                "effect_size_cross_type",
                ColumnType.DOUBLE,
                nullable=True,
                # Pearson correlation for matched pairs with different effect types.
                # Range -1.0 to 1.0. Less reliable due to harmonization assumptions.
                # Nullable when insufficient cross-type pairs (<3).
            ),
            ColumnDef(
                "n_within_type_pairs",
                ColumnType.INTEGER,
                nullable=False,
                # Number of matched pairs with same effect type. Used for assessing
                # reliability of within-type similarity metric.
            ),
            ColumnDef(
                "n_cross_type_pairs",
                ColumnType.INTEGER,
                nullable=False,
                # Number of matched pairs with different effect types. Used for
                # sensitivity analysis of effect size harmonization.
            ),
            ColumnDef(
                "similar_publication_year",
                ColumnType.INTEGER,
                nullable=True,
                # Publication year of similar paper. Used for temporal gap analysis
                # and flagging study pairs with large time differences.
            ),
            ColumnDef(
                "query_completeness",
                ColumnType.DOUBLE,
                nullable=False,
                # Data completeness of query study (same as query_combinations.data_completeness).
                # Stored here for convenience in similarity analysis.
            ),
            ColumnDef(
                "similar_completeness",
                ColumnType.DOUBLE,
                nullable=False,
                # Data completeness of similar study. Used for quality-weighted similarity
                # and filtering comparisons involving low-quality data.
            ),
            ColumnDef(
                "composite_similarity_equal",
                ColumnType.DOUBLE,
                nullable=True,
                # Composite similarity with equal weighting and quality adjustment.
                # Computed as average of available normalized metrics, then multiplied by
                # min(query_completeness, similar_completeness). Range 0.0 to 1.0.
                # Nullable when insufficient non-null metrics (<2).
            ),
            ColumnDef(
                "composite_similarity_direction",
                ColumnType.DOUBLE,
                nullable=True,
                # Composite similarity prioritizing direction with quality adjustment.
                # Weights: 0.50*direction + 0.20*effect_size + 0.15*consistency + 0.15*overlap
                # (weights adjusted if metrics missing). Then multiplied by
                # min(query_completeness, similar_completeness). Range 0.0 to 1.0.
                # Nullable when insufficient non-null metrics (<2). Recommended for main analyses.
            ),
            ColumnDef(
                "query_result_count",
                ColumnType.INTEGER,
                nullable=False,
                # Total number of results in the query combination. Stored for analysis of how
                # evidence quantity affects similarity patterns and filtering by study size.
            ),
            ColumnDef(
                "similar_result_count",
                ColumnType.INTEGER,
                nullable=False,
                # Total number of results in the similar combination. Used to understand
                # relationship between result count differences and similarity scores.
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
# Focuses on commonly queried columns for evidence similarity analysis and model filtering
EVIDENCE_PROFILE_INDEXES = [
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
    # Evidence similarities indexes
    IndexDef(
        "idx_evidence_similarities_query_id",
        "evidence_similarities",
        ["query_combination_id"],
        # Optimizes foreign key lookups and queries for all similarities of a specific combination
        # Built after evidence_similarities table population with foreign key relationships
    ),
    IndexDef(
        "idx_evidence_similarities_similar_pmid",
        "evidence_similarities",
        ["similar_pmid"],
        # Enables reverse lookup to find what combinations have similar evidence to a specific paper
        # Built after evidence_similarities table with similar_pmid extraction
    ),
    IndexDef(
        "idx_evidence_similarities_similar_model",
        "evidence_similarities",
        ["similar_model"],
        # Supports model-specific similarity analysis and filtering
        # Built after evidence_similarities table with model filtering validation
    ),
    IndexDef(
        "idx_evidence_similarities_composite_equal",
        "evidence_similarities",
        ["composite_similarity_equal"],
        # Optimizes queries filtering by equal-weighted composite similarity thresholds
        # Built after evidence_similarities table with composite score computation
    ),
    IndexDef(
        "idx_evidence_similarities_composite_direction",
        "evidence_similarities",
        ["composite_similarity_direction"],
        # Speeds up queries filtering by direction-prioritized composite similarity
        # Built after evidence_similarities table with composite score computation
        # Primary index for main analysis queries
    ),
    IndexDef(
        "idx_evidence_similarities_direction_concordance",
        "evidence_similarities",
        ["direction_concordance"],
        # Optimizes queries filtering by directional concordance thresholds
        # Built after evidence_similarities table with concordance computation
        # Used for finding highly concordant or discordant study pairs
    ),
]

# Define expected views for common evidence similarity analysis operations
# Build process (create_views):
# Creates analytical views using JOINs and window functions for similarity analysis
# Views are built after all tables and indexes are created for optimal performance
EVIDENCE_PROFILE_VIEWS = [
    ViewDef(
        name="evidence_similarity_analysis",
        # Build process: JOINs query_combinations with evidence_similarities
        # Adds RANK() window function for similarity ranking within each query
        # Built after both tables are populated and foreign keys are established
        # Comprehensive analysis view combining query and similarity data.
        # Ranks similarities within each query combination by direction-prioritized score.
        # Includes all metadata and metrics for both query and similar combinations.
        # Useful for detailed evidence consistency analysis and comparison workflows.
        sql="""
        SELECT
            qc.pmid as query_pmid,
            qc.model as query_model,
            qc.title as query_title,
            qc.result_count as query_result_count,
            qc.data_completeness as query_completeness,
            es.similar_pmid,
            es.similar_model,
            es.similar_title,
            es.similar_result_count,
            es.matched_pairs,
            es.effect_size_similarity,
            es.direction_concordance,
            es.statistical_consistency,
            es.evidence_overlap,
            es.composite_similarity_equal,
            es.composite_similarity_direction,
            RANK() OVER (
                PARTITION BY qc.id 
                ORDER BY es.composite_similarity_direction DESC
            ) as similarity_rank
        FROM query_combinations qc
        JOIN evidence_similarities es ON qc.id = es.query_combination_id
        ORDER BY qc.pmid, qc.model, es.composite_similarity_direction DESC
        """,
    ),
    ViewDef(
        name="model_evidence_stats",
        # Build process: Aggregates query_combinations by model
        # Computes statistics about result extraction patterns and data quality per model
        # Built after query_combinations table population with complete model data
        # Statistical summary of each model's performance and characteristics.
        # Shows result extraction patterns, data completeness, and similarity pair totals.
        # Useful for comparing model capabilities and understanding data distribution.
        # Assumes each query has exactly 10 similarity pairs for total calculation.
        sql="""
        SELECT
            model,
            COUNT(*) as total_combinations,
            AVG(result_count) as avg_result_count,
            AVG(data_completeness) as avg_completeness,
            MIN(result_count) as min_result_count,
            MAX(result_count) as max_result_count,
            COUNT(*) * 10 as total_similarity_pairs
        FROM query_combinations
        GROUP BY model
        ORDER BY model
        """,
    ),
    ViewDef(
        name="high_concordance_pairs",
        # Build process: JOINs tables and filters by high directional concordance (>=0.8)
        # Orders by model and concordance for finding highly consistent evidence
        # Built after evidence_similarities table with computed concordance scores
        # High-quality similarity pairs filtered by directional concordance threshold.
        # Focuses on study pairs with strong agreement in causal directions (>=0.8).
        # Includes all similarity metrics for comprehensive evidence consistency assessment.
        # Useful for finding studies with reproducible findings and validating results.
        sql="""
        SELECT
            es.similar_model as model,
            qc.pmid as query_pmid,
            es.similar_pmid,
            qc.title as query_title,
            es.similar_title,
            es.direction_concordance,
            es.effect_size_similarity,
            es.evidence_overlap,
            es.matched_pairs,
            qc.result_count as query_result_count,
            es.similar_result_count
        FROM evidence_similarities es
        JOIN query_combinations qc ON es.query_combination_id = qc.id
        WHERE es.direction_concordance >= 0.8
        ORDER BY es.similar_model, es.direction_concordance DESC
        """,
    ),
    ViewDef(
        name="discordant_evidence_pairs",
        # Build process: JOINs tables and filters by negative directional concordance
        # Orders by model and ascending concordance to show most discordant pairs first
        # Built after evidence_similarities table with computed concordance scores
        # Study pairs with contradictory evidence showing opposite causal directions.
        # Filters for direction_concordance < 0, indicating predominantly discordant results.
        # Critical for identifying conflicting findings that require investigation.
        # Useful for detecting potential heterogeneity, bias, or context-specific effects.
        sql="""
        SELECT
            es.similar_model as model,
            qc.pmid as query_pmid,
            es.similar_pmid,
            qc.title as query_title,
            es.similar_title,
            es.direction_concordance,
            es.matched_pairs,
            es.evidence_overlap,
            qc.result_count as query_result_count,
            es.similar_result_count
        FROM evidence_similarities es
        JOIN query_combinations qc ON es.query_combination_id = qc.id
        WHERE es.direction_concordance < 0
        ORDER BY es.similar_model, es.direction_concordance ASC
        """,
    ),
]
