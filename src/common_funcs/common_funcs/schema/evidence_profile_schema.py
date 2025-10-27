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
                # Total number of matched exposure-outcome pairs between the two studies.
                # Sum of exact + fuzzy + EFO matches. Minimum of 1 required for comparison.
                # Higher values increase reliability of similarity metrics.
            ),
            ColumnDef(
                "match_type_exact",
                ColumnType.BOOLEAN,
                nullable=False,
                # Boolean flag indicating presence of exact trait index matches.
                # TRUE if at least one pair matched by exact trait index correspondence.
                # Highest precision matching tier. Both exposure and outcome trait indices
                # must be identical. Most reliable match type for direct comparisons.
            ),
            ColumnDef(
                "match_type_fuzzy",
                ColumnType.BOOLEAN,
                nullable=False,
                # Boolean flag indicating presence of fuzzy semantic matches.
                # TRUE if at least one pair matched by semantic similarity (cosine >= 0.80).
                # Second tier matching using 200-dim trait embeddings from vector_store.db.
                # Captures semantic variants like "BMI" vs "body mass index" or "BMI" vs "obesity".
            ),
            ColumnDef(
                "match_type_efo",
                ColumnType.BOOLEAN,
                nullable=False,
                # Boolean flag indicating presence of EFO category matches.
                # TRUE if at least one pair matched by EFO category correspondence.
                # Third tier matching using ontology-based category mapping from trait_efo_similarity_search.
                # Enables category-level meta-analysis by matching traits mapped to same EFO terms.
            ),
            ColumnDef(
                "direction_concordance",
                ColumnType.DOUBLE,
                nullable=False,
                # Proportion of matched pairs with concordant effect directions.
                # Range -1.0 (all discordant) to 1.0 (all concordant). Primary interpretable
                # metric for evidence consistency. 100% data availability. Positive values
                # indicate similar causal directions, negative indicate contradictory findings.
            ),
            ColumnDef(
                "composite_similarity_equal",
                ColumnType.DOUBLE,
                nullable=True,
                # Composite similarity with equal weighting (no quality adjustment).
                # Computed as average of available normalized metrics: direction_concordance,
                # effect_size_similarity, statistical_consistency. Range 0.0 to 1.0.
                # Nullable when insufficient non-null metrics. 100% availability (always computed).
            ),
            ColumnDef(
                "composite_similarity_direction",
                ColumnType.DOUBLE,
                nullable=True,
                # Composite similarity prioritizing direction concordance.
                # Weighted combination of available metrics with direction_concordance weighted
                # more heavily. Range 0.0 to 1.0. Nullable when insufficient metrics.
                # 100% availability (always computed). Recommended for main analyses.
            ),
            ColumnDef(
                "effect_size_similarity",
                ColumnType.DOUBLE,
                nullable=True,
                # Pearson correlation coefficient of harmonized effect sizes across matched pairs.
                # Range -1.0 to 1.0. Measures consistency of effect magnitudes. Nullable when
                # insufficient pairs (<3) for correlation computation. Only 3.82% data availability
                # due to abstract-only extraction limiting complete effect size data.
            ),
            ColumnDef(
                "statistical_consistency",
                ColumnType.DOUBLE,
                nullable=True,
                # Cohen's kappa measuring agreement in statistical significance patterns.
                # Range -1.0 to 1.0. Assesses whether studies agree on which relationships
                # are statistically significant (p<0.05). Only 0.27% data availability due to
                # requiring >=3 matched pairs (82% of comparisons have only 1 pair).
            ),
            ColumnDef(
                "precision_concordance",
                ColumnType.DOUBLE,
                nullable=True,
                # Spearman correlation of confidence interval widths for matched pairs.
                # Range -1.0 to 1.0. Measures consistency of precision/uncertainty across studies.
                # Only 3.33% data availability due to compound limitation: requires complete CI data
                # (abstract extraction limit) AND sufficient matched pairs.
            ),
            ColumnDef(
                "similar_publication_year",
                ColumnType.INTEGER,
                nullable=True,
                # Publication year of similar paper. Used for temporal gap analysis
                # and flagging study pairs with large time differences.
            ),
        ],
        foreign_keys=[
            ForeignKeyDef("query_combination_id", "query_combinations", "id"),
        ],
    ),
}

# Define expected indexes for query performance optimization
# Build process (create_indexes):
# Creates 6 essential indexes after table population for optimal performance
# Removed: NULL-heavy composite indexes and redundant single-column indexes
# Focus: High-value query patterns for evidence similarity analysis
EVIDENCE_PROFILE_INDEXES = [
    # Compound index for primary lookup pattern (query combinations)
    IndexDef(
        "idx_query_combinations_pmid_model",
        "query_combinations",
        ["pmid", "model"],
        # Compound index for unique constraint enforcement and primary lookup pattern
        # Replaces separate pmid and model indexes with more efficient compound index
        # Built after query_combinations table to support UNIQUE constraint
    ),
    # Foreign key relationship
    IndexDef(
        "idx_evidence_similarities_query_id",
        "evidence_similarities",
        ["query_combination_id"],
        # Optimizes foreign key lookups and queries for all similarities of a specific combination
        # Built after evidence_similarities table population with foreign key relationships
    ),
    # Similar paper lookups
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
    # Primary metric for sorting (100% availability)
    IndexDef(
        "idx_evidence_similarities_direction_concordance",
        "evidence_similarities",
        ["direction_concordance"],
        # Primary index for sorting by directional concordance (100% data availability)
        # Optimizes queries filtering by concordance thresholds
        # Built after evidence_similarities table with concordance computation
        # Note: May fail in DuckDB due to ENUM type, but queries still work without it
    ),
    # Match quality filtering
    IndexDef(
        "idx_evidence_similarities_matched_pairs",
        "evidence_similarities",
        ["matched_pairs"],
        # Optimizes queries filtering by number of matched exposure-outcome pairs
        # Built after evidence_similarities table with pair counting
        # Used for quality control and filtering by match quantity
    ),
]

# Define expected views for common evidence similarity analysis operations
# Build process (create_views):
# Creates 5 analytical views using JOINs and window functions for similarity analysis
# Views are built after all tables and indexes are created for optimal performance
# Focus on reliable metrics with high data availability (direction_concordance, effect_size_similarity)
EVIDENCE_PROFILE_VIEWS = [
    ViewDef(
        name="evidence_similarity_analysis",
        # Build process: JOINs query_combinations with evidence_similarities
        # Adds RANK() window function for similarity ranking within each query
        # Built after both tables are populated and foreign keys are established
        # Comprehensive analysis view combining query and similarity data.
        # Ranks similarities within each query combination by direction concordance (100% availability).
        # Includes all metadata and available metrics for both query and similar combinations.
        # Useful for detailed evidence consistency analysis and comparison workflows.
        sql="""
        SELECT
            qc.pmid as query_pmid,
            qc.model as query_model,
            qc.title as query_title,
            qc.result_count as query_result_count,
            qc.data_completeness as query_completeness,
            qc.publication_year as query_publication_year,
            es.similar_pmid,
            es.similar_model,
            es.similar_title,
            es.similar_publication_year,
            es.matched_pairs,
            es.match_type_exact,
            es.match_type_fuzzy,
            es.match_type_efo,
            es.direction_concordance,
            es.effect_size_similarity,
            es.composite_similarity_equal,
            es.composite_similarity_direction,
            RANK() OVER (
                PARTITION BY qc.id 
                ORDER BY es.direction_concordance DESC
            ) as similarity_rank
        FROM query_combinations qc
        JOIN evidence_similarities es ON qc.id = es.query_combination_id
        ORDER BY qc.pmid, qc.model, es.direction_concordance DESC
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
        # Includes publication years for temporal gap analysis and filtering.
        # Useful for finding studies with reproducible findings and validating results.
        sql="""
        SELECT
            es.similar_model as model,
            qc.pmid as query_pmid,
            es.similar_pmid,
            qc.title as query_title,
            es.similar_title,
            qc.publication_year as query_publication_year,
            es.similar_publication_year,
            es.direction_concordance,
            es.effect_size_similarity,
            es.matched_pairs,
            es.match_type_exact,
            es.match_type_fuzzy,
            es.match_type_efo,
            qc.result_count as query_result_count,
            qc.data_completeness as query_completeness
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
            qc.publication_year as query_publication_year,
            es.similar_publication_year,
            es.direction_concordance,
            es.matched_pairs,
            es.match_type_exact,
            es.match_type_fuzzy,
            es.match_type_efo,
            qc.result_count as query_result_count,
            qc.data_completeness as query_completeness
        FROM evidence_similarities es
        JOIN query_combinations qc ON es.query_combination_id = qc.id
        WHERE es.direction_concordance < 0
        ORDER BY es.similar_model, es.direction_concordance ASC
        """,
    ),
    ViewDef(
        name="metric_availability",
        # Build process: Aggregates evidence_similarities to summarize metric availability
        # Computes counts and percentages for each similarity metric
        # Built after evidence_similarities table with all metrics populated
        # Data availability summary for each similarity metric across all comparisons.
        # Shows absolute counts and percentages of non-NULL values for each metric.
        # Critical for understanding which metrics are reliable for analysis.
        # Expected availability: direction_concordance 100%, effect_size_similarity 3.82%,
        # statistical_consistency 0.27%, precision_concordance 3.33%.
        sql="""
        SELECT
            COUNT(*) as total_comparisons,
            COUNT(direction_concordance) as direction_concordance_available,
            COUNT(effect_size_similarity) as effect_size_similarity_available,
            COUNT(composite_similarity_equal) as composite_equal_available,
            COUNT(composite_similarity_direction) as composite_direction_available,
            COUNT(statistical_consistency) as statistical_consistency_available,
            COUNT(precision_concordance) as precision_concordance_available,
            ROUND(COUNT(direction_concordance)::DOUBLE / COUNT(*) * 100, 2) 
                as direction_concordance_pct,
            ROUND(COUNT(effect_size_similarity)::DOUBLE / COUNT(*) * 100, 2) 
                as effect_size_similarity_pct,
            ROUND(COUNT(composite_similarity_equal)::DOUBLE / COUNT(*) * 100, 2) 
                as composite_equal_pct,
            ROUND(COUNT(composite_similarity_direction)::DOUBLE / COUNT(*) * 100, 2) 
                as composite_direction_pct,
            ROUND(COUNT(statistical_consistency)::DOUBLE / COUNT(*) * 100, 2) 
                as statistical_consistency_pct,
            ROUND(COUNT(precision_concordance)::DOUBLE / COUNT(*) * 100, 2) 
                as precision_concordance_pct
        FROM evidence_similarities
        """,
    ),
]
