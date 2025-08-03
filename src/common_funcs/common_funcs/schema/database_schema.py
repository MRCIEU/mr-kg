"""Database schema definitions for the vector store.

This module defines the expected database schema. Validation utilities
are provided in database_schema_utils.py.
"""

from .database_schema_utils import (
    ColumnType,
    ColumnDef,
    ForeignKeyDef,
    IndexDef,
    ViewDef,
    TableDef,
)


# Define the complete database schema
#
# DATA FLOW OVERVIEW:
# 1. unique_traits.csv provides canonical trait indices and labels
# 2. trait_embeddings: Stores embeddings for traits that have valid vector representations
# 3. efo_embeddings: Stores EFO ontology terms with embeddings for semantic mapping
# 4. model_results: Raw LLM outputs with metadata containing exposure/outcome trait references
# 5. model_result_traits: Links model outputs to canonical traits via trait_index validation
#
# The build process:
# - Loads unique_traits.csv as the authoritative trait index
# - Creates embeddings only for traits with valid vector representations
# - Processes model results and extracts trait references from metadata.exposures/outcomes
# - Validates trait references against available embeddings before creating linkages
# - Combines exposure and outcome traits without role distinction in model_result_traits
DATABASE_SCHEMA = {
    # ==== trait_embeddings ====
    # Build process (create_trait_embeddings_table):
    # 1. Load unique_traits.csv as authoritative source (trait_index -> trait_label mapping)
    # 2. Load trait embeddings from traits.json (trait_label -> vector mapping)
    # 3. Create table with trait_index as PRIMARY KEY from unique_traits
    # 4. For each unique trait, lookup its embedding by trait_label
    # 5. Insert (trait_index, trait_label, vector) if embedding exists
    # 6. Log missing embeddings for traits without vector representations
    # Result: Only traits with both unique_traits entry AND embedding vector are included
    "trait_embeddings": TableDef(
        name="trait_embeddings",
        description="Trait embeddings indexed by unique_traits.csv indices",
        columns=[
            ColumnDef(
                "trait_index",
                ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
                # From unique_traits.csv index column. Serves as canonical trait identifier
                # across all tables. Only traits with valid embeddings are included.
            ),
            ColumnDef(
                "trait_label",
                ColumnType.VARCHAR,
                nullable=False,
                # From unique_traits.csv trait column. Human-readable trait name used for
                # embedding lookup and display. Must match labels in traits.json exactly.
            ),
            ColumnDef(
                "vector",
                ColumnType.FLOAT_ARRAY_200,
                nullable=False,
                # 200-dimensional embedding vector from traits.json. Used for cosine similarity
                # calculations in trait-to-trait and trait-to-EFO similarity searches.
            ),
        ],
    ),
    # ==== efo_embeddings ====
    # Build process (create_efo_embeddings_table):
    # 1. Load EFO embeddings from efo.json (contains id, label, vector fields)
    # 2. Create table with EFO id as PRIMARY KEY
    # 3. Insert all EFO records directly (id, label, vector)
    # Result: Complete EFO ontology terms with embeddings for trait mapping
    "efo_embeddings": TableDef(
        name="efo_embeddings",
        description="EFO (Experimental Factor Ontology) term embeddings",
        columns=[
            ColumnDef(
                "id",
                ColumnType.VARCHAR,
                nullable=False,
                primary_key=True,
                # EFO ontology identifier (e.g., "EFO_0004340"). Loaded directly from efo.json.
                # Used as primary key for EFO term lookups and trait-to-EFO mapping.
            ),
            ColumnDef(
                "label",
                ColumnType.VARCHAR,
                nullable=False,
                # Human-readable EFO term label from efo.json. Used for display and
                # text-based searches. Enables mapping traits to standardized ontology terms.
            ),
            ColumnDef(
                "vector",
                ColumnType.FLOAT_ARRAY_200,
                nullable=False,
                # 200-dimensional embedding vector from efo.json. Used for cosine similarity
                # with trait vectors to find semantically related EFO terms.
            ),
        ],
    ),
    # ==== model_results ====
    # Build process (create_model_results_tables - part 1):
    # 1. Load processed_model_results.json containing model outputs by PMID
    # 2. Create table with auto-incrementing id as PRIMARY KEY
    # 3. For each model -> data_item loop:
    #    - Extract: model name, PMID, metadata (exposures/outcomes), results
    #    - Assign sequential current_result_id
    #    - Store metadata and results as JSON strings
    # 4. Batch insert all model_results_data using executemany
    # Result: Complete model outputs with structured metadata for trait extraction
    "model_results": TableDef(
        name="model_results",
        description="Extracted structural data from model results organized by PMID",
        columns=[
            ColumnDef(
                "id",
                ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
                # Auto-incrementing primary key (current_result_id in build script).
                # Uniquely identifies each model result entry across all models and PMIDs.
            ),
            ColumnDef(
                "model",
                ColumnType.VARCHAR,
                nullable=False,
                # Name of the LLM model that generated this result (e.g., "gpt-4", "deepseek-r1").
                # Used to compare performance across different models and filter results by model type.
            ),
            ColumnDef(
                "pmid",
                ColumnType.VARCHAR,
                nullable=False,
                # PubMed ID of the research paper this result was extracted from.
                # Links model results back to source publications for validation and analysis.
            ),
            ColumnDef(
                "metadata",
                ColumnType.JSON,
                nullable=False,
                # JSON containing structured metadata including exposures and outcomes lists.
                # Each exposure/outcome may have linked_index pointing to trait_embeddings.trait_index
                # and original id from the model's output. Used to extract trait relationships.
            ),
            ColumnDef(
                "results",
                ColumnType.JSON,
                nullable=False,
                # JSON containing the complete raw model output and extracted results.
                # Preserves full model response for detailed analysis and debugging.
            ),
        ],
    ),
    # ==== model_result_traits ====
    # Build process (create_model_results_tables - part 2):
    # 1. For each model result, extract trait indices from metadata.exposures and metadata.outcomes
    # 2. Use _extract_trait_indices_from_items() to get (trait_index, trait_id_in_result) tuples
    # 3. Combine exposure and outcome traits using set union (automatic deduplication)
    # 4. Batch validate all trait indices using _get_valid_trait_indices():
    #    - Query trait_embeddings table for trait_index and trait_label
    #    - Only include traits that exist in trait_embeddings (have valid embeddings)
    # 5. Create (trait_link_id, model_result_id, trait_index, trait_label, trait_id_in_result) tuples
    # 6. Batch insert all model_result_traits_data using executemany
    # Result: Links between model results and validated traits, no role distinction
    "model_result_traits": TableDef(
        name="model_result_traits",
        description="Links model results to traits based on unique_traits indices",
        columns=[
            ColumnDef(
                "id",
                ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
                # Auto-incrementing primary key (trait_link_id in build script).
                # Uniquely identifies each trait-to-result relationship.
            ),
            ColumnDef(
                "model_result_id",
                ColumnType.INTEGER,
                nullable=False,
                # Foreign key to model_results.id. Links this trait relationship
                # to a specific model result, enabling queries for all traits
                # associated with a particular model output.
            ),
            ColumnDef(
                "trait_index",
                ColumnType.INTEGER,
                nullable=False,
                # Foreign key to trait_embeddings.trait_index. References the canonical
                # trait from unique_traits.csv that this model result mentions.
                # Enables consistent trait identification across all model results.
            ),
            ColumnDef(
                "trait_label",
                ColumnType.VARCHAR,
                nullable=False,
                # Denormalized trait label from trait_embeddings.trait_label.
                # Stored here for query performance to avoid joins when displaying
                # human-readable trait names. Updated during trait validation process.
            ),
            ColumnDef(
                "trait_id_in_result",
                ColumnType.VARCHAR,
                nullable=True,
                # Original trait identifier from the model's output (exposure.id or outcome.id).
                # May be empty if model didn't provide explicit IDs. Used to trace back
                # to the exact model output and understand how models identify traits.
            ),
        ],
        foreign_keys=[
            ForeignKeyDef("model_result_id", "model_results", "id"),
            ForeignKeyDef("trait_index", "trait_embeddings", "trait_index"),
        ],
    ),
}

# Define expected indexes for query performance optimization
# Build process (create_indexes):
# Creates all indexes after table population for optimal performance
# Uses individual CREATE INDEX statements for each commonly queried column
DATABASE_INDEXES = [
    IndexDef(
        "idx_trait_embeddings_label",
        "trait_embeddings",
        ["trait_label"],
        # Enables fast text-based trait searches and lookups by human-readable names
        # Built after trait_embeddings table population from unique_traits + embeddings
    ),
    IndexDef(
        "idx_trait_embeddings_index",
        "trait_embeddings",
        ["trait_index"],
        # Optimizes foreign key lookups from model_result_traits table
        # Built after trait_embeddings table with validated trait indices
    ),
    IndexDef(
        "idx_efo_embeddings_label",
        "efo_embeddings",
        ["label"],
        # Speeds up EFO term searches by human-readable labels
        # Built after complete EFO embeddings insertion from efo.json
    ),
    IndexDef(
        "idx_model_results_model",
        "model_results",
        ["model"],
        # Enables efficient filtering and comparison by model type (gpt-4, deepseek-r1, etc.)
        # Built after all model results insertion from processed_model_results.json
    ),
    IndexDef(
        "idx_model_results_pmid",
        "model_results",
        ["pmid"],
        # Allows fast retrieval of all model results for a specific research paper
        # Built after model results table population with PMID extraction
    ),
    IndexDef(
        "idx_model_result_traits_trait_index",
        "model_result_traits",
        ["trait_index"],
        # Optimizes queries finding all model results mentioning a specific trait
        # Built after trait validation and linking process completion
    ),
    IndexDef(
        "idx_model_result_traits_model_result_id",
        "model_result_traits",
        ["model_result_id"],
        # Speeds up queries for all traits mentioned in a specific model result
        # Built after model_result_traits linking table population
    ),
    IndexDef(
        "idx_model_result_traits_trait_label",
        "model_result_traits",
        ["trait_label"],
        # Enables fast text-based searches within the denormalized trait labels
        # Built after trait label denormalization from trait_embeddings
    ),
]

# Define expected views for common similarity search operations
# Build process (create_similarity_functions):
# Creates pre-computed similarity views using DuckDB's array_cosine_similarity function
# Views are built after all tables and indexes are created for optimal performance
DATABASE_VIEWS = [
    ViewDef(
        name="trait_similarity_search",
        # Build process: Creates CROSS JOIN between trait_embeddings and itself
        # Calculates cosine similarity for all trait pairs (excluding self-comparisons)
        # Built after trait_embeddings table contains validated traits with embeddings
        # Pre-computed similarity matrix for all trait-to-trait comparisons.
        # Uses cosine similarity on 200-dimensional embeddings to find semantically related traits.
        # Excludes self-comparisons (t1.trait_index != t2.trait_index).
        # Useful for discovering related traits and clustering analysis.
        sql="""
        SELECT
            t1.trait_index as query_id,
            t1.trait_label as query_label,
            t2.trait_index as result_id,
            t2.trait_label as result_label,
            array_cosine_similarity(t1.vector, t2.vector) as similarity
        FROM trait_embeddings t1
        CROSS JOIN trait_embeddings t2
        WHERE t1.trait_index != t2.trait_index
        """,
    ),
    ViewDef(
        name="trait_efo_similarity_search",
        # Build process: Creates CROSS JOIN between trait_embeddings and efo_embeddings
        # Calculates cosine similarity between all trait-EFO term pairs
        # Built after both trait_embeddings and efo_embeddings tables are populated
        # Cross-reference matrix between traits and EFO ontology terms.
        # Uses cosine similarity to map traits to relevant EFO terms for ontology alignment.
        # Enables automatic trait categorization and standardization against biomedical ontologies.
        # Results can be filtered by similarity threshold to find best EFO matches.
        sql="""
        SELECT
            t.trait_index as trait_index,
            t.trait_label as trait_label,
            e.id as efo_id,
            e.label as efo_label,
            array_cosine_similarity(t.vector, e.vector) as similarity
        FROM trait_embeddings t
        CROSS JOIN efo_embeddings e
        """,
    ),
]
