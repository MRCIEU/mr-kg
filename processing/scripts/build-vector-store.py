"""Build a vector store using DuckDB for trait and EFO embeddings.

This script creates a DuckDB database containing:
1. Trait embeddings from model results
2. EFO term embeddings
3. Model results data with trait linkings
4. Functions for similarity search

The database enables finding:
- Most similar traits from model results
- Most similar EFO terms for a given trait
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import duckdb
from common_funcs.schema.embedding_schema import EmbeddingRecord
from common_funcs.schema.processed_data_schema import ProcessModelResults
from loguru import logger
from yiutils.project_utils import find_project_root


def make_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Perform a dry run without actually creating the database",
    )
    parser.add_argument(
        "--database-name",
        "-db",
        type=str,
        help="Custom database name (without .db extension). If not provided, uses timestamp",
    )
    return parser.parse_args()


def load_embeddings(file_path: Path) -> List[EmbeddingRecord]:
    """Load embedding records from JSON file.

    Args:
        file_path: Path to the embedding JSON file

    Returns:
        List of embedding records
    """
    logger.info(f"Loading embeddings from: {file_path}")
    with file_path.open("r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} embedding records")
    return data


def load_model_results(file_path: Path) -> List[ProcessModelResults]:
    """Load processed model results from JSON file.

    Args:
        file_path: Path to the model results JSON file

    Returns:
        List of processed model results
    """
    logger.info(f"Loading model results from: {file_path}")
    with file_path.open("r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} model result sets")
    return data


def create_trait_embeddings_table(
    conn: duckdb.DuckDBPyConnection, trait_embeddings: List[EmbeddingRecord]
):
    """Create and populate the trait embeddings table.

    Args:
        conn: DuckDB connection
        trait_embeddings: List of trait embedding records
    """
    logger.info("Creating trait embeddings table...")

    # Create table with vector column
    conn.execute("""
        CREATE TABLE trait_embeddings (
            id VARCHAR PRIMARY KEY,
            label VARCHAR NOT NULL,
            vector FLOAT[200] NOT NULL
        )
    """)

    # Insert data
    logger.info(f"Inserting {len(trait_embeddings)} trait embeddings...")
    for record in trait_embeddings:
        conn.execute(
            """
            INSERT INTO trait_embeddings (id, label, vector)
            VALUES (?, ?, ?)
        """,
            (record["id"], record["label"], record["vector"]),
        )

    logger.info("✓ Trait embeddings table created and populated")


def create_efo_embeddings_table(
    conn: duckdb.DuckDBPyConnection, efo_embeddings: List[EmbeddingRecord]
):
    """Create and populate the EFO embeddings table.

    Args:
        conn: DuckDB connection
        efo_embeddings: List of EFO embedding records
    """
    logger.info("Creating EFO embeddings table...")

    # Create table with vector column
    conn.execute("""
        CREATE TABLE efo_embeddings (
            id VARCHAR PRIMARY KEY,
            label VARCHAR NOT NULL,
            vector FLOAT[200] NOT NULL
        )
    """)

    # Insert data
    logger.info(f"Inserting {len(efo_embeddings)} EFO embeddings...")
    for record in efo_embeddings:
        conn.execute(
            """
            INSERT INTO efo_embeddings (id, label, vector)
            VALUES (?, ?, ?)
        """,
            (record["id"], record["label"], record["vector"]),
        )

    logger.info("✓ EFO embeddings table created and populated")


def create_model_results_tables(
    conn: duckdb.DuckDBPyConnection, model_results: List[ProcessModelResults]
):
    """Create and populate tables for model results data.

    Args:
        conn: DuckDB connection
        model_results: List of processed model results
    """
    logger.info("Creating model results tables...")

    # Create main model results table
    conn.execute("""
        CREATE TABLE model_results (
            id INTEGER PRIMARY KEY,
            model VARCHAR NOT NULL,
            pmid VARCHAR NOT NULL
        )
    """)

    # Create traits table for exposures and outcomes
    conn.execute("""
        CREATE TABLE model_traits (
            id INTEGER PRIMARY KEY,
            model_result_id INTEGER NOT NULL,
            trait_id VARCHAR NOT NULL,
            trait VARCHAR NOT NULL,
            category VARCHAR NOT NULL,
            linked_index INTEGER NOT NULL,
            trait_type VARCHAR NOT NULL,  -- 'exposure' or 'outcome'
            FOREIGN KEY (model_result_id) REFERENCES model_results(id)
        )
    """)

    # Pre-collect all data to avoid manual ID management
    model_results_data = []
    model_traits_data = []

    for result_id, model_result in enumerate(model_results):
        model_name = model_result["model"]

        for data_item in model_result["data"]:
            pmid = str(data_item["pmid"])
            current_result_id = len(model_results_data)

            # Collect model result data
            model_results_data.append((current_result_id, model_name, pmid))

            # Extract and validate trait data
            exposure_traits = _extract_valid_traits(
                data_item["metadata"]["exposures"], "exposure"
            )
            outcome_traits = _extract_valid_traits(
                data_item["metadata"]["outcomes"], "outcome"
            )

            # Collect trait data with pre-calculated IDs
            for trait_data in exposure_traits + outcome_traits:
                trait_id = len(model_traits_data)
                model_traits_data.append((
                    trait_id,
                    current_result_id,
                    str(trait_data["id"]),
                    trait_data["trait"],
                    trait_data["category"],
                    trait_data["linked_index"],
                    trait_data["trait_type"]
                ))

    # Batch insert model results
    logger.info(f"Inserting {len(model_results_data)} model results...")
    conn.executemany(
        "INSERT INTO model_results (id, model, pmid) VALUES (?, ?, ?)",
        model_results_data
    )

    # Batch insert model traits
    logger.info(f"Inserting {len(model_traits_data)} model traits...")
    conn.executemany(
        """INSERT INTO model_traits
           (id, model_result_id, trait_id, trait, category, linked_index, trait_type)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        model_traits_data
    )

    logger.info(
        f"✓ Model results tables created with {len(model_results_data)} results and {len(model_traits_data)} traits"
    )


def _extract_valid_traits(traits_list, trait_type: str):
    """Extract valid traits from exposures or outcomes list.

    Args:
        traits_list: List of trait dictionaries
        trait_type: Either 'exposure' or 'outcome'

    Returns:
        List of valid trait dictionaries with trait_type added
    """
    valid_traits = []
    required_keys = ["id", "trait", "category", "linked_index"]

    for trait in traits_list:
        if isinstance(trait, dict) and all(key in trait for key in required_keys):
            trait_data = {
                "id": trait.get("id"),
                "trait": trait.get("trait"),
                "category": trait.get("category"),
                "linked_index": trait.get("linked_index"),
                "trait_type": trait_type
            }
            valid_traits.append(trait_data)

    return valid_traits


def create_similarity_functions(conn: duckdb.DuckDBPyConnection):
    """Create utility functions for similarity search.

    Args:
        conn: DuckDB connection
    """
    logger.info("Creating similarity search functions...")

    # Create a view for easy trait similarity search
    conn.execute("""
        CREATE VIEW trait_similarity_search AS
        SELECT
            t1.id as query_id,
            t1.label as query_label,
            t2.id as result_id,
            t2.label as result_label,
            array_cosine_similarity(t1.vector, t2.vector) as similarity
        FROM trait_embeddings t1
        CROSS JOIN trait_embeddings t2
        WHERE t1.id != t2.id
    """)

    # Create a view for trait-to-EFO similarity search
    conn.execute("""
        CREATE VIEW trait_efo_similarity_search AS
        SELECT
            t.id as trait_id,
            t.label as trait_label,
            e.id as efo_id,
            e.label as efo_label,
            array_cosine_similarity(t.vector, e.vector) as similarity
        FROM trait_embeddings t
        CROSS JOIN efo_embeddings e
    """)

    logger.info("✓ Similarity search views created")


def create_indexes(conn: duckdb.DuckDBPyConnection):
    """Create indexes for better query performance.

    Args:
        conn: DuckDB connection
    """
    logger.info("Creating indexes...")

    # Create indexes on commonly queried columns
    conn.execute(
        "CREATE INDEX idx_trait_embeddings_label ON trait_embeddings(label)"
    )
    conn.execute(
        "CREATE INDEX idx_efo_embeddings_label ON efo_embeddings(label)"
    )
    conn.execute("CREATE INDEX idx_model_traits_trait ON model_traits(trait)")
    conn.execute(
        "CREATE INDEX idx_model_traits_linked_index ON model_traits(linked_index)"
    )
    conn.execute(
        "CREATE INDEX idx_model_results_model ON model_results(model)"
    )

    logger.info("✓ Indexes created")


def main():
    """Main function to build the vector store database."""
    args = make_args()

    # Project configuration
    PROJECT_ROOT = find_project_root("docker-compose.yml")
    DATA_DIR = PROJECT_ROOT / "data"

    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Data directory: {DATA_DIR}")

    # Check if required files exist
    trait_embeddings_path = (
        DATA_DIR / "processed" / "embeddings" / "traits.json"
    )
    efo_embeddings_path = DATA_DIR / "processed" / "embeddings" / "efo.json"
    model_results_path = (
        DATA_DIR
        / "processed"
        / "model_results"
        / "processed_model_results.json"
    )

    for path in [
        trait_embeddings_path,
        efo_embeddings_path,
        model_results_path,
    ]:
        if not path.exists():
            logger.error(f"Required file not found: {path}")
            return 1
        logger.info(f"✓ Found required file: {path}")

    if args.dry_run:
        logger.info("Dry run completed. All required files found.")
        return 0

    # Generate database name
    if args.database_name:
        db_name = f"{args.database_name}.db"
    else:
        timestamp = int(time.time())
        db_name = f"database-{timestamp}.db"

    db_path = DATA_DIR / "db" / db_name
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating database: {db_path}")

    # Load data
    trait_embeddings = load_embeddings(trait_embeddings_path)
    efo_embeddings = load_embeddings(efo_embeddings_path)
    model_results = load_model_results(model_results_path)

    # Create database and tables
    with duckdb.connect(str(db_path)) as conn:
        logger.info("Connected to DuckDB database")

        # Create tables and populate data
        create_trait_embeddings_table(conn, trait_embeddings)
        create_efo_embeddings_table(conn, efo_embeddings)
        create_model_results_tables(conn, model_results)
        create_similarity_functions(conn)
        create_indexes(conn)

        # Verify data
        trait_count = conn.execute(
            "SELECT COUNT(*) FROM trait_embeddings"
        ).fetchone()
        efo_count = conn.execute(
            "SELECT COUNT(*) FROM efo_embeddings"
        ).fetchone()
        result_count = conn.execute(
            "SELECT COUNT(*) FROM model_results"
        ).fetchone()
        model_trait_count = conn.execute(
            "SELECT COUNT(*) FROM model_traits"
        ).fetchone()

        logger.info("Database created successfully:")
        logger.info(
            f"  - {trait_count[0] if trait_count else 0} trait embeddings"
        )
        logger.info(f"  - {efo_count[0] if efo_count else 0} EFO embeddings")
        logger.info(
            f"  - {result_count[0] if result_count else 0} model results"
        )
        logger.info(
            f"  - {model_trait_count[0] if model_trait_count else 0} model traits"
        )
        logger.info(f"  - Database saved to: {db_path}")

    return 0


if __name__ == "__main__":
    exit(main())
