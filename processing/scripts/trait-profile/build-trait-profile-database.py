"""Build a DuckDB database for trait profile similarity data.

This script creates a DuckDB database containing:
1. Query combinations (PMID-model pairs with trait profiles)
2. Similarity relationships between combinations within the same model
3. Indexes and views for efficient querying

The database enables:
- Finding most similar trait profiles for a given PMID-model combination
- Analyzing similarity patterns within specific models
- Exploring trait profile relationships and clustering
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import duckdb
from loguru import logger
from yiutils.project_utils import find_project_root

# Project configuration
PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"


def make_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to the trait profile similarities JSON file",
        default=DATA_DIR
        / "processed"
        / "trait-profile-similarities"
        / "trait-profile-similarities.json",
    )
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


def load_trait_profile_data(file_path: Path) -> List[Dict]:
    """Load trait profile similarity data from JSON file.

    Args:
        file_path: Path to the trait profile similarities JSON file

    Returns:
        List of trait profile similarity records
    """
    logger.info(f"Loading trait profile data from: {file_path}")
    with file_path.open("r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} trait profile records")
    return data


def create_query_combinations_table(
    conn: duckdb.DuckDBPyConnection, trait_profile_data: List[Dict]
):
    """Create and populate the query combinations table.

    Args:
        conn: DuckDB connection
        trait_profile_data: List of trait profile similarity records
    """
    logger.info("Creating query combinations table...")

    # Create table for query combinations
    conn.execute("""
        CREATE TABLE query_combinations (
            id INTEGER PRIMARY KEY,
            pmid VARCHAR NOT NULL,
            model VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            trait_count INTEGER NOT NULL,
            UNIQUE(pmid, model)
        )
    """)

    # Prepare data for batch insert
    query_combinations_data = []
    for idx, record in enumerate(trait_profile_data):
        query_combinations_data.append(
            (
                idx,  # id
                record["query_pmid"],
                record["query_model"],
                record["query_title"],
                record["query_trait_count"],
            )
        )

    # Batch insert
    logger.info(
        f"Inserting {len(query_combinations_data)} query combinations..."
    )
    conn.executemany(
        """INSERT INTO query_combinations (id, pmid, model, title, trait_count)
           VALUES (?, ?, ?, ?, ?)""",
        query_combinations_data,
    )

    logger.info(
        f"Query combinations table created with {len(query_combinations_data)} records"
    )


def create_trait_similarities_table(
    conn: duckdb.DuckDBPyConnection, trait_profile_data: List[Dict]
):
    """Create and populate the trait similarities table.

    Args:
        conn: DuckDB connection
        trait_profile_data: List of trait profile similarity records
    """
    logger.info("Creating trait similarities table...")

    # Create table for similarity relationships
    conn.execute("""
        CREATE TABLE trait_similarities (
            id INTEGER PRIMARY KEY,
            query_combination_id INTEGER NOT NULL,
            similar_pmid VARCHAR NOT NULL,
            similar_model VARCHAR NOT NULL,
            similar_title VARCHAR NOT NULL,
            trait_profile_similarity DOUBLE NOT NULL,
            trait_jaccard_similarity DOUBLE NOT NULL,
            query_trait_count INTEGER NOT NULL,
            similar_trait_count INTEGER NOT NULL,
            FOREIGN KEY (query_combination_id) REFERENCES query_combinations(id)
        )
    """)

    # Prepare data for batch insert
    similarities_data = []
    similarity_id = 0

    for query_idx, record in enumerate(trait_profile_data):
        for similarity in record["top_similarities"]:
            similarities_data.append(
                (
                    similarity_id,  # id
                    query_idx,  # query_combination_id (matches query_combinations.id)
                    similarity["similar_pmid"],
                    similarity["similar_model"],
                    similarity["similar_title"],
                    similarity["trait_profile_similarity"],
                    similarity["trait_jaccard_similarity"],
                    similarity["query_trait_count"],
                    similarity["similar_trait_count"],
                )
            )
            similarity_id += 1

    # Batch insert
    logger.info(
        f"Inserting {len(similarities_data)} similarity relationships..."
    )
    conn.executemany(
        """INSERT INTO trait_similarities 
           (id, query_combination_id, similar_pmid, similar_model, similar_title,
            trait_profile_similarity, trait_jaccard_similarity, 
            query_trait_count, similar_trait_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        similarities_data,
    )

    logger.info(
        f"Trait similarities table created with {len(similarities_data)} records"
    )


def create_indexes(conn: duckdb.DuckDBPyConnection):
    """Create indexes for better query performance.

    Args:
        conn: DuckDB connection
    """
    logger.info("Creating indexes...")

    # Indexes for query_combinations table
    conn.execute(
        "CREATE INDEX idx_query_combinations_pmid ON query_combinations(pmid)"
    )
    conn.execute(
        "CREATE INDEX idx_query_combinations_model ON query_combinations(model)"
    )
    conn.execute(
        "CREATE INDEX idx_query_combinations_pmid_model ON query_combinations(pmid, model)"
    )

    # Indexes for trait_similarities table
    conn.execute(
        "CREATE INDEX idx_trait_similarities_query_id ON trait_similarities(query_combination_id)"
    )
    conn.execute(
        "CREATE INDEX idx_trait_similarities_similar_pmid ON trait_similarities(similar_pmid)"
    )
    conn.execute(
        "CREATE INDEX idx_trait_similarities_similar_model ON trait_similarities(similar_model)"
    )
    conn.execute(
        "CREATE INDEX idx_trait_similarities_trait_profile_sim ON trait_similarities(trait_profile_similarity)"
    )
    conn.execute(
        "CREATE INDEX idx_trait_similarities_jaccard_sim ON trait_similarities(trait_jaccard_similarity)"
    )

    logger.info("Indexes created")


def create_views(conn: duckdb.DuckDBPyConnection):
    """Create utility views for common queries.

    Args:
        conn: DuckDB connection
    """
    logger.info("Creating views...")

    # View for comprehensive similarity analysis
    conn.execute("""
        CREATE VIEW trait_similarity_analysis AS
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
    """)

    # View for model-specific similarity statistics
    conn.execute("""
        CREATE VIEW model_similarity_stats AS
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
    """)

    # View for top similarity pairs by model
    conn.execute("""
        CREATE VIEW top_similarity_pairs AS
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
    """)

    logger.info("Views created")


def validate_database(conn: duckdb.DuckDBPyConnection):
    """Validate the created database by running basic queries.

    Args:
        conn: DuckDB connection
    """
    logger.info("Validating database...")

    # Check table counts
    query_result = conn.execute(
        "SELECT COUNT(*) FROM query_combinations"
    ).fetchone()
    query_count = query_result[0] if query_result else 0

    similarity_result = conn.execute(
        "SELECT COUNT(*) FROM trait_similarities"
    ).fetchone()
    similarity_count = similarity_result[0] if similarity_result else 0

    logger.info(f"Query combinations: {query_count}")
    logger.info(f"Similarity records: {similarity_count}")

    # Check data integrity
    expected_similarities = (
        query_count * 10
    )  # Each query should have 10 similarities
    if similarity_count == expected_similarities:
        logger.info("✓ Data integrity check passed")
    else:
        logger.warning(
            f"⚠ Expected {expected_similarities} similarities, found {similarity_count}"
        )

    # Test views
    try:
        view_result = conn.execute(
            "SELECT COUNT(*) FROM trait_similarity_analysis"
        ).fetchone()
        view_count = view_result[0] if view_result else 0
        logger.info(
            f"✓ trait_similarity_analysis view working ({view_count} records)"
        )

        model_result = conn.execute(
            "SELECT COUNT(*) FROM model_similarity_stats"
        ).fetchone()
        model_stats = model_result[0] if model_result else 0
        logger.info(
            f"✓ model_similarity_stats view working ({model_stats} models)"
        )

        pairs_result = conn.execute(
            "SELECT COUNT(*) FROM top_similarity_pairs"
        ).fetchone()
        top_pairs = pairs_result[0] if pairs_result else 0
        logger.info(
            f"✓ top_similarity_pairs view working ({top_pairs} high-similarity pairs)"
        )

    except Exception as e:
        logger.error(f"✗ View validation failed: {e}")


def main():
    """Main function to build the trait profile similarity database."""
    args = make_args()

    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Data directory: {DATA_DIR}")

    # Check if input file exists
    input_file = args.input_file
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    logger.info(f"Found input file: {input_file}")

    if args.dry_run:
        logger.info("Dry run completed. Input file found.")
        return 0

    # Generate database name
    if args.database_name:
        db_name = f"{args.database_name}.db"
    else:
        timestamp = int(time.time())
        db_name = f"trait-profile-db-{timestamp}.db"

    db_path = DATA_DIR / "db" / db_name
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating database: {db_path}")

    # Load data
    trait_profile_data = load_trait_profile_data(input_file)

    # Create database and tables
    with duckdb.connect(str(db_path)) as conn:
        logger.info("Connected to DuckDB database")

        # Create tables and populate data
        create_query_combinations_table(conn, trait_profile_data)
        create_trait_similarities_table(conn, trait_profile_data)
        create_indexes(conn)
        create_views(conn)

        # Validate the database
        validate_database(conn)

    logger.info(f"Database successfully created: {db_path}")
    return 0


if __name__ == "__main__":
    exit(main())
