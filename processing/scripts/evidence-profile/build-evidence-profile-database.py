"""Build a DuckDB database for evidence profile similarity data.

This script creates a DuckDB database containing:
1. Query combinations (PMID-model pairs with evidence profiles and data quality metrics)
2. Similarity relationships between combinations within the same model
3. Indexes and views for efficient querying

The database enables:
- Finding most similar evidence profiles for a given PMID-model combination
- Analyzing evidence concordance patterns within specific models
- Exploring reproducibility and contradictory findings
- Filtering by data quality and completeness metrics
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import duckdb
from loguru import logger
from yiutils.project_utils import find_project_root

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"


def make_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --input-file ----
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to the evidence profile similarities JSON file",
        default=DATA_DIR
        / "processed"
        / "evidence-profile-similarities"
        / "evidence-similarities.json",
    )

    # ---- --dry-run ----
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Perform a dry run without actually creating the database",
    )

    # ---- --database-name ----
    parser.add_argument(
        "--database-name",
        "-db",
        type=str,
        help="Custom database name (without .db extension). If not provided, uses timestamp",
    )

    # ---- --skip-indexes ----
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Skip index creation (useful for troubleshooting)",
    )

    # ---- --memory-limit ----
    parser.add_argument(
        "--memory-limit",
        type=str,
        default="4GB",
        help="Memory limit for DuckDB (default: 4GB)",
    )

    # ---- --force-write ----
    parser.add_argument(
        "--force-write",
        action="store_true",
        default=False,
        help="Remove existing database and create a new one if database already exists",
    )

    return parser.parse_args()


def load_evidence_profile_data(file_path: Path) -> List[Dict]:
    """Load evidence profile similarity data from JSON file.

    Args:
        file_path: Path to the evidence profile similarities JSON file

    Returns:
        List of evidence profile similarity records
    """
    logger.info(f"Loading evidence profile data from: {file_path}")
    with file_path.open("r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} evidence profile records")
    return data


def create_query_combinations_table(
    conn: duckdb.DuckDBPyConnection, evidence_profile_data: List[Dict]
):
    """Create and populate the query combinations table.

    Args:
        conn: DuckDB connection
        evidence_profile_data: List of evidence profile similarity records
    """
    logger.info("Creating query combinations table...")

    conn.execute("""
        CREATE TABLE query_combinations (
            id INTEGER PRIMARY KEY,
            pmid VARCHAR NOT NULL,
            model VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            result_count INTEGER NOT NULL,
            complete_result_count INTEGER NOT NULL,
            data_completeness DOUBLE NOT NULL,
            publication_year INTEGER,
            UNIQUE(pmid, model)
        )
    """)

    query_combinations_data = []
    for idx, record in enumerate(evidence_profile_data):
        query_combinations_data.append(
            (
                idx,
                record["query_pmid"],
                record["query_model"],
                record["query_title"],
                record["query_result_count"],
                record["complete_result_count"],
                record["data_completeness"],
                record.get("query_publication_year"),
            )
        )

    logger.info(
        f"Inserting {len(query_combinations_data)} query combinations..."
    )
    conn.executemany(
        """INSERT INTO query_combinations 
           (id, pmid, model, title, result_count, complete_result_count, 
            data_completeness, publication_year)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        query_combinations_data,
    )

    logger.info(
        f"Query combinations table created with {len(query_combinations_data)} records"
    )


def create_evidence_similarities_table(
    conn: duckdb.DuckDBPyConnection, evidence_profile_data: List[Dict]
):
    """Create and populate the evidence similarities table.

    Args:
        conn: DuckDB connection
        evidence_profile_data: List of evidence profile similarity records
    """
    logger.info("Creating evidence similarities table...")

    conn.execute("""
        CREATE TABLE evidence_similarities (
            id INTEGER PRIMARY KEY,
            query_combination_id INTEGER NOT NULL,
            similar_pmid VARCHAR NOT NULL,
            similar_model VARCHAR NOT NULL,
            similar_title VARCHAR NOT NULL,
            matched_pairs INTEGER NOT NULL,
            effect_size_similarity DOUBLE,
            effect_size_within_type DOUBLE,
            effect_size_cross_type DOUBLE,
            n_within_type_pairs INTEGER NOT NULL,
            n_cross_type_pairs INTEGER NOT NULL,
            direction_concordance DOUBLE NOT NULL,
            statistical_consistency DOUBLE,
            evidence_overlap DOUBLE NOT NULL,
            null_concordance DOUBLE NOT NULL,
            composite_similarity_equal DOUBLE,
            composite_similarity_direction DOUBLE,
            query_result_count INTEGER NOT NULL,
            similar_result_count INTEGER NOT NULL,
            query_completeness DOUBLE NOT NULL,
            similar_completeness DOUBLE NOT NULL,
            similar_publication_year INTEGER,
            FOREIGN KEY (query_combination_id) REFERENCES query_combinations(id)
        )
    """)

    similarities_data = []
    similarity_id = 0

    for query_idx, record in enumerate(evidence_profile_data):
        for similarity in record["top_similarities"]:
            similarities_data.append(
                (
                    similarity_id,
                    query_idx,
                    similarity["similar_pmid"],
                    similarity["similar_model"],
                    similarity["similar_title"],
                    similarity["matched_pairs"],
                    similarity.get("effect_size_similarity"),
                    similarity.get("effect_size_within_type"),
                    similarity.get("effect_size_cross_type"),
                    similarity.get("n_within_type_pairs", 0),
                    similarity.get("n_cross_type_pairs", 0),
                    similarity["direction_concordance"],
                    similarity.get("statistical_consistency"),
                    similarity["evidence_overlap"],
                    similarity.get("null_concordance", 0.0),
                    similarity.get("composite_similarity_equal"),
                    similarity.get("composite_similarity_direction"),
                    similarity["query_result_count"],
                    similarity["similar_result_count"],
                    similarity.get("query_completeness", 1.0),
                    similarity.get("similar_completeness", 1.0),
                    similarity.get("similar_publication_year"),
                )
            )
            similarity_id += 1

    logger.info(
        f"Inserting {len(similarities_data)} similarity relationships..."
    )

    if len(similarities_data) > 0:
        conn.executemany(
            """INSERT INTO evidence_similarities
               (id, query_combination_id, similar_pmid, similar_model, similar_title,
                matched_pairs, effect_size_similarity, effect_size_within_type, 
                effect_size_cross_type, n_within_type_pairs, n_cross_type_pairs,
                direction_concordance, statistical_consistency, evidence_overlap,
                null_concordance, composite_similarity_equal, composite_similarity_direction,
                query_result_count, similar_result_count, query_completeness,
                similar_completeness, similar_publication_year)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            similarities_data,
        )
        logger.info(
            f"Evidence similarities table created with {len(similarities_data)} records"
        )
    else:
        logger.warning(
            "No similarity relationships to insert. The table will be empty."
        )


def create_indexes(conn: duckdb.DuckDBPyConnection):
    """Create indexes for better query performance.

    Args:
        conn: DuckDB connection
    """
    logger.info("Creating indexes...")

    indexes = [
        ("idx_query_combinations_pmid", "query_combinations(pmid)"),
        ("idx_query_combinations_model", "query_combinations(model)"),
        (
            "idx_evidence_similarities_query_id",
            "evidence_similarities(query_combination_id)",
        ),
        (
            "idx_evidence_similarities_similar_pmid",
            "evidence_similarities(similar_pmid)",
        ),
        (
            "idx_evidence_similarities_similar_model",
            "evidence_similarities(similar_model)",
        ),
        (
            "idx_query_combinations_pmid_model",
            "query_combinations(pmid, model)",
        ),
        (
            "idx_evidence_similarities_composite_equal",
            "evidence_similarities(composite_similarity_equal)",
        ),
        (
            "idx_evidence_similarities_composite_direction",
            "evidence_similarities(composite_similarity_direction)",
        ),
        (
            "idx_evidence_similarities_direction_concordance",
            "evidence_similarities(direction_concordance)",
        ),
    ]

    logger.info(f"Creating {len(indexes)} indexes...")

    successful_indexes = 0
    failed_indexes = []

    for index_name, index_definition in indexes:
        try:
            logger.info(f"Creating index: {index_name}")
            conn.execute(f"CREATE INDEX {index_name} ON {index_definition}")
            successful_indexes += 1
            logger.info(f"Successfully created index: {index_name}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to create index {index_name}: {error_msg}")
            failed_indexes.append((index_name, error_msg))
            logger.warning("Continuing with next index...")
            continue

    logger.info(
        f"Index creation completed: {successful_indexes} successful, {len(failed_indexes)} failed"
    )

    if failed_indexes:
        logger.warning("Failed indexes:")
        for index_name, error in failed_indexes:
            logger.warning(f"  - {index_name}: {error}")

    return successful_indexes, failed_indexes


def create_views(conn: duckdb.DuckDBPyConnection):
    """Create utility views for common queries.

    Args:
        conn: DuckDB connection
    """
    logger.info("Creating views...")

    # ---- evidence_similarity_analysis ----
    conn.execute("""
        CREATE VIEW evidence_similarity_analysis AS
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
    """)

    # ---- model_evidence_stats ----
    conn.execute("""
        CREATE VIEW model_evidence_stats AS
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
    """)

    # ---- high_concordance_pairs ----
    conn.execute("""
        CREATE VIEW high_concordance_pairs AS
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
    """)

    # ---- discordant_evidence_pairs ----
    conn.execute("""
        CREATE VIEW discordant_evidence_pairs AS
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
    """)

    logger.info("Views created")


def validate_database(conn: duckdb.DuckDBPyConnection):
    """Validate the created database by running basic queries.

    Args:
        conn: DuckDB connection
    """
    logger.info("Validating database...")

    query_result = conn.execute(
        "SELECT COUNT(*) FROM query_combinations"
    ).fetchone()
    query_count = query_result[0] if query_result else 0

    similarity_result = conn.execute(
        "SELECT COUNT(*) FROM evidence_similarities"
    ).fetchone()
    similarity_count = similarity_result[0] if similarity_result else 0

    logger.info(f"Query combinations: {query_count}")
    logger.info(f"Similarity records: {similarity_count}")

    expected_similarities = query_count * 10
    if similarity_count == expected_similarities:
        logger.info("Data integrity check passed")
    else:
        logger.warning(
            f"Expected {expected_similarities} similarities, found {similarity_count}"
        )

    try:
        view_result = conn.execute(
            "SELECT COUNT(*) FROM evidence_similarity_analysis"
        ).fetchone()
        view_count = view_result[0] if view_result else 0
        logger.info(
            f"evidence_similarity_analysis view working ({view_count} records)"
        )

        model_result = conn.execute(
            "SELECT COUNT(*) FROM model_evidence_stats"
        ).fetchone()
        model_stats = model_result[0] if model_result else 0
        logger.info(
            f"model_evidence_stats view working ({model_stats} models)"
        )

        concordance_result = conn.execute(
            "SELECT COUNT(*) FROM high_concordance_pairs"
        ).fetchone()
        concordance_pairs = concordance_result[0] if concordance_result else 0
        logger.info(
            f"high_concordance_pairs view working ({concordance_pairs} pairs)"
        )

        discordant_result = conn.execute(
            "SELECT COUNT(*) FROM discordant_evidence_pairs"
        ).fetchone()
        discordant_pairs = discordant_result[0] if discordant_result else 0
        logger.info(
            f"discordant_evidence_pairs view working ({discordant_pairs} pairs)"
        )

    except Exception as e:
        logger.error(f"View validation failed: {e}")


def main():
    """Main function to build the evidence profile similarity database."""
    args = make_args()

    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Data directory: {DATA_DIR}")

    input_file = args.input_file
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    logger.info(f"Found input file: {input_file}")

    if args.dry_run:
        logger.info("Dry run completed. Input file found.")
        return 0

    if args.database_name:
        db_name = f"{args.database_name}.db"
    else:
        timestamp = int(time.time())
        db_name = f"evidence-profile-db-{timestamp}.db"

    db_path = DATA_DIR / "db" / db_name
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        if args.force_write:
            logger.info(f"Removing existing database: {db_path}")
            db_path.unlink()
        else:
            logger.error(f"Database already exists: {db_path}")
            logger.error(
                "Use --force-write to overwrite or choose a different --database-name"
            )
            return 1

    logger.info(f"Creating database: {db_path}")

    evidence_profile_data = load_evidence_profile_data(input_file)

    with duckdb.connect(str(db_path)) as conn:
        logger.info("Connected to DuckDB database")

        logger.info(f"Setting memory limit to {args.memory_limit}")
        conn.execute(f"SET memory_limit='{args.memory_limit}'")
        conn.execute("SET threads=4")
        conn.execute("SET enable_progress_bar=true")
        conn.execute("SET checkpoint_threshold='1GB'")

        logger.info("DuckDB configuration completed")

        create_query_combinations_table(conn, evidence_profile_data)
        create_evidence_similarities_table(conn, evidence_profile_data)

        if args.skip_indexes:
            logger.info("Skipping index creation as requested")
        else:
            successful_indexes, failed_indexes = create_indexes(conn)
            if failed_indexes:
                logger.warning(
                    "Some indexes failed to create, but continuing with database creation"
                )
            logger.info(
                f"Index creation summary: {successful_indexes} successful, {len(failed_indexes)} failed"
            )

        create_views(conn)

        validate_database(conn)

    logger.info(f"Database successfully created: {db_path}")
    return 0


if __name__ == "__main__":
    exit(main())
