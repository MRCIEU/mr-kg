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
from typing import Dict, List

import duckdb
import pandas as pd
from common_funcs.schema.database_schema import (
    DATABASE_SCHEMA,
    DATABASE_INDEXES,
    DATABASE_VIEWS,
)
from common_funcs.schema.database_schema_utils import (
    print_validation_report,
    validate_database_schema,
)
from common_funcs.schema.embedding_schema import EmbeddingRecord
from common_funcs.schema.mr_pubmed_schema import RawMrData
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


def load_unique_traits(file_path: Path) -> Dict[int, str]:
    """Load unique traits from CSV file.

    Args:
        file_path: Path to the unique_traits.csv file

    Returns:
        Dictionary mapping trait index to trait label
    """
    logger.info(f"Loading unique traits from: {file_path}")
    df = pd.read_csv(file_path)
    traits_dict = dict(zip(df["index"], df["trait"]))
    logger.info(f"Loaded {len(traits_dict)} unique traits")
    return traits_dict


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


def load_mr_pubmed_data(file_path: Path) -> RawMrData:
    """Load raw MR PubMed data from JSON file.

    Args:
        file_path: Path to the MR PubMed data JSON file

    Returns:
        List of raw MR PubMed records
    """
    logger.info(f"Loading MR PubMed data from: {file_path}")
    with file_path.open("r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} MR PubMed records")
    return data


def create_trait_embeddings_table(
    conn: duckdb.DuckDBPyConnection,
    trait_embeddings: List[EmbeddingRecord],
    unique_traits: Dict[int, str],
):
    """Create and populate the trait embeddings table using unique traits as the basis.

    Args:
        conn: DuckDB connection
        trait_embeddings: List of trait embedding records
        unique_traits: Dictionary mapping trait index to trait label
    """
    logger.info("Creating trait embeddings table...")

    # Create table with trait_index as primary key
    conn.execute("""
        CREATE TABLE trait_embeddings (
            trait_index INTEGER PRIMARY KEY,
            trait_label VARCHAR NOT NULL,
            vector FLOAT[200] NOT NULL
        )
    """)

    # Create a mapping from trait labels to embeddings
    embedding_map = {
        record["label"]: record["vector"] for record in trait_embeddings
    }

    # Insert data using unique_traits as the basis
    logger.info(f"Inserting {len(unique_traits)} trait embeddings...")
    inserted_count = 0
    missing_embeddings = []

    for trait_index, trait_label in unique_traits.items():
        if trait_label in embedding_map:
            conn.execute(
                """
                INSERT INTO trait_embeddings (trait_index, trait_label, vector)
                VALUES (?, ?, ?)
                """,
                (trait_index, trait_label, embedding_map[trait_label]),
            )
            inserted_count += 1
        else:
            missing_embeddings.append((trait_index, trait_label))

    logger.info(
        f"Trait embeddings table created with {inserted_count} embeddings"
    )
    if missing_embeddings:
        logger.warning(
            f"Missing embeddings for {len(missing_embeddings)} traits"
        )


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

    logger.info("EFO embeddings table created and populated")


def _extract_trait_indices_from_items(items) -> set[tuple[int, str]]:
    """Extract trait indices from exposure or outcome items.

    Args:
        items: List of exposure or outcome items (can be strings or dictionaries)

    Returns:
        Set of (trait_index, trait_id_in_result) tuples where:
        - trait_index (int): The linked index from unique_traits
        - trait_id_in_result (str): The original trait ID from the model result
    """
    trait_indices = set()
    for item in items:
        if (
            isinstance(item, dict)
            and "linked_index" in item
            and item["linked_index"] is not None
        ):
            trait_indices.add((item["linked_index"], str(item.get("id", ""))))
    return trait_indices


def _get_valid_trait_indices(
    conn: duckdb.DuckDBPyConnection, trait_indices: set
) -> set:
    """Filter trait indices to only include those with embeddings and get their labels.

    Args:
        conn: DuckDB connection
        trait_indices: Set of (trait_index, trait_id_in_result) tuples

    Returns:
        Set of valid (trait_index, trait_label, trait_id_in_result) tuples
    """
    if not trait_indices:
        return set()

    # Get all trait indices that have embeddings in a single query
    trait_index_list = [idx for idx, _ in trait_indices]
    placeholders = ",".join("?" * len(trait_index_list))
    query = f"SELECT trait_index, trait_label FROM trait_embeddings WHERE trait_index IN ({placeholders})"

    valid_traits = {
        row[0]: row[1]
        for row in conn.execute(query, trait_index_list).fetchall()
    }

    # Return only the trait indices that are valid, including their labels
    return {
        (idx, valid_traits[idx], trait_id)
        for idx, trait_id in trait_indices
        if idx in valid_traits
    }


def create_model_results_tables(
    conn: duckdb.DuckDBPyConnection, model_results: List[ProcessModelResults]
):
    """Create and populate tables for model results data focused on LinkedModelData.

    Creates two tables:
    1. model_results: Contains the complete structural data for each PMID and model
    2. model_result_traits: Links model results to traits based on unique_traits indices,
       combining both exposure and outcome traits without role distinction

    Args:
        conn: DuckDB connection
        model_results: List of processed model results
    """
    logger.info("Creating model results tables...")

    # Create main model results table for extracted structural data
    conn.execute("""
        CREATE TABLE model_results (
            id INTEGER PRIMARY KEY,
            model VARCHAR NOT NULL,
            pmid VARCHAR NOT NULL,
            metadata JSON NOT NULL,
            results JSON NOT NULL
        )
    """)

    # Create table to link model results to traits
    conn.execute("""
        CREATE TABLE model_result_traits (
            id INTEGER PRIMARY KEY,
            model_result_id INTEGER NOT NULL,
            trait_index INTEGER NOT NULL,
            trait_label VARCHAR NOT NULL,  -- trait label from unique_traits
            trait_id_in_result VARCHAR,  -- original trait id from the model result
            FOREIGN KEY (model_result_id) REFERENCES model_results(id),
            FOREIGN KEY (trait_index) REFERENCES trait_embeddings(trait_index)
        )
    """)

    # Calculate total number of data items for logging
    total_data_items = sum(
        len(model_result["data"]) for model_result in model_results
    )
    logger.info(
        f"Processing {total_data_items} data items from {len(model_results)} models"
    )

    # Prepare data collections
    model_results_data = []
    all_trait_data = []  # Will store (result_id, trait_indices_set) tuples

    # Process model results data (single loop, no nesting)
    current_result_id = 0

    for model_result in model_results:
        model_name = model_result["model"]

        for data_item in model_result["data"]:
            pmid = str(data_item["pmid"])
            metadata = data_item["metadata"]
            results = data_item["results"]

            # Store the complete structural data for this PMID
            model_results_data.append(
                (
                    current_result_id,
                    model_name,
                    pmid,
                    json.dumps(metadata),
                    json.dumps(results),
                )
            )

            # Extract all trait indices for this result (combining exposures and outcomes)
            exposure_traits = _extract_trait_indices_from_items(
                metadata.get("exposures", [])
            )
            outcome_traits = _extract_trait_indices_from_items(
                metadata.get("outcomes", [])
            )

            # Combine and deduplicate trait indices
            all_trait_indices = exposure_traits | outcome_traits

            if all_trait_indices:
                all_trait_data.append((current_result_id, all_trait_indices))

            current_result_id += 1

    # Batch validate trait indices
    logger.info("Validating trait indices against embeddings...")
    model_result_traits_data = []
    trait_link_id = 0

    for result_id, trait_indices in all_trait_data:
        valid_trait_indices = _get_valid_trait_indices(conn, trait_indices)

        for (
            trait_index,
            trait_label,
            trait_id_in_result,
        ) in valid_trait_indices:
            model_result_traits_data.append(
                (
                    trait_link_id,
                    result_id,
                    trait_index,
                    trait_label,
                    trait_id_in_result,
                )
            )
            trait_link_id += 1

    # Batch insert model results
    logger.info(f"Inserting {len(model_results_data)} model results...")
    conn.executemany(
        "INSERT INTO model_results (id, model, pmid, metadata, results) VALUES (?, ?, ?, ?, ?)",
        model_results_data,
    )

    # Batch insert trait linkings
    logger.info(f"Inserting {len(model_result_traits_data)} trait linkings...")
    conn.executemany(
        """INSERT INTO model_result_traits
           (id, model_result_id, trait_index, trait_label, trait_id_in_result)
           VALUES (?, ?, ?, ?, ?)""",
        model_result_traits_data,
    )

    logger.info(
        f"Model results tables created with {len(model_results_data)} results and {len(model_result_traits_data)} trait linkings"
    )


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
            t1.trait_index as query_id,
            t1.trait_label as query_label,
            t2.trait_index as result_id,
            t2.trait_label as result_label,
            array_cosine_similarity(t1.vector, t2.vector) as similarity
        FROM trait_embeddings t1
        CROSS JOIN trait_embeddings t2
        WHERE t1.trait_index != t2.trait_index
    """)

    # Create a view for trait-to-EFO similarity search
    conn.execute("""
        CREATE VIEW trait_efo_similarity_search AS
        SELECT
            t.trait_index as trait_index,
            t.trait_label as trait_label,
            e.id as efo_id,
            e.label as efo_label,
            array_cosine_similarity(t.vector, e.vector) as similarity
        FROM trait_embeddings t
        CROSS JOIN efo_embeddings e
    """)

    # Create a comprehensive view for PMID and model analysis
    conn.execute("""
        CREATE VIEW pmid_model_analysis AS
        SELECT
            mr.pmid,
            mr.model,
            mr.id as model_result_id,
            mr.metadata,
            mr.results,
            mpd.title,
            mpd.abstract,
            mpd.pub_date,
            mpd.journal,
            mpd.journal_issn,
            mpd.author_affil,
            COALESCE(
                LIST(
                    STRUCT_PACK(
                        trait_index := mrt.trait_index,
                        trait_label := mrt.trait_label,
                        trait_id_in_result := mrt.trait_id_in_result
                    )
                ) FILTER (WHERE mrt.trait_index IS NOT NULL),
                []
            ) as traits
        FROM model_results mr
        LEFT JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        LEFT JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
        GROUP BY 
            mr.pmid, mr.model, mr.id, mr.metadata, mr.results,
            mpd.title, mpd.abstract, mpd.pub_date, mpd.journal, 
            mpd.journal_issn, mpd.author_affil
    """)

    logger.info("Similarity search views created")


def create_mr_pubmed_data_table(
    conn: duckdb.DuckDBPyConnection, mr_pubmed_data: RawMrData
):
    """Create and populate the MR PubMed data table.

    Args:
        conn: DuckDB connection
        mr_pubmed_data: List of raw MR PubMed records
    """
    logger.info("Creating MR PubMed data table...")

    # Create table
    conn.execute("""
        CREATE TABLE mr_pubmed_data (
            pmid VARCHAR PRIMARY KEY,
            title VARCHAR NOT NULL,
            abstract VARCHAR NOT NULL,
            pub_date VARCHAR NOT NULL,
            journal VARCHAR NOT NULL,
            journal_issn VARCHAR,
            author_affil VARCHAR
        )
    """)

    # Prepare data for batch insert
    logger.info(f"Inserting {len(mr_pubmed_data)} MR PubMed records...")
    pubmed_data_rows = []

    for record in mr_pubmed_data:
        pubmed_data_rows.append(
            (
                record["pmid"],
                record["title"],
                record["ab"],  # abstract
                record["pub_date"],
                record["journal"],
                record.get(
                    "journal_issn", ""
                ),  # Use get() in case field is missing
                record.get(
                    "author_affil", ""
                ),  # Use get() in case field is missing
            )
        )

    # Batch insert
    conn.executemany(
        """INSERT INTO mr_pubmed_data
           (pmid, title, abstract, pub_date, journal, journal_issn, author_affil)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        pubmed_data_rows,
    )

    logger.info(
        f"MR PubMed data table created with {len(pubmed_data_rows)} records"
    )


def create_indexes(conn: duckdb.DuckDBPyConnection):
    """Create indexes for better query performance.

    Args:
        conn: DuckDB connection
    """
    logger.info("Creating indexes...")

    # Create indexes on commonly queried columns
    conn.execute(
        "CREATE INDEX idx_trait_embeddings_label ON trait_embeddings(trait_label)"
    )
    conn.execute(
        "CREATE INDEX idx_trait_embeddings_index ON trait_embeddings(trait_index)"
    )
    conn.execute(
        "CREATE INDEX idx_efo_embeddings_label ON efo_embeddings(label)"
    )
    conn.execute(
        "CREATE INDEX idx_model_results_model ON model_results(model)"
    )
    conn.execute("CREATE INDEX idx_model_results_pmid ON model_results(pmid)")
    conn.execute(
        "CREATE INDEX idx_model_result_traits_trait_index ON model_result_traits(trait_index)"
    )
    conn.execute(
        "CREATE INDEX idx_model_result_traits_model_result_id ON model_result_traits(model_result_id)"
    )
    conn.execute(
        "CREATE INDEX idx_model_result_traits_trait_label ON model_result_traits(trait_label)"
    )

    # Indexes for MR PubMed data table
    conn.execute(
        "CREATE INDEX idx_mr_pubmed_data_pmid ON mr_pubmed_data(pmid)"
    )
    conn.execute(
        "CREATE INDEX idx_mr_pubmed_data_journal ON mr_pubmed_data(journal)"
    )
    conn.execute(
        "CREATE INDEX idx_mr_pubmed_data_pub_date ON mr_pubmed_data(pub_date)"
    )

    logger.info("Indexes created")


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
    unique_traits_path = (
        DATA_DIR / "processed" / "traits" / "unique_traits.csv"
    )
    mr_pubmed_data_path = (
        DATA_DIR / "raw" / "mr-pubmed-data" / "mr-pubmed-data.json"
    )

    for path in [
        trait_embeddings_path,
        efo_embeddings_path,
        model_results_path,
        unique_traits_path,
        mr_pubmed_data_path,
    ]:
        if not path.exists():
            logger.error(f"Required file not found: {path}")
            return 1
        logger.info(f"Found required file: {path}")

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
    unique_traits = load_unique_traits(unique_traits_path)
    mr_pubmed_data = load_mr_pubmed_data(mr_pubmed_data_path)

    # Create database and tables
    with duckdb.connect(str(db_path)) as conn:
        logger.info("Connected to DuckDB database")

        # Create tables and populate data
        create_trait_embeddings_table(conn, trait_embeddings, unique_traits)
        create_efo_embeddings_table(conn, efo_embeddings)
        create_model_results_tables(conn, model_results)
        create_mr_pubmed_data_table(conn, mr_pubmed_data)
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
            "SELECT COUNT(*) FROM model_result_traits"
        ).fetchone()
        pubmed_count = conn.execute(
            "SELECT COUNT(*) FROM mr_pubmed_data"
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
            f"  - {model_trait_count[0] if model_trait_count else 0} trait linkings"
        )
        logger.info(
            f"  - {pubmed_count[0] if pubmed_count else 0} MR PubMed records"
        )
        logger.info(f"  - Database saved to: {db_path}")

    # Validate database schema
    logger.info("Validating database schema...")
    try:
        with duckdb.connect(str(db_path)) as conn:
            validation_results = validate_database_schema(
                conn, DATABASE_SCHEMA, DATABASE_INDEXES, DATABASE_VIEWS
            )
            if validation_results["valid"]:
                logger.info("[OK] Database schema validation passed")
                print_validation_report(validation_results)
            else:
                logger.warning(
                    "[WARN] Database schema validation found issues"
                )
                print_validation_report(validation_results)
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
