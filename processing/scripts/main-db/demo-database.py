"""Example usage of the vector store for trait and EFO similarity search.

This script demonstrates various use cases for the vector store:
1. Finding similar traits for given traits from model results
2. Finding related EFO terms for traits
3. Exploring trait linkings and model statistics

Usage:
    # Use automatic database discovery (preferred restructured-vector-store.db)
    python demo-vector-store.py

    # Use specific database file
    python demo-vector-store.py --database restructured-vector-store.db
    python demo-vector-store.py -db /path/to/database.db
"""

import argparse
from pathlib import Path

import duckdb
from common_funcs.schema.database_schema import (
    DATABASE_SCHEMA,
    DATABASE_INDEXES,
    DATABASE_VIEWS,
)
from common_funcs.schema.database_schema_utils import (
    validate_database_schema,
)
from loguru import logger
from yiutils.project_utils import find_project_root


def connect_to_latest_db() -> duckdb.DuckDBPyConnection:
    """Connect to the most recently created database.

    Searches for databases in this order:
    1. restructured-vector-store.db (preferred)
    2. Most recent database-*.db file

    Returns:
        DuckDB connection to the latest database
    """
    PROJECT_ROOT = find_project_root("docker-compose.yml")
    db_dir = PROJECT_ROOT / "data" / "db"

    # First try to find the restructured database
    restructured_db = db_dir / "restructured-vector-store.db"
    if restructured_db.exists():
        logger.info(f"Connecting to restructured database: {restructured_db}")
        return duckdb.connect(str(restructured_db))

    # Fall back to finding the most recent database file
    db_files = list(db_dir.glob("database-*.db"))
    if not db_files:
        raise FileNotFoundError("No database files found in data/db/")

    latest_db = max(db_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Connecting to: {latest_db}")

    return duckdb.connect(str(latest_db))


def demo_trait_similarity_search(conn: duckdb.DuckDBPyConnection):
    """Demonstrate trait similarity search."""
    logger.info("=== TRAIT SIMILARITY SEARCH DEMO ===")

    # Example traits to search for
    example_traits = ["coffee intake", "diabetes", "migraine", "obesity"]

    for trait in example_traits:
        print(f"\nFinding traits similar to '{trait}':")

        # Check if trait exists
        trait_check = conn.execute(
            """
            SELECT COUNT(*) FROM trait_embeddings WHERE trait_label = ?
        """,
            (trait,),
        ).fetchone()

        if trait_check and trait_check[0] == 0:
            print(f"   ERROR: Trait '{trait}' not found in database")
            continue

        # Get top 3 similar traits
        similar = conn.execute(
            """
            SELECT result_label, similarity
            FROM trait_similarity_search
            WHERE query_label = ?
            ORDER BY similarity DESC
            LIMIT 3
        """,
            (trait,),
        ).fetchall()

        for label, similarity in similar:
            print(f"   {similarity:.3f} - {label}")


def demo_efo_similarity_search(conn: duckdb.DuckDBPyConnection):
    """Demonstrate EFO similarity search."""
    logger.info("=== EFO SIMILARITY SEARCH DEMO ===")

    # Example traits to find EFO terms for
    example_traits = ["coffee intake", "BMI"]

    for trait in example_traits:
        print(f"\nFinding EFO terms similar to '{trait}':")

        # Check if trait exists
        trait_check = conn.execute(
            """
            SELECT COUNT(*) FROM trait_embeddings WHERE trait_label = ?
        """,
            (trait,),
        ).fetchone()

        if trait_check and trait_check[0] == 0:
            print(f"   ERROR: Trait '{trait}' not found in database")
            continue

        # Get top 3 similar EFO terms
        similar = conn.execute(
            """
            SELECT efo_label, similarity
            FROM trait_efo_similarity_search
            WHERE trait_label = ?
            ORDER BY similarity DESC
            LIMIT 3
        """,
            (trait,),
        ).fetchall()

        for label, similarity in similar:
            print(f"   {similarity:.3f} - {label}")


def demo_model_analysis(conn: duckdb.DuckDBPyConnection):
    """Demonstrate model analysis capabilities."""
    logger.info("=== MODEL ANALYSIS DEMO ===")

    # Model statistics
    print("\nModel Statistics:")
    models = conn.execute("""
        SELECT
            model,
            COUNT(DISTINCT mr.id) as papers,
            COUNT(DISTINCT mrt.trait_index) as unique_traits
        FROM model_results mr
        LEFT JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
        GROUP BY model
        ORDER BY papers DESC
    """).fetchall()

    for model, papers, traits in models:
        print(f"   {model}: {papers} papers, {traits} unique traits")

    # Most common traits across all models
    print("\nMost Common Traits Across Models:")
    common_traits = conn.execute("""
        SELECT
            te.trait_label,
            COUNT(*) as frequency,
            COUNT(DISTINCT mr.model) as models_count
        FROM model_result_traits mrt
        JOIN model_results mr ON mrt.model_result_id = mr.id
        JOIN trait_embeddings te ON mrt.trait_index = te.trait_index
        GROUP BY te.trait_label
        HAVING frequency > 20
        ORDER BY frequency DESC
        LIMIT 10
    """).fetchall()

    for trait, freq, model_count in common_traits:
        print(f"   {trait}: {freq} occurrences across {model_count} models")


def demo_pubmed_analysis(conn: duckdb.DuckDBPyConnection):
    """Demonstrate PubMed data analysis capabilities."""
    logger.info("=== PUBMED DATA ANALYSIS DEMO ===")

    # PubMed data statistics
    print("\nPubMed Data Statistics:")
    pubmed_stats = conn.execute("""
        SELECT 
            COUNT(*) as total_papers,
            COUNT(DISTINCT journal) as unique_journals,
            MIN(pub_date) as earliest_date,
            MAX(pub_date) as latest_date
        FROM mr_pubmed_data
    """).fetchone()

    if pubmed_stats:
        total, journals, earliest, latest = pubmed_stats
        print(f"   Total papers: {total}")
        print(f"   Unique journals: {journals}")
        print(f"   Publication date range: {earliest} to {latest}")

    # Top journals by paper count
    print("\nTop Journals by Paper Count:")
    top_journals = conn.execute("""
        SELECT 
            journal,
            COUNT(*) as paper_count
        FROM mr_pubmed_data
        GROUP BY journal
        ORDER BY paper_count DESC
        LIMIT 10
    """).fetchall()

    for journal, count in top_journals:
        print(f"   {journal}: {count} papers")

    # Papers with model results vs without
    print("\nPapers with Model Results:")
    coverage_stats = conn.execute("""
        SELECT 
            COUNT(DISTINCT mpd.pmid) as total_pubmed_papers,
            COUNT(DISTINCT mr.pmid) as papers_with_results,
            COUNT(DISTINCT mr.pmid) * 100.0 / COUNT(DISTINCT mpd.pmid) as coverage_percent
        FROM mr_pubmed_data mpd
        LEFT JOIN model_results mr ON mpd.pmid = mr.pmid
    """).fetchone()

    if coverage_stats:
        total, with_results, coverage = coverage_stats
        print(f"   Total PubMed papers: {total}")
        print(f"   Papers with model results: {with_results}")
        print(f"   Coverage: {coverage:.1f}%")

    # Example: Show paper details for a few PMIDs
    print("\nSample Paper Details:")
    sample_papers = conn.execute("""
        SELECT 
            mpd.pmid,
            mpd.title,
            mpd.journal,
            mpd.pub_date,
            COUNT(mr.id) as model_result_count
        FROM mr_pubmed_data mpd
        LEFT JOIN model_results mr ON mpd.pmid = mr.pmid
        GROUP BY mpd.pmid, mpd.title, mpd.journal, mpd.pub_date
        ORDER BY model_result_count DESC, mpd.pmid
        LIMIT 3
    """).fetchall()

    for pmid, title, journal, pub_date, result_count in sample_papers:
        truncated_title = title[:70] + "..." if len(title) > 70 else title
        print(f"   PMID {pmid}: {truncated_title}")
        print(f"     Journal: {journal} ({pub_date})")
        print(f"     Model results: {result_count}")
        print()  # Empty line for spacing


def demo_pmid_model_analysis(conn: duckdb.DuckDBPyConnection):
    """Demonstrate the comprehensive PMID and model analysis view."""
    logger.info("=== PMID MODEL ANALYSIS DEMO ===")

    # Find a few PMIDs that have model results for demonstration
    print("\nSample PMID Analysis:")
    sample_pmids = conn.execute("""
        SELECT pmid, model, LENGTH(traits) as trait_count
        FROM pmid_model_analysis
        WHERE pmid IS NOT NULL AND LENGTH(traits) > 0
        ORDER BY trait_count DESC
        LIMIT 3
    """).fetchall()

    for pmid, model, trait_count in sample_pmids:
        print(f"\n--- Analysis for PMID {pmid} with model {model} ---")

        # Get comprehensive data for this PMID and model
        analysis_data = conn.execute(
            """
            SELECT 
                title,
                journal,
                pub_date,
                metadata,
                results,
                traits
            FROM pmid_model_analysis
            WHERE pmid = ? AND model = ?
        """,
            (pmid, model),
        ).fetchone()

        if analysis_data:
            title, journal, pub_date, metadata, results, traits = analysis_data
            truncated_title = title[:80] + "..." if len(title) > 80 else title

            print(f"Paper: {truncated_title}")
            print(f"Journal: {journal} ({pub_date})")
            print(f"Total extracted traits: {len(traits)}")

            print("Extracted traits:")
            for i, trait in enumerate(traits[:5]):  # Show first 5
                trait_id_display = (
                    trait["trait_id_in_result"]
                    if trait["trait_id_in_result"]
                    else "N/A"
                )
                print(
                    f"  - {trait['trait_label']} (index: {trait['trait_index']}, id: {trait_id_display})"
                )

            if len(traits) > 5:
                print(f"  ... and {len(traits) - 5} more traits")

    # Show statistics about the view
    print("\nPMID Model Analysis View Statistics:")
    view_stats = conn.execute("""
        SELECT 
            COUNT(DISTINCT pmid) as unique_pmids,
            COUNT(DISTINCT model) as unique_models,
            COUNT(*) as unique_pmid_model_pairs,
            SUM(LENGTH(traits)) as total_trait_instances
        FROM pmid_model_analysis
        WHERE pmid IS NOT NULL
    """).fetchone()

    if view_stats:
        unique_pmids, unique_models, unique_pairs, total_traits = view_stats
        print(f"  Unique PMIDs: {unique_pmids}")
        print(f"  Unique models: {unique_models}")
        print(f"  PMID-model pairs: {unique_pairs}")
        print(f"  Total trait instances: {total_traits}")


def demo_trait_exploration(conn: duckdb.DuckDBPyConnection):
    """Demonstrate trait exploration by analyzing trait usage patterns."""
    logger.info("=== TRAIT EXPLORATION DEMO ===")

    # Trait usage by role (exposure vs outcome)
    print("\nTrait Usage by Role:")
    role_stats = conn.execute("""
        SELECT
            trait_role,
            COUNT(DISTINCT trait_index) as unique_traits,
            COUNT(*) as total_occurrences
        FROM model_result_traits
        GROUP BY trait_role
        ORDER BY unique_traits DESC
    """).fetchall()

    for role, unique, total in role_stats:
        print(f"   {role}: {unique} unique traits, {total} total occurrences")

    # Example: Find traits that appear as both exposures and outcomes
    print("\nTraits used as both Exposures and Outcomes:")
    dual_role = conn.execute("""
        SELECT 
            te.trait_label,
            COUNT(CASE WHEN mrt.trait_role = 'exposure' THEN 1 END) as exposure_count,
            COUNT(CASE WHEN mrt.trait_role = 'outcome' THEN 1 END) as outcome_count
        FROM model_result_traits mrt
        JOIN trait_embeddings te ON mrt.trait_index = te.trait_index
        GROUP BY mrt.trait_index, te.trait_label
        HAVING exposure_count > 0 AND outcome_count > 0
        ORDER BY (exposure_count + outcome_count) DESC
        LIMIT 10
    """).fetchall()

    for trait, exp_count, out_count in dual_role:
        print(f"   {trait}: {exp_count} exposures, {out_count} outcomes")

    # Example: Find traits similar to a common trait
    print("\nFinding traits similar to 'BMI':")
    try:
        similar_to_bmi = conn.execute("""
            SELECT 
                t2.trait_label as result_label,
                array_cosine_similarity(t1.vector, t2.vector) as similarity
            FROM trait_embeddings t1
            CROSS JOIN trait_embeddings t2
            WHERE t1.trait_label = 'BMI' AND t1.trait_index != t2.trait_index
            ORDER BY similarity DESC
            LIMIT 5
        """).fetchall()

        for trait, similarity in similar_to_bmi:
            print(f"   {similarity:.3f} - {trait}")
    except Exception:
        print("   No similarity data available for 'BMI'")


def main():
    """Run the demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        "-db",
        type=str,
        help="Path to the DuckDB database file (if not provided, will search for latest)",
    )
    parser.add_argument(
        "--skip-trait-search",
        action="store_true",
        help="Skip trait similarity search demo",
    )
    parser.add_argument(
        "--skip-efo-search",
        action="store_true",
        help="Skip EFO similarity search demo",
    )
    parser.add_argument(
        "--skip-model-analysis",
        action="store_true",
        help="Skip model analysis demo",
    )
    parser.add_argument(
        "--skip-trait-exploration",
        action="store_true",
        help="Skip trait exploration demo",
    )
    parser.add_argument(
        "--skip-pubmed-analysis",
        action="store_true",
        help="Skip PubMed data analysis demo",
    )
    parser.add_argument(
        "--skip-pmid-model-analysis",
        action="store_true",
        help="Skip PMID model analysis demo",
    )
    args = parser.parse_args()

    try:
        # Connect to database
        if args.database:
            # Use specified database path
            db_path = Path(args.database)
            if not db_path.is_absolute():
                PROJECT_ROOT = find_project_root("docker-compose.yml")
                DATA_DIR = PROJECT_ROOT / "data"
                db_path = DATA_DIR / "db" / args.database

            if not db_path.exists():
                logger.error(f"Database file not found: {db_path}")
                return 1

            logger.info(f"Connecting to specified database: {db_path}")
            conn = duckdb.connect(str(db_path))
        else:
            # Use automatic discovery
            conn = connect_to_latest_db()

        # Validate database schema before running demos
        logger.info("Validating database schema...")
        validation_results = validate_database_schema(
            conn, DATABASE_SCHEMA, DATABASE_INDEXES, DATABASE_VIEWS
        )
        if not validation_results["valid"]:
            logger.error("Database schema validation failed")
            for error in validation_results["errors"]:
                logger.error(f"   {error}")
            return 1
        logger.info("Database schema validation passed")

        print("Vector Store Demo")
        print("=" * 50)

        # Run demos
        if not args.skip_trait_search:
            demo_trait_similarity_search(conn)

        if not args.skip_efo_search:
            demo_efo_similarity_search(conn)

        if not args.skip_model_analysis:
            demo_model_analysis(conn)

        if not args.skip_trait_exploration:
            demo_trait_exploration(conn)

        if not args.skip_pubmed_analysis:
            demo_pubmed_analysis(conn)

        if not args.skip_pmid_model_analysis:
            demo_pmid_model_analysis(conn)

        print("\\nDemo completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1

    finally:
        if "conn" in locals():
            conn.close()

    return 0


if __name__ == "__main__":
    exit(main())
