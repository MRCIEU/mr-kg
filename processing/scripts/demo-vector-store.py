"""Example usage of the vector store for trait and EFO similarity search.

This script demonstrates various use cases for the vector store:
1. Finding similar traits for given traits from model results
2. Finding related EFO terms for traits
3. Exploring trait linkings and model statistics
"""

import argparse

import duckdb
from loguru import logger
from yiutils.project_utils import find_project_root


def connect_to_latest_db() -> duckdb.DuckDBPyConnection:
    """Connect to the most recently created database.

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
    args = parser.parse_args()

    try:
        # Connect to database
        conn = connect_to_latest_db()

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
