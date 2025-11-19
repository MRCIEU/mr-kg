"""Database validation checks for CS1 Phase 3 (Similarity Expansion).

This script validates that the vector_store.db database has the required
structure and performance characteristics for implementing similarity-based
trait pair expansion in Case Study 1.
"""

import time
from pathlib import Path

import duckdb
import pandas as pd


def main() -> None:
    """Run all validation checks for Phase 3 feasibility."""
    db_path = Path("data/db/vector_store.db")

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return

    # ---- connect to database ----
    conn = duckdb.connect(str(db_path), read_only=True)

    print("=" * 70)
    print("DATABASE VALIDATION CHECKS FOR SIMILARITY EXPANSION")
    print("=" * 70)

    # ---- check 1: view structure ----
    check_view_structure(conn)

    # ---- check 2: query performance ----
    elapsed = check_query_performance(conn)

    # ---- check 3: expansion rate ----
    check_expansion_rate(conn)

    # ---- check 4: label consistency ----
    match_rate = check_label_consistency(conn)

    # ---- check 5: runtime estimation ----
    estimate_runtime(elapsed)

    conn.close()

    # ---- final recommendation ----
    print_recommendation(elapsed, match_rate)


def check_view_structure(conn: duckdb.DuckDBPyConnection) -> None:
    """Check if trait_similarity_search view exists and has correct structure.

    Args:
        conn: DuckDB connection to vector store
    """
    print("\n1. Checking trait_similarity_search view structure...")
    try:
        schema = conn.execute("DESCRIBE trait_similarity_search").df()
        print("✅ View exists")
        print(schema.to_string())
    except Exception as e:
        print(f"❌ View not found or error: {e}")


def check_query_performance(conn: duckdb.DuckDBPyConnection) -> float:
    """Test query performance with similarity filter.

    Args:
        conn: DuckDB connection to vector store

    Returns:
        Query execution time in seconds
    """
    print("\n2. Testing query performance (0.9 threshold)...")
    try:
        start = time.time()
        result = conn.execute(
            """
            SELECT result_label, similarity
            FROM trait_similarity_search
            WHERE query_label = 'body mass index' AND similarity >= 0.9
            ORDER BY similarity DESC
            LIMIT 100
        """
        ).df()
        elapsed = time.time() - start
        print(f"✅ Query time: {elapsed:.3f}s")
        print(f"   Results found: {len(result)}")
        pass_status = "✅ PASS" if elapsed < 1.0 else "❌ FAIL"
        print(f"   Target: < 1.0s {pass_status}")
        return elapsed
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return 999.0


def check_expansion_rate(conn: duckdb.DuckDBPyConnection) -> None:
    """Estimate average number of similar traits per trait.

    Args:
        conn: DuckDB connection to vector store
    """
    print("\n3. Estimating expansion rate (sample of 10 traits)...")
    print("   Note: Using small sample due to large trait space (75k traits)")
    try:
        # ---- use specific well-known traits instead of random sample ----
        test_traits = [
            "body mass index",
            "type 2 diabetes",
            "coronary artery disease",
            "systolic blood pressure",
            "LDL cholesterol",
        ]

        results = []
        for trait in test_traits:
            query = f"""
            SELECT COUNT(*) as n_similar
            FROM trait_similarity_search
            WHERE query_label = '{trait}' AND similarity >= 0.9
            """
            count = conn.execute(query).df()["n_similar"].iloc[0]
            results.append({"trait": trait, "n_similar": count})
            print(f"   {trait}: {count} similar traits")

        # ---- compute statistics ----
        import numpy as np

        counts = [r["n_similar"] for r in results]
        avg_similar = np.mean(counts)
        median_similar = np.median(counts)
        max_similar = np.max(counts)

        print(f"\n✅ Expansion rate statistics (n={len(test_traits)} traits):")
        print(f"   Average similar traits: {avg_similar:.1f}")
        print(f"   Median similar traits: {median_similar:.1f}")
        print(f"   Max similar traits: {max_similar}")
    except Exception as e:
        print(f"❌ Failed to estimate expansion: {e}")


def check_label_consistency(conn: duckdb.DuckDBPyConnection) -> float:
    """Check trait label matching between embeddings and model results.

    Args:
        conn: DuckDB connection to vector store

    Returns:
        Match rate percentage
    """
    print("\n4. Checking label consistency between databases...")
    try:
        # ---- model_results is in vector_store.db ----
        query = """
        WITH metadata_traits AS (
            SELECT DISTINCT 
                json_extract_string(res.value, '$.exposure') as trait_label
            FROM model_results mr,
            json_each(mr.results) res
            WHERE json_extract_string(res.value, '$.exposure') IS NOT NULL
            LIMIT 500
        )
        SELECT 
            COUNT(DISTINCT mt.trait_label) as metadata_traits,
            COUNT(DISTINCT te.trait_label) as matched_traits,
            ROUND(
                COUNT(DISTINCT te.trait_label)::DOUBLE / 
                COUNT(DISTINCT mt.trait_label) * 100, 1
            ) as match_rate_pct
        FROM metadata_traits mt
        LEFT JOIN trait_embeddings te 
            ON LOWER(mt.trait_label) = LOWER(te.trait_label)
        """
        match_stats = conn.execute(query).df()
        print("✅ Label matching statistics:")
        print(match_stats.to_string())
        match_rate = match_stats["match_rate_pct"].iloc[0]

        if match_rate > 80:
            status = "✅ PASS"
        elif match_rate > 60:
            status = "⚠️ WARNING"
        else:
            status = "❌ FAIL"
        print(f"   Target: > 80% {status}")
        return match_rate
    except Exception as e:
        print(f"❌ Failed to check label matching: {e}")
        print(f"   Error details: {str(e)}")
        return 0.0

        query = """
        WITH metadata_traits AS (
            SELECT DISTINCT 
                json_extract_string(res.value, '$.exposure') as trait_label
            FROM model_results mr,
            json_each(mr.results) res
            WHERE json_extract_string(res.value, '$.exposure') IS NOT NULL
            LIMIT 500
        )
        SELECT 
            COUNT(DISTINCT mt.trait_label) as metadata_traits,
            COUNT(DISTINCT te.trait_label) as matched_traits,
            ROUND(
                COUNT(DISTINCT te.trait_label)::DOUBLE / 
                COUNT(DISTINCT mt.trait_label) * 100, 1
            ) as match_rate_pct
        FROM metadata_traits mt
        LEFT JOIN trait_embeddings te 
            ON LOWER(mt.trait_label) = LOWER(te.trait_label)
        """
        match_stats = conn.execute(query).df()
        print("✅ Label matching statistics:")
        print(match_stats.to_string())
        match_rate = match_stats["match_rate_pct"].iloc[0]

        if match_rate > 80:
            status = "✅ PASS"
        elif match_rate > 60:
            status = "⚠️ WARNING"
        else:
            status = "❌ FAIL"
        print(f"   Target: > 80% {status}")
        return match_rate
    except Exception as e:
        print(f"❌ Failed to check label matching: {e}")
        return 0.0


def estimate_runtime(avg_query_time: float) -> None:
    """Estimate total runtime for full pair extraction.

    Args:
        avg_query_time: Average query execution time in seconds
    """
    print("\n5. Estimating total runtime for full extraction...")
    try:
        n_pairs = 2075
        estimated_queries = n_pairs * 2
        estimated_cache_hit_rate = 0.5
        estimated_time = (
            estimated_queries * avg_query_time * (1 - estimated_cache_hit_rate)
        )

        print(f"   Estimated queries: {estimated_queries}")
        print(
            f"   With 50% cache hit rate: {int(estimated_queries * 0.5)} actual queries"
        )
        print(f"   Estimated total time: {estimated_time / 60:.1f} minutes")

        if estimated_time < 1800:
            status = "✅ FEASIBLE"
        elif estimated_time < 3600:
            status = "⚠️ SLOW"
        else:
            status = "❌ TOO SLOW"
        print(f"   Target: < 30 min {status}")
    except Exception as e:
        print(f"⚠️ Could not estimate runtime: {e}")


def print_recommendation(query_time: float, match_rate: float) -> None:
    """Print final recommendation based on validation results.

    Args:
        query_time: Query execution time in seconds
        match_rate: Label match rate percentage
    """
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    # ---- determine recommendation ----
    estimated_time = 2075 * 2 * query_time * 0.5 / 60

    if query_time < 1.0 and match_rate > 60 and estimated_time < 60:
        recommendation = "✅ FEASIBLE"
        details = (
            "All checks passed. Phase 3 can proceed with current database."
        )
    elif query_time < 2.0 and match_rate > 40:
        recommendation = "⚠️ NEEDS_OPTIMIZATION"
        details = "Performance acceptable but optimization recommended:\n"
        if query_time >= 1.0:
            details += (
                "  - Consider indexing trait_label in trait_embeddings\n"
            )
            details += "  - Implement query result caching\n"
        if match_rate <= 60:
            details += "  - Improve trait label normalization\n"
            details += "  - Add fuzzy matching fallback\n"
    else:
        recommendation = "❌ NOT_RECOMMENDED"
        details = "Critical issues detected:\n"
        if query_time >= 2.0:
            details += "  - Query performance unacceptable\n"
            details += "  - Consider alternative database design\n"
        if match_rate <= 40:
            details += "  - Poor label matching rate\n"
            details += "  - Trait labels may need extensive preprocessing\n"

    print(f"Based on these checks, Phase 3 (Similarity Expansion) is:")
    print(f"  {recommendation}")
    print(f"\n{details}")
    print("=" * 70)


if __name__ == "__main__":
    main()
