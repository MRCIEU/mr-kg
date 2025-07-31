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
    
    # Find the most recent database file
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
        print(f"\n🔍 Finding traits similar to '{trait}':")
        
        # Check if trait exists
        trait_check = conn.execute("""
            SELECT COUNT(*) FROM trait_embeddings WHERE label = ?
        """, (trait,)).fetchone()
        
        if trait_check and trait_check[0] == 0:
            print(f"   ❌ Trait '{trait}' not found in database")
            continue
        
        # Get top 3 similar traits
        similar = conn.execute("""
            SELECT result_label, similarity
            FROM trait_similarity_search
            WHERE query_label = ?
            ORDER BY similarity DESC
            LIMIT 3
        """, (trait,)).fetchall()
        
        for label, similarity in similar:
            print(f"   📊 {similarity:.3f} - {label}")


def demo_efo_similarity_search(conn: duckdb.DuckDBPyConnection):
    """Demonstrate EFO similarity search."""
    logger.info("=== EFO SIMILARITY SEARCH DEMO ===")
    
    # Example traits to find EFO terms for
    example_traits = ["coffee intake", "BMI"]
    
    for trait in example_traits:
        print(f"\n🔍 Finding EFO terms similar to '{trait}':")
        
        # Check if trait exists
        trait_check = conn.execute("""
            SELECT COUNT(*) FROM trait_embeddings WHERE label = ?
        """, (trait,)).fetchone()
        
        if trait_check and trait_check[0] == 0:
            print(f"   ❌ Trait '{trait}' not found in database")
            continue
        
        # Get top 3 similar EFO terms
        similar = conn.execute("""
            SELECT efo_label, similarity
            FROM trait_efo_similarity_search
            WHERE trait_label = ?
            ORDER BY similarity DESC
            LIMIT 3
        """, (trait,)).fetchall()
        
        for label, similarity in similar:
            print(f"   🎯 {similarity:.3f} - {label}")


def demo_model_analysis(conn: duckdb.DuckDBPyConnection):
    """Demonstrate model analysis capabilities."""
    logger.info("=== MODEL ANALYSIS DEMO ===")
    
    # Model statistics
    print("\\n📈 Model Statistics:")
    models = conn.execute("""
        SELECT 
            model,
            COUNT(DISTINCT mr.id) as papers,
            COUNT(DISTINCT mt.trait) as unique_traits
        FROM model_results mr
        JOIN model_traits mt ON mr.id = mt.model_result_id
        GROUP BY model
        ORDER BY papers DESC
    """).fetchall()
    
    for model, papers, traits in models:
        print(f"   {model}: {papers} papers, {traits} unique traits")
    
    # Most common traits across all models
    print("\\n🏆 Most Common Traits Across Models:")
    common_traits = conn.execute("""
        SELECT 
            trait,
            COUNT(*) as frequency,
            COUNT(DISTINCT model) as models_count
        FROM model_traits mt
        JOIN model_results mr ON mt.model_result_id = mr.id
        GROUP BY trait
        HAVING frequency > 50
        ORDER BY frequency DESC
        LIMIT 10
    """).fetchall()
    
    for trait, freq, model_count in common_traits:
        print(f"   {trait}: {freq} occurrences across {model_count} models")


def demo_trait_exploration(conn: duckdb.DuckDBPyConnection):
    """Demonstrate trait exploration by category."""
    logger.info("=== TRAIT EXPLORATION DEMO ===")
    
    # Trait categories
    print("\\n📂 Trait Categories:")
    categories = conn.execute("""
        SELECT 
            category,
            COUNT(DISTINCT trait) as unique_traits,
            COUNT(*) as total_occurrences
        FROM model_traits
        WHERE category != ''
        GROUP BY category
        ORDER BY unique_traits DESC
        LIMIT 10
    """).fetchall()
    
    for category, unique, total in categories:
        print(f"   {category}: {unique} unique traits, {total} total occurrences")
    
    # Example: Find disease traits similar to a behavioral trait
    print("\\n🔗 Cross-Category Similarity (Behavioral → Disease):")
    cross_category = conn.execute("""
        SELECT 
            tss.result_label,
            mt_result.category as result_category,
            tss.similarity
        FROM trait_similarity_search tss
        JOIN model_traits mt_query ON tss.query_label = mt_query.trait
        JOIN model_traits mt_result ON tss.result_label = mt_result.trait
        WHERE mt_query.category = 'behavioural'
        AND mt_result.category LIKE '%disease%'
        AND tss.query_label = 'coffee intake'
        ORDER BY tss.similarity DESC
        LIMIT 5
    """).fetchall()
    
    print("   Similar disease traits to 'coffee intake':")
    for trait, category, similarity in cross_category:
        print(f"   🔄 {similarity:.3f} - {trait} ({category})")


def main():
    """Run the demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-trait-search",
        action="store_true",
        help="Skip trait similarity search demo"
    )
    parser.add_argument(
        "--skip-efo-search", 
        action="store_true",
        help="Skip EFO similarity search demo"
    )
    parser.add_argument(
        "--skip-model-analysis",
        action="store_true", 
        help="Skip model analysis demo"
    )
    parser.add_argument(
        "--skip-trait-exploration",
        action="store_true",
        help="Skip trait exploration demo"
    )
    args = parser.parse_args()
    
    try:
        # Connect to database
        conn = connect_to_latest_db()
        
        print("🚀 Vector Store Demo")
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
        
        print("\\n✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
    
    finally:
        if 'conn' in locals():
            conn.close()
    
    return 0


if __name__ == "__main__":
    exit(main())
