"""Query the vector store for similarity searches.

This script provides utilities to query the DuckDB vector store for:
1. Finding most similar traits for a given trait
2. Finding most similar EFO terms for a given trait
3. Exploring model results and trait linkings
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import duckdb
from loguru import logger
from yiutils.project_utils import find_project_root


def make_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        "-db",
        type=str,
        required=True,
        help="Path to the DuckDB database file",
    )
    parser.add_argument(
        "--query-trait",
        "-qt",
        type=str,
        help="Trait label to search for similar traits",
    )
    parser.add_argument(
        "--query-efo",
        "-qe",
        type=str,
        help="Trait label to search for similar EFO terms",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=10,
        help="Number of top results to return (default: 10)",
    )
    parser.add_argument(
        "--list-traits",
        action="store_true",
        help="List all available traits in the database",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all models in the database with their statistics",
    )
    parser.add_argument(
        "--trait-by-index",
        "-ti",
        type=int,
        help="Find trait by linked_index",
    )
    return parser.parse_args()


def find_similar_traits(conn: duckdb.DuckDBPyConnection, query_trait: str, limit: int = 10) -> List[Tuple[str, str, float]]:
    """Find the most similar traits for a given query trait.
    
    Args:
        conn: DuckDB connection
        query_trait: The trait label to search for
        limit: Number of top results to return
        
    Returns:
        List of tuples (trait_id, trait_label, similarity_score)
    """
    result = conn.execute("""
        SELECT result_id, result_label, similarity
        FROM trait_similarity_search
        WHERE query_label = ?
        ORDER BY similarity DESC
        LIMIT ?
    """, (query_trait, limit)).fetchall()
    
    return result


def find_similar_efo_terms(conn: duckdb.DuckDBPyConnection, query_trait: str, limit: int = 10) -> List[Tuple[str, str, float]]:
    """Find the most similar EFO terms for a given trait.
    
    Args:
        conn: DuckDB connection
        query_trait: The trait label to search for
        limit: Number of top results to return
        
    Returns:
        List of tuples (efo_id, efo_label, similarity_score)
    """
    result = conn.execute("""
        SELECT efo_id, efo_label, similarity
        FROM trait_efo_similarity_search
        WHERE trait_label = ?
        ORDER BY similarity DESC
        LIMIT ?
    """, (query_trait, limit)).fetchall()
    
    return result


def list_all_traits(conn: duckdb.DuckDBPyConnection) -> List[Tuple[str, str]]:
    """List all traits in the database.
    
    Args:
        conn: DuckDB connection
        
    Returns:
        List of tuples (trait_id, trait_label)
    """
    result = conn.execute("""
        SELECT id, label
        FROM trait_embeddings
        ORDER BY label
    """).fetchall()
    
    return result


def list_models_stats(conn: duckdb.DuckDBPyConnection) -> List[Tuple[str, int, int]]:
    """List all models with their statistics.
    
    Args:
        conn: DuckDB connection
        
    Returns:
        List of tuples (model_name, num_results, num_traits)
    """
    result = conn.execute("""
        SELECT 
            model,
            COUNT(DISTINCT mr.id) as num_results,
            COUNT(DISTINCT mt.id) as num_traits
        FROM model_results mr
        LEFT JOIN model_traits mt ON mr.id = mt.model_result_id
        GROUP BY model
        ORDER BY model
    """).fetchall()
    
    return result


def find_trait_by_index(conn: duckdb.DuckDBPyConnection, linked_index: int) -> List[Tuple[str, str]]:
    """Find trait information by linked_index.
    
    Args:
        conn: DuckDB connection
        linked_index: The linked_index to search for
        
    Returns:
        List of tuples (trait_label, category)
    """
    result = conn.execute("""
        SELECT DISTINCT trait, category
        FROM model_traits
        WHERE linked_index = ?
    """, (linked_index,)).fetchall()
    
    return result


def get_trait_embedding_by_label(conn: duckdb.DuckDBPyConnection, trait_label: str) -> Tuple[str, List[float]] | None:
    """Get embedding vector for a specific trait label.
    
    Args:
        conn: DuckDB connection
        trait_label: The trait label to search for
        
    Returns:
        Tuple of (trait_id, embedding_vector) or None if not found
    """
    result = conn.execute("""
        SELECT id, vector
        FROM trait_embeddings
        WHERE label = ?
        LIMIT 1
    """, (trait_label,)).fetchone()
    
    if result:
        return result[0], result[1]
    return None


def main():
    """Main function to query the vector store."""
    args = make_args()
    
    # Resolve database path
    db_path = Path(args.database)
    if not db_path.is_absolute():
        PROJECT_ROOT = find_project_root("docker-compose.yml")
        DATA_DIR = PROJECT_ROOT / "data"
        db_path = DATA_DIR / "db" / args.database
    
    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        return 1
    
    logger.info(f"Connecting to database: {db_path}")
    
    # Connect to database and execute queries
    with duckdb.connect(str(db_path)) as conn:
        
        if args.list_models:
            logger.info("Listing models and their statistics...")
            models = list_models_stats(conn)
            print("\\nModels in database:")
            print("Model\\t\\tResults\\tTraits")
            print("-" * 40)
            for model, num_results, num_traits in models:
                print(f"{model}\\t\\t{num_results}\\t{num_traits}")
        
        if args.list_traits:
            logger.info("Listing all traits...")
            traits = list_all_traits(conn)
            print(f"\\nFound {len(traits)} traits:")
            for trait_id, trait_label in traits[:50]:  # Show first 50
                print(f"{trait_id}: {trait_label}")
            if len(traits) > 50:
                print(f"... and {len(traits) - 50} more traits")
        
        if args.trait_by_index is not None:
            logger.info(f"Finding trait with linked_index {args.trait_by_index}...")
            traits = find_trait_by_index(conn, args.trait_by_index)
            if traits:
                print(f"\\nTrait(s) with linked_index {args.trait_by_index}:")
                for trait, category in traits:
                    print(f"  - {trait} (category: {category})")
            else:
                print(f"\\nNo trait found with linked_index {args.trait_by_index}")
        
        if args.query_trait:
            logger.info(f"Finding similar traits for: {args.query_trait}")
            
            # First check if the trait exists
            trait_result = get_trait_embedding_by_label(conn, args.query_trait)
            if trait_result is None:
                print(f"\\nTrait '{args.query_trait}' not found in database.")
                # Suggest similar trait names
                similar_names = conn.execute("""
                    SELECT label
                    FROM trait_embeddings
                    WHERE label ILIKE ?
                    LIMIT 5
                """, (f"%{args.query_trait}%",)).fetchall()
                
                if similar_names:
                    print("Did you mean one of these?")
                    for (name,) in similar_names:
                        print(f"  - {name}")
            else:
                trait_id, embedding = trait_result
                similar_traits = find_similar_traits(conn, args.query_trait, args.limit)
                print(f"\\nTop {len(similar_traits)} most similar traits to '{args.query_trait}':")
                print("Similarity\\tTrait ID\\t\\tTrait Label")
                print("-" * 80)
                for trait_id, trait_label, similarity in similar_traits:
                    print(f"{similarity:.4f}\\t\\t{trait_id}\\t{trait_label}")
        
        if args.query_efo:
            logger.info(f"Finding similar EFO terms for: {args.query_efo}")
            
            # First check if the trait exists
            trait_result = get_trait_embedding_by_label(conn, args.query_efo)
            if trait_result is None:
                print(f"\\nTrait '{args.query_efo}' not found in database.")
                # Suggest similar trait names
                similar_names = conn.execute("""
                    SELECT label
                    FROM trait_embeddings
                    WHERE label ILIKE ?
                    LIMIT 5
                """, (f"%{args.query_efo}%",)).fetchall()
                
                if similar_names:
                    print("Did you mean one of these?")
                    for (name,) in similar_names:
                        print(f"  - {name}")
            else:
                trait_id, embedding = trait_result
                similar_efo = find_similar_efo_terms(conn, args.query_efo, args.limit)
                print(f"\\nTop {len(similar_efo)} most similar EFO terms to '{args.query_efo}':")
                print("Similarity\\tEFO ID\\t\\t\\t\\t\\tEFO Label")
                print("-" * 120)
                for efo_id, efo_label, similarity in similar_efo:
                    # Truncate long IDs for display
                    short_id = efo_id if len(efo_id) <= 40 else efo_id[:37] + "..."
                    print(f"{similarity:.4f}\\t\\t{short_id}\\t{efo_label}")
    
    return 0


if __name__ == "__main__":
    exit(main())
