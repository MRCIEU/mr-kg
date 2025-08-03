"""Query the vector store for similarity searches.

This script provides utilities to query the DuckDB vector store for:
1. Finding most similar traits for a given trait
2. Finding most similar EFO terms for a given trait
3. Exploring model results and trait linkings
"""

import argparse
from pathlib import Path
from typing import List, Tuple

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
        help="Find trait by trait_index",
    )
    parser.add_argument(
        "--query-pmid",
        "-qp",
        type=str,
        help="Query PubMed data by PMID",
    )
    parser.add_argument(
        "--query-journal",
        "-qj",
        type=str,
        help="Query papers by journal name (supports partial matching)",
    )
    parser.add_argument(
        "--list-journals",
        action="store_true",
        help="List all journals in the database with paper counts",
    )
    return parser.parse_args()


def find_similar_traits(
    conn: duckdb.DuckDBPyConnection, query_trait: str, limit: int = 10
) -> List[Tuple[str, str, float]]:
    """Find the most similar traits for a given query trait.

    Args:
        conn: DuckDB connection
        query_trait: The trait label to search for
        limit: Number of top results to return

    Returns:
        List of tuples (trait_index, trait_label, similarity_score)
    """
    # Use direct SQL query instead of view for better compatibility
    result = conn.execute(
        """
        SELECT
            t2.trait_index as result_id,
            t2.trait_label as result_label,
            array_cosine_similarity(t1.vector, t2.vector) as similarity
        FROM trait_embeddings t1
        CROSS JOIN trait_embeddings t2
        WHERE t1.trait_label = ? AND t1.trait_index != t2.trait_index
        ORDER BY similarity DESC
        LIMIT ?
    """,
        (query_trait, limit),
    ).fetchall()

    return result


def find_similar_efo_terms(
    conn: duckdb.DuckDBPyConnection, query_trait: str, limit: int = 10
) -> List[Tuple[str, str, float]]:
    """Find the most similar EFO terms for a given trait.

    Args:
        conn: DuckDB connection
        query_trait: The trait label to search for
        limit: Number of top results to return

    Returns:
        List of tuples (efo_id, efo_label, similarity_score)
    """
    # Use direct SQL query instead of view for better compatibility
    result = conn.execute(
        """
        SELECT
            e.id as efo_id,
            e.label as efo_label,
            array_cosine_similarity(t.vector, e.vector) as similarity
        FROM trait_embeddings t
        CROSS JOIN efo_embeddings e
        WHERE t.trait_label = ?
        ORDER BY similarity DESC
        LIMIT ?
    """,
        (query_trait, limit),
    ).fetchall()

    return result


def list_all_traits(conn: duckdb.DuckDBPyConnection) -> List[Tuple[int, str]]:
    """List all traits in the database.

    Args:
        conn: DuckDB connection

    Returns:
        List of tuples (trait_index, trait_label)
    """
    result = conn.execute("""
        SELECT trait_index, trait_label
        FROM trait_embeddings
        ORDER BY trait_label
    """).fetchall()

    return result


def list_models_stats(
    conn: duckdb.DuckDBPyConnection,
) -> List[Tuple[str, int, int]]:
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
            COUNT(DISTINCT mrt.trait_index) as num_traits
        FROM model_results mr
        LEFT JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
        GROUP BY model
        ORDER BY model
    """).fetchall()

    return result


def query_pubmed_by_pmid(
    conn: duckdb.DuckDBPyConnection, pmid: str
) -> Tuple | None:
    """Query PubMed data by PMID.

    Args:
        conn: DuckDB connection
        pmid: PubMed ID to search for

    Returns:
        Tuple with PubMed data or None if not found
    """
    result = conn.execute(
        """
        SELECT pmid, title, abstract, pub_date, journal, journal_issn, author_affil
        FROM mr_pubmed_data
        WHERE pmid = ?
    """,
        (pmid,),
    ).fetchone()

    return result


def query_papers_by_journal(
    conn: duckdb.DuckDBPyConnection, journal_pattern: str, limit: int = 10
) -> List[Tuple]:
    """Query papers by journal name (supports partial matching).

    Args:
        conn: DuckDB connection
        journal_pattern: Journal name or pattern to search for
        limit: Number of results to return

    Returns:
        List of tuples with paper information
    """
    result = conn.execute(
        """
        SELECT pmid, title, journal, pub_date
        FROM mr_pubmed_data
        WHERE journal ILIKE ?
        ORDER BY pub_date DESC
        LIMIT ?
    """,
        (f"%{journal_pattern}%", limit),
    ).fetchall()

    return result


def list_journals_stats(
    conn: duckdb.DuckDBPyConnection,
) -> List[Tuple[str, int]]:
    """List all journals with paper counts.

    Args:
        conn: DuckDB connection

    Returns:
        List of tuples (journal_name, paper_count)
    """
    result = conn.execute("""
        SELECT journal, COUNT(*) as paper_count
        FROM mr_pubmed_data
        GROUP BY journal
        ORDER BY paper_count DESC, journal
    """).fetchall()

    return result


def find_trait_by_index(
    conn: duckdb.DuckDBPyConnection, trait_index: int
) -> List[Tuple[str, str, str]]:
    """Find trait information by trait_index.

    Args:
        conn: DuckDB connection
        trait_index: The trait_index to search for

    Returns:
        List of tuples (trait_label, trait_role, count)
    """
    # First get the trait label
    trait_info = conn.execute(
        """
        SELECT trait_label
        FROM trait_embeddings
        WHERE trait_index = ?
    """,
        (trait_index,),
    ).fetchone()

    if not trait_info:
        return []

    trait_label = trait_info[0]

    # Get usage statistics
    result = conn.execute(
        """
        SELECT 
            ? as trait_label,
            trait_role,
            COUNT(*) as count
        FROM model_result_traits
        WHERE trait_index = ?
        GROUP BY trait_role
        ORDER BY count DESC
    """,
        (trait_label, trait_index),
    ).fetchall()

    return result


def get_trait_embedding_by_label(
    conn: duckdb.DuckDBPyConnection, trait_label: str
) -> Tuple[int, List[float]] | None:
    """Get embedding vector for a specific trait label.

    Args:
        conn: DuckDB connection
        trait_label: The trait label to search for

    Returns:
        Tuple of (trait_index, embedding_vector) or None if not found
    """
    result = conn.execute(
        """
        SELECT trait_index, vector
        FROM trait_embeddings
        WHERE trait_label = ?
        LIMIT 1
    """,
        (trait_label,),
    ).fetchone()

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
        # Validate database schema before executing queries
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
        if args.list_models:
            logger.info("Listing models and their statistics...")
            models = list_models_stats(conn)
            print("\\nModels in database:")
            print("Model\t\tResults\tTraits")
            print("-" * 40)
            for model, num_results, num_traits in models:
                print(f"{model}\t\t{num_results}\t{num_traits}")

        if args.list_traits:
            logger.info("Listing all traits...")
            traits = list_all_traits(conn)
            print(f"\nFound {len(traits)} traits:")
            for trait_index, trait_label in traits[:50]:  # Show first 50
                print(f"{trait_index}: {trait_label}")
            if len(traits) > 50:
                print(f"... and {len(traits) - 50} more traits")

        if args.trait_by_index is not None:
            logger.info(
                f"Finding trait with trait_index {args.trait_by_index}..."
            )
            traits = find_trait_by_index(conn, args.trait_by_index)
            if traits:
                print(f"\nTrait with trait_index {args.trait_by_index}:")
                for trait_label, trait_role, count in traits:
                    print(
                        f"  - {trait_label} (role: {trait_role}, count: {count})"
                    )
            else:
                print(
                    f"\nNo trait found with trait_index {args.trait_by_index}"
                )

        if args.query_trait:
            logger.info(f"Finding similar traits for: {args.query_trait}")

            # First check if the trait exists
            trait_result = get_trait_embedding_by_label(conn, args.query_trait)
            if trait_result is None:
                print(f"\\nTrait '{args.query_trait}' not found in database.")
                # Suggest similar trait names
                similar_names = conn.execute(
                    """
                    SELECT trait_label
                    FROM trait_embeddings
                    WHERE trait_label ILIKE ?
                    LIMIT 5
                """,
                    (f"%{args.query_trait}%",),
                ).fetchall()

                if similar_names:
                    print("Did you mean one of these?")
                    for (name,) in similar_names:
                        print(f"  - {name}")
            else:
                trait_index, embedding = trait_result
                similar_traits = find_similar_traits(
                    conn, args.query_trait, args.limit
                )
                print(
                    f"\nTop {len(similar_traits)} most similar traits to '{args.query_trait}':"
                )
                print("Similarity\tTrait Index\t\tTrait Label")
                print("-" * 80)
                for trait_index, trait_label, similarity in similar_traits:
                    print(
                        f"{similarity:.4f}\t\t{trait_index}\t\t{trait_label}"
                    )

        if args.query_efo:
            logger.info(f"Finding similar EFO terms for: {args.query_efo}")

            # First check if the trait exists
            trait_result = get_trait_embedding_by_label(conn, args.query_efo)
            if trait_result is None:
                print(f"\\nTrait '{args.query_efo}' not found in database.")
                # Suggest similar trait names
                similar_names = conn.execute(
                    """
                    SELECT trait_label
                    FROM trait_embeddings
                    WHERE trait_label ILIKE ?
                    LIMIT 5
                """,
                    (f"%{args.query_efo}%",),
                ).fetchall()

                if similar_names:
                    print("Did you mean one of these?")
                    for (name,) in similar_names:
                        print(f"  - {name}")
            else:
                trait_index, embedding = trait_result
                similar_efo = find_similar_efo_terms(
                    conn, args.query_efo, args.limit
                )
                print(
                    f"\nTop {len(similar_efo)} most similar EFO terms to '{args.query_efo}':"
                )
                print("Similarity\tEFO ID\t\t\t\t\tEFO Label")
                print("-" * 120)
                for efo_id, efo_label, similarity in similar_efo:
                    # Truncate long IDs for display
                    short_id = (
                        efo_id if len(efo_id) <= 40 else efo_id[:37] + "..."
                    )
                    print(f"{similarity:.4f}\t\t{short_id}\t{efo_label}")

        if args.query_pmid:
            logger.info(f"Querying PubMed data for PMID: {args.query_pmid}")
            pubmed_data = query_pubmed_by_pmid(conn, args.query_pmid)
            if pubmed_data:
                (
                    pmid,
                    title,
                    abstract,
                    pub_date,
                    journal,
                    journal_issn,
                    author_affil,
                ) = pubmed_data
                print(f"\nPubMed Data for PMID {pmid}:")
                print(f"Title: {title}")
                print(f"Journal: {journal} ({pub_date})")
                if journal_issn:
                    print(f"ISSN: {journal_issn}")
                if author_affil:
                    print(f"Author Affiliation: {author_affil}")
                print(f"\nAbstract:\n{abstract}")
            else:
                print(f"\nNo PubMed data found for PMID {args.query_pmid}")

        if args.query_journal:
            logger.info(f"Querying papers by journal: {args.query_journal}")
            papers = query_papers_by_journal(
                conn, args.query_journal, args.limit
            )
            if papers:
                print(
                    f"\nPapers from journals matching '{args.query_journal}':"
                )
                print("PMID\t\tTitle\t\t\t\t\t\t\tJournal\t\t\tDate")
                print("-" * 150)
                for pmid, title, journal, pub_date in papers:
                    truncated_title = (
                        title[:50] + "..." if len(title) > 50 else title
                    )
                    truncated_journal = (
                        journal[:20] + "..." if len(journal) > 20 else journal
                    )
                    print(
                        f"{pmid}\t{truncated_title:<53}\t{truncated_journal:<23}\t{pub_date}"
                    )
            else:
                print(
                    f"\nNo papers found for journal pattern '{args.query_journal}'"
                )

        if args.list_journals:
            logger.info("Listing all journals...")
            journals = list_journals_stats(conn)
            print(f"\nFound {len(journals)} journals:")
            print("Papers\tJournal")
            print("-" * 80)
            for journal, count in journals[:20]:  # Show top 20
                print(f"{count}\t{journal}")
            if len(journals) > 20:
                print(f"... and {len(journals) - 20} more journals")

    return 0


if __name__ == "__main__":
    exit(main())
