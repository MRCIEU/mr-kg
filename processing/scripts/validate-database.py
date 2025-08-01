#!/usr/bin/env python3
"""Database validation script for the vector store.

This script validates the structure of the vector store database against
the expected schema definitions.
"""

import argparse
from pathlib import Path
import duckdb

from common_funcs.schema.database_schema import (
    validate_database_schema,
    print_validation_report,
    get_schema_documentation,
)
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
        help="Path to the DuckDB database file to validate (default: restructured-vector-store.db)",
        default="restructured-vector-store.db",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="Show the expected schema documentation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate, don't show detailed report",
    )
    return parser.parse_args()


def main():
    """Main function to validate the database schema."""
    args = make_args()

    if args.show_schema:
        print(get_schema_documentation())
        return 0

    # Resolve database path
    db_path = Path(args.database)
    if not db_path.is_absolute():
        PROJECT_ROOT = find_project_root("docker-compose.yml")
        DATA_DIR = PROJECT_ROOT / "data"
        db_path = DATA_DIR / "db" / args.database

    if not db_path.exists():
        print(f"[ERROR] Database file not found: {db_path}")
        return 1

    print(f"Validating database: {db_path}")
    print(f"Database size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Connect to database and validate schema
    try:
        with duckdb.connect(str(db_path)) as conn:
            validation_results = validate_database_schema(conn)

            if args.validate_only:
                if validation_results["valid"]:
                    print("[OK] Database schema is valid")
                    return 0
                else:
                    print("[ERROR] Database schema validation failed")
                    for error in validation_results["errors"]:
                        print(f"   {error}")
                    return 1
            else:
                print_validation_report(validation_results)
                return 0 if validation_results["valid"] else 1

    except Exception as e:
        print(f"[ERROR] Error connecting to database: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
