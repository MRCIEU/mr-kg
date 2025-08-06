#!/usr/bin/env python3
"""Script to describe the structure of the MR-KG DuckDB databases.

This script connects to both the vector store and trait profile databases
and provides detailed information about their structure, including:
- Database overview
- Tables and their schemas
- Views and their definitions
- Index information
- Row counts for each table
"""

# Import after path setup
import duckdb

from common_funcs.database_utils.utils import get_database_paths


def print_section_header(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 60}")
    print(f"{title.center(60)}")
    print(f"{char * 60}")


def print_subsection_header(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-' * 40}")
    print(f"{title}")
    print(f"{'-' * 40}")


def describe_database(conn, db_name: str):
    """Describe a single database connection."""
    print_section_header(f"{db_name.upper()} DATABASE")

    # Get database version and info
    try:
        version_info = conn.execute("SELECT version()").fetchone()
        if version_info:
            print(f"DuckDB Version: {version_info[0]}")
    except Exception as e:
        print(f"Could not get version info: {e}")

    # List all tables
    print_subsection_header("TABLES")
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        if tables:
            print(f"Found {len(tables)} tables:")
            for table in tables:
                table_name = table[0]
                print(f"  - {table_name}")

                # Get row count for each table
                try:
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM {table_name}"
                    ).fetchone()
                    row_count = count[0] if count else 0
                    print(f"    Rows: {row_count:,}")
                except Exception as e:
                    print(f"    Error getting row count: {e}")
        else:
            print("No tables found.")
    except Exception as e:
        print(f"Error listing tables: {e}")

    # List all views
    print_subsection_header("VIEWS")
    try:
        views = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'"
        ).fetchall()
        if views:
            print(f"Found {len(views)} views:")
            for view in views:
                view_name = view[0]
                print(f"  - {view_name}")

                # Get row count for each view
                try:
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM {view_name}"
                    ).fetchone()
                    row_count = count[0] if count else 0
                    print(f"    Rows: {row_count:,}")
                except Exception as e:
                    print(f"    Error getting row count: {e}")
        else:
            print("No views found.")
    except Exception as e:
        print(f"Error listing views: {e}")

    # Describe table schemas
    print_subsection_header("TABLE SCHEMAS")
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        for table in tables:
            table_name = table[0]
            print(f"\n📋 Table: {table_name}")
            try:
                schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
                if schema:
                    print("  Columns:")
                    for col in schema:
                        col_name = col[0]
                        col_type = col[1]
                        nullable = "NULL" if col[2] else "NOT NULL"
                        key_info = ""
                        if len(col) > 3 and col[3]:  # Primary key info
                            key_info = " (PRIMARY KEY)"
                        print(
                            f"    {col_name}: {col_type} {nullable}{key_info}"
                        )
                else:
                    print("    No schema information available")
            except Exception as e:
                print(f"    Error describing table: {e}")
    except Exception as e:
        print(f"Error getting table schemas: {e}")

    # List indexes
    print_subsection_header("INDEXES")
    try:
        # DuckDB doesn't have a standard way to list all indexes, so we'll try a few approaches
        indexes = conn.execute("""
            SELECT table_name, index_name, is_unique
            FROM duckdb_indexes()
            ORDER BY table_name, index_name
        """).fetchall()

        if indexes:
            print(f"Found {len(indexes)} indexes:")
            current_table = None
            for index in indexes:
                table_name, index_name, is_unique = index
                if table_name != current_table:
                    print(f"\n  Table: {table_name}")
                    current_table = table_name
                unique_str = " (UNIQUE)" if is_unique else ""
                print(f"    - {index_name}{unique_str}")
        else:
            print(
                "No custom indexes found (or index information not available)."
            )
    except Exception as e:
        print(f"Index information not available: {e}")

    # Database file size and other metadata
    print_subsection_header("DATABASE METADATA")
    try:
        # Get database size if possible
        db_info = conn.execute("PRAGMA database_list").fetchall()
        if db_info:
            for db in db_info:
                print(f"Database: {db[1]}")
    except Exception as e:
        print(f"Could not get database metadata: {e}")


def main():
    """Main function to describe both databases."""
    print_section_header("MR-KG DATABASE DESCRIPTION TOOL")
    print(
        "This tool provides detailed information about the structure of the MR-KG databases."
    )

    # Set up database paths similar to the webapp setup
    vector_store_db_path, trait_profile_db_path = get_database_paths(
        profile="local"
    )
    # Connect to databases
    vector_conn = duckdb.connect(str(vector_store_db_path), read_only=True)
    trait_conn = duckdb.connect(str(trait_profile_db_path), read_only=True)

    # Describe vector store database
    describe_database(vector_conn, "Vector Store")

    # Describe trait profile database
    describe_database(trait_conn, "Trait Profile")

    print_section_header("DESCRIPTION COMPLETE")
    print("Database description completed successfully!")


if __name__ == "__main__":
    main()
