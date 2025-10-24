"""Generate database schema documentation from schema definitions.

This script generates a comprehensive markdown documentation file from the
database schema definitions in common_funcs. The output includes:
- Overview and quick reference for both databases
- Detailed table descriptions with mermaid ER diagrams
- Index documentation
- View definitions with SQL
- Live database statistics (row counts, version info)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import duckdb
from yiutils.project_utils import find_project_root

PROJECT_ROOT = find_project_root("docker-compose.yml")
COMMON_FUNCS_PATH = PROJECT_ROOT / "src" / "common_funcs"
sys.path.insert(0, str(COMMON_FUNCS_PATH))

from common_funcs.database_utils.utils import get_database_paths  # noqa: E402
from common_funcs.schema.database_schema import (  # noqa: E402
    DATABASE_SCHEMA,
    DATABASE_INDEXES,
    DATABASE_VIEWS,
)
from common_funcs.schema.database_schema_utils import (  # noqa: E402
    ColumnType,
    IndexDef,
    TableDef,
    ViewDef,
)
from common_funcs.schema.trait_profile_schema import (  # noqa: E402
    TRAIT_PROFILE_INDEXES,
    TRAIT_PROFILE_SCHEMA,
    TRAIT_PROFILE_VIEWS,
)
from common_funcs.schema.evidence_profile_schema import (  # noqa: E402
    EVIDENCE_PROFILE_INDEXES,
    EVIDENCE_PROFILE_SCHEMA,
    EVIDENCE_PROFILE_VIEWS,
)


def make_args():
    """Parse command line arguments.

    Returns:
        Parsed arguments with output_file option
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "processing" / "db-schema.md"),
        help="Output markdown file path",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing file",
    )

    return parser.parse_args()


def format_column_type(col_type: ColumnType) -> str:
    """Format column type for markdown display.

    Args:
        col_type: Column type enum value

    Returns:
        Human-readable type string
    """
    type_map = {
        ColumnType.INTEGER: "INTEGER",
        ColumnType.VARCHAR: "VARCHAR",
        ColumnType.JSON: "JSON",
        ColumnType.FLOAT_ARRAY_200: "FLOAT[200]",
        ColumnType.DOUBLE: "DOUBLE",
    }
    res = type_map.get(col_type, str(col_type))
    return res


def parse_sql_select_columns(sql: str) -> List[tuple]:
    """Parse SQL SELECT statement to extract column definitions.

    Args:
        sql: SQL query string

    Returns:
        List of (column_name, source_expr) tuples
    """
    import re

    sql_normalized = " ".join(sql.split())
    select_match = re.search(
        r"SELECT\s+(.*?)\s+FROM", sql_normalized, re.IGNORECASE | re.DOTALL
    )

    if not select_match:
        return []

    select_clause = select_match.group(1)
    columns = []

    parts = []
    paren_depth = 0
    current = []

    for char in select_clause:
        if char == "(":
            paren_depth += 1
            current.append(char)
        elif char == ")":
            paren_depth -= 1
            current.append(char)
        elif char == "," and paren_depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        parts.append("".join(current).strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue

        as_match = re.search(r"\s+as\s+(\w+)\s*$", part, re.IGNORECASE)
        if as_match:
            col_name = as_match.group(1)
            source_expr = part[: as_match.start()].strip()
        else:
            dot_match = re.search(r"(\w+)\.(\w+)$", part)
            if dot_match:
                col_name = dot_match.group(2)
                source_expr = part
            else:
                col_name = part.split()[-1] if part else "unknown"
                source_expr = part

        columns.append((col_name, source_expr))

    return columns


def extract_source_table_column(expr: str, tables: Dict[str, TableDef]) -> str:
    """Extract source table.column from a SQL expression.

    Args:
        expr: SQL expression (e.g., "t1.trait_index", "mr.pmid")
        tables: Dictionary of table definitions

    Returns:
        Source in format "table.column", "computed", or "aggregated"
    """
    import re

    table_aliases = {
        "t1": "trait_embeddings",
        "t2": "trait_embeddings",
        "t": "trait_embeddings",
        "e": "efo_embeddings",
        "mr": "model_results",
        "mpd": "mr_pubmed_data",
        "mrt": "model_result_traits",
    }

    if "array_cosine_similarity" in expr.lower():
        return "computed_similarity"

    if "COALESCE" in expr.upper() and "LIST" in expr.upper():
        if "mrt." in expr:
            return "aggregated"

    simple_match = re.search(r"(\w+)\.(\w+)", expr)
    if simple_match:
        alias = simple_match.group(1)
        column = simple_match.group(2)
        table = table_aliases.get(alias, alias)
        return f"{table}.{column}"

    if "(" in expr or "+" in expr or "-" in expr or "*" in expr or "/" in expr:
        return "computed"

    return "unknown"


def infer_column_type(
    col_name: str, source: str, tables: Dict[str, TableDef]
) -> str:
    """Infer SQL type for a view column.

    Args:
        col_name: Column name
        source: Source expression (e.g., "table.column", "computed")
        tables: Dictionary of table definitions

    Returns:
        SQL type string
    """
    if source == "computed_similarity":
        return "FLOAT"

    if source == "aggregated":
        return "JSON"

    if source == "computed":
        if "similarity" in col_name.lower():
            return "FLOAT"
        return "VARCHAR"

    if "." in source:
        table_name, source_col = source.split(".", 1)
        if table_name in tables:
            for col in tables[table_name].columns:
                if col.name == source_col:
                    return format_column_type(col.type)

    if "id" in col_name.lower() and "index" in col_name.lower():
        return "INTEGER"
    if "pmid" in col_name.lower() or "label" in col_name.lower():
        return "VARCHAR"
    if "metadata" in col_name.lower() or "results" in col_name.lower():
        return "JSON"

    return "VARCHAR"


def extract_view_columns(
    views: List[ViewDef], tables: Dict[str, TableDef]
) -> Dict[str, List[tuple]]:
    """Extract column definitions from view SQL statements.

    Args:
        views: List of view definitions
        tables: Dictionary of table definitions

    Returns:
        Dictionary mapping view names to list of (type, name, source) tuples
    """
    result = {}

    for view in views:
        columns = parse_sql_select_columns(view.sql)
        view_cols = []

        for col_name, source_expr in columns:
            source = extract_source_table_column(source_expr, tables)
            col_type = infer_column_type(col_name, source, tables)
            view_cols.append((col_type, col_name, source))

        result[view.name] = view_cols

    return result


def extract_view_table_refs(views: List[ViewDef]) -> Dict[str, List[str]]:
    """Extract table references from view SQL statements.

    Args:
        views: List of view definitions

    Returns:
        Dictionary mapping view names to list of referenced table names
    """
    import re

    result = {}

    table_aliases = {
        "t1": "trait_embeddings",
        "t2": "trait_embeddings",
        "t": "trait_embeddings",
        "e": "efo_embeddings",
        "mr": "model_results",
        "mpd": "mr_pubmed_data",
        "mrt": "model_result_traits",
    }

    for view in views:
        sql_normalized = " ".join(view.sql.split())

        from_match = re.search(r"FROM\s+(\w+)", sql_normalized, re.IGNORECASE)
        join_matches = re.findall(
            r"(?:LEFT\s+JOIN|CROSS\s+JOIN|JOIN)\s+(\w+)",
            sql_normalized,
            re.IGNORECASE,
        )

        tables = set()
        if from_match:
            table = from_match.group(1)
            tables.add(table_aliases.get(table, table))

        for match in join_matches:
            table = match
            tables.add(table_aliases.get(table, table))

        result[view.name] = sorted(tables)

    return result


def generate_mermaid_diagram(
    tables: Dict[str, TableDef], views: List[ViewDef]
) -> str:
    """Generate mermaid ER diagram for all tables and views.

    Args:
        tables: Dictionary of table definitions
        views: List of view definitions

    Returns:
        Mermaid diagram as string
    """
    lines = ["```mermaid", "erDiagram"]

    for table_name, table_def in tables.items():
        lines.append(f"    {table_name} {{")

        for col in table_def.columns:
            type_str = format_column_type(col.type)
            constraints = []
            if col.primary_key:
                constraints.append("PK")
            if table_def.foreign_keys and any(
                fk.column == col.name for fk in table_def.foreign_keys
            ):
                constraints.append("FK")

            constraint_str = f" {' '.join(constraints)}" if constraints else ""
            lines.append(f"        {type_str} {col.name}{constraint_str}")

        lines.append("    }")

    view_columns = extract_view_columns(views, tables)

    for view in views:
        view_name = view.name
        lines.append(f"    {view_name} {{")

        if view_name in view_columns:
            for col_type, col_name, source in view_columns[view_name]:
                lines.append(f'        {col_type} {col_name} "from_{source}"')

        lines.append("    }")

    for table_def in tables.values():
        if table_def.foreign_keys:
            for fk in table_def.foreign_keys:
                lines.append(
                    f"    {table_def.name} }}o--|| "
                    f"{fk.ref_table} : "
                    f'"{fk.column} references {fk.ref_column}"'
                )

    view_table_refs = extract_view_table_refs(views)

    for view in views:
        if view.name in view_table_refs:
            for table_name in view_table_refs[view.name]:
                lines.append(f'    {view.name} }}o..o{{ {table_name} : "uses"')

    lines.append("")
    lines.append("    %% Styling")
    for view in views:
        lines.append(
            f"    style {view.name} fill:#e1f5ff,stroke:#0288d1,stroke-width:2px"
        )

    lines.append("```")
    res = "\n".join(lines)
    return res


def generate_table_documentation(table_name: str, table_def: TableDef) -> str:
    """Generate detailed markdown documentation for a single table.

    Args:
        table_name: Name of the table
        table_def: Table definition object

    Returns:
        Markdown documentation string
    """
    lines = [
        f"#### {table_name}",
        "",
        table_def.description,
        "",
        "**Columns:**",
        "",
    ]

    for col in table_def.columns:
        type_str = format_column_type(col.type)
        nullable_str = "nullable" if col.nullable else "NOT NULL"
        pk_str = " (PRIMARY KEY)" if col.primary_key else ""

        lines.append(
            f"- **`{col.name}`** ({type_str}, {nullable_str}){pk_str}"
        )
        lines.append("")

    if table_def.foreign_keys:
        lines.append("**Foreign Keys:**")
        lines.append("")
        for fk in table_def.foreign_keys:
            lines.append(
                f"- `{fk.column}` -> `{fk.ref_table}.{fk.ref_column}`"
            )
        lines.append("")

    res = "\n".join(lines)
    return res


def generate_index_documentation(indexes: List[IndexDef]) -> str:
    """Generate documentation for database indexes.

    Args:
        indexes: List of index definitions

    Returns:
        Markdown documentation string
    """
    lines = ["## Indexes", "", "Performance optimization indexes:", ""]

    by_table: Dict[str, List[IndexDef]] = {}
    for idx in indexes:
        if idx.table not in by_table:
            by_table[idx.table] = []
        by_table[idx.table].append(idx)

    for table_name in sorted(by_table.keys()):
        lines.append(f"#### {table_name}")
        lines.append("")

        for idx in by_table[table_name]:
            columns_str = ", ".join(idx.columns)
            lines.append(f"- **`{idx.name}`** on ({columns_str})")
            lines.append("")

    res = "\n".join(lines)
    return res


def generate_view_documentation(views: List[ViewDef]) -> str:
    """Generate documentation for database views.

    Args:
        views: List of view definitions

    Returns:
        Markdown documentation string
    """
    lines = ["## Views", "", "Pre-computed views for common queries:", ""]

    for view in views:
        lines.append(f"#### {view.name}")
        lines.append("")

        if hasattr(view, "description") and view.description:
            lines.append(view.description)
            lines.append("")

        lines.append("**SQL Definition:**")
        lines.append("")
        lines.append("```sql")
        lines.append(view.sql.strip())
        lines.append("```")
        lines.append("")

    res = "\n".join(lines)
    return res


def generate_quick_reference(tables: Dict[str, TableDef]) -> str:
    """Generate a quick reference table summary.

    Args:
        tables: Dictionary of table definitions

    Returns:
        Markdown table string
    """
    lines = [
        "### Quick reference",
        "",
        "| Table | Description | Key Columns |",
        "|-------|-------------|-------------|",
    ]

    for table_name, table_def in tables.items():
        pk_cols = [c.name for c in table_def.columns if c.primary_key]
        pk_str = ", ".join(pk_cols) if pk_cols else "none"
        desc = table_def.description.split(".")[0]
        lines.append(f"| `{table_name}` | {desc} | {pk_str} |")

    lines.append("")
    res = "\n".join(lines)
    return res


def get_database_statistics(
    conn: duckdb.DuckDBPyConnection, db_name: str
) -> Dict:
    """Get comprehensive statistics for a database.

    Args:
        conn: DuckDB connection
        db_name: Database name for display

    Returns:
        Dictionary with version, table counts, and index info
    """
    stats = {
        "version": None,
        "tables": {},
        "indexes": {},
    }

    try:
        version_info = conn.execute("SELECT version()").fetchone()
        if version_info:
            stats["version"] = version_info[0]
    except Exception:
        pass

    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        for table in tables:
            table_name = table[0]
            try:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                ).fetchone()
                stats["tables"][table_name] = count[0] if count else 0
            except Exception:
                stats["tables"][table_name] = -1

        views = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_type = 'VIEW'"
        ).fetchall()
        for view in views:
            view_name = view[0]
            try:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {view_name}"
                ).fetchone()
                stats["tables"][view_name] = count[0] if count else 0
            except Exception:
                stats["tables"][view_name] = -1
    except Exception as e:
        print(f"Warning: Could not get statistics for {db_name}: {e}")

    try:
        indexes = conn.execute(
            "SELECT table_name, index_name, is_unique "
            "FROM duckdb_indexes() "
            "ORDER BY table_name, index_name"
        ).fetchall()
        for table_name, index_name, is_unique in indexes:
            if table_name not in stats["indexes"]:
                stats["indexes"][table_name] = []
            stats["indexes"][table_name].append(
                {"name": index_name, "unique": is_unique}
            )
    except Exception:
        pass

    return stats


def generate_database_statistics_section(
    vector_stats: Dict, trait_stats: Dict, evidence_stats: Dict
) -> str:
    """Generate markdown section with live database statistics.

    Args:
        vector_stats: Statistics dict for vector_store.db
        trait_stats: Statistics dict for trait_profile_db.db
        evidence_stats: Statistics dict for evidence_profile_db.db

    Returns:
        Markdown section with statistics
    """
    lines = [
        "## Database statistics",
        "",
        "Live statistics from the actual database files.",
        "",
    ]

    if vector_stats:
        lines.extend(
            [
                "### Vector store database",
                "",
            ]
        )
        if vector_stats.get("version"):
            lines.append(f"**DuckDB Version:** {vector_stats['version']}")
            lines.append("")

        lines.extend(
            [
                "| Table/View | Row Count |",
                "|------------|-----------|",
            ]
        )
        for name in sorted(vector_stats.get("tables", {}).keys()):
            count = vector_stats["tables"][name]
            count_str = f"{count:,}" if count >= 0 else "Error"
            lines.append(f"| `{name}` | {count_str} |")
        lines.append("")

    if trait_stats:
        lines.extend(
            [
                "### Trait profile database",
                "",
            ]
        )
        if trait_stats.get("version"):
            lines.append(f"**DuckDB Version:** {trait_stats['version']}")
            lines.append("")

        lines.extend(
            [
                "| Table/View | Row Count |",
                "|------------|-----------|",
            ]
        )
        for name in sorted(trait_stats.get("tables", {}).keys()):
            count = trait_stats["tables"][name]
            count_str = f"{count:,}" if count >= 0 else "Error"
            lines.append(f"| `{name}` | {count_str} |")
        lines.append("")

    if evidence_stats:
        lines.extend(
            [
                "### Evidence profile database",
                "",
            ]
        )
        if evidence_stats.get("version"):
            lines.append(f"**DuckDB Version:** {evidence_stats['version']}")
            lines.append("")

        lines.extend(
            [
                "| Table/View | Row Count |",
                "|------------|-----------|",
            ]
        )
        for name in sorted(evidence_stats.get("tables", {}).keys()):
            count = evidence_stats["tables"][name]
            count_str = f"{count:,}" if count >= 0 else "Error"
            lines.append(f"| `{name}` | {count_str} |")
        lines.append("")

    res = "\n".join(lines)
    return res


def generate_schema_documentation(
    vector_tables: Dict[str, TableDef],
    vector_indexes: List[IndexDef],
    vector_views: List[ViewDef],
    trait_tables: Dict[str, TableDef],
    trait_indexes: List[IndexDef],
    trait_views: List[ViewDef],
    evidence_tables: Dict[str, TableDef],
    evidence_indexes: List[IndexDef],
    evidence_views: List[ViewDef],
    vector_stats: Dict[str, int],
    trait_stats: Dict[str, int],
    evidence_stats: Dict[str, int],
) -> str:
    """Generate complete schema documentation for all databases.

    Args:
        vector_tables: Vector store table definitions
        vector_indexes: Vector store index definitions
        vector_views: Vector store view definitions
        trait_tables: Trait profile table definitions
        trait_indexes: Trait profile index definitions
        trait_views: Trait profile view definitions
        evidence_tables: Evidence profile table definitions
        evidence_indexes: Evidence profile index definitions
        evidence_views: Evidence profile view definitions
        vector_stats: Vector store database statistics
        trait_stats: Trait profile database statistics
        evidence_stats: Evidence profile database statistics

    Returns:
        Complete markdown documentation
    """
    sections = [
        "# Database schema",
        "",
        "Auto-generated documentation from schema definitions and live database statistics.",
        "",
        "This document covers three databases:",
        "- **Vector store database** (`vector_store.db`): MR-KG embeddings and analysis",
        "- **Trait profile database** (`trait_profile_db.db`): Trait similarity profiles",
        "- **Evidence profile database** (`evidence_profile_db.db`): Evidence similarity profiles",
        "",
        generate_database_statistics_section(
            vector_stats, trait_stats, evidence_stats
        ),
        "",
        "## Vector store database",
        "",
        "### Overview",
        "",
        generate_mermaid_diagram(vector_tables, vector_views),
        "",
        generate_quick_reference(vector_tables),
        "",
        "### Tables",
        "",
    ]

    for table_name, table_def in vector_tables.items():
        sections.append(generate_table_documentation(table_name, table_def))

    sections.append(
        generate_index_documentation(vector_indexes).replace(
            "## Indexes", "### Indexes"
        )
    )
    sections.append(
        generate_view_documentation(vector_views).replace(
            "## Views", "### Views"
        )
    )

    sections.extend(
        [
            "",
            "## Trait profile database",
            "",
            "### Overview",
            "",
            generate_mermaid_diagram(trait_tables, trait_views),
            "",
            generate_quick_reference(trait_tables),
            "",
            "### Tables",
            "",
        ]
    )

    for table_name, table_def in trait_tables.items():
        sections.append(generate_table_documentation(table_name, table_def))

    sections.append(
        generate_index_documentation(trait_indexes).replace(
            "## Indexes", "### Indexes"
        )
    )
    sections.append(
        generate_view_documentation(trait_views).replace(
            "## Views", "### Views"
        )
    )

    sections.extend(
        [
            "",
            "## Evidence profile database",
            "",
            "### Overview",
            "",
            generate_mermaid_diagram(evidence_tables, evidence_views),
            "",
            generate_quick_reference(evidence_tables),
            "",
            "### Tables",
            "",
        ]
    )

    for table_name, table_def in evidence_tables.items():
        sections.append(generate_table_documentation(table_name, table_def))

    sections.append(
        generate_index_documentation(evidence_indexes).replace(
            "## Indexes", "### Indexes"
        )
    )
    sections.append(
        generate_view_documentation(evidence_views).replace(
            "## Views", "### Views"
        )
    )

    res = "\n".join(sections)
    return res


def main():
    """Main function to generate schema documentation.

    This function loads the schema definitions for both databases and generates
    a comprehensive markdown documentation file with live statistics.
    """
    args = make_args()

    print("Generating schema documentation for all databases...")
    print(f"  Vector store - Tables: {len(DATABASE_SCHEMA)}")
    print(f"  Vector store - Indexes: {len(DATABASE_INDEXES)}")
    print(f"  Vector store - Views: {len(DATABASE_VIEWS)}")
    print(f"  Trait profile - Tables: {len(TRAIT_PROFILE_SCHEMA)}")
    print(f"  Trait profile - Indexes: {len(TRAIT_PROFILE_INDEXES)}")
    print(f"  Trait profile - Views: {len(TRAIT_PROFILE_VIEWS)}")
    print(f"  Evidence profile - Tables: {len(EVIDENCE_PROFILE_SCHEMA)}")
    print(f"  Evidence profile - Indexes: {len(EVIDENCE_PROFILE_INDEXES)}")
    print(f"  Evidence profile - Views: {len(EVIDENCE_PROFILE_VIEWS)}")

    vector_stats: Dict[str, int] = {}
    trait_stats: Dict[str, int] = {}
    evidence_stats: Dict[str, int] = {}

    try:
        print("\nConnecting to databases for live statistics...")
        vector_db_path, trait_db_path = get_database_paths(profile="local")

        vector_conn = duckdb.connect(str(vector_db_path), read_only=True)
        vector_stats = get_database_statistics(vector_conn, "Vector Store")
        vector_conn.close()
        print(
            f"  Vector store: {len(vector_stats.get('tables', {}))} tables/views found"
        )

        trait_conn = duckdb.connect(str(trait_db_path), read_only=True)
        trait_stats = get_database_statistics(trait_conn, "Trait Profile")
        trait_conn.close()
        print(
            f"  Trait profile: {len(trait_stats.get('tables', {}))} tables/views found"
        )

        evidence_db_path = (
            PROJECT_ROOT / "data" / "db" / "evidence_profile_db.db"
        )
        if evidence_db_path.exists():
            evidence_conn = duckdb.connect(
                str(evidence_db_path), read_only=True
            )
            evidence_stats = get_database_statistics(
                evidence_conn, "Evidence Profile"
            )
            evidence_conn.close()
            print(
                f"  Evidence profile: {len(evidence_stats.get('tables', {}))} tables/views found"
            )
        else:
            print(
                "  Warning: Evidence profile database not found, skipping statistics"
            )
    except Exception as e:
        print(f"Warning: Could not connect to databases: {e}")
        print("Continuing without live statistics...")

    doc_content = generate_schema_documentation(
        DATABASE_SCHEMA,
        DATABASE_INDEXES,
        DATABASE_VIEWS,
        TRAIT_PROFILE_SCHEMA,
        TRAIT_PROFILE_INDEXES,
        TRAIT_PROFILE_VIEWS,
        EVIDENCE_PROFILE_SCHEMA,
        EVIDENCE_PROFILE_INDEXES,
        EVIDENCE_PROFILE_VIEWS,
        vector_stats,
        trait_stats,
        evidence_stats,
    )

    if args.dry_run:
        print("\nDry run - would generate:")
        print(doc_content[:500] + "...")
        return 0

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write(doc_content)

    print(f"\nDocumentation written to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
