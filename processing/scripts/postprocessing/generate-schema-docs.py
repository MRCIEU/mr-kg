"""Generate database schema documentation from schema definitions.

This script generates a comprehensive markdown documentation file from the
database schema definitions in common_funcs. The output includes:
- Overview and quick reference
- Detailed table descriptions with mermaid ER diagrams
- Index documentation
- View definitions with SQL
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from yiutils.project_utils import find_project_root

PROJECT_ROOT = find_project_root("docker-compose.yml")
COMMON_FUNCS_PATH = PROJECT_ROOT / "src" / "common_funcs"
sys.path.insert(0, str(COMMON_FUNCS_PATH))

from common_funcs.schema.database_schema import (
    DATABASE_SCHEMA,
    DATABASE_INDEXES,
    DATABASE_VIEWS,
)
from common_funcs.schema.database_schema_utils import (
    TableDef,
    IndexDef,
    ViewDef,
    ColumnType,
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
        default=str(PROJECT_ROOT / "processing" / "docs" / "db_schema.md"),
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
    }
    res = type_map.get(col_type, str(col_type))
    return res


def generate_mermaid_diagram(tables: Dict[str, TableDef]) -> str:
    """Generate mermaid ER diagram for all tables.

    Args:
        tables: Dictionary of table definitions

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
            if not col.nullable:
                constraints.append("NOT NULL")

            constraint_str = (
                f" {','.join(constraints)}" if constraints else ""
            )
            lines.append(f"        {type_str} {col.name}{constraint_str}")

        lines.append("    }")

    for table_def in tables.values():
        if table_def.foreign_keys:
            for fk in table_def.foreign_keys:
                lines.append(
                    f"    {table_def.name} ||--o{{ "
                    f"{fk.ref_table} : "
                    f'"{fk.column} -> {fk.ref_column}"'
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
        f"### {table_name}",
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

        lines.append(f"- **`{col.name}`** ({type_str}, {nullable_str}){pk_str}")
        lines.append("")

    if table_def.foreign_keys:
        lines.append("**Foreign Keys:**")
        lines.append("")
        for fk in table_def.foreign_keys:
            lines.append(
                f"- `{fk.column}` -> " f"`{fk.ref_table}.{fk.ref_column}`"
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
        lines.append(f"### {table_name}")
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
        lines.append(f"### {view.name}")
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
        "## Quick reference",
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


def generate_schema_documentation(
    tables: Dict[str, TableDef],
    indexes: List[IndexDef],
    views: List[ViewDef],
) -> str:
    """Generate complete schema documentation.

    Args:
        tables: Dictionary of table definitions
        indexes: List of index definitions
        views: List of view definitions

    Returns:
        Complete markdown documentation
    """
    sections = [
        "# Database schema",
        "",
        "Auto-generated documentation from schema definitions.",
        "",
        "## Overview",
        "",
        generate_mermaid_diagram(tables),
        "",
        generate_quick_reference(tables),
        "",
        "## Tables",
        "",
    ]

    for table_name, table_def in tables.items():
        sections.append(generate_table_documentation(table_name, table_def))

    sections.append(generate_index_documentation(indexes))
    sections.append(generate_view_documentation(views))

    res = "\n".join(sections)
    return res


def main():
    """Main function to generate schema documentation.

    This function loads the schema definitions and generates a comprehensive
    markdown documentation file.
    """
    args = make_args()

    print("Generating schema documentation...")
    print(f"  Tables: {len(DATABASE_SCHEMA)}")
    print(f"  Indexes: {len(DATABASE_INDEXES)}")
    print(f"  Views: {len(DATABASE_VIEWS)}")

    doc_content = generate_schema_documentation(
        DATABASE_SCHEMA, DATABASE_INDEXES, DATABASE_VIEWS
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
