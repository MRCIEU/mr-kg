"""Database schema definitions and validation for the vector store.

This module defines the expected database schema and provides validation
functions to ensure the database structure is correct.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import duckdb


class ColumnType(Enum):
    """Supported column types in the database schema."""

    INTEGER = "INTEGER"
    VARCHAR = "VARCHAR"
    JSON = "JSON"
    FLOAT_ARRAY_200 = "FLOAT[200]"


@dataclass
class ColumnDef:
    """Definition of a database column."""

    name: str
    type: ColumnType
    nullable: bool = True
    primary_key: bool = False
    unique: bool = False

    def __str__(self) -> str:
        parts = [f"{self.name} {self.type.value}"]
        if self.primary_key:
            parts.append("PRIMARY KEY")
        elif not self.nullable:
            parts.append("NOT NULL")
        if self.unique and not self.primary_key:
            parts.append("UNIQUE")
        return " ".join(parts)


@dataclass
class ForeignKeyDef:
    """Definition of a foreign key constraint."""

    column: str
    ref_table: str
    ref_column: str

    def __str__(self) -> str:
        return f"FOREIGN KEY ({self.column}) REFERENCES {self.ref_table}({self.ref_column})"


@dataclass
class IndexDef:
    """Definition of a database index."""

    name: str
    table: str
    columns: List[str]
    unique: bool = False

    def __str__(self) -> str:
        unique_part = "UNIQUE " if self.unique else ""
        columns_part = ", ".join(self.columns)
        return f"CREATE {unique_part}INDEX {self.name} ON {self.table}({columns_part})"


@dataclass
class ViewDef:
    """Definition of a database view."""

    name: str
    sql: str

    def __str__(self) -> str:
        return f"CREATE VIEW {self.name} AS {self.sql}"


@dataclass
class TableDef:
    """Definition of a database table."""

    name: str
    columns: List[ColumnDef]
    foreign_keys: Optional[List[ForeignKeyDef]] = None
    description: str = ""

    def __post_init__(self):
        if self.foreign_keys is None:
            self.foreign_keys = []

    def get_create_sql(self) -> str:
        """Generate CREATE TABLE SQL statement."""
        column_defs = [str(col) for col in self.columns]
        fk_defs = [str(fk) for fk in (self.foreign_keys or [])]
        all_defs = column_defs + fk_defs

        return f"""CREATE TABLE {self.name} (
    {",\n    ".join(all_defs)}
)"""


# Define the complete database schema
#
# DATA FLOW OVERVIEW:
# 1. unique_traits.csv provides canonical trait indices and labels
# 2. trait_embeddings: Stores embeddings for traits that have valid vector representations
# 3. efo_embeddings: Stores EFO ontology terms with embeddings for semantic mapping
# 4. model_results: Raw LLM outputs with metadata containing exposure/outcome trait references
# 5. model_result_traits: Links model outputs to canonical traits via trait_index validation
#
# The build process:
# - Loads unique_traits.csv as the authoritative trait index
# - Creates embeddings only for traits with valid vector representations
# - Processes model results and extracts trait references from metadata.exposures/outcomes
# - Validates trait references against available embeddings before creating linkages
# - Combines exposure and outcome traits without role distinction in model_result_traits
DATABASE_SCHEMA = {
    "trait_embeddings": TableDef(
        name="trait_embeddings",
        description="Trait embeddings indexed by unique_traits.csv indices",
        columns=[
            ColumnDef(
                "trait_index",
                ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
                # Primary key from unique_traits.csv - serves as the authoritative index
                # for all trait references across the database. Used to link model results
                # to specific traits and enable consistent trait identification.
            ),
            ColumnDef(
                "trait_label",
                ColumnType.VARCHAR,
                nullable=False,
                # Human-readable trait name from unique_traits.csv (e.g., "body mass index").
                # This is the canonical trait label used for display and search purposes.
                # Serves as the basis for creating embeddings and linking to model results.
            ),
            ColumnDef(
                "vector",
                ColumnType.FLOAT_ARRAY_200,
                nullable=False,
                # 200-dimensional embedding vector for the trait, generated from trait_label.
                # Used for similarity searches to find semantically related traits.
                # Only traits with valid embeddings are included in this table.
            ),
        ],
    ),
    "efo_embeddings": TableDef(
        name="efo_embeddings",
        description="EFO (Experimental Factor Ontology) term embeddings",
        columns=[
            ColumnDef(
                "id",
                ColumnType.VARCHAR,
                nullable=False,
                primary_key=True,
                # EFO ontology identifier (e.g., "EFO_0004340"). Serves as the unique
                # identifier for EFO terms in the ontology hierarchy.
            ),
            ColumnDef(
                "label",
                ColumnType.VARCHAR,
                nullable=False,
                # Human-readable EFO term label (e.g., "body mass index"). Used for
                # display and to generate embeddings for semantic similarity matching.
            ),
            ColumnDef(
                "vector",
                ColumnType.FLOAT_ARRAY_200,
                nullable=False,
                # 200-dimensional embedding vector for the EFO term, generated from label.
                # Enables finding EFO terms semantically similar to traits for ontology mapping.
            ),
        ],
    ),
    "model_results": TableDef(
        name="model_results",
        description="Extracted structural data from model results organized by PMID",
        columns=[
            ColumnDef(
                "id",
                ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
                # Auto-incrementing primary key (current_result_id in build script).
                # Uniquely identifies each model result entry across all models and PMIDs.
            ),
            ColumnDef(
                "model",
                ColumnType.VARCHAR,
                nullable=False,
                # Name of the LLM model that generated this result (e.g., "gpt-4", "deepseek-r1").
                # Used to compare performance across different models and filter results by model type.
            ),
            ColumnDef(
                "pmid",
                ColumnType.VARCHAR,
                nullable=False,
                # PubMed ID of the research paper this result was extracted from.
                # Links model results back to source publications for validation and analysis.
            ),
            ColumnDef(
                "metadata",
                ColumnType.JSON,
                nullable=False,
                # JSON containing structured metadata including exposures and outcomes lists.
                # Each exposure/outcome may have linked_index pointing to trait_embeddings.trait_index
                # and original id from the model's output. Used to extract trait relationships.
            ),
            ColumnDef(
                "results",
                ColumnType.JSON,
                nullable=False,
                # JSON containing the complete raw model output and extracted results.
                # Preserves full model response for detailed analysis and debugging.
            ),
        ],
    ),
    "model_result_traits": TableDef(
        name="model_result_traits",
        description="Links model results to traits based on unique_traits indices",
        columns=[
            ColumnDef(
                "id",
                ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
                # Auto-incrementing primary key (trait_link_id in build script).
                # Uniquely identifies each trait-to-result relationship.
            ),
            ColumnDef(
                "model_result_id",
                ColumnType.INTEGER,
                nullable=False,
                # Foreign key to model_results.id. Links this trait relationship
                # to a specific model result, enabling queries for all traits
                # associated with a particular model output.
            ),
            ColumnDef(
                "trait_index",
                ColumnType.INTEGER,
                nullable=False,
                # Foreign key to trait_embeddings.trait_index. References the canonical
                # trait from unique_traits.csv that this model result mentions.
                # Enables consistent trait identification across all model results.
            ),
            ColumnDef(
                "trait_label",
                ColumnType.VARCHAR,
                nullable=False,
                # Denormalized trait label from trait_embeddings.trait_label.
                # Stored here for query performance to avoid joins when displaying
                # human-readable trait names. Updated during trait validation process.
            ),
            ColumnDef(
                "trait_id_in_result",
                ColumnType.VARCHAR,
                nullable=True,
                # Original trait identifier from the model's output (exposure.id or outcome.id).
                # May be empty if model didn't provide explicit IDs. Used to trace back
                # to the exact model output and understand how models identify traits.
            ),
        ],
        foreign_keys=[
            ForeignKeyDef("model_result_id", "model_results", "id"),
            ForeignKeyDef("trait_index", "trait_embeddings", "trait_index"),
        ],
    ),
}

# Define expected indexes for query performance optimization
DATABASE_INDEXES = [
    IndexDef(
        "idx_trait_embeddings_label", "trait_embeddings", ["trait_label"]
        # Enables fast text-based trait searches and lookups by human-readable names
    ),
    IndexDef(
        "idx_trait_embeddings_index", "trait_embeddings", ["trait_index"]
        # Optimizes foreign key lookups from model_result_traits table
    ),
    IndexDef(
        "idx_efo_embeddings_label", "efo_embeddings", ["label"]
        # Speeds up EFO term searches by human-readable labels
    ),
    IndexDef(
        "idx_model_results_model", "model_results", ["model"]
        # Enables efficient filtering and comparison by model type (gpt-4, deepseek-r1, etc.)
    ),
    IndexDef(
        "idx_model_results_pmid", "model_results", ["pmid"]
        # Allows fast retrieval of all model results for a specific research paper
    ),
    IndexDef(
        "idx_model_result_traits_trait_index",
        "model_result_traits",
        ["trait_index"],
        # Optimizes queries finding all model results mentioning a specific trait
    ),
    IndexDef(
        "idx_model_result_traits_model_result_id",
        "model_result_traits",
        ["model_result_id"],
        # Speeds up queries for all traits mentioned in a specific model result
    ),
    IndexDef(
        "idx_model_result_traits_trait_label",
        "model_result_traits",
        ["trait_label"],
        # Enables fast text-based searches within the denormalized trait labels
    ),
]

# Define expected views for common similarity search operations
DATABASE_VIEWS = [
    ViewDef(
        name="trait_similarity_search",
        # Pre-computed similarity matrix for all trait-to-trait comparisons.
        # Uses cosine similarity on 200-dimensional embeddings to find semantically related traits.
        # Excludes self-comparisons (t1.trait_index != t2.trait_index).
        # Useful for discovering related traits and clustering analysis.
        sql="""
        SELECT
            t1.trait_index as query_id,
            t1.trait_label as query_label,
            t2.trait_index as result_id,
            t2.trait_label as result_label,
            array_cosine_similarity(t1.vector, t2.vector) as similarity
        FROM trait_embeddings t1
        CROSS JOIN trait_embeddings t2
        WHERE t1.trait_index != t2.trait_index
        """,
    ),
    ViewDef(
        name="trait_efo_similarity_search",
        # Cross-reference matrix between traits and EFO ontology terms.
        # Uses cosine similarity to map traits to relevant EFO terms for ontology alignment.
        # Enables automatic trait categorization and standardization against biomedical ontologies.
        # Results can be filtered by similarity threshold to find best EFO matches.
        sql="""
        SELECT
            t.trait_index as trait_index,
            t.trait_label as trait_label,
            e.id as efo_id,
            e.label as efo_label,
            array_cosine_similarity(t.vector, e.vector) as similarity
        FROM trait_embeddings t
        CROSS JOIN efo_embeddings e
        """,
    ),
]


def validate_database_schema(
    conn: duckdb.DuckDBPyConnection,
) -> Dict[str, Any]:
    """Validate that the database matches the expected schema.

    Args:
        conn: DuckDB connection to validate

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "valid": True,
        "tables": {},
        "indexes": {},
        "views": {},
        "errors": [],
        "warnings": [],
    }

    try:
        # Check tables
        for table_name, table_def in DATABASE_SCHEMA.items():
            table_result = validate_table(conn, table_def)
            validation_results["tables"][table_name] = table_result
            if not table_result["valid"]:
                validation_results["valid"] = False
                validation_results["errors"].extend(table_result["errors"])

        # Check indexes
        for index_def in DATABASE_INDEXES:
            index_result = validate_index(conn, index_def)
            validation_results["indexes"][index_def.name] = index_result
            if not index_result["valid"]:
                validation_results["warnings"].extend(index_result["errors"])

        # Check views
        for view_def in DATABASE_VIEWS:
            view_result = validate_view(conn, view_def)
            validation_results["views"][view_def.name] = view_result
            if not view_result["valid"]:
                validation_results["warnings"].extend(view_result["errors"])

    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Validation error: {str(e)}")

    return validation_results


def validate_table(
    conn: duckdb.DuckDBPyConnection, table_def: TableDef
) -> Dict[str, Any]:
    """Validate a single table against its definition.

    Args:
        conn: DuckDB connection
        table_def: Table definition to validate against

    Returns:
        Dictionary with validation results for the table
    """
    result = {
        "valid": True,
        "exists": False,
        "columns": {},
        "row_count": 0,
        "errors": [],
    }

    try:
        # Check if table exists
        table_check = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            (table_def.name,),
        ).fetchone()

        if not table_check or table_check[0] == 0:
            result["valid"] = False
            result["errors"].append(f"Table {table_def.name} does not exist")
            return result

        result["exists"] = True

        # Get table schema - PRAGMA statements don't support parameterized queries
        columns_info = conn.execute(
            f"PRAGMA table_info({table_def.name})"
        ).fetchall()

        actual_columns = {
            col[1]: {  # col[1] is column name
                "type": col[2],  # col[2] is type
                "nullable": not bool(col[3]),  # col[3] is notnull
                "primary_key": bool(col[5]),  # col[5] is pk
            }
            for col in columns_info
        }

        # Validate each expected column
        for col_def in table_def.columns:
            if col_def.name not in actual_columns:
                result["valid"] = False
                result["errors"].append(
                    f"Column {col_def.name} missing from {table_def.name}"
                )
                continue

            actual_col = actual_columns[col_def.name]
            col_result = {
                "exists": True,
                "type_match": True,
                "nullable_match": True,
                "pk_match": True,
            }

            # Check type (simplified check)
            expected_type = col_def.type.value
            if expected_type == "FLOAT[200]":
                # DuckDB may show this as different variants
                if (
                    "FLOAT" not in actual_col["type"]
                    and "DOUBLE" not in actual_col["type"]
                ):
                    col_result["type_match"] = False
                    result["errors"].append(
                        f"Column {col_def.name} type mismatch: expected {expected_type}, got {actual_col['type']}"
                    )
            elif expected_type not in actual_col["type"]:
                col_result["type_match"] = False
                result["errors"].append(
                    f"Column {col_def.name} type mismatch: expected {expected_type}, got {actual_col['type']}"
                )

            # Check nullable
            if col_def.nullable != actual_col["nullable"]:
                col_result["nullable_match"] = False
                result["errors"].append(
                    f"Column {col_def.name} nullable mismatch: expected {col_def.nullable}, got {actual_col['nullable']}"
                )

            # Check primary key
            if col_def.primary_key != actual_col["primary_key"]:
                col_result["pk_match"] = False
                result["errors"].append(
                    f"Column {col_def.name} primary key mismatch: expected {col_def.primary_key}, got {actual_col['primary_key']}"
                )

            result["columns"][col_def.name] = col_result

            if not all(col_result.values()):
                result["valid"] = False

        # Get row count
        row_count = conn.execute(
            f"SELECT COUNT(*) FROM {table_def.name}"
        ).fetchone()
        result["row_count"] = row_count[0] if row_count else 0

    except Exception as e:
        result["valid"] = False
        result["errors"].append(
            f"Error validating table {table_def.name}: {str(e)}"
        )

    return result


def validate_index(
    conn: duckdb.DuckDBPyConnection, index_def: IndexDef
) -> Dict[str, Any]:
    """Validate a single index against its definition.

    Args:
        conn: DuckDB connection
        index_def: Index definition to validate against

    Returns:
        Dictionary with validation results for the index
    """
    result = {
        "valid": True,
        "exists": False,
        "errors": [],
    }

    try:
        # Check if index exists (simplified check)
        # DuckDB doesn't have a standard way to list indexes, so we'll try to create it
        # and see if it already exists
        try:
            conn.execute(str(index_def))
            result["exists"] = False  # If no error, index didn't exist
        except Exception as e:
            if "already exists" in str(e).lower():
                result["exists"] = True
            else:
                result["valid"] = False
                result["errors"].append(
                    f"Error checking index {index_def.name}: {str(e)}"
                )

    except Exception as e:
        result["valid"] = False
        result["errors"].append(
            f"Error validating index {index_def.name}: {str(e)}"
        )

    return result


def validate_view(
    conn: duckdb.DuckDBPyConnection, view_def: ViewDef
) -> Dict[str, Any]:
    """Validate a single view against its definition.

    Args:
        conn: DuckDB connection
        view_def: View definition to validate against

    Returns:
        Dictionary with validation results for the view
    """
    result = {
        "valid": True,
        "exists": False,
        "errors": [],
    }

    try:
        # Check if view exists
        view_check = conn.execute(
            "SELECT COUNT(*) FROM information_schema.views WHERE table_name = ?",
            (view_def.name,),
        ).fetchone()

        if view_check and view_check[0] > 0:
            result["exists"] = True
        else:
            result["valid"] = False
            result["errors"].append(f"View {view_def.name} does not exist")

    except Exception as e:
        result["valid"] = False
        result["errors"].append(
            f"Error validating view {view_def.name}: {str(e)}"
        )

    return result


def print_validation_report(validation_results: Dict[str, Any]) -> None:
    """Print a formatted validation report.

    Args:
        validation_results: Results from validate_database_schema()
    """
    print("=" * 80)
    print("DATABASE SCHEMA VALIDATION REPORT")
    print("=" * 80)

    if validation_results["valid"]:
        print("[OK] Overall Status: VALID")
    else:
        print("[ERROR] Overall Status: INVALID")

    print("\n" + "=" * 40)
    print("TABLES")
    print("=" * 40)

    for table_name, table_result in validation_results["tables"].items():
        status = "[OK]" if table_result["valid"] else "[ERROR]"
        row_count = table_result.get("row_count", 0)
        print(f"{status} {table_name}: {row_count:,} rows")

        if table_result["errors"]:
            for error in table_result["errors"]:
                print(f"    [ERROR] {error}")

    print("\n" + "=" * 40)
    print("INDEXES")
    print("=" * 40)

    for index_name, index_result in validation_results["indexes"].items():
        status = "[OK]" if index_result["exists"] else "[WARN]"
        print(f"{status} {index_name}")

        if index_result["errors"]:
            for error in index_result["errors"]:
                print(f"    [WARN] {error}")

    print("\n" + "=" * 40)
    print("VIEWS")
    print("=" * 40)

    for view_name, view_result in validation_results["views"].items():
        status = "[OK]" if view_result["exists"] else "[WARN]"
        print(f"{status} {view_name}")

        if view_result["errors"]:
            for error in view_result["errors"]:
                print(f"    [WARN] {error}")

    if validation_results["errors"]:
        print("\n" + "=" * 40)
        print("CRITICAL ERRORS")
        print("=" * 40)
        for error in validation_results["errors"]:
            print(f"[ERROR] {error}")

    if validation_results["warnings"]:
        print("\n" + "=" * 40)
        print("WARNINGS")
        print("=" * 40)
        for warning in validation_results["warnings"]:
            print(f"[WARN] {warning}")

    print("\n" + "=" * 80)


def get_schema_documentation() -> str:
    """Generate documentation for the database schema.

    Returns:
        Formatted schema documentation
    """
    doc = []
    doc.append("DATABASE SCHEMA DOCUMENTATION")
    doc.append("=" * 80)
    doc.append("")

    for table_name, table_def in DATABASE_SCHEMA.items():
        doc.append(f"TABLE: {table_name}")
        doc.append("-" * 40)
        if table_def.description:
            doc.append(f"Description: {table_def.description}")
        doc.append("")
        doc.append("Columns:")
        for col in table_def.columns:
            nullable = "NULL" if col.nullable else "NOT NULL"
            pk = " (PRIMARY KEY)" if col.primary_key else ""
            doc.append(f"  - {col.name}: {col.type.value} {nullable}{pk}")

        if table_def.foreign_keys:
            doc.append("")
            doc.append("Foreign Keys:")
            for fk in table_def.foreign_keys:
                doc.append(f"  - {fk}")

        doc.append("")

    doc.append("INDEXES:")
    doc.append("-" * 40)
    for index in DATABASE_INDEXES:
        unique = "UNIQUE " if index.unique else ""
        doc.append(
            f"  - {unique}{index.name}: {index.table}({', '.join(index.columns)})"
        )

    doc.append("")
    doc.append("VIEWS:")
    doc.append("-" * 40)
    for view in DATABASE_VIEWS:
        doc.append(f"  - {view.name}")

    return "\n".join(doc)
