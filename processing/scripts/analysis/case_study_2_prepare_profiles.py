"""Prepare trait co-occurrence profiles for case study 2 analyses.

This script builds per-study trait lists from the trait profile and vector
store databases, computes trait co-occurrence statistics, and exports
CSV/JSON datasets (with an optional sparse matrix NPZ file) for downstream
network, hotspot, and concordance analyses.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import duckdb
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from yiutils.project_utils import find_project_root

PROJECT_ROOT = find_project_root("docker-compose.yml")
CONFIG_DIR = PROJECT_ROOT / "processing" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "case_studies.yml"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "case-study-cs2" / "cooccurrence"
)


def parse_args() -> argparse.Namespace:
    """Create the command-line parser and parse arguments."""

    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Validate configuration and paths without executing",
    )

    # ---- --config ----
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to YAML configuration (default: {DEFAULT_CONFIG})",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Override output directory; defaults to cooccurrence directory "
            "from configuration"
        ),
    )

    # ---- --min-shared-studies ----
    parser.add_argument(
        "--min-shared-studies",
        type=int,
        help="Minimum number of shared studies required for a trait pair",
    )

    # ---- --min-semantic-similarity ----
    parser.add_argument(
        "--min-semantic-similarity",
        type=float,
        help="Minimum cosine similarity between trait embeddings",
    )

    # ---- --sample-fraction ----
    parser.add_argument(
        "--sample-fraction",
        type=float,
        help="Optional sampling fraction for combinations (0 < f <= 1)",
    )

    # ---- --limit ----
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of study profiles processed (post-sampling)",
    )

    # ---- --include-model ----
    parser.add_argument(
        "--include-model",
        action="append",
        help="Restrict processing to specified extraction models",
    )

    # ---- --exclude-model ----
    parser.add_argument(
        "--exclude-model",
        action="append",
        help="Exclude specified extraction models from processing",
    )

    # ---- --no-npz ----
    parser.add_argument(
        "--no-npz",
        action="store_true",
        help="Skip generation of the sparse co-occurrence matrix",
    )

    res = parser.parse_args()
    return res


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    return config


def attach_databases(
    trait_db: Path,
    vector_db: Path,
) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection and attach source databases.

    Args:
        trait_db: Path to trait profile DuckDB database.
        vector_db: Path to vector store DuckDB database.

    Returns:
        DuckDB connection with both databases attached read-only.
    """

    conn = duckdb.connect(database=":memory:")
    trait_path = str(trait_db.resolve()).replace("'", "''")
    vector_path = str(vector_db.resolve()).replace("'", "''")
    conn.execute(f"ATTACH DATABASE '{trait_path}' AS trait_db (READ_ONLY)")
    conn.execute(f"ATTACH DATABASE '{vector_path}' AS vector_db (READ_ONLY)")
    return conn


def build_condition_clause(
    min_trait_count: int,
    max_trait_count: int,
    include_models: Optional[Sequence[str]] = None,
    exclude_models: Optional[Sequence[str]] = None,
) -> Tuple[str, List[Any]]:
    """Build SQL WHERE clause fragments for trait profile selection."""

    statements = ["qc.trait_count >= ?", "qc.trait_count <= ?"]
    params: List[Any] = [min_trait_count, max_trait_count]

    if include_models:
        placeholders = ", ".join(["?"] * len(include_models))
        statements.append(f"qc.model IN ({placeholders})")
        params.extend(include_models)

    if exclude_models:
        placeholders = ", ".join(["?"] * len(exclude_models))
        statements.append(f"qc.model NOT IN ({placeholders})")
        params.extend(exclude_models)

    clause = " AND ".join(statements)
    return clause, params


def fetch_trait_rows(
    conn: duckdb.DuckDBPyConnection,
    clause: str,
    params: Sequence[Any],
    limit: Optional[int],
) -> pd.DataFrame:
    """Fetch long-form trait rows for each study combination."""

    limit_sql = ""
    if limit is not None:
        limit_sql = " LIMIT ?"

    query = f"""
        SELECT DISTINCT
            qc.id AS combination_id,
            qc.pmid,
            qc.model,
            qc.title,
            qc.trait_count,
            mrt.trait_label,
            mrt.trait_index
        FROM trait_db.query_combinations qc
        JOIN vector_db.model_results mr
            ON mr.pmid = qc.pmid
           AND mr.model = qc.model
        JOIN vector_db.model_result_traits mrt
            ON mrt.model_result_id = mr.id
        WHERE {clause}
        ORDER BY qc.pmid, qc.model, mrt.trait_label{limit_sql}
    """

    query_params: List[Any] = list(params)
    if limit is not None:
        query_params.append(limit)

    logger.debug("Executing trait profile query")
    trait_rows = conn.execute(query, query_params).fetch_df()
    logger.info("Retrieved %s trait records", len(trait_rows))
    return trait_rows


def aggregate_trait_profiles(
    trait_rows: pd.DataFrame,
    min_trait_count: int,
    max_traits_per_study: int,
    sample_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    """Aggregate long-form trait rows into per-study trait lists."""

    if trait_rows.empty:
        return pd.DataFrame()

    deduped = (
        trait_rows.dropna(subset=["trait_label"])
        .drop_duplicates(
            subset=["combination_id", "pmid", "model", "trait_label"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    grouped = deduped.groupby(
        ["combination_id", "pmid", "model", "title", "trait_count"],
        as_index=False,
    ).agg(
        trait_labels=("trait_label", list),
        trait_indices=(
            "trait_index",
            lambda vals: sorted({int(v) for v in vals if pd.notna(v)}),
        ),
    )
    grouped = cast(pd.DataFrame, grouped)

    grouped["trait_labels"] = grouped["trait_labels"].apply(
        lambda vals: sorted({v for v in vals if v})
    )
    grouped["observed_trait_count"] = grouped["trait_labels"].apply(len)

    filtered = grouped[
        grouped["observed_trait_count"].between(
            min_trait_count, max_traits_per_study
        )
    ].copy()
    filtered = cast(pd.DataFrame, filtered)

    if sample_fraction < 1.0 and len(filtered) > 0:
        filtered = filtered.sample(
            frac=sample_fraction,
            random_state=random_seed,
            ignore_index=True,
        )
        filtered = cast(pd.DataFrame, filtered)

    filtered = filtered.reset_index(drop=True)
    return filtered


def load_trait_embeddings(
    conn: duckdb.DuckDBPyConnection,
    trait_labels: Iterable[str],
) -> Dict[str, np.ndarray]:
    """Load normalised trait embedding vectors for the provided trait labels."""

    unique_traits = sorted({label for label in trait_labels if label})
    if not unique_traits:
        return {}

    placeholders = ", ".join(["?"] * len(unique_traits))
    query = (
        "SELECT trait_label, vector FROM vector_db.trait_embeddings "
        f"WHERE trait_label IN ({placeholders})"
    )

    rows = conn.execute(query, unique_traits).fetchall()
    embeddings: Dict[str, np.ndarray] = {}
    for trait_label, vector in rows:
        array = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(array)
        if norm == 0.0:
            continue
        embeddings[trait_label] = array / norm

    return embeddings


def compute_pair_metrics(
    profiles: pd.DataFrame,
    min_shared_studies: int,
    min_jaccard_similarity: float,
    min_semantic_similarity: float,
    embeddings: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame, Counter[str]]:
    """Compute trait pair metrics and trait frequency counts."""

    trait_counter: Counter[str] = Counter()
    pair_counter: Counter[Tuple[str, str]] = Counter()

    for _, record in profiles.iterrows():
        traits = list(record["trait_labels"])
        if not traits:
            continue

        seen = sorted(set(traits))
        for trait in seen:
            trait_counter[trait] += 1

        for trait_a, trait_b in combinations(seen, 2):
            ordered = tuple(sorted((trait_a, trait_b)))
            pair_counter[ordered] += 1

    total_profiles = max(len(profiles), 1)
    pair_records: List[Dict[str, Any]] = []

    for (trait_a, trait_b), shared_studies in pair_counter.items():
        freq_a = trait_counter[trait_a]
        freq_b = trait_counter[trait_b]
        union = freq_a + freq_b - shared_studies
        jaccard = shared_studies / union if union else 0.0
        dice = (
            (2.0 * shared_studies) / (freq_a + freq_b)
            if (freq_a + freq_b) > 0
            else 0.0
        )
        expected = (freq_a * freq_b) / total_profiles
        lift = shared_studies / expected if expected > 0 else 0.0
        co_rate = shared_studies / total_profiles

        embedding_a = embeddings.get(trait_a)
        embedding_b = embeddings.get(trait_b)
        semantic_similarity = (
            float(np.dot(embedding_a, embedding_b))
            if embedding_a is not None and embedding_b is not None
            else np.nan
        )

        passes_shared = shared_studies >= min_shared_studies
        passes_jaccard = jaccard >= min_jaccard_similarity
        passes_semantic = (
            np.isnan(semantic_similarity)
            or semantic_similarity >= min_semantic_similarity
        )

        if not (passes_shared and passes_jaccard and passes_semantic):
            continue

        pair_records.append(
            {
                "trait_a": trait_a,
                "trait_b": trait_b,
                "shared_studies": int(shared_studies),
                "frequency_a": int(freq_a),
                "frequency_b": int(freq_b),
                "jaccard": jaccard,
                "dice": dice,
                "lift": lift,
                "cooccurrence_rate": co_rate,
                "semantic_similarity": (
                    None
                    if np.isnan(semantic_similarity)
                    else semantic_similarity
                ),
            }
        )

    pair_df = pd.DataFrame(pair_records)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(
            by=["shared_studies", "jaccard"], ascending=[False, False]
        ).reset_index(drop=True)

    trait_df = (
        pd.DataFrame(
            {
                "trait_label": list(trait_counter.keys()),
                "study_count": list(trait_counter.values()),
            }
        )
        .sort_values(
            by=["study_count", "trait_label"], ascending=[False, True]
        )
        .reset_index(drop=True)
    )

    return pair_df, trait_df, trait_counter


def export_profiles(
    profiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export aggregated trait profiles to CSV and JSON files."""

    profile_csv = output_dir / "trait_profiles.csv"
    profile_json = output_dir / "trait_profiles.json"

    export_df = profiles.copy()
    export_df["trait_labels_pipe"] = export_df["trait_labels"].apply(
        lambda values: "|".join(values)
    )
    export_df["trait_indices_pipe"] = export_df["trait_indices"].apply(
        lambda values: "|".join(str(v) for v in values)
    )

    export_columns = [
        "combination_id",
        "pmid",
        "model",
        "title",
        "trait_count",
        "observed_trait_count",
        "trait_labels_pipe",
        "trait_indices_pipe",
    ]

    export_df.to_csv(profile_csv, index=False, columns=export_columns)
    logger.info("Saved trait profile table: %s", profile_csv)

    with profile_json.open("w", encoding="utf-8") as handle:
        json_payload = cast(
            str, export_df[export_columns].to_json(orient="records")
        )
        records = cast(List[Dict[str, Any]], json.loads(json_payload))
        json.dump(records, handle, indent=2)
    logger.info("Saved trait profile metadata: %s", profile_json)


def export_pair_metrics(
    pair_df: pd.DataFrame,
    trait_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write trait pair metrics and trait frequencies to disk."""

    pair_csv = output_dir / "trait_pair_metrics.csv"
    pair_json = output_dir / "trait_pair_metrics.json"
    trait_csv = output_dir / "trait_frequency.csv"

    pair_df.to_csv(pair_csv, index=False)
    logger.info("Saved trait pair metrics: %s", pair_csv)

    with pair_json.open("w", encoding="utf-8") as handle:
        pair_json = cast(str, pair_df.to_json(orient="records"))
        pair_records = cast(List[Dict[str, Any]], json.loads(pair_json))
        json.dump(pair_records, handle, indent=2)
    logger.info("Saved trait pair metrics JSON: %s", pair_json)

    trait_df.to_csv(trait_csv, index=False)
    logger.info("Saved trait frequency table: %s", trait_csv)


def export_sparse_matrix(
    pair_df: pd.DataFrame,
    trait_df: pd.DataFrame,
    output_dir: Path,
    filename: str,
) -> Optional[Path]:
    """Create a symmetric sparse matrix NPZ file from pair metrics."""

    if pair_df.empty or trait_df.empty:
        logger.warning("Skipping sparse matrix export due to empty inputs")
        return None

    trait_order = list(trait_df["trait_label"])
    trait_index_map = {label: idx for idx, label in enumerate(trait_order)}

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for _, record in pair_df.iterrows():
        idx_a = trait_index_map[record["trait_a"]]
        idx_b = trait_index_map[record["trait_b"]]
        weight = float(record["shared_studies"])

        rows.extend([idx_a, idx_b])
        cols.extend([idx_b, idx_a])
        data.extend([weight, weight])

    matrix_path = output_dir / filename
    np.savez_compressed(
        matrix_path,
        rows=np.asarray(rows, dtype=np.int32),
        cols=np.asarray(cols, dtype=np.int32),
        data=np.asarray(data, dtype=np.float32),
        trait_labels=np.asarray(trait_order, dtype=object),
    )
    logger.info("Saved sparse co-occurrence matrix: %s", matrix_path)
    return matrix_path


def export_metadata(
    output_dir: Path,
    parameters: Dict[str, Any],
    profiles: pd.DataFrame,
    pair_df: pd.DataFrame,
    trait_df: pd.DataFrame,
    matrix_path: Optional[Path],
) -> None:
    """Write a metadata summary for downstream traceability."""

    metadata = {
        "parameters": parameters,
        "total_profiles": int(len(profiles)),
        "total_unique_traits": int(len(trait_df)),
        "total_trait_pairs": int(len(pair_df)),
        "matrix_path": str(matrix_path) if matrix_path else None,
    }

    metadata_path = output_dir / "trait_cooccurrence_summary.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Saved metadata summary: %s", metadata_path)


def main() -> int:
    """Run trait profile preparation workflow."""

    args = parse_args()
    config = load_config(args.config)

    cs2_config = config["case_study_2"]
    co_config = cs2_config["cooccurrence"]
    prep_config = cs2_config.get("preparation", {})
    paths_config = config["databases"]
    output_config = config["output"]["case_study_2"]["cooccurrence"]

    trait_db = PROJECT_ROOT / paths_config["trait_profile"]
    vector_db = PROJECT_ROOT / paths_config["vector_store"]

    output_dir = args.output_dir or (PROJECT_ROOT / output_config)

    parameters = {
        "min_shared_studies": args.min_shared_studies
        if args.min_shared_studies is not None
        else co_config["min_shared_studies"],
        "min_jaccard_similarity": co_config["min_jaccard_similarity"],
        "min_semantic_similarity": args.min_semantic_similarity
        if args.min_semantic_similarity is not None
        else co_config["min_semantic_similarity"],
        "min_trait_count": co_config["min_trait_count"],
        "max_trait_count": co_config["max_trait_count"],
        "max_traits_per_study": co_config["max_traits_per_study"],
        "sample_fraction": args.sample_fraction
        if args.sample_fraction is not None
        else co_config["sample_fraction"],
        "random_seed": config.get("random_seed", 42),
        "include_models": args.include_model
        if args.include_model is not None
        else co_config.get("include_models", []),
        "exclude_models": args.exclude_model
        if args.exclude_model is not None
        else co_config.get("exclude_models", []),
        "limit": args.limit,
        "export_sparse_matrix": prep_config.get("export_sparse_matrix", True)
        and not args.no_npz,
        "npz_filename": prep_config.get(
            "npz_filename", "trait_cooccurrence_matrix.npz"
        ),
    }

    logger.info("Trait co-occurrence preparation parameters: %s", parameters)

    if args.dry_run:
        logger.info("Dry run complete")
        res = 0
        return res

    if not trait_db.exists():
        logger.error("Trait profile database not found: %s", trait_db)
        res = 1
        return res

    if not vector_db.exists():
        logger.error("Vector store database not found: %s", vector_db)
        res = 1
        return res

    output_dir.mkdir(parents=True, exist_ok=True)

    conn = attach_databases(trait_db, vector_db)

    clause, clause_params = build_condition_clause(
        parameters["min_trait_count"],
        parameters["max_trait_count"],
        parameters["include_models"],
        parameters["exclude_models"],
    )

    trait_rows = fetch_trait_rows(
        conn,
        clause,
        clause_params,
        parameters["limit"],
    )

    profiles = aggregate_trait_profiles(
        trait_rows,
        parameters["min_trait_count"],
        parameters["max_traits_per_study"],
        parameters["sample_fraction"],
        parameters["random_seed"],
    )

    if profiles.empty:
        logger.warning("No trait profiles met the selection criteria")
        conn.close()
        res = 0
        return res

    all_traits = [
        trait for traits in profiles["trait_labels"] for trait in traits
    ]
    embeddings = load_trait_embeddings(conn, all_traits)

    pair_df, trait_df, _ = compute_pair_metrics(
        profiles,
        parameters["min_shared_studies"],
        co_config["min_jaccard_similarity"],
        parameters["min_semantic_similarity"],
        embeddings,
    )

    if pair_df.empty:
        logger.warning("No trait pairs passed the filtering thresholds")

    export_profiles(profiles, output_dir)
    export_pair_metrics(pair_df, trait_df, output_dir)

    matrix_path = None
    if parameters["export_sparse_matrix"]:
        matrix_path = export_sparse_matrix(
            pair_df,
            trait_df,
            output_dir,
            parameters["npz_filename"],
        )

    export_metadata(
        output_dir,
        parameters,
        profiles,
        pair_df,
        trait_df,
        matrix_path,
    )

    conn.close()
    logger.info("Trait profile preparation complete")
    res = 0
    return res


if __name__ == "__main__":
    raise SystemExit(main())
