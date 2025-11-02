"""Overlay trait similarity metrics with evidence concordance statistics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from yiutils.project_utils import find_project_root

PROJECT_ROOT = find_project_root("docker-compose.yml")
CONFIG_DIR = PROJECT_ROOT / "processing" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "case_studies.yml"
OVERLAY_DIR = (
    PROJECT_ROOT / "data" / "processed" / "case-study-cs2" / "overlays"
)
FIGURE_DIR = PROJECT_ROOT / "data" / "processed" / "case-study-cs2" / "figures"
COOCCURRENCE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "case-study-cs2" / "cooccurrence"
)
PAIRS_FILENAME = "trait_similarity_concordance_pairs.csv"
SUMMARY_FILENAME = "trait_similarity_concordance_summary.json"
STUDY_FILENAME = "trait_similarity_concordance.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for concordance overlay."""

    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Validate configuration and inputs without executing queries",
    )

    # ---- --config ----
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Configuration file path (default: {DEFAULT_CONFIG})",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OVERLAY_DIR,
        help="Directory to store overlay outputs",
    )

    # ---- --figures-dir ----
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Directory to store generated figures",
    )

    # ---- --cooccurrence-dir ----
    parser.add_argument(
        "--cooccurrence-dir",
        type=Path,
        default=COOCCURRENCE_DIR,
        help="Directory containing co-occurrence outputs (for metadata)",
    )

    # ---- --min-shared-pairs ----
    parser.add_argument(
        "--min-shared-pairs",
        type=int,
        help="Minimum matched pairs required for inclusion",
    )

    # ---- --limit ----
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of joined records",
    )

    res = parser.parse_args()
    return res


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    return config


def attach_databases(config: Dict[str, Any]) -> duckdb.DuckDBPyConnection:
    """Attach trait, evidence, and vector databases to DuckDB connection."""

    db_config = config["databases"]
    trait_db = PROJECT_ROOT / db_config["trait_profile"]
    evidence_db = PROJECT_ROOT / db_config["evidence_profile"]
    vector_db = PROJECT_ROOT / db_config["vector_store"]

    missing: List[str] = [
        str(path)
        for path in (trait_db, evidence_db, vector_db)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing database files: " + ", ".join(sorted(missing))
        )

    conn = duckdb.connect(database=":memory:")
    trait_path = str(trait_db.resolve()).replace("'", "''")
    evidence_path = str(evidence_db.resolve()).replace("'", "''")
    vector_path = str(vector_db.resolve()).replace("'", "''")
    conn.execute(f"ATTACH DATABASE '{trait_path}' AS trait_db (READ_ONLY)")
    conn.execute(
        f"ATTACH DATABASE '{evidence_path}' AS evidence_db (READ_ONLY)"
    )
    conn.execute(f"ATTACH DATABASE '{vector_path}' AS vector_db (READ_ONLY)")
    return conn


def fetch_similarity_concordance(
    conn: duckdb.DuckDBPyConnection,
    min_shared_pairs: int,
    limit: Optional[int],
) -> pd.DataFrame:
    """Fetch joined trait similarity and evidence concordance records."""

    limit_sql = ""
    params: List[Any] = [min_shared_pairs]
    if limit is not None and limit > 0:
        limit_sql = " LIMIT ?"
        params.append(limit)

    query = f"""
        SELECT
            qc_query.pmid AS query_pmid,
            qc_query.model AS query_model,
            qc_query.id AS query_trait_id,
            qc_similar.pmid AS similar_pmid,
            qc_similar.model AS similar_model,
            qc_similar.id AS similar_trait_id,
            ts.trait_profile_similarity,
            ts.trait_jaccard_similarity,
            es.direction_concordance,
            es.matched_pairs,
            es.match_type_exact,
            es.match_type_fuzzy,
            es.match_type_efo,
            es.id AS evidence_similarity_id
        FROM trait_db.trait_similarities ts
        JOIN trait_db.query_combinations qc_query
            ON qc_query.id = ts.query_combination_id
        JOIN evidence_db.evidence_similarities es
            ON es.query_combination_id = ts.query_combination_id
           AND es.similar_pmid = ts.similar_pmid
           AND es.similar_model = ts.similar_model
        JOIN trait_db.query_combinations qc_similar
            ON qc_similar.pmid = ts.similar_pmid
           AND qc_similar.model = ts.similar_model
        WHERE es.matched_pairs >= ?{limit_sql}
    """

    logger.debug("Executing similarity-concordance join query")
    df = conn.execute(query, params).fetch_df()
    return df


def classify_direction(direction_str: Any) -> int:
    """Map textual direction annotations to {-1, 0, 1}."""

    if not direction_str:
        return 0

    direction = str(direction_str).strip().lower()
    positive_terms = {
        "positive",
        "pos",
        "increase",
        "increased",
        "increases",
        "promotes",
        "exacerbates",
        "adversely affects",
        "potentially increases",
        "positive genetic causal association",
    }
    negative_terms = {
        "negative",
        "neg",
        "decrease",
        "decreased",
        "decreases",
        "protective",
        "inverse",
        "negatively associated",
        "does not increase",
        "reduces",
        "reduced",
    }
    null_terms = {
        "null",
        "no association",
        "not associated",
        "no effect",
        "bidirectional",
        "no significant impact",
        "not causally connected",
        "does not increase or decrease",
    }

    if direction in positive_terms:
        return 1
    if direction in negative_terms:
        return -1
    if direction in null_terms:
        return 0
    return 0


def build_trait_index_map(
    conn: duckdb.DuckDBPyConnection,
    pmid: str,
    model: str,
) -> Dict[str, int]:
    """Return mapping from trait label to trait index for a combination."""

    query = """
        SELECT mrt.trait_label, mrt.trait_index
        FROM vector_db.model_result_traits mrt
        JOIN vector_db.model_results mr
            ON mrt.model_result_id = mr.id
        WHERE mr.pmid = ? AND mr.model = ?
    """
    rows = conn.execute(query, [pmid, model]).fetchall()
    res = {
        str(label): int(index) for label, index in rows if label is not None
    }
    return res


def load_model_results(
    conn: duckdb.DuckDBPyConnection,
    pmid: str,
    model: str,
) -> List[Dict[str, Any]]:
    """Load raw model results JSON for a PMID-model combination."""

    query = """
        SELECT results
        FROM vector_db.model_results
        WHERE pmid = ? AND model = ?
        LIMIT 1
    """
    row = conn.execute(query, [pmid, model]).fetchone()
    if row is None:
        return []
    payload = row[0]
    if isinstance(payload, str):
        try:
            res = cast(List[Dict[str, Any]], json.loads(payload))
            return res
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to decode model results for %s/%s: %s",
                pmid,
                model,
                exc,
            )
            return []
    if isinstance(payload, list):
        res = cast(List[Dict[str, Any]], payload)
        return res
    return []


def tokenize_label(label: str) -> Set[str]:
    """Tokenize a trait label into normalized lowercase terms."""

    tokens = set(re.findall(r"[a-z0-9]+", label.lower()))
    return tokens


def build_trait_lookup(trait_map: Dict[str, int]) -> Dict[str, Any]:
    """Construct helper structures for matching trait labels."""

    lower_map = {label.lower(): index for label, index in trait_map.items()}
    token_map = {label: tokenize_label(label) for label in trait_map}
    res = {
        "direct": trait_map,
        "lower": lower_map,
        "tokens": token_map,
    }
    return res


def resolve_trait_index(label: str, lookup: Dict[str, Any]) -> Optional[int]:
    """Resolve a raw label to trait index using direct and fuzzy matching."""

    direct_map = cast(Dict[str, int], lookup["direct"])
    lower_map = cast(Dict[str, int], lookup["lower"])
    tokens_map = cast(Dict[str, Set[str]], lookup["tokens"])

    if label in direct_map:
        return direct_map[label]

    lower_label = label.lower()
    if lower_label in lower_map:
        return lower_map[lower_label]

    raw_tokens = tokenize_label(label)
    if not raw_tokens:
        return None

    best_index: Optional[int] = None
    best_score = 0.0
    for source_label, tokens in tokens_map.items():
        if not tokens:
            continue
        overlap = raw_tokens & tokens
        if not overlap:
            continue
        coverage = len(overlap) / len(raw_tokens)
        if coverage < 0.5:
            continue
        if coverage > best_score:
            best_score = coverage
            best_index = direct_map[source_label]
            if coverage == 1.0:
                break
    return best_index


def harmonize_result(
    raw: Dict[str, Any],
    lookup: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Convert raw model result into harmonized structure with trait indices."""

    exposure_label = str(raw.get("exposure", "")).strip()
    outcome_label = str(raw.get("outcome", "")).strip()
    if not exposure_label or not outcome_label:
        return None

    exposure_index = resolve_trait_index(exposure_label, lookup)
    outcome_index = resolve_trait_index(outcome_label, lookup)
    if exposure_index is None or outcome_index is None:
        return None

    direction = classify_direction(raw.get("direction"))

    res = {
        "exposure_label": exposure_label,
        "outcome_label": outcome_label,
        "exposure_trait_index": int(exposure_index),
        "outcome_trait_index": int(outcome_index),
        "direction": direction,
    }
    return res


def prepare_combination_results(
    conn: duckdb.DuckDBPyConnection,
    pmid: str,
    model: str,
    cache: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    """Load and cache harmonized results for a PMID-model pair."""

    key = (pmid, model)
    if key in cache:
        res = cache[key]
        return res

    trait_map = build_trait_index_map(conn, pmid, model)
    lookup = build_trait_lookup(trait_map)
    raw_results = load_model_results(conn, pmid, model)

    processed: List[Dict[str, Any]] = []
    trait_indices: Set[int] = set()
    for raw in raw_results:
        harmonized = harmonize_result(raw, lookup)
        if harmonized is None:
            continue
        processed.append(harmonized)
        trait_indices.add(harmonized["exposure_trait_index"])
        trait_indices.add(harmonized["outcome_trait_index"])

    cache[key] = {
        "results": processed,
        "trait_indices": trait_indices,
    }
    return cache[key]


def load_trait_embeddings(
    conn: duckdb.DuckDBPyConnection,
    trait_indices: Iterable[int],
) -> Dict[int, np.ndarray]:
    """Load embedding vectors for the provided trait indices."""

    unique_indices = sorted({int(idx) for idx in trait_indices})
    if not unique_indices:
        return {}

    placeholders = ", ".join(["?"] * len(unique_indices))
    query = (
        "SELECT trait_index, vector FROM vector_db.trait_embeddings "
        f"WHERE trait_index IN ({placeholders})"
    )

    rows = conn.execute(query, unique_indices).fetchall()
    embeddings = {
        int(index): np.asarray(vector, dtype=float)
        for index, vector in rows
        if vector is not None
    }
    return embeddings


def load_trait_efo_map(
    conn: duckdb.DuckDBPyConnection,
    trait_indices: Iterable[int],
    min_similarity: float,
) -> Dict[int, str]:
    """Load top EFO identifier per trait index with similarity threshold."""

    unique_indices = sorted({int(idx) for idx in trait_indices})
    if not unique_indices:
        return {}

    placeholders = ", ".join(["?"] * len(unique_indices))
    query = f"""
        WITH ranked AS (
            SELECT
                trait_index,
                efo_id,
                similarity,
                ROW_NUMBER() OVER (
                    PARTITION BY trait_index
                    ORDER BY similarity DESC
                ) AS rn
            FROM vector_db.trait_efo_similarity_search
            WHERE trait_index IN ({placeholders})
        )
        SELECT trait_index, efo_id
        FROM ranked
        WHERE rn = 1 AND similarity >= ?
    """
    params: List[Any] = [*unique_indices, float(min_similarity)]
    rows = conn.execute(query, params).fetchall()
    res = {int(index): str(efo_id) for index, efo_id in rows}
    return res


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""

    dot_product = float(np.dot(vec1, vec2))
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    res = dot_product / (norm1 * norm2)
    return res


def match_exposure_outcome_pairs(
    query_results: Sequence[Dict[str, Any]],
    similar_results: Sequence[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Match exposure-outcome pairs by exact trait indices."""

    similar_dict = {
        (
            res["exposure_trait_index"],
            res["outcome_trait_index"],
        ): res
        for res in similar_results
    }

    matched_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for query_res in query_results:
        key = (
            query_res["exposure_trait_index"],
            query_res["outcome_trait_index"],
        )
        if key in similar_dict:
            matched_pairs.append((query_res, similar_dict[key]))
    return matched_pairs


def match_exposure_outcome_pairs_fuzzy(
    query_results: Sequence[Dict[str, Any]],
    similar_results: Sequence[Dict[str, Any]],
    trait_embeddings: Dict[int, np.ndarray],
    threshold: float,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Match exposure-outcome pairs using embedding similarity."""

    matched_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    used_similar_indices: Set[int] = set()

    for query_res in query_results:
        exp_idx = query_res["exposure_trait_index"]
        out_idx = query_res["outcome_trait_index"]
        if exp_idx not in trait_embeddings or out_idx not in trait_embeddings:
            continue
        exp_vec = trait_embeddings[exp_idx]
        out_vec = trait_embeddings[out_idx]

        best_match: Optional[Tuple[int, Dict[str, Any], float]] = None
        for idx, similar_res in enumerate(similar_results):
            if idx in used_similar_indices:
                continue
            s_exp_idx = similar_res["exposure_trait_index"]
            s_out_idx = similar_res["outcome_trait_index"]
            if (
                s_exp_idx not in trait_embeddings
                or s_out_idx not in trait_embeddings
            ):
                continue
            exp_sim = cosine_similarity(exp_vec, trait_embeddings[s_exp_idx])
            out_sim = cosine_similarity(out_vec, trait_embeddings[s_out_idx])
            if exp_sim >= threshold and out_sim >= threshold:
                combined = (exp_sim + out_sim) / 2
                if best_match is None or combined > best_match[2]:
                    best_match = (idx, similar_res, combined)

        if best_match is not None:
            used_similar_indices.add(best_match[0])
            matched_pairs.append((query_res, best_match[1]))

    return matched_pairs


def match_exposure_outcome_pairs_efo(
    query_results: Sequence[Dict[str, Any]],
    similar_results: Sequence[Dict[str, Any]],
    trait_efo_map: Dict[int, str],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Match exposure-outcome pairs via shared EFO identifiers."""

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for similar_res in similar_results:
        exp_idx = similar_res["exposure_trait_index"]
        out_idx = similar_res["outcome_trait_index"]
        exp_efo = trait_efo_map.get(exp_idx)
        out_efo = trait_efo_map.get(out_idx)
        if not exp_efo or not out_efo:
            continue
        key = (exp_efo, out_efo)
        groups.setdefault(key, []).append(similar_res)

    matched_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for query_res in query_results:
        exp_idx = query_res["exposure_trait_index"]
        out_idx = query_res["outcome_trait_index"]
        exp_efo = trait_efo_map.get(exp_idx)
        out_efo = trait_efo_map.get(out_idx)
        if not exp_efo or not out_efo:
            continue
        key = (exp_efo, out_efo)
        for similar_res in groups.get(key, []):
            matched_pairs.append((query_res, similar_res))

    return matched_pairs


def match_exposure_outcome_pairs_tiered(
    query_results: Sequence[Dict[str, Any]],
    similar_results: Sequence[Dict[str, Any]],
    trait_embeddings: Dict[int, np.ndarray],
    trait_efo_map: Dict[int, str],
    fuzzy_threshold: float,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    """Apply exact, fuzzy, then EFO matching tiers with provenance labels."""

    matched_pairs: List[Tuple[Dict[str, Any], Dict[str, Any], str]] = []
    exact_pairs = match_exposure_outcome_pairs(query_results, similar_results)
    matched_query_keys: Set[Tuple[int, int]] = set()

    for query_res, similar_res in exact_pairs:
        key = (
            query_res["exposure_trait_index"],
            query_res["outcome_trait_index"],
        )
        if key in matched_query_keys:
            continue
        matched_pairs.append((query_res, similar_res, "exact"))
        matched_query_keys.add(key)

    remaining_query = [
        res
        for res in query_results
        if (
            res["exposure_trait_index"],
            res["outcome_trait_index"],
        )
        not in matched_query_keys
    ]
    if remaining_query and trait_embeddings:
        fuzzy_pairs = match_exposure_outcome_pairs_fuzzy(
            remaining_query,
            similar_results,
            trait_embeddings,
            fuzzy_threshold,
        )
        for query_res, similar_res in fuzzy_pairs:
            key = (
                query_res["exposure_trait_index"],
                query_res["outcome_trait_index"],
            )
            if key in matched_query_keys:
                continue
            matched_pairs.append((query_res, similar_res, "fuzzy"))
            matched_query_keys.add(key)

    remaining_query = [
        res
        for res in query_results
        if (
            res["exposure_trait_index"],
            res["outcome_trait_index"],
        )
        not in matched_query_keys
    ]
    if remaining_query and trait_efo_map:
        efo_pairs = match_exposure_outcome_pairs_efo(
            remaining_query,
            similar_results,
            trait_efo_map,
        )
        for query_res, similar_res in efo_pairs:
            key = (
                query_res["exposure_trait_index"],
                query_res["outcome_trait_index"],
            )
            if key in matched_query_keys:
                continue
            matched_pairs.append((query_res, similar_res, "efo"))
            matched_query_keys.add(key)

    return matched_pairs


def classify_quadrant(
    similarity: float,
    concordance: float,
    thresholds: Dict[str, float],
) -> str:
    """Assign quadrant label based on similarity and concordance thresholds."""

    high_sim = thresholds["similarity_high"]
    low_sim = thresholds["similarity_low"]
    high_conc = thresholds["concordance_high"]
    low_conc = thresholds["concordance_low"]

    if similarity >= high_sim and concordance >= high_conc:
        return "high_similarity_high_concordance"
    if similarity >= high_sim and concordance <= low_conc:
        return "high_similarity_low_concordance"
    if similarity <= low_sim and concordance >= high_conc:
        return "low_similarity_high_concordance"
    if similarity <= low_sim and concordance <= low_conc:
        return "low_similarity_low_concordance"
    return "mid_zone"


def compute_correlations(
    df: pd.DataFrame,
    method: str,
) -> Dict[str, Any]:
    """Compute correlation metrics between similarity and concordance."""

    subset = df.dropna(
        subset=["trait_profile_similarity", "direction_concordance"]
    )
    if subset.empty:
        return {
            "pearson": None,
            "spearman": None,
            "selected": None,
            "method": method,
        }

    pearson = subset["trait_profile_similarity"].corr(
        other=subset["direction_concordance"],
        method="pearson",
    )
    spearman = subset["trait_profile_similarity"].corr(
        other=subset["direction_concordance"],
        method="spearman",
    )

    selected = pearson if method == "pearson" else spearman
    res = {
        "pearson": float(pearson) if pearson is not None else None,
        "spearman": float(spearman) if spearman is not None else None,
        "selected": float(selected) if selected is not None else None,
        "method": method,
    }
    return res


def render_scatter_plot(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    figure_path: Path,
    max_points: int,
    include_regression: bool,
    point_alpha: float,
) -> None:
    """Render scatter plot of similarity versus concordance."""

    plot_df = df.dropna(
        subset=["trait_profile_similarity", "direction_concordance"]
    ).copy()
    if max_points > 0 and len(plot_df) > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=42)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        plot_df["trait_profile_similarity"],
        plot_df["direction_concordance"],
        alpha=point_alpha,
        s=30,
        c="tab:blue",
    )

    ax.axvline(
        thresholds["similarity_high"],
        color="tab:red",
        linestyle="--",
        linewidth=1,
    )
    ax.axvline(
        thresholds["similarity_low"],
        color="tab:red",
        linestyle=":",
        linewidth=1,
    )
    ax.axhline(
        thresholds["concordance_high"],
        color="tab:green",
        linestyle="--",
        linewidth=1,
    )
    ax.axhline(
        thresholds["concordance_low"],
        color="tab:green",
        linestyle=":",
        linewidth=1,
    )

    if include_regression and len(plot_df) > 1:
        z = pd.Series(plot_df["trait_profile_similarity"])
        y = pd.Series(plot_df["direction_concordance"])
        variance = z.var()
        if variance > 0:
            coef = pd.Series(z).cov(y) / variance
            intercept = y.mean() - coef * z.mean()
            xs = pd.Series(sorted(z))
            ax.plot(xs, coef * xs + intercept, color="tab:purple", linewidth=1)

    ax.set_xlabel("Trait profile similarity")
    ax.set_ylabel("Direction concordance")
    ax.set_title("Trait similarity versus evidence concordance")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    alt_path = figure_path.with_suffix(".svg")
    fig.savefig(alt_path)
    plt.close(fig)


def compute_pair_direction_agreement(
    query_direction: int,
    similar_direction: int,
) -> int:
    """Return agreement indicator for matched pair directions."""

    if query_direction == 0 or similar_direction == 0:
        return 0
    if query_direction == similar_direction:
        return 1
    return -1


def build_pair_outputs(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    fuzzy_threshold: float,
    efo_min_similarity: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Construct pair-level records and enrich study-level DataFrame."""

    if df.empty:
        logger.warning("Joined similarity dataframe is empty")
        empty_pair_df = pd.DataFrame(
            columns=[
                "query_pmid",
                "query_model",
                "similar_pmid",
                "similar_model",
                "trait_a",
                "trait_b",
                "trait_a_index",
                "trait_b_index",
                "pair_match_type",
                "pair_direction_agreement",
            ]
        )
        res = {
            "pair_records": 0,
            "traits_with_pairs": 0,
            "concordant_pairs": 0,
            "discordant_pairs": 0,
        }
        return df.copy(), empty_pair_df, res

    combination_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    trait_indices_needed: Set[int] = set()
    pending: List[Dict[str, Any]] = []

    df = df.copy()

    for idx in range(len(df)):
        row = df.iloc[idx]
        query_key = (str(row["query_pmid"]), str(row["query_model"]))
        similar_key = (str(row["similar_pmid"]), str(row["similar_model"]))

        query_info = prepare_combination_results(
            conn,
            *query_key,
            cache=combination_cache,
        )
        similar_info = prepare_combination_results(
            conn,
            *similar_key,
            cache=combination_cache,
        )

        trait_indices_needed.update(query_info["trait_indices"])
        trait_indices_needed.update(similar_info["trait_indices"])

        pending.append(
            {
                "idx": idx,
                "query_key": query_key,
                "similar_key": similar_key,
                "trait_profile_similarity": float(
                    row["trait_profile_similarity"]
                ),
                "trait_jaccard_similarity": float(
                    row["trait_jaccard_similarity"]
                ),
                "direction_concordance": float(row["direction_concordance"]),
                "matched_pairs_total": int(row["matched_pairs"]),
                "match_type_exact": bool(row["match_type_exact"]),
                "match_type_fuzzy": bool(row["match_type_fuzzy"]),
                "match_type_efo": bool(row["match_type_efo"]),
                "match_type": row["match_type"],
                "quadrant": row["quadrant"],
                "evidence_similarity_id": int(row["evidence_similarity_id"]),
            }
        )

    trait_embeddings = load_trait_embeddings(conn, trait_indices_needed)
    trait_efo_map = load_trait_efo_map(
        conn,
        trait_indices_needed,
        efo_min_similarity,
    )

    pair_records: List[Dict[str, Any]] = []
    trait_counter: Counter[str] = Counter()
    concordant_pairs = 0
    discordant_pairs = 0

    for item in pending:
        idx = item["idx"]
        query_key = cast(Tuple[str, str], item["query_key"])
        similar_key = cast(Tuple[str, str], item["similar_key"])

        query_info = combination_cache[query_key]
        similar_info = combination_cache[similar_key]
        matched_pairs = match_exposure_outcome_pairs_tiered(
            query_info["results"],
            similar_info["results"],
            trait_embeddings,
            trait_efo_map,
            fuzzy_threshold,
        )

        pair_labels: Set[str] = set()
        agreement_scores: List[int] = []

        for query_res, similar_res, match_type in matched_pairs:
            agreement = compute_pair_direction_agreement(
                query_res["direction"],
                similar_res["direction"],
            )
            agreement_scores.append(agreement)
            if agreement > 0:
                concordant_pairs += 1
            elif agreement < 0:
                discordant_pairs += 1

            trait_a = query_res["exposure_label"]
            trait_b = query_res["outcome_label"]
            trait_pair_id = (
                f"{query_res['exposure_trait_index']}::"
                f"{query_res['outcome_trait_index']}"
            )
            trait_a_efo = trait_efo_map.get(query_res["exposure_trait_index"])
            trait_b_efo = trait_efo_map.get(query_res["outcome_trait_index"])

            pair_records.append(
                {
                    "query_pmid": query_key[0],
                    "query_model": query_key[1],
                    "similar_pmid": similar_key[0],
                    "similar_model": similar_key[1],
                    "trait_profile_similarity": item[
                        "trait_profile_similarity"
                    ],
                    "trait_jaccard_similarity": item[
                        "trait_jaccard_similarity"
                    ],
                    "direction_concordance": item["direction_concordance"],
                    "matched_pairs_total": item["matched_pairs_total"],
                    "match_type_exact": item["match_type_exact"],
                    "match_type_fuzzy": item["match_type_fuzzy"],
                    "match_type_efo": item["match_type_efo"],
                    "match_type": item["match_type"],
                    "quadrant": item["quadrant"],
                    "trait_a": trait_a,
                    "trait_b": trait_b,
                    "trait_a_index": int(query_res["exposure_trait_index"]),
                    "trait_b_index": int(query_res["outcome_trait_index"]),
                    "similar_trait_a": similar_res["exposure_label"],
                    "similar_trait_b": similar_res["outcome_label"],
                    "pair_match_type": match_type,
                    "pair_direction_agreement": int(agreement),
                    "pair_direction_query": int(query_res["direction"]),
                    "pair_direction_similar": int(similar_res["direction"]),
                    "trait_pair_id": trait_pair_id,
                    "trait_a_efo": trait_a_efo,
                    "trait_b_efo": trait_b_efo,
                    "evidence_similarity_id": item["evidence_similarity_id"],
                }
            )
            pair_labels.update({trait_a, trait_b})
            trait_counter.update({trait_a: 1, trait_b: 1})

        pair_count = len(agreement_scores)
        shared_traits = "|".join(sorted(pair_labels)) if pair_labels else ""
        df.loc[idx, "matched_trait_pairs"] = int(pair_count)
        df.loc[idx, "shared_traits"] = shared_traits
        if pair_count > 0:
            df.loc[idx, "pair_direction_mean"] = float(
                sum(agreement_scores) / pair_count
            )
        else:
            df.loc[idx, "pair_direction_mean"] = 0.0

    pair_df = pd.DataFrame(pair_records)
    pair_count_total = int(
        df.get("matched_trait_pairs", pd.Series(dtype=float)).fillna(0).sum()
    )
    summary = {
        "pair_records": int(len(pair_df)),
        "traits_with_pairs": int(len(trait_counter)),
        "concordant_pairs": int(concordant_pairs),
        "discordant_pairs": int(discordant_pairs),
        "pair_count_total": pair_count_total,
    }
    return df, pair_df, summary


def main() -> int:
    """Execute concordance overlay workflow."""

    args = parse_args()
    config = load_config(args.config)
    cs2_overlay = config["case_study_2"]["concordance_overlay"]

    thresholds = {
        "similarity_high": cs2_overlay.get("high_similarity_threshold", 0.6),
        "similarity_low": cs2_overlay.get("low_similarity_threshold", 0.2),
        "concordance_high": cs2_overlay.get("high_concordance_threshold", 0.6),
        "concordance_low": cs2_overlay.get("low_concordance_threshold", 0.0),
    }

    parameters = {
        "min_shared_pairs": args.min_shared_pairs
        if args.min_shared_pairs is not None
        else cs2_overlay.get("min_shared_pairs", 1),
        "correlation_method": cs2_overlay.get(
            "correlation_method", "spearman"
        ),
        "thresholds": thresholds,
        "figure": cs2_overlay.get("scatter", {}),
        "limit": args.limit,
        "fuzzy_threshold": cs2_overlay.get("fuzzy_similarity_threshold", 0.8),
        "efo_min_similarity": cs2_overlay.get("efo_min_similarity", 0.5),
    }

    logger.info("Overlay parameters: %s", parameters)

    if args.dry_run:
        logger.info("Dry run complete")
        return 0

    conn = attach_databases(config)

    df = fetch_similarity_concordance(
        conn,
        parameters["min_shared_pairs"],
        parameters["limit"],
    )

    if df.empty:
        logger.warning("No joined records met the inclusion criteria")
        conn.close()
        return 0

    df["match_type"] = df.apply(
        lambda row: "exact"
        if row["match_type_exact"]
        else (
            "fuzzy"
            if row["match_type_fuzzy"]
            else ("efo" if row["match_type_efo"] else "mixed")
        ),
        axis=1,
    )

    df["quadrant"] = df.apply(
        lambda row: classify_quadrant(
            float(row["trait_profile_similarity"]),
            float(row["direction_concordance"]),
            thresholds,
        ),
        axis=1,
    )

    df, pair_df, pair_summary = build_pair_outputs(
        conn,
        df,
        parameters["fuzzy_threshold"],
        parameters["efo_min_similarity"],
    )

    correlation_stats = compute_correlations(
        df, parameters["correlation_method"]
    )
    quadrant_counts = cast(
        Dict[str, int], df["quadrant"].value_counts().to_dict()
    )
    match_type_counts = cast(
        Dict[str, int], df["match_type"].value_counts().to_dict()
    )

    scatter_cfg = parameters["figure"]
    figure_name = scatter_cfg.get(
        "figure_name", "trait_similarity_vs_concordance"
    )
    figure_path = args.figures_dir / f"{figure_name}.png"
    render_scatter_plot(
        df,
        thresholds,
        figure_path,
        int(scatter_cfg.get("max_points", 2000)),
        bool(scatter_cfg.get("include_regression", True)),
        float(scatter_cfg.get("point_alpha", 0.6)),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    study_path = output_dir / STUDY_FILENAME
    df.to_csv(study_path, index=False)

    pair_path = output_dir / PAIRS_FILENAME
    pair_df.to_csv(pair_path, index=False)

    metadata = {
        "parameters": parameters,
        "study_record_count": int(len(df)),
        "pair_record_count": int(len(pair_df)),
        "quadrant_counts": quadrant_counts,
        "match_type_counts": match_type_counts,
        "correlation": correlation_stats,
        "figure": str(figure_path),
        "pair_summary": pair_summary,
        "study_path": str(study_path),
        "pair_path": str(pair_path),
    }

    metadata_path = output_dir / SUMMARY_FILENAME
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info(
        "Concordance overlay complete. Study CSV: %s, pair CSV: %s, metadata: %s",
        study_path,
        pair_path,
        metadata_path,
    )
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
