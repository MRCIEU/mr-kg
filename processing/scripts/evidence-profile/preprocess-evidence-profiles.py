"""Extract and harmonize evidence profiles from model results.

This script processes model results to:
1. Extract exposure-outcome results with effect sizes
2. Harmonize effect sizes to common scale (log transformation for OR/HR)
3. Classify statistical significance
4. Assess data quality and completeness
5. Output structured evidence profiles for similarity computation

The script queries the vector_store.db to extract all results for each
PMID-model combination and creates evidence profiles containing harmonized
effect sizes, directions, and metadata. Only studies meeting minimum quality
thresholds are included in the output.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import duckdb
from loguru import logger
from tqdm import tqdm
from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "db"
DEFAULT_DB_PATH = DB_DIR / "vector_store.db"
DEFAULT_OUTPUT_DIR = DATA_DIR / "processed" / "evidence-profiles"

# Global tracker for unrecognized directions
UNRECOGNIZED_DIRECTIONS: Dict[str, int] = {}

# ==== Type definitions ====


class HarmonizedResult(TypedDict):
    """Structure for a single harmonized result.

    Attributes:
        exposure_trait_index: Index of exposure trait in trait embeddings
        outcome_trait_index: Index of outcome trait in trait embeddings
        exposure_label: Human-readable exposure trait label
        outcome_label: Human-readable outcome trait label
        effect_size_type: Type of effect size (beta, OR, HR)
        original_effect_size: Original effect size value before harmonization
        harmonized_effect_size: Effect size transformed to common scale
        direction: Direction indicator (1=positive, -1=negative, 0=null)
        is_significant: Whether result is statistically significant (p<0.05)
        p_value: P-value (nullable)
        se: Standard error (nullable)
        ci_lower: Lower 95% confidence interval bound (nullable)
        ci_upper: Upper 95% confidence interval bound (nullable)
    """

    exposure_trait_index: int
    outcome_trait_index: int
    exposure_label: str
    outcome_label: str
    effect_size_type: str
    original_effect_size: float
    harmonized_effect_size: float
    direction: int
    is_significant: bool
    p_value: Optional[float]
    se: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]


class EvidenceProfile(TypedDict):
    """Structure for a complete evidence profile.

    Attributes:
        pmid: PubMed ID
        model: Model name
        title: Study title
        publication_year: Year of publication (nullable)
        result_count: Total number of results
        complete_result_count: Number of results with complete data
        data_completeness: Proportion of complete results (0-1)
        results: List of harmonized results
    """

    pmid: str
    model: str
    title: str
    publication_year: Optional[int]
    result_count: int
    complete_result_count: int
    data_completeness: float
    results: List[HarmonizedResult]


class PreprocessingStats(TypedDict):
    """Statistics from preprocessing operation.

    Attributes:
        total_combinations: Total PMID-model combinations processed
        included_combinations: Combinations passing quality filters
        excluded_combinations: Combinations failing quality filters
        total_results: Total number of results
        complete_results: Number of results with complete data
        results_by_effect_type: Count of results by effect size type
        completeness_distribution: Distribution statistics of data completeness
        exclusion_reasons: Count of combinations by exclusion reason
        unrecognized_directions: Count and examples of unrecognized direction strings
    """

    total_combinations: int
    included_combinations: int
    excluded_combinations: int
    total_results: int
    complete_results: int
    results_by_effect_type: Dict[str, int]
    completeness_distribution: Dict[str, float]
    exclusion_reasons: Dict[str, int]
    unrecognized_directions: Dict[str, Any]


# ==== Argument parsing ====


def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --dry-run ----
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Perform dry run without processing",
    )

    # ---- --database-path ----
    parser.add_argument(
        "--database-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to vector_store.db database (default: {DEFAULT_DB_PATH})",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for evidence profiles (default: {DEFAULT_OUTPUT_DIR})",
    )

    # ---- --min-results ----
    parser.add_argument(
        "--min-results",
        type=int,
        default=3,
        help="Minimum number of complete results required per study (default: 3)",
    )

    # ---- --max-missing-rate ----
    parser.add_argument(
        "--max-missing-rate",
        type=float,
        default=0.75,
        help="Maximum allowed proportion of missing data (default: 0.75)",
    )

    # ---- --limit ----
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of combinations to process (for testing)",
    )

    res = parser.parse_args()
    return res


# ==== Data extraction functions ====


def get_pmid_model_combinations(
    conn: duckdb.DuckDBPyConnection,
) -> List[Dict[str, Any]]:
    """Extract all PMID-model combinations from database.

    Args:
        conn: DuckDB connection to vector_store.db

    Returns:
        List of dictionaries containing pmid, model, title, and publication_year
    """
    query = """
    SELECT DISTINCT
        mr.pmid,
        mr.model,
        mpd.title,
        CAST(SUBSTR(mpd.pub_date, 1, 4) AS INTEGER) as publication_year
    FROM model_results mr
    LEFT JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
    ORDER BY mr.model, mr.pmid
    """
    res = conn.execute(query).fetchall()
    combinations = [
        {
            "pmid": row[0],
            "model": row[1],
            "title": row[2] or "Unknown Title",
            "publication_year": row[3],
        }
        for row in res
    ]
    return combinations


def extract_results_for_combination(
    conn: duckdb.DuckDBPyConnection, pmid: str, model: str
) -> List[Dict[str, Any]]:
    """Extract all results for a specific PMID-model combination.

    Args:
        conn: DuckDB connection
        pmid: PubMed ID
        model: Model name

    Returns:
        List of result dictionaries with metadata and results JSON
    """
    query = """
    SELECT
        mr.results,
        mr.metadata
    FROM model_results mr
    WHERE mr.pmid = ?
    AND mr.model = ?
    """
    res = conn.execute(query, [pmid, model]).fetchall()

    if not res:
        return []

    # Extract results from first row (should only be one per PMID-model)
    results_json = res[0][0]
    _ = res[0][1]  # metadata_json not used in this function

    # Parse JSON if needed
    if isinstance(results_json, str):
        results_data = json.loads(results_json)
    else:
        results_data = results_json

    # Results column directly contains list of results
    if isinstance(results_data, list):
        return results_data
    elif isinstance(results_data, dict) and "results" in results_data:
        # Handle case where results are wrapped in dict
        return results_data["results"]
    else:
        logger.warning(f"Unexpected results format for {pmid}-{model}")
        return []


# ==== Effect size harmonization functions ====


def harmonize_effect_size(effect_type: str, value: float) -> Optional[float]:
    """Transform effect size to common scale.

    For odds ratios and hazard ratios, applies log transformation.
    For beta coefficients, returns value as-is.

    Args:
        effect_type: Type of effect size (beta, OR, HR)
        value: Original effect size value

    Returns:
        Harmonized effect size, or None if transformation fails
    """
    if value is None:
        return None

    try:
        if effect_type in ["OR", "HR"]:
            # Log transformation for ratio measures
            # Handle edge cases
            if value <= 0:
                logger.warning(
                    f"Invalid {effect_type} value {value} (must be >0). Returning None."
                )
                return None
            res = math.log(value)
            return res
        elif effect_type == "beta":
            # Beta coefficients kept as-is
            res = value
            return res
        else:
            logger.warning(f"Unknown effect type: {effect_type}")
            return None
    except (ValueError, TypeError) as e:
        logger.warning(f"Error harmonizing {effect_type} value {value}: {e}")
        return None


def classify_direction(direction_str: str) -> int:
    """Classify effect direction to numeric indicator.

    Args:
        direction_str: Direction string from result (e.g., "positive", "negative")

    Returns:
        Direction indicator: 1 (positive), -1 (negative), 0 (null/unclear)

    Side effects:
        Tracks unrecognized direction strings in UNRECOGNIZED_DIRECTIONS global dict
    """
    if not direction_str:
        return 0

    direction_lower = direction_str.lower().strip()

    # Positive direction indicators
    positive_terms = [
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
    ]

    # Negative direction indicators
    negative_terms = [
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
    ]

    # Null/unclear indicators
    null_terms = [
        "null",
        "no association",
        "not associated",
        "no effect",
        "bidirectional",
        "no significant impact",
        "not causally connected",
        "does not increase or decrease",
    ]

    if direction_lower in positive_terms:
        return 1
    elif direction_lower in negative_terms:
        return -1
    elif direction_lower in null_terms:
        return 0
    else:
        # Track unrecognized direction for reporting
        UNRECOGNIZED_DIRECTIONS[direction_str] = (
            UNRECOGNIZED_DIRECTIONS.get(direction_str, 0) + 1
        )
        logger.debug(f"Unrecognized direction: '{direction_str}'")
        return 0


def classify_significance(p_value: Optional[float]) -> bool:
    """Classify statistical significance based on p-value.

    Args:
        p_value: P-value (nullable, can be float or string)

    Returns:
        True if p < 0.05, False otherwise
    """
    if p_value is None:
        return False

    # Handle string p-values
    if isinstance(p_value, str):
        try:
            p_value = float(p_value)
        except (ValueError, TypeError):
            logger.debug(f"Could not parse p-value: '{p_value}'")
            return False

    # Check if numeric
    if not isinstance(p_value, (int, float)):
        return False

    res = p_value < 0.05
    return res


# ==== Result processing functions ====


def extract_effect_size_info(
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Extract effect size information from a result.

    Args:
        result: Single result dictionary from model results

    Returns:
        Dictionary with effect_type, original_value, harmonized_value, or None
    """
    # Check for beta
    beta = result.get("beta")
    if beta is not None:
        harmonized = harmonize_effect_size("beta", beta)
        if harmonized is not None:
            return {
                "effect_type": "beta",
                "original_value": beta,
                "harmonized_value": harmonized,
            }

    # Check for odds ratio
    odds_ratio = result.get("odds ratio")
    if odds_ratio is not None:
        harmonized = harmonize_effect_size("OR", odds_ratio)
        if harmonized is not None:
            return {
                "effect_type": "OR",
                "original_value": odds_ratio,
                "harmonized_value": harmonized,
            }

    # Check for hazard ratio
    hazard_ratio = result.get("hazard ratio")
    if hazard_ratio is not None:
        harmonized = harmonize_effect_size("HR", hazard_ratio)
        if harmonized is not None:
            return {
                "effect_type": "HR",
                "original_value": hazard_ratio,
                "harmonized_value": harmonized,
            }

    # No valid effect size found
    return None


def get_trait_indices_from_db(
    conn: duckdb.DuckDBPyConnection, pmid: str, model: str
) -> Dict[str, int]:
    """Get trait label to index mapping for a PMID-model combination.

    Args:
        conn: DuckDB connection
        pmid: PubMed ID
        model: Model name

    Returns:
        Dictionary mapping trait labels to trait indices
    """
    query = """
    SELECT DISTINCT
        mrt.trait_label,
        mrt.trait_index
    FROM model_result_traits mrt
    JOIN model_results mr ON mrt.model_result_id = mr.id
    WHERE mr.pmid = ?
    AND mr.model = ?
    """
    res = conn.execute(query, [pmid, model]).fetchall()
    trait_map = {row[0]: row[1] for row in res}
    return trait_map


def process_single_result(
    result: Dict[str, Any], trait_map: Dict[str, int]
) -> Optional[HarmonizedResult]:
    """Process a single result into harmonized format.

    Args:
        result: Raw result dictionary
        trait_map: Mapping from trait labels to indices

    Returns:
        HarmonizedResult or None if result cannot be processed
    """
    # Extract exposure and outcome
    exposure_label = result.get("exposure")
    outcome_label = result.get("outcome")

    if not exposure_label or not outcome_label:
        return None

    # Get trait indices
    exposure_index = trait_map.get(exposure_label)
    outcome_index = trait_map.get(outcome_label)

    if exposure_index is None or outcome_index is None:
        return None

    # Extract effect size
    effect_info = extract_effect_size_info(result)
    if effect_info is None:
        return None

    # Extract direction
    direction_str = result.get("direction", "")
    direction = classify_direction(direction_str)

    if direction == 0:
        # Direction is required
        return None

    # Extract p-value and significance
    p_value = result.get("P-value")
    is_significant = classify_significance(p_value)

    # Extract uncertainty measures
    se = result.get("SE")
    ci = result.get("95% CI")
    ci_lower = None
    ci_upper = None

    if ci is not None and isinstance(ci, list) and len(ci) == 2:
        ci_lower = ci[0]
        ci_upper = ci[1]

    # Create harmonized result
    res = HarmonizedResult(
        exposure_trait_index=exposure_index,
        outcome_trait_index=outcome_index,
        exposure_label=exposure_label,
        outcome_label=outcome_label,
        effect_size_type=effect_info["effect_type"],
        original_effect_size=effect_info["original_value"],
        harmonized_effect_size=effect_info["harmonized_value"],
        direction=direction,
        is_significant=is_significant,
        p_value=p_value,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )

    return res


def process_combination(
    conn: duckdb.DuckDBPyConnection,
    pmid: str,
    model: str,
    title: str,
    publication_year: Optional[int],
    min_results: int,
    max_missing_rate: float,
) -> tuple[Optional[EvidenceProfile], Optional[str]]:
    """Process a single PMID-model combination.

    Args:
        conn: DuckDB connection
        pmid: PubMed ID
        model: Model name
        title: Study title
        publication_year: Year of publication (nullable)
        min_results: Minimum complete results required
        max_missing_rate: Maximum allowed missing data rate

    Returns:
        Tuple of (EvidenceProfile, exclusion_reason)
        If included, returns (profile, None)
        If excluded, returns (None, reason)
    """
    # Extract raw results
    raw_results = extract_results_for_combination(conn, pmid, model)

    if not raw_results:
        return None, "no_results"

    # Get trait mapping
    trait_map = get_trait_indices_from_db(conn, pmid, model)

    # Process each result
    harmonized_results = []
    for result in raw_results:
        harmonized = process_single_result(result, trait_map)
        if harmonized is not None:
            harmonized_results.append(harmonized)

    # Calculate completeness
    result_count = len(raw_results)
    complete_result_count = len(harmonized_results)

    if result_count == 0:
        return None, "no_results"

    data_completeness = complete_result_count / result_count

    # Apply quality filters
    if complete_result_count < min_results:
        return None, "insufficient_complete_results"

    if data_completeness < (1 - max_missing_rate):
        return None, "low_data_completeness"

    # Create evidence profile
    profile = EvidenceProfile(
        pmid=pmid,
        model=model,
        title=title,
        publication_year=publication_year,
        result_count=result_count,
        complete_result_count=complete_result_count,
        data_completeness=data_completeness,
        results=harmonized_results,
    )

    return profile, None


# ==== Statistics computation ====


def compute_preprocessing_stats(
    evidence_profiles: List[EvidenceProfile],
    excluded_combinations: List[Dict[str, str]],
    total_combinations: int,
) -> PreprocessingStats:
    """Compute statistics about preprocessing results.

    Args:
        evidence_profiles: List of included evidence profiles
        excluded_combinations: List of excluded combinations with reasons
        total_combinations: Total combinations processed

    Returns:
        PreprocessingStats dictionary
    """
    included_count = len(evidence_profiles)
    excluded_count = len(excluded_combinations)

    # Count results by effect type
    effect_type_counts: Dict[str, int] = {}
    total_results = 0
    complete_results = 0

    for profile in evidence_profiles:
        total_results += profile["result_count"]
        complete_results += profile["complete_result_count"]

        for result in profile["results"]:
            effect_type = result["effect_size_type"]
            effect_type_counts[effect_type] = (
                effect_type_counts.get(effect_type, 0) + 1
            )

    # Completeness distribution
    if evidence_profiles:
        completeness_values = [
            p["data_completeness"] for p in evidence_profiles
        ]
        completeness_dist = {
            "mean": sum(completeness_values) / len(completeness_values),
            "min": min(completeness_values),
            "max": max(completeness_values),
            "median": sorted(completeness_values)[
                len(completeness_values) // 2
            ],
        }
    else:
        completeness_dist = {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    # Exclusion reasons
    exclusion_reasons: Dict[str, int] = {}
    for exc in excluded_combinations:
        reason = exc["reason"]
        exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

    # Prepare unrecognized directions report
    total_unrecognized = sum(UNRECOGNIZED_DIRECTIONS.values())
    unrecognized_report = {
        "total_count": total_unrecognized,
        "unique_strings": len(UNRECOGNIZED_DIRECTIONS),
        "examples": dict(
            sorted(
                UNRECOGNIZED_DIRECTIONS.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:20]
        ),
    }

    res = PreprocessingStats(
        total_combinations=total_combinations,
        included_combinations=included_count,
        excluded_combinations=excluded_count,
        total_results=total_results,
        complete_results=complete_results,
        results_by_effect_type=effect_type_counts,
        completeness_distribution=completeness_dist,
        exclusion_reasons=exclusion_reasons,
        unrecognized_directions=unrecognized_report,
    )

    return res


# ==== Main function ====


def main():
    """Main function to preprocess evidence profiles."""
    args = parse_args()

    # ==== Initialization and validation ====

    logger.info("Starting evidence profile preprocessing")
    logger.info(f"Database path: {args.database_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Minimum results: {args.min_results}")
    logger.info(f"Maximum missing rate: {args.max_missing_rate}")

    # Check database exists
    if not args.database_path.exists():
        logger.error(f"Database not found: {args.database_path}")
        raise FileNotFoundError(f"Database not found: {args.database_path}")

    # Dry run validation
    if args.dry_run:
        logger.info("Dry run mode: Checking database and setup...")
        logger.info(f"✓ Database exists: {args.database_path}")
        logger.info(f"✓ Output directory will be: {args.output_dir}")
        logger.info("Dry run completed. Exiting without processing.")
        return

    # ==== Connect to database ====

    logger.info("Connecting to database...")
    conn = duckdb.connect(str(args.database_path), read_only=True)

    # ==== Extract PMID-model combinations ====

    logger.info("Extracting PMID-model combinations...")
    combinations = get_pmid_model_combinations(conn)
    logger.info(f"Found {len(combinations)} PMID-model combinations")

    # Apply limit if specified
    if args.limit is not None:
        combinations = combinations[: args.limit]
        logger.info(f"Limited to {len(combinations)} combinations for testing")

    # ==== Process each combination ====

    logger.info("Processing combinations...")
    evidence_profiles: List[EvidenceProfile] = []
    excluded_combinations: List[Dict[str, str]] = []

    for combo in tqdm(combinations, desc="Processing combinations"):
        pmid = combo["pmid"]
        model = combo["model"]
        title = combo["title"]
        publication_year = combo["publication_year"]

        profile, exclusion_reason = process_combination(
            conn,
            pmid,
            model,
            title,
            publication_year,
            args.min_results,
            args.max_missing_rate,
        )

        if profile is not None:
            evidence_profiles.append(profile)
        else:
            # exclusion_reason is guaranteed to be str when profile is None
            assert exclusion_reason is not None
            excluded_combinations.append(
                {"pmid": pmid, "model": model, "reason": exclusion_reason}
            )

    logger.info(
        f"Processing complete. Included: {len(evidence_profiles)}, "
        f"Excluded: {len(excluded_combinations)}"
    )

    # ==== Compute statistics ====

    logger.info("Computing preprocessing statistics...")
    stats = compute_preprocessing_stats(
        evidence_profiles, excluded_combinations, len(combinations)
    )

    # ==== Write output files ====

    logger.info("Writing output files...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write evidence profiles
    output_path_profiles = args.output_dir / "evidence-profiles.json"
    with open(output_path_profiles, "w") as f:
        json.dump(evidence_profiles, f, indent=2)
    logger.info(f"Evidence profiles written to {output_path_profiles}")

    # Write statistics
    output_path_stats = args.output_dir / "preprocessing-stats.json"
    with open(output_path_stats, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics written to {output_path_stats}")

    # ==== Final summary ====

    logger.info("=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total combinations: {stats['total_combinations']}")
    logger.info(
        f"Included: {stats['included_combinations']} "
        f"({100 * stats['included_combinations'] / stats['total_combinations']:.1f}%)"
    )
    logger.info(
        f"Excluded: {stats['excluded_combinations']} "
        f"({100 * stats['excluded_combinations'] / stats['total_combinations']:.1f}%)"
    )
    logger.info(f"Total results: {stats['total_results']}")
    if stats["total_results"] > 0:
        logger.info(
            f"Complete results: {stats['complete_results']} "
            f"({100 * stats['complete_results'] / stats['total_results']:.1f}%)"
        )
    else:
        logger.info(f"Complete results: {stats['complete_results']}")
    logger.info(
        f"Data completeness (mean): {stats['completeness_distribution']['mean']:.3f}"
    )
    logger.info("=" * 60)

    conn.close()
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
