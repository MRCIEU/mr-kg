#!/usr/bin/env python3
"""
Debug script to test processing a single PMID-model combination.

This script helps identify why combinations are being excluded during
preprocessing by showing detailed information about each processing step.
"""

import json
import math
from typing import Any, Dict, Optional

import duckdb
from yiutils.project_utils import find_project_root

# Setup paths
PROJECT_ROOT = find_project_root("docker-compose.yml")
DB_PATH = PROJECT_ROOT / "data" / "db" / "vector_store.db"

# Import helper functions (copy from main script to avoid import issues)


def harmonize_effect_size(effect_type: str, value: float) -> Optional[float]:
    """Transform effect size to common scale."""
    if value is None:
        return None

    try:
        if effect_type in ["OR", "HR"]:
            if value <= 0:
                print(
                    f"  WARNING: Invalid {effect_type} value {value} (must be >0)"
                )
                return None
            return math.log(value)
        elif effect_type == "beta":
            return value
        else:
            print(f"  WARNING: Unknown effect type: {effect_type}")
            return None
    except (ValueError, TypeError) as e:
        print(f"  WARNING: Error harmonizing {effect_type} value {value}: {e}")
        return None


def extract_effect_size_info(
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Extract effect size information from a result."""
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

    return None


def classify_direction(direction_str: str) -> int:
    """Classify effect direction to numeric indicator."""
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
    ]

    if direction_lower in positive_terms:
        return 1
    elif direction_lower in negative_terms:
        return -1
    else:
        return 0


def debug_single_result(
    result: Dict[str, Any], trait_map: Dict[str, int], idx: int
) -> None:
    """Debug a single result and show why it passes/fails."""
    print(f"\n{'=' * 60}")
    print(f"Result #{idx}")
    print(f"{'=' * 60}")

    # Show raw result (limited fields)
    print("\nRaw result (key fields):")
    print(f"  exposure: {result.get('exposure')}")
    print(f"  outcome: {result.get('outcome')}")
    print(f"  beta: {result.get('beta')}")
    print(f"  odds ratio: {result.get('odds ratio')}")
    print(f"  hazard ratio: {result.get('hazard ratio')}")
    print(f"  direction: {result.get('direction')}")
    print(f"  P-value: {result.get('P-value')}")

    # Check exposure/outcome
    exposure_label = result.get("exposure")
    outcome_label = result.get("outcome")

    if not exposure_label or not outcome_label:
        print("\nFAIL: Missing exposure or outcome")
        return

    # Check trait mapping
    exposure_index = trait_map.get(exposure_label)
    outcome_index = trait_map.get(outcome_label)
    print(
        f"\nTrait indices: exposure={exposure_index}, outcome={outcome_index}"
    )

    if exposure_index is None or outcome_index is None:
        print("FAIL: Trait not in mapping")
        return

    # Check effect size
    effect_info = extract_effect_size_info(result)
    if effect_info is None:
        print("FAIL: No valid effect size")
        return
    else:
        print(
            f"PASS: Effect size extracted: {effect_info['effect_type']} = {effect_info['original_value']}"
        )

    # Check direction
    direction_str = result.get("direction", "")
    direction = classify_direction(direction_str)
    print(f"Direction: '{direction_str}' -> {direction}")

    if direction == 0:
        print("FAIL: Direction is null/unclear")
        return
    else:
        print("PASS: Valid direction")

    print("\nSUCCESS: Result would be included")


def main():
    """Debug a single PMID-model combination."""
    # Test case from summary
    pmid = "37845420"
    model = "llama3"

    print(f"Debugging PMID: {pmid}, Model: {model}")
    print(f"{'=' * 60}\n")

    # Connect to database
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    # Get results
    query = """
    SELECT mr.results
    FROM model_results mr
    WHERE mr.pmid = ?
    AND mr.model = ?
    """
    res = conn.execute(query, [pmid, model]).fetchall()

    if not res:
        print("ERROR: No results found for this combination")
        return

    results_json = res[0][0]
    if isinstance(results_json, str):
        results = json.loads(results_json)
    else:
        results = results_json

    # Handle list vs dict
    if isinstance(results, dict) and "results" in results:
        results = results["results"]

    print(f"Found {len(results)} results\n")

    # Get trait mapping
    query_traits = """
    SELECT DISTINCT
        mrt.trait_label,
        mrt.trait_index
    FROM model_result_traits mrt
    JOIN model_results mr ON mrt.model_result_id = mr.id
    WHERE mr.pmid = ?
    AND mr.model = ?
    """
    trait_rows = conn.execute(query_traits, [pmid, model]).fetchall()
    trait_map = {row[0]: row[1] for row in trait_rows}

    print(f"Trait mapping has {len(trait_map)} entries:")
    for trait_label, trait_idx in sorted(trait_map.items()):
        print(f"  {trait_label} -> {trait_idx}")

    # Debug each result
    for idx, result in enumerate(results, 1):
        debug_single_result(result, trait_map, idx)

    conn.close()


if __name__ == "__main__":
    main()
