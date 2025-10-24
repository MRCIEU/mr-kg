"""Unit tests for evidence profile similarity pipeline.

Tests cover:
- Effect size harmonization (log transformations, direction extraction)
- Similarity metric computations (concordance, Jaccard, Pearson, Cohen's kappa)
- Trait index matching and minimum pairs filtering
"""

import math
from typing import Dict, List, Tuple

import pytest

# Import functions from evidence profile scripts
import importlib.util
import sys
from pathlib import Path

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts" / "evidence-profile"


def load_module_from_path(module_name: str, file_path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load module from {file_path}")
    if spec.loader is None:
        raise ImportError(f"Module spec has no loader for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load modules
preprocess = load_module_from_path(
    "preprocess_evidence_profiles",
    SCRIPTS_DIR / "preprocess-evidence-profiles.py",
)
compute_sim = load_module_from_path(
    "compute_evidence_similarity",
    SCRIPTS_DIR / "compute-evidence-similarity.py",
)

# Import specific functions
harmonize_effect_size = preprocess.harmonize_effect_size
classify_direction = preprocess.classify_direction
classify_significance = preprocess.classify_significance

compute_direction_concordance = compute_sim.compute_direction_concordance
compute_effect_size_similarity = compute_sim.compute_effect_size_similarity
compute_evidence_overlap = compute_sim.compute_evidence_overlap
compute_statistical_consistency = compute_sim.compute_statistical_consistency
match_exposure_outcome_pairs = compute_sim.match_exposure_outcome_pairs


# ==== TestHarmonization ====


class TestHarmonization:
    """Test effect size transformations and harmonization functions."""

    def test_harmonize_beta_unchanged(self):
        """Beta coefficients should remain unchanged."""
        res = harmonize_effect_size("beta", 0.5)
        assert res == 0.5

        res = harmonize_effect_size("beta", -0.3)
        assert res == -0.3

        res = harmonize_effect_size("beta", 0.0)
        assert res == 0.0

    def test_harmonize_or_log_transform(self):
        """Odds ratios should be log-transformed."""
        res = harmonize_effect_size("OR", 2.0)
        assert res == pytest.approx(math.log(2.0))

        res = harmonize_effect_size("OR", 0.5)
        assert res == pytest.approx(math.log(0.5))

        res = harmonize_effect_size("OR", 1.0)
        assert res == pytest.approx(0.0)

    def test_harmonize_hr_log_transform(self):
        """Hazard ratios should be log-transformed."""
        res = harmonize_effect_size("HR", 3.0)
        assert res == pytest.approx(math.log(3.0))

        res = harmonize_effect_size("HR", 0.25)
        assert res == pytest.approx(math.log(0.25))

    def test_harmonize_invalid_or_hr(self):
        """Invalid OR/HR values should return None."""
        res = harmonize_effect_size("OR", 0.0)
        assert res is None

        res = harmonize_effect_size("OR", -1.0)
        assert res is None

        res = harmonize_effect_size("HR", -0.5)
        assert res is None

    def test_harmonize_none_value(self):
        """None values should return None."""
        res = harmonize_effect_size("beta", None)
        assert res is None

        res = harmonize_effect_size("OR", None)
        assert res is None

    def test_harmonize_unknown_type(self):
        """Unknown effect types should return None."""
        res = harmonize_effect_size("unknown", 1.5)
        assert res is None

    def test_classify_direction_positive(self):
        """Positive direction terms should return 1."""
        positive_cases = [
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

        for term in positive_cases:
            res = classify_direction(term)
            assert res == 1, f"Failed for term: {term}"

    def test_classify_direction_negative(self):
        """Negative direction terms should return -1."""
        negative_cases = [
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

        for term in negative_cases:
            res = classify_direction(term)
            assert res == -1, f"Failed for term: {term}"

    def test_classify_direction_null(self):
        """Null/unclear direction terms should return 0."""
        null_cases = [
            "null",
            "no association",
            "not associated",
            "no effect",
            "bidirectional",
            "no significant impact",
            "not causally connected",
            "does not increase or decrease",
        ]

        for term in null_cases:
            res = classify_direction(term)
            assert res == 0, f"Failed for term: {term}"

    def test_classify_direction_empty(self):
        """Empty direction should return 0."""
        res = classify_direction("")
        assert res == 0

        res = classify_direction(None)
        assert res == 0

    def test_classify_direction_case_insensitive(self):
        """Direction classification should be case-insensitive."""
        res = classify_direction("POSITIVE")
        assert res == 1

        res = classify_direction("Negative")
        assert res == -1

        res = classify_direction("NULL")
        assert res == 0

    def test_classify_significance_threshold(self):
        """P-values should be classified at 0.05 threshold."""
        res = classify_significance(0.04)
        assert res is True

        res = classify_significance(0.049)
        assert res is True

        res = classify_significance(0.05)
        assert res is False

        res = classify_significance(0.051)
        assert res is False

        res = classify_significance(0.5)
        assert res is False

    def test_classify_significance_string_pvalue(self):
        """String p-values should be parsed and classified."""
        res = classify_significance("0.03")
        assert res is True

        res = classify_significance("0.1")
        assert res is False

    def test_classify_significance_invalid(self):
        """Invalid p-values should return False."""
        res = classify_significance(None)
        assert res is False

        res = classify_significance("not_a_number")
        assert res is False

        res = classify_significance("N/A")
        assert res is False


# ==== TestSimilarityMetrics ====


class TestSimilarityMetrics:
    """Test similarity metric computation functions."""

    @pytest.fixture
    def matched_pairs_basic(self) -> List[Tuple[Dict, Dict]]:
        """Create basic matched pairs fixture."""
        return [
            (
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 1,
                    "harmonized_effect_size": 0.5,
                    "direction": 1,
                    "is_significant": True,
                },
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 1,
                    "harmonized_effect_size": 0.6,
                    "direction": 1,
                    "is_significant": True,
                },
            ),
            (
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 2,
                    "harmonized_effect_size": -0.3,
                    "direction": -1,
                    "is_significant": False,
                },
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 2,
                    "harmonized_effect_size": -0.4,
                    "direction": -1,
                    "is_significant": False,
                },
            ),
            (
                {
                    "exposure_trait_index": 1,
                    "outcome_trait_index": 2,
                    "harmonized_effect_size": 0.2,
                    "direction": 1,
                    "is_significant": True,
                },
                {
                    "exposure_trait_index": 1,
                    "outcome_trait_index": 2,
                    "harmonized_effect_size": 0.25,
                    "direction": 1,
                    "is_significant": False,
                },
            ),
        ]

    def test_effect_size_similarity_perfect_correlation(self):
        """Perfect correlation should yield r=1.0."""
        pairs = [
            ({"harmonized_effect_size": 0.5}, {"harmonized_effect_size": 1.0}),
            ({"harmonized_effect_size": 1.0}, {"harmonized_effect_size": 2.0}),
            ({"harmonized_effect_size": 1.5}, {"harmonized_effect_size": 3.0}),
        ]

        res = compute_effect_size_similarity(pairs)
        assert res == pytest.approx(1.0, abs=1e-6)

    def test_effect_size_similarity_negative_correlation(self):
        """Negative correlation should yield r=-1.0."""
        pairs = [
            (
                {"harmonized_effect_size": 0.5},
                {"harmonized_effect_size": -0.5},
            ),
            (
                {"harmonized_effect_size": 1.0},
                {"harmonized_effect_size": -1.0},
            ),
            (
                {"harmonized_effect_size": 1.5},
                {"harmonized_effect_size": -1.5},
            ),
        ]

        res = compute_effect_size_similarity(pairs)
        assert res == pytest.approx(-1.0, abs=1e-6)

    def test_effect_size_similarity_insufficient_pairs(self):
        """Fewer than 3 pairs should return None."""
        pairs = [
            ({"harmonized_effect_size": 0.5}, {"harmonized_effect_size": 0.6}),
            ({"harmonized_effect_size": 1.0}, {"harmonized_effect_size": 1.1}),
        ]

        res = compute_effect_size_similarity(pairs)
        assert res is None

    def test_direction_concordance_perfect_agreement(self):
        """All concordant directions should yield 1.0."""
        pairs = [
            ({"direction": 1}, {"direction": 1}),
            ({"direction": -1}, {"direction": -1}),
            ({"direction": 1}, {"direction": 1}),
        ]

        res = compute_direction_concordance(pairs)
        assert res == 1.0

    def test_direction_concordance_perfect_disagreement(self):
        """All discordant directions should yield -1.0."""
        pairs = [
            ({"direction": 1}, {"direction": -1}),
            ({"direction": -1}, {"direction": 1}),
            ({"direction": 1}, {"direction": -1}),
        ]

        res = compute_direction_concordance(pairs)
        assert res == -1.0

    def test_direction_concordance_mixed(self):
        """Mixed concordance should yield proportional score."""
        pairs = [
            ({"direction": 1}, {"direction": 1}),
            ({"direction": 1}, {"direction": -1}),
            ({"direction": -1}, {"direction": -1}),
        ]

        res = compute_direction_concordance(pairs)
        expected = (2 - 1) / 3
        assert res == pytest.approx(expected)

    def test_direction_concordance_ignores_zero(self):
        """Direction=0 should be ignored in concordance calculation."""
        pairs = [
            ({"direction": 1}, {"direction": 1}),
            ({"direction": 0}, {"direction": 1}),
            ({"direction": 1}, {"direction": 0}),
            ({"direction": -1}, {"direction": -1}),
        ]

        res = compute_direction_concordance(pairs)
        assert res == 1.0

    def test_direction_concordance_all_zero(self):
        """All zero directions should return 0.0."""
        pairs = [
            ({"direction": 0}, {"direction": 0}),
            ({"direction": 0}, {"direction": 1}),
            ({"direction": 1}, {"direction": 0}),
        ]

        res = compute_direction_concordance(pairs)
        assert res == 0.0

    def test_statistical_consistency_perfect_agreement(
        self, matched_pairs_basic
    ):
        """Perfect agreement should yield high Cohen's kappa."""
        pairs = [
            ({"is_significant": True}, {"is_significant": True}),
            ({"is_significant": False}, {"is_significant": False}),
            ({"is_significant": True}, {"is_significant": True}),
        ]

        res = compute_statistical_consistency(pairs)
        assert res == 1.0

    def test_statistical_consistency_no_agreement(self):
        """Complete disagreement should yield negative kappa."""
        pairs = [
            ({"is_significant": True}, {"is_significant": False}),
            ({"is_significant": False}, {"is_significant": True}),
            ({"is_significant": True}, {"is_significant": False}),
        ]

        res = compute_statistical_consistency(pairs)
        assert res < 0.0

    def test_statistical_consistency_insufficient_pairs(self):
        """Fewer than 3 pairs should return None."""
        pairs = [
            ({"is_significant": True}, {"is_significant": True}),
            ({"is_significant": False}, {"is_significant": False}),
        ]

        res = compute_statistical_consistency(pairs)
        assert res is None

    def test_statistical_consistency_no_variance(self):
        """No variance in labels should return None."""
        pairs = [
            ({"is_significant": True}, {"is_significant": True}),
            ({"is_significant": True}, {"is_significant": True}),
            ({"is_significant": True}, {"is_significant": True}),
        ]

        res = compute_statistical_consistency(pairs)
        assert res is None

    def test_evidence_overlap_perfect_overlap(self):
        """Identical significant sets should yield Jaccard=1.0."""
        pairs = [
            (
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 1,
                    "is_significant": True,
                },
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 1,
                    "is_significant": True,
                },
            ),
            (
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 2,
                    "is_significant": True,
                },
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 2,
                    "is_significant": True,
                },
            ),
        ]

        res = compute_evidence_overlap(pairs)
        assert res == 1.0

    def test_evidence_overlap_no_overlap(self):
        """Disjoint significant sets should yield Jaccard=0.0."""
        pairs = [
            (
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 1,
                    "is_significant": True,
                },
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 1,
                    "is_significant": False,
                },
            ),
            (
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 2,
                    "is_significant": False,
                },
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 2,
                    "is_significant": True,
                },
            ),
        ]

        res = compute_evidence_overlap(pairs)
        assert res == 0.0

    def test_evidence_overlap_partial(self):
        """Partial overlap should yield proportional Jaccard score."""
        pairs = [
            (
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 1,
                    "is_significant": True,
                },
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 1,
                    "is_significant": True,
                },
            ),
            (
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 2,
                    "is_significant": True,
                },
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 2,
                    "is_significant": False,
                },
            ),
            (
                {
                    "exposure_trait_index": 1,
                    "outcome_trait_index": 2,
                    "is_significant": False,
                },
                {
                    "exposure_trait_index": 1,
                    "outcome_trait_index": 2,
                    "is_significant": True,
                },
            ),
        ]

        res = compute_evidence_overlap(pairs)
        expected = 1 / 3
        assert res == pytest.approx(expected)

    def test_evidence_overlap_both_null_edge_case(self):
        """Both studies with zero significant findings should return 0.0."""
        pairs = [
            (
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 1,
                    "is_significant": False,
                },
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 1,
                    "is_significant": False,
                },
            ),
            (
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 2,
                    "is_significant": False,
                },
                {
                    "exposure_trait_index": 0,
                    "outcome_trait_index": 2,
                    "is_significant": False,
                },
            ),
        ]

        res = compute_evidence_overlap(pairs)
        assert res == 0.0

    def test_evidence_overlap_empty_pairs(self):
        """Empty pairs list should return 0.0."""
        res = compute_evidence_overlap([])
        assert res == 0.0


# ==== TestMatching ====


class TestMatching:
    """Test trait index matching and filtering functions."""

    @pytest.fixture
    def query_results(self) -> List[Dict]:
        """Create query results fixture."""
        return [
            {
                "exposure_trait_index": 0,
                "outcome_trait_index": 1,
                "value": 1.0,
            },
            {
                "exposure_trait_index": 0,
                "outcome_trait_index": 2,
                "value": 2.0,
            },
            {
                "exposure_trait_index": 1,
                "outcome_trait_index": 2,
                "value": 3.0,
            },
            {
                "exposure_trait_index": 2,
                "outcome_trait_index": 3,
                "value": 4.0,
            },
        ]

    @pytest.fixture
    def similar_results(self) -> List[Dict]:
        """Create similar results fixture."""
        return [
            {
                "exposure_trait_index": 0,
                "outcome_trait_index": 1,
                "value": 1.5,
            },
            {
                "exposure_trait_index": 0,
                "outcome_trait_index": 2,
                "value": 2.5,
            },
            {
                "exposure_trait_index": 1,
                "outcome_trait_index": 3,
                "value": 5.0,
            },
        ]

    def test_match_exposure_outcome_pairs_full_overlap(self):
        """Full overlap should match all pairs."""
        query = [
            {"exposure_trait_index": 0, "outcome_trait_index": 1},
            {"exposure_trait_index": 0, "outcome_trait_index": 2},
        ]

        similar = [
            {"exposure_trait_index": 0, "outcome_trait_index": 1},
            {"exposure_trait_index": 0, "outcome_trait_index": 2},
        ]

        res = match_exposure_outcome_pairs(query, similar)
        assert len(res) == 2

    def test_match_exposure_outcome_pairs_partial_overlap(
        self, query_results, similar_results
    ):
        """Partial overlap should match only common pairs."""
        res = match_exposure_outcome_pairs(query_results, similar_results)
        assert len(res) == 2

        matched_keys = [
            (pair[0]["exposure_trait_index"], pair[0]["outcome_trait_index"])
            for pair in res
        ]
        assert (0, 1) in matched_keys
        assert (0, 2) in matched_keys
        assert (1, 2) not in matched_keys
        assert (2, 3) not in matched_keys

    def test_match_exposure_outcome_pairs_no_overlap(self):
        """No overlap should return empty list."""
        query = [
            {"exposure_trait_index": 0, "outcome_trait_index": 1},
        ]

        similar = [
            {"exposure_trait_index": 1, "outcome_trait_index": 2},
        ]

        res = match_exposure_outcome_pairs(query, similar)
        assert len(res) == 0

    def test_match_exposure_outcome_pairs_empty_inputs(self):
        """Empty inputs should return empty list."""
        res = match_exposure_outcome_pairs([], [])
        assert len(res) == 0

        query = [{"exposure_trait_index": 0, "outcome_trait_index": 1}]
        res = match_exposure_outcome_pairs(query, [])
        assert len(res) == 0

        similar = [{"exposure_trait_index": 0, "outcome_trait_index": 1}]
        res = match_exposure_outcome_pairs([], similar)
        assert len(res) == 0

    def test_match_preserves_order(self, query_results, similar_results):
        """Matched pairs should follow query order."""
        res = match_exposure_outcome_pairs(query_results, similar_results)

        assert res[0][0]["outcome_trait_index"] == 1
        assert res[1][0]["outcome_trait_index"] == 2

    def test_match_preserves_values(self, query_results, similar_results):
        """Matched pairs should preserve original values."""
        res = match_exposure_outcome_pairs(query_results, similar_results)

        assert res[0][0]["value"] == 1.0
        assert res[0][1]["value"] == 1.5

        assert res[1][0]["value"] == 2.0
        assert res[1][1]["value"] == 2.5
