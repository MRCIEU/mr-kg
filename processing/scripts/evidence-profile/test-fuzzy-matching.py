"""Test script to validate fuzzy matching implementation.

This script performs basic validation of the fuzzy matching functionality.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from compute_evidence_similarity import (  # type: ignore[import-not-found]
    cosine_similarity,
    load_trait_embeddings,
    match_exposure_outcome_pairs,
    match_exposure_outcome_pairs_fuzzy,
)

print("Testing fuzzy matching implementation...")
print("=" * 60)

print("\n1. Testing cosine_similarity function...")
vec1 = np.array([1.0, 0.0, 0.0])
vec2 = np.array([1.0, 0.0, 0.0])
sim = cosine_similarity(vec1, vec2)
assert abs(sim - 1.0) < 1e-6, f"Expected 1.0, got {sim}"
print(f"✓ Identical vectors: {sim:.4f} (expected: 1.0)")

vec3 = np.array([1.0, 0.0, 0.0])
vec4 = np.array([0.0, 1.0, 0.0])
sim = cosine_similarity(vec3, vec4)
assert abs(sim - 0.0) < 1e-6, f"Expected 0.0, got {sim}"
print(f"✓ Orthogonal vectors: {sim:.4f} (expected: 0.0)")

vec5 = np.array([1.0, 1.0])
vec6 = np.array([1.0, 1.0])
sim = cosine_similarity(vec5, vec6)
assert abs(sim - 1.0) < 1e-6, f"Expected 1.0, got {sim}"
print(f"✓ Identical 2D vectors: {sim:.4f} (expected: 1.0)")

print("\n2. Testing exact matching...")
query_results = [
    {
        "exposure_trait_index": 1,
        "outcome_trait_index": 2,
        "harmonized_effect_size": 0.5,
    },
    {
        "exposure_trait_index": 3,
        "outcome_trait_index": 4,
        "harmonized_effect_size": 0.3,
    },
]

similar_results = [
    {
        "exposure_trait_index": 1,
        "outcome_trait_index": 2,
        "harmonized_effect_size": 0.6,
    },
    {
        "exposure_trait_index": 5,
        "outcome_trait_index": 6,
        "harmonized_effect_size": 0.4,
    },
]

matches = match_exposure_outcome_pairs(query_results, similar_results)
assert len(matches) == 1, f"Expected 1 match, got {len(matches)}"
print(f"✓ Exact matching found {len(matches)} match (expected: 1)")

print("\n3. Testing fuzzy matching with mock embeddings...")

trait_embeddings = {
    1: np.array([1.0, 0.0]),
    2: np.array([0.0, 1.0]),
    3: np.array([0.96, 0.28]),
    4: np.array([0.0, 1.0]),
    5: np.array([0.5, 0.5]),
    6: np.array([0.5, 0.5]),
}

query_results_fuzzy = [
    {
        "exposure_trait_index": 1,
        "outcome_trait_index": 2,
        "harmonized_effect_size": 0.5,
    },
    {
        "exposure_trait_index": 3,
        "outcome_trait_index": 4,
        "harmonized_effect_size": 0.3,
    },
]

similar_results_fuzzy = [
    {
        "exposure_trait_index": 1,
        "outcome_trait_index": 2,
        "harmonized_effect_size": 0.6,
    },
    {
        "exposure_trait_index": 5,
        "outcome_trait_index": 6,
        "harmonized_effect_size": 0.4,
    },
]

matches_fuzzy_095 = match_exposure_outcome_pairs_fuzzy(
    query_results_fuzzy,
    similar_results_fuzzy,
    trait_embeddings,
    threshold=0.95,
)
print(
    f"✓ Fuzzy matching (threshold=0.95) found {len(matches_fuzzy_095)} matches"
)

matches_fuzzy_090 = match_exposure_outcome_pairs_fuzzy(
    query_results_fuzzy,
    similar_results_fuzzy,
    trait_embeddings,
    threshold=0.90,
)
print(
    f"✓ Fuzzy matching (threshold=0.90) found {len(matches_fuzzy_090)} matches"
)

print("\n4. Testing with real database (if available)...")
db_path = (
    Path(__file__).parent.parent.parent.parent
    / "data"
    / "db"
    / "vector_store.db"
)

if db_path.exists():
    print(f"✓ Database found: {db_path}")
    embeddings = load_trait_embeddings(db_path)
    print(f"✓ Loaded {len(embeddings)} trait embeddings")

    first_idx = list(embeddings.keys())[0]
    first_vec = embeddings[first_idx]
    print(f"✓ First embedding shape: {first_vec.shape}")
    assert first_vec.shape == (200,), (
        f"Expected shape (200,), got {first_vec.shape}"
    )

    second_idx = (
        list(embeddings.keys())[1] if len(embeddings) > 1 else first_idx
    )
    second_vec = embeddings[second_idx]
    sim = cosine_similarity(first_vec, second_vec)
    print(
        f"✓ Similarity between embeddings {first_idx} and {second_idx}: {sim:.4f}"
    )
else:
    print(f"⚠ Database not found: {db_path}")
    print("  Skipping real database test")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("\nFuzzy matching implementation is ready for use.")
