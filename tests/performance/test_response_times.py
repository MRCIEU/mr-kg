"""Performance tests for MR-KG API response times.

This script measures response times for key API endpoints to verify
they meet performance targets. Run with the API service active.

Usage:
    python tests/performance/test_response_times.py

Targets:
    - Autocomplete: <300ms
    - Study search: <500ms
    - Similarity queries: <1000ms
"""

import os
import statistics
import time

import httpx


# ==== Configuration ====


def get_api_base_url() -> str:
    """Get API base URL from environment or default."""
    return os.environ.get("API_URL", "http://localhost:8000")


# ==== Test runner ====


class ResponseTimeTest:
    """Test runner for measuring API response times."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results: dict[str, list[float]] = {}

    def measure(
        self,
        name: str,
        endpoint: str,
        params: dict | None = None,
        iterations: int = 5,
    ) -> None:
        """Measure response time for an endpoint.

        Args:
            name: Test name for reporting
            endpoint: API endpoint path
            params: Query parameters
            iterations: Number of test iterations
        """
        times = []

        with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
            for _ in range(iterations):
                start = time.perf_counter()
                try:
                    response = client.get(endpoint, params=params)
                    end = time.perf_counter()

                    if response.status_code in (200, 404):
                        elapsed_ms = (end - start) * 1000
                        times.append(elapsed_ms)
                except Exception as e:
                    print(f"  Error: {e}")

        self.results[name] = times

    def print_results(self) -> None:
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("Performance Test Results")
        print("=" * 60)

        for name, times in self.results.items():
            if times:
                avg = statistics.mean(times)
                min_t = min(times)
                max_t = max(times)
                std = statistics.stdev(times) if len(times) > 1 else 0

                print(f"\n{name}:")
                print(f"  Samples: {len(times)}")
                print(f"  Average: {avg:.1f}ms")
                print(f"  Min:     {min_t:.1f}ms")
                print(f"  Max:     {max_t:.1f}ms")
                print(f"  Std Dev: {std:.1f}ms")
            else:
                print(f"\n{name}:")
                print("  No successful measurements")

        print("\n" + "=" * 60)


# ==== Test definitions ====


def run_autocomplete_tests(tester: ResponseTimeTest) -> None:
    """Run autocomplete endpoint performance tests."""
    print("\nRunning autocomplete tests...")

    # ---- Trait autocomplete ----
    tester.measure(
        "Trait autocomplete (short)",
        "/traits/autocomplete",
        params={"q": "body", "limit": 20},
    )

    tester.measure(
        "Trait autocomplete (longer)",
        "/traits/autocomplete",
        params={"q": "blood pressure", "limit": 20},
    )

    # ---- Study autocomplete ----
    tester.measure(
        "Study autocomplete (short)",
        "/studies/autocomplete",
        params={"q": "diabetes", "limit": 20},
    )

    tester.measure(
        "Study autocomplete (longer)",
        "/studies/autocomplete",
        params={"q": "mendelian randomization", "limit": 20},
    )


def run_search_tests(tester: ResponseTimeTest) -> None:
    """Run study search endpoint performance tests."""
    print("\nRunning search tests...")

    # ---- Basic search ----
    tester.measure(
        "Study search (no filter)",
        "/studies",
        params={"model": "gpt-5", "limit": 20},
    )

    # ---- Text query search ----
    tester.measure(
        "Study search (text query)",
        "/studies",
        params={"q": "diabetes", "model": "gpt-5", "limit": 20},
    )

    # ---- Trait filter search ----
    tester.measure(
        "Study search (trait filter)",
        "/studies",
        params={"trait": "body mass index", "model": "gpt-5", "limit": 20},
    )


def run_extraction_tests(tester: ResponseTimeTest, pmid: str | None) -> None:
    """Run extraction endpoint performance tests."""
    print("\nRunning extraction tests...")

    if not pmid:
        print("  Skipping - no PMID available")
        return

    tester.measure(
        "Get extraction",
        f"/studies/{pmid}/extraction",
        params={"model": "gpt-5"},
    )


def run_similarity_tests(tester: ResponseTimeTest, pmid: str | None) -> None:
    """Run similarity endpoint performance tests."""
    print("\nRunning similarity tests...")

    if not pmid:
        print("  Skipping - no PMID available")
        return

    # ---- Trait similarity ----
    tester.measure(
        "Trait similarity (limit 10)",
        f"/studies/{pmid}/similar/trait",
        params={"model": "gpt-5", "limit": 10},
    )

    tester.measure(
        "Trait similarity (limit 50)",
        f"/studies/{pmid}/similar/trait",
        params={"model": "gpt-5", "limit": 50},
    )

    # ---- Evidence similarity ----
    tester.measure(
        "Evidence similarity (limit 10)",
        f"/studies/{pmid}/similar/evidence",
        params={"model": "gpt-5", "limit": 10},
    )

    tester.measure(
        "Evidence similarity (limit 50)",
        f"/studies/{pmid}/similar/evidence",
        params={"model": "gpt-5", "limit": 50},
    )


def run_statistics_tests(tester: ResponseTimeTest) -> None:
    """Run statistics endpoint performance tests."""
    print("\nRunning statistics tests...")

    tester.measure(
        "Get statistics",
        "/statistics",
    )


# ==== Main ====


def get_sample_pmid(base_url: str) -> str | None:
    """Get a sample PMID from the API for testing.

    Args:
        base_url: API base URL

    Returns:
        A PMID string or None if unavailable
    """
    try:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            response = client.get(
                "/studies",
                params={"model": "gpt-5", "limit": 1},
            )
            if response.status_code == 200:
                data = response.json()
                studies = data.get("studies", [])
                if studies:
                    return studies[0]["pmid"]
    except Exception:
        pass
    return None


def check_api_available(base_url: str) -> bool:
    """Check if the API is available.

    Args:
        base_url: API base URL

    Returns:
        True if API is reachable
    """
    try:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            response = client.get("/health")
            return response.status_code == 200
    except Exception:
        return False


def main() -> None:
    """Run all performance tests."""
    base_url = get_api_base_url()

    print(f"MR-KG API Performance Tests")
    print(f"API URL: {base_url}")
    print("-" * 60)

    # ---- Check API availability ----
    if not check_api_available(base_url):
        print("\nError: API is not available at", base_url)
        print("Please start the API service before running performance tests.")
        return

    # ---- Get sample PMID for testing ----
    pmid = get_sample_pmid(base_url)
    if pmid:
        print(f"Using sample PMID: {pmid}")
    else:
        print("Warning: Could not get sample PMID, some tests will be skipped")

    # ---- Run tests ----
    tester = ResponseTimeTest(base_url)

    run_autocomplete_tests(tester)
    run_search_tests(tester)
    run_extraction_tests(tester, pmid)
    run_similarity_tests(tester, pmid)
    run_statistics_tests(tester)

    # ---- Print results ----
    tester.print_results()

    # ---- Check against targets ----
    print("\nPerformance Targets Check:")
    print("-" * 60)

    targets = {
        "Trait autocomplete (short)": 300,
        "Trait autocomplete (longer)": 300,
        "Study autocomplete (short)": 300,
        "Study autocomplete (longer)": 300,
        "Study search (no filter)": 500,
        "Study search (text query)": 500,
        "Study search (trait filter)": 500,
        "Get extraction": 500,
        "Trait similarity (limit 10)": 1000,
        "Trait similarity (limit 50)": 1000,
        "Evidence similarity (limit 10)": 1000,
        "Evidence similarity (limit 50)": 1000,
        "Get statistics": 500,
    }

    passed = 0
    failed = 0

    for name, target in targets.items():
        times = tester.results.get(name, [])
        if times:
            avg = statistics.mean(times)
            status = "PASS" if avg <= target else "FAIL"
            if status == "PASS":
                passed += 1
            else:
                failed += 1
            print(f"  {name}: {avg:.0f}ms (target: {target}ms) [{status}]")
        else:
            print(f"  {name}: N/A")

    print(f"\nSummary: {passed} passed, {failed} failed")


if __name__ == "__main__":
    main()
