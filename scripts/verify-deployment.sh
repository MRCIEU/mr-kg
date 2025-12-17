#!/bin/bash
# Deployment verification script for MR-KG web services
#
# This script verifies that the MR-KG deployment is working correctly by
# checking service health, API endpoints, and database connectivity.
#
# Usage:
#   ./scripts/verify-deployment.sh [--api-url URL] [--webapp-url URL]
#
# Options:
#   --api-url URL      API base URL (default: http://localhost:8000)
#   --webapp-url URL   Webapp base URL (default: http://localhost:8501)

set -euo pipefail

# ==== Configuration ====

API_URL="${API_URL:-http://localhost:8000/mr-kg/api}"
WEBAPP_URL="${WEBAPP_URL:-http://localhost:8501/mr-kg}"
TIMEOUT=10
PASSED=0
FAILED=0

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --webapp-url)
            WEBAPP_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==== Helper functions ====

log_info() {
    echo "[INFO] $1"
}

log_pass() {
    echo "[PASS] $1"
    ((PASSED++))
}

log_fail() {
    echo "[FAIL] $1"
    ((FAILED++))
}

check_endpoint() {
    local name="$1"
    local url="$2"
    local expected_status="${3:-200}"

    local status
    status=$(curl -sf -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$url" 2>/dev/null) || status="000"

    if [[ "$status" == "$expected_status" ]]; then
        log_pass "$name (HTTP $status)"
        return 0
    else
        log_fail "$name (HTTP $status, expected $expected_status)"
        return 1
    fi
}

check_json_response() {
    local name="$1"
    local url="$2"
    local field="$3"

    local response
    response=$(curl -sf --max-time "$TIMEOUT" "$url" 2>/dev/null) || {
        log_fail "$name (request failed)"
        return 1
    }

    if echo "$response" | grep -q "\"$field\""; then
        log_pass "$name"
        return 0
    else
        log_fail "$name (field '$field' not found)"
        return 1
    fi
}

# ==== Main verification ====

echo "========================================"
echo "MR-KG Deployment Verification"
echo "========================================"
echo ""
echo "API URL: $API_URL"
echo "Webapp URL: $WEBAPP_URL"
echo ""

# ---- Check API health ----
log_info "Checking API health..."
check_endpoint "API health endpoint" "$API_URL/health"

# ---- Check API documentation ----
log_info "Checking API documentation..."
check_endpoint "API OpenAPI docs" "$API_URL/"

# ---- Check database connectivity via health endpoint ----
log_info "Checking database connectivity..."
check_json_response "Database status in health" "$API_URL/health" "databases"

# ---- Check API endpoints ----
log_info "Checking API endpoints..."
check_endpoint "Studies endpoint" "$API_URL/studies"

# ---- Check webapp health ----
log_info "Checking webapp health..."
check_endpoint "Webapp health endpoint" "$WEBAPP_URL/_stcore/health"

# ---- Check webapp main page ----
log_info "Checking webapp main page..."
check_endpoint "Webapp main page" "$WEBAPP_URL"

# ==== Summary ====

echo ""
echo "========================================"
echo "Verification Summary"
echo "========================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo "All checks passed!"
    exit 0
else
    echo "Some checks failed. Please review the output above."
    exit 1
fi
