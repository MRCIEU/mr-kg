#!/bin/bash
# Monitor preprocessing progress

OUTPUT_DIR="/Users/ik18445/local-projects/+dmer/+mr-paper-data-extraction/mr-kg/data/processed/evidence-profiles"
STATS_FILE="$OUTPUT_DIR/preprocessing-stats.json"

echo "=== Evidence Profile Preprocessing Monitor ==="
echo ""

# Check if process is running
if pgrep -f "preprocess-evidence-profiles.py" > /dev/null; then
    echo "✓ Process is RUNNING"
    echo ""
    
    # Get process info
    echo "Process details:"
    ps aux | grep "preprocess-evidence-profiles.py" | grep -v grep | awk '{print "  PID: " $2 "  CPU: " $3 "%  MEM: " $4 "%  TIME: " $10}'
    echo ""
else
    echo "✗ Process is NOT running"
    echo ""
fi

# Check if output file exists and show stats
if [ -f "$STATS_FILE" ]; then
    echo "Latest statistics:"
    cat "$STATS_FILE" | python3 -c "
import json
import sys
data = json.load(sys.stdin)
print(f\"  Total combinations: {data['total_combinations']:,}\")
print(f\"  Included: {data['included_combinations']:,} ({data['included_combinations']/data['total_combinations']*100:.1f}%)\")
print(f\"  Excluded: {data['excluded_combinations']:,} ({data['excluded_combinations']/data['total_combinations']*100:.1f}%)\")
print(f\"  Total results: {data['total_results']:,}\")
print(f\"  Complete results: {data['complete_results']:,}\")
print(f\"\\n  Effect types:\")
for etype, count in data['results_by_effect_type'].items():
    print(f\"    {etype}: {count}\")
print(f\"\\n  Exclusion reasons:\")
for reason, count in data['exclusion_reasons'].items():
    print(f\"    {reason}: {count}\")
" 2>/dev/null
else
    echo "No statistics file found yet"
fi

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To view full output in real-time:"
echo "  # Run preprocessing in a separate terminal and monitor output"
echo ""
