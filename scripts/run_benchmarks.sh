#!/bin/bash
# Run benchmark suite with different profiles

set -e

echo "Running vLLM Benchmark Suite"
echo "=============================="
echo ""

# Ensure server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✗ Server is not running!"
    echo "Start it with: ./scripts/start_server.sh"
    exit 1
fi

echo "✓ Server is running"
echo ""

# Install dependencies if needed
if ! python -c "import requests" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run benchmarks
profiles=("quick" "light" "medium")

for profile in "${profiles[@]}"; do
    echo "Running $profile profile..."
    python -m bench.benchmark --profile "$profile"
    echo ""
    sleep 2
done

echo "=============================="
echo "Benchmark suite completed!"
echo "Results saved to results/"
echo ""
echo "View results:"
echo "  ls -lh results/"
