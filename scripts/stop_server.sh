#!/bin/bash
# Stop vLLM server

set -e

echo "Stopping vLLM server..."
docker compose down

echo "✓ Server stopped"
