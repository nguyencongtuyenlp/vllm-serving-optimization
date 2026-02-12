#!/bin/bash
# Start vLLM server using docker-compose

set -e

echo "Starting vLLM OpenAI-compatible server..."

# Check if .env exists, if not copy from example
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please review .env and adjust MODEL_ID and other settings if needed."
fi

# Pull latest vLLM image
echo "Pulling latest vLLM image..."
docker compose pull

# Start server
echo "Starting server..."
docker compose up -d

# Wait for server to be ready
echo "Waiting for server to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        echo ""
        echo "Server URL: http://localhost:8000"
        echo "API Docs: http://localhost:8000/docs"
        echo ""
        echo "Test with:"
        echo "  curl http://localhost:8000/v1/models"
        echo ""
        echo "View logs:"
        echo "  docker compose logs -f"
        exit 0
    fi
    
    attempt=$((attempt + 1))
    echo "Attempt $attempt/$max_attempts..."
    sleep 2
done

echo "✗ Server failed to start within timeout"
echo "Check logs with: docker compose logs"
exit 1
