.PHONY: help install start stop test benchmark baseline clean

help:
	@echo "vLLM Serving Optimization - Available Commands"
	@echo "=============================================="
	@echo "  make install    - Install Python dependencies"
	@echo "  make start      - Start vLLM server (Docker)"
	@echo "  make stop       - Stop vLLM server"
	@echo "  make test       - Quick server test"
	@echo "  make benchmark  - Run benchmark suite"
	@echo "  make baseline   - Run HF baseline inference"
	@echo "  make clean      - Clean results and cache"

install:
	pip install -r requirements.txt

start:
	@bash scripts/start_server.sh

stop:
	@bash scripts/stop_server.sh

test:
	@echo "Testing vLLM server..."
	@curl -s http://localhost:8000/v1/models | python -m json.tool
	@echo ""
	@python -m bench.benchmark --profile quick

benchmark:
	@bash scripts/run_benchmarks.sh

baseline:
	@echo "Running HF baseline inference..."
	@python -m baseline.hf_inference --num-requests 10 --prompt-tokens 128 --max-new-tokens 64

clean:
	rm -rf results/*.json results/*.csv
	rm -rf __pycache__ bench/__pycache__ baseline/__pycache__
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
