# Project Structure

```
vllm/
├── bench/                          # Benchmark client package
│   ├── __init__.py
│   └── benchmark.py               # Main benchmark tool
│       ├── BenchmarkClient        # HTTP client with streaming
│       ├── VRAMMonitor            # GPU memory tracking
│       ├── RequestMetrics         # Per-request metrics
│       └── BenchmarkResults       # Aggregated results
│
├── baseline/                       # HuggingFace baseline
│   ├── __init__.py
│   └── hf_inference.py            # Direct inference comparison
│       └── HFInferenceBaseline    # Simple baseline runner
│
├── configs/                        # Configuration files
│   ├── server_config.yaml         # vLLM tuning profiles
│   │   ├── low_vram (4GB)
│   │   ├── medium_vram (8GB)
│   │   └── high_vram (16GB+)
│   └── benchmark_config.yaml      # Benchmark profiles
│       ├── quick (validation)
│       ├── light (baseline)
│       ├── medium (stress test)
│       └── heavy (max load)
│
├── docs/                           # Documentation
│   ├── TUNING_GUIDE.md            # Parameter tuning guide
│   └── MODEL_SELECTION.md         # Model recommendations
│
├── examples/                       # Example scripts
│   ├── custom_benchmark.py        # Programmatic usage
│   └── api_usage.py               # Direct API calls
│
├── results/                        # Benchmark outputs
│   ├── .gitkeep
│   ├── run_*.json                 # Full results with raw data
│   └── run_*.csv                  # Summary statistics
│
├── scripts/                        # Helper scripts
│   ├── start_server.sh            # Start vLLM server
│   ├── stop_server.sh             # Stop server
│   └── run_benchmarks.sh          # Run benchmark suite
│
├── docker-compose.yml              # vLLM server deployment
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── Makefile                       # Convenient commands
├── LICENSE                        # MIT license
└── README.md                      # Main documentation
```

## Key Files

### Core Implementation

- **bench/benchmark.py** (500+ lines)
  - Streaming HTTP client
  - TTFT measurement (first token latency)
  - Concurrent request handling
  - Percentile calculations (p50, p95, p99)
  - VRAM monitoring via nvidia-smi
  - JSON + CSV output

- **baseline/hf_inference.py** (200+ lines)
  - HuggingFace Transformers baseline
  - Direct model inference
  - Simple comparison metrics

### Configuration

- **docker-compose.yml**
  - vLLM OpenAI server
  - GPU passthrough
  - Environment-based configuration
  - Health checks

- **configs/*.yaml**
  - Server tuning profiles
  - Benchmark load profiles
  - Predefined safe defaults

### Documentation

- **README.md**
  - Architecture diagram (Mermaid)
  - Quick start guide
  - Configuration reference
  - Troubleshooting section
  - CV-friendly summary

- **docs/TUNING_GUIDE.md**
  - Parameter explanations
  - Tuning workflow
  - Trade-off matrix
  - Common scenarios

- **docs/MODEL_SELECTION.md**
  - Model recommendations by VRAM
  - Size estimation formulas
  - Testing procedures
  - Fallback strategies

## Design Principles

### 1. Modularity
- Separate concerns: client, server, config, docs
- Reusable components (BenchmarkClient, VRAMMonitor)
- Clear interfaces

### 2. Configurability
- Environment variables for runtime config
- YAML profiles for different scenarios
- No hardcoded values

### 3. Robustness
- Retry logic with exponential backoff
- OOM detection and error messages
- Graceful degradation (VRAM monitoring optional)

### 4. Reproducibility
- Docker-first deployment
- Versioned dependencies
- Detailed documentation
- Example commands

### 5. Observability
- Detailed logging
- Metrics collection (TTFT, latency, throughput)
- VRAM tracking
- JSON + CSV output for analysis

## Workflow

### Development Workflow

1. **Setup**: `make install` → Install dependencies
2. **Configure**: Edit `.env` → Set model and parameters
3. **Start**: `make start` → Launch vLLM server
4. **Test**: `make test` → Quick validation
5. **Benchmark**: `make benchmark` → Full suite
6. **Analyze**: Review `results/*.csv` → Metrics
7. **Tune**: Adjust `.env` → Optimize
8. **Iterate**: Repeat 3-7

### CI/CD Ready

- Makefile targets for automation
- Scripts with exit codes
- JSON output for parsing
- Docker for consistency

## Metrics Collected

### Per-Request Metrics
- Request ID
- TTFT (Time To First Token)
- End-to-end latency
- Tokens generated
- Success/failure status
- Error messages

### Aggregated Metrics
- Total/successful/failed requests
- TTFT: mean, median, p95, p99, std
- Latency: mean, median, p50, p95, p99, std
- Total tokens generated
- Throughput (tokens/sec)
- VRAM peak (if available)

### Output Format

**JSON** (`results/run_*.json`):
```json
{
  "timestamp": "2026-02-12T19:15:00",
  "config": {...},
  "ttft_mean": 0.045,
  "latency_p95": 2.45,
  "throughput_tokens_per_sec": 124.7,
  "vram_peak_mb": 3200,
  "request_metrics": [...]
}
```

**CSV** (`results/run_*.csv`):
```csv
timestamp,total_requests,successful_requests,ttft_mean,latency_p95,throughput_tokens_per_sec,vram_peak_mb
2026-02-12T19:15:00,100,100,0.045,2.45,124.7,3200
```

## Extension Points

### Easy Extensions

1. **Add new benchmark profile**
   - Edit `configs/benchmark_config.yaml`
   - Add new profile section

2. **Add new server profile**
   - Edit `configs/server_config.yaml`
   - Add VRAM-specific settings

3. **Custom metrics**
   - Extend `RequestMetrics` dataclass
   - Update `aggregate_results()` function

### Advanced Extensions

1. **Grafana dashboard**
   - Export metrics to Prometheus
   - Create visualization

2. **Multi-model comparison**
   - Loop over models in script
   - Aggregate comparison table

3. **Automated tuning**
   - Grid search over parameters
   - Find optimal configuration

## Testing Strategy

### Unit Tests (Future)
- `test_benchmark_client.py`
- `test_metrics_calculation.py`
- `test_vram_monitor.py`

### Integration Tests
- `make test` (quick profile)
- Server health check
- End-to-end request

### Performance Tests
- `make benchmark` (full suite)
- Concurrency scaling
- Load testing

## Dependencies

### Runtime
- Docker + Docker Compose
- NVIDIA Container Runtime
- Python 3.10+

### Python Packages
- requests (HTTP client)
- numpy, pandas (metrics)
- pyyaml (config)
- tqdm (progress)
- transformers, torch (baseline)

### Optional
- matplotlib, seaborn (plotting)
- nvidia-smi (VRAM monitoring)

## Performance Targets

### 4GB VRAM (GTX 1650)
- Model: Qwen2.5-0.5B or 1.5B-AWQ
- Throughput: 100-150 tokens/s
- TTFT: <100ms
- Concurrency: 2-4

### 8GB VRAM (RTX 3060)
- Model: Qwen2.5-3B-AWQ or 7B-AWQ
- Throughput: 200-300 tokens/s
- TTFT: <80ms
- Concurrency: 4-8

### 16GB+ VRAM (RTX 4090)
- Model: Qwen2.5-7B or Llama-3.1-8B
- Throughput: 400-600 tokens/s
- TTFT: <50ms
- Concurrency: 8-16

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| OOM on startup | Reduce MAX_MODEL_LEN or use smaller model |
| OOM during inference | Reduce concurrency or MAX_MODEL_LEN |
| Low throughput | Increase GPU_MEMORY_UTILIZATION or concurrency |
| High latency | Reduce concurrency or MAX_MODEL_LEN |
| CUDA error | Check nvidia-smi, update drivers |
| Model not found | Check HF_TOKEN for gated models |

## CV/Resume Talking Points

1. **Systems Engineering**
   - "Designed and deployed production-ready LLM serving infrastructure"
   - "Optimized inference performance under resource constraints"

2. **Performance Engineering**
   - "Implemented comprehensive benchmarking suite measuring TTFT, latency percentiles, and throughput"
   - "Achieved X% improvement in tokens/sec through parameter tuning"

3. **DevOps/Infrastructure**
   - "Containerized deployment with Docker and GPU passthrough"
   - "Reproducible experiments with configuration management"

4. **Technical Skills**
   - vLLM, PagedAttention, continuous batching
   - Docker, NVIDIA Container Runtime
   - Python, asyncio, streaming HTTP
   - Performance profiling, metrics collection

## License

MIT License - See LICENSE file
