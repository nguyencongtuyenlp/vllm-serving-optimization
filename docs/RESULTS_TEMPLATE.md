# Benchmark Results Template

Fill this template after running benchmarks on Lightning.ai or other GPU platform.

## Environment

- **Platform**: Lightning.ai / Google Colab / Local
- **GPU**: T4 / A10G / RTX 3060 / etc.
- **VRAM**: 16GB / 8GB / 4GB
- **Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Quantization**: None / AWQ / GPTQ
- **Date**: YYYY-MM-DD

---

## vLLM Results

### Configuration

```yaml
MAX_MODEL_LEN: 1024
GPU_MEMORY_UTILIZATION: 0.75
DTYPE: auto
QUANTIZATION: 
```

### Quick Profile (10 requests, concurrency 1)

| Metric | Value |
|--------|-------|
| Total Requests | 10 |
| Successful | 10 |
| Failed | 0 |
| **TTFT Mean** | ___ ms |
| **TTFT Median** | ___ ms |
| **TTFT P95** | ___ ms |
| **TTFT P99** | ___ ms |
| **Latency Mean** | ___ s |
| **Latency Median** | ___ s |
| **Latency P50** | ___ s |
| **Latency P95** | ___ s |
| **Latency P99** | ___ s |
| **Total Tokens** | ___ |
| **Throughput** | ___ tokens/s |
| **VRAM Peak** | ___ MB |

### Light Profile (50 requests, concurrency 2)

| Metric | Value |
|--------|-------|
| Total Requests | 50 |
| Successful | ___ |
| Failed | ___ |
| **TTFT Mean** | ___ ms |
| **TTFT P95** | ___ ms |
| **Latency P95** | ___ s |
| **Throughput** | ___ tokens/s |
| **VRAM Peak** | ___ MB |

### Medium Profile (100 requests, concurrency 4)

| Metric | Value |
|--------|-------|
| Total Requests | 100 |
| Successful | ___ |
| Failed | ___ |
| **TTFT Mean** | ___ ms |
| **TTFT P95** | ___ ms |
| **Latency P95** | ___ s |
| **Throughput** | ___ tokens/s |
| **VRAM Peak** | ___ MB |

---

## HuggingFace Baseline Results

### Configuration

```yaml
Model: Qwen/Qwen2.5-0.5B-Instruct
Device: cuda
Dtype: float16
Num Requests: 10
```

| Metric | Value |
|--------|-------|
| **Latency Mean** | ___ s |
| **Latency Median** | ___ s |
| **Total Tokens** | ___ |
| **Throughput** | ___ tokens/s |

---

## Comparison

| Backend | Throughput (tok/s) | Latency P95 (s) | VRAM (GB) | Speedup |
|---------|-------------------|-----------------|-----------|---------|
| **vLLM** | ___ | ___ | ___ | - |
| **HF Transformers** | ___ | ___ | ___ | - |
| **Improvement** | ___x | ___x | +___% | ___x |

---

## Observations

### Performance

- [ ] vLLM achieved ___x throughput improvement over baseline
- [ ] TTFT was under 100ms for ___% of requests
- [ ] P95 latency was acceptable at ___ seconds
- [ ] No OOM errors occurred

### Resource Usage

- [ ] VRAM peak was ___ GB (___% of total)
- [ ] GPU utilization was high (check with `nvidia-smi dmon`)
- [ ] No throttling observed

### Issues Encountered

- [ ] None
- [ ] OOM errors → Solution: ___
- [ ] High latency → Solution: ___
- [ ] Server crashes → Solution: ___

---

## Screenshots

Add screenshots here:

1. **Server startup logs**:
   - ![Server startup](path/to/screenshot1.png)

2. **Benchmark running**:
   - ![Benchmark progress](path/to/screenshot2.png)

3. **Results summary**:
   - ![Results output](path/to/screenshot3.png)

4. **GPU utilization**:
   - ![nvidia-smi output](path/to/screenshot4.png)

---

## Raw Data Files

- `results/run_YYYYMMDD_HHMMSS.json` - Full results with all request metrics
- `results/run_YYYYMMDD_HHMMSS.csv` - Summary statistics
- `results/baseline_hf_*.json` - Baseline comparison

---

## Notes

Add any additional observations or notes here:

- 
- 
- 

---

## Next Steps

- [ ] Update README.md with results
- [ ] Add comparison table
- [ ] Include screenshots in walkthrough
- [ ] Push to GitHub
- [ ] Update CV with metrics
