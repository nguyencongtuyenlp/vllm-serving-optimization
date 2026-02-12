# Tuning Guide for vLLM Serving

## Overview

This guide explains how to tune vLLM for optimal performance under different hardware constraints.

## Key Parameters

### 1. MAX_MODEL_LEN

**What it controls**: Maximum context length (prompt + generation)

**Impact**:
- Higher → More VRAM usage, lower batch size, lower throughput
- Lower → Less VRAM usage, higher batch size, higher throughput

**Recommendations**:
- 4GB VRAM: 1024-2048
- 8GB VRAM: 2048-4096
- 16GB+ VRAM: 4096-8192

**When to reduce**:
- Getting OOM errors
- Need higher concurrency
- Short prompts/responses typical

### 2. GPU_MEMORY_UTILIZATION

**What it controls**: Fraction of GPU memory allocated to vLLM

**Impact**:
- Higher → More KV cache, higher batch size, higher throughput
- Lower → Safety margin for memory spikes

**Recommendations**:
- 4GB VRAM: 0.80-0.85 (conservative)
- 8GB VRAM: 0.85-0.90
- 16GB+ VRAM: 0.90-0.95

**When to reduce**:
- Frequent OOM errors
- Sharing GPU with other processes
- Using large models

### 3. QUANTIZATION

**What it controls**: Model weight compression method

**Options**:
- `awq`: Activation-aware Weight Quantization (recommended)
- `gptq`: GPTQ quantization
- `squeezellm`: SqueezeLLM quantization
- (empty): No quantization

**Impact**:
- Quantized → 2-4x less VRAM, similar quality, slight speedup
- Unquantized → More VRAM, full precision

**Recommendations**:
- 4GB VRAM: **Required** (use AWQ or GPTQ)
- 8GB VRAM: Recommended for 7B+ models
- 16GB+ VRAM: Optional

### 4. DTYPE

**What it controls**: Floating point precision

**Options**:
- `auto`: Automatic selection
- `float16`: Half precision
- `bfloat16`: Brain float (better for training)
- `float32`: Full precision (not recommended)

**Recommendations**:
- Use `auto` (vLLM will choose best option)
- If manual: `float16` for most GPUs

## Tuning Workflow

### Step 1: Start Conservative

```bash
# .env
MAX_MODEL_LEN=1024
GPU_MEMORY_UTILIZATION=0.80
QUANTIZATION=awq
DTYPE=auto
```

### Step 2: Run Quick Benchmark

```bash
make test
```

### Step 3: Check Results

- **No OOM**: Increase `GPU_MEMORY_UTILIZATION` by 0.05
- **OOM**: Decrease `MAX_MODEL_LEN` by 512
- **Low throughput**: Check GPU utilization with `nvidia-smi dmon`

### Step 4: Scale Concurrency

```bash
python -m bench.benchmark --profile light
python -m bench.benchmark --profile medium
```

### Step 5: Find Sweet Spot

Iterate on parameters until you find the best balance:
- No OOM errors
- High throughput
- Acceptable latency

## Common Scenarios

### Scenario 1: Maximize Throughput (Batch Processing)

```bash
MAX_MODEL_LEN=1024          # Lower context
GPU_MEMORY_UTILIZATION=0.90 # Higher utilization
```

Run with high concurrency:
```bash
python -m bench.benchmark --concurrency 8 --num-requests 200
```

### Scenario 2: Minimize Latency (Interactive Chat)

```bash
MAX_MODEL_LEN=2048          # Moderate context
GPU_MEMORY_UTILIZATION=0.85 # Moderate utilization
```

Run with low concurrency:
```bash
python -m bench.benchmark --concurrency 1 --num-requests 50
```

### Scenario 3: Long Context (Document Q&A)

```bash
MAX_MODEL_LEN=4096          # Higher context
GPU_MEMORY_UTILIZATION=0.85 # Moderate utilization
```

Reduce concurrency to avoid OOM:
```bash
python -m bench.benchmark --concurrency 2 --num-requests 50
```

## Monitoring

### GPU Utilization

```bash
# Real-time monitoring
nvidia-smi dmon -s u

# Target: 80-95% GPU utilization
```

### VRAM Usage

```bash
# Check current usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Monitor during benchmark
watch -n 1 nvidia-smi
```

### Throughput vs Latency

Plot results from `results/*.csv`:
- X-axis: Concurrency
- Y-axis: Throughput (tokens/s) and Latency P95

## Trade-off Matrix

| Goal | MAX_MODEL_LEN | GPU_MEM_UTIL | Concurrency |
|------|---------------|--------------|-------------|
| **Max Throughput** | Low (1024) | High (0.90) | High (8+) |
| **Min Latency** | Medium (2048) | Medium (0.85) | Low (1-2) |
| **Long Context** | High (4096+) | Medium (0.85) | Low (1-2) |
| **Stability** | Low (1024) | Low (0.80) | Medium (4) |

## Troubleshooting

### OOM During Startup

**Cause**: Model too large for VRAM

**Solution**:
1. Use smaller model
2. Use quantized variant (AWQ/GPTQ)
3. Reduce `MAX_MODEL_LEN`

### OOM During Inference

**Cause**: Batch size too large

**Solution**:
1. Reduce `MAX_MODEL_LEN`
2. Reduce benchmark concurrency
3. Lower `GPU_MEMORY_UTILIZATION`

### Low GPU Utilization

**Cause**: Batch size too small, CPU bottleneck

**Solution**:
1. Increase concurrency
2. Increase `GPU_MEMORY_UTILIZATION`
3. Check CPU usage

### High Latency

**Cause**: Queue buildup, large batch size

**Solution**:
1. Reduce concurrency
2. Reduce `MAX_MODEL_LEN`
3. Use smaller model

## Advanced: Profiling

### vLLM Metrics

Enable vLLM metrics endpoint:
```bash
# Add to docker-compose.yml command:
--disable-log-requests
```

Query metrics:
```bash
curl http://localhost:8000/metrics
```

### Detailed Profiling

Use NVIDIA Nsight Systems:
```bash
nsys profile -o vllm_profile docker compose up
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [AWQ Quantization](https://arxiv.org/abs/2306.00978)
