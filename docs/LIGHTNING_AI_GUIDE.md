# Running on Lightning.ai

This guide shows how to run the vLLM Serving Optimization project on [Lightning.ai](https://lightning.ai) with free GPU access.

## Why Lightning.ai?

- ✅ **Free GPU access** (T4, A10G, or better)
- ✅ **Pre-configured environment** with Docker support
- ✅ **Persistent storage** for results
- ✅ **Public URLs** for demo
- ✅ **Easy setup** (< 10 minutes)

---

## Quick Start (10 minutes)

### Step 1: Create Lightning.ai Account

1. Go to https://lightning.ai
2. Sign up with GitHub (free tier)
3. Verify email

### Step 2: Create Studio with GPU

1. Click **"New Studio"**
2. Select **GPU** (choose T4 or A10G if available)
3. Wait for Studio to start (~1 minute)

### Step 3: Clone Repository

In the Lightning.ai terminal:

```bash
# Clone your repo (replace with your GitHub URL)
git clone https://github.com/YOUR_USERNAME/vllm.git
cd vllm
```

### Step 4: Run Setup Script

```bash
# Make script executable
chmod +x scripts/lightning_setup.sh

# Run automated setup
bash scripts/lightning_setup.sh
```

**This script will**:
- ✅ Check GPU availability
- ✅ Install Docker (if needed)
- ✅ Install NVIDIA Container Toolkit
- ✅ Setup environment (.env)
- ✅ Install Python dependencies
- ✅ Start vLLM server
- ✅ Wait for server ready
- ✅ Test server endpoint

**Expected time**: 5-8 minutes (first time)

### Step 5: Run Benchmarks

```bash
# Quick test (10 requests)
python -m bench.benchmark --profile quick

# Light load (50 requests)
python -m bench.benchmark --profile light

# Medium load (100 requests)
python -m bench.benchmark --profile medium
```

### Step 6: Collect Results

```bash
# View results
ls -lh results/

# Check latest CSV
cat results/run_*.csv | tail -n 5

# Copy results to your local machine
# (Download via Lightning.ai file browser)
```

---

## Expected Results (T4 GPU)

Based on similar benchmarks, you should see:

| Metric | Quick Profile | Light Profile | Medium Profile |
|--------|--------------|---------------|----------------|
| **Model** | Qwen2.5-0.5B | Qwen2.5-0.5B | Qwen2.5-0.5B |
| **Requests** | 10 | 50 | 100 |
| **Concurrency** | 1 | 2 | 4 |
| **TTFT Mean** | ~40-60ms | ~45-70ms | ~50-80ms |
| **TTFT P95** | ~70-100ms | ~80-120ms | ~90-140ms |
| **Latency P95** | ~1.5-2.5s | ~2.0-3.0s | ~2.5-4.0s |
| **Throughput** | ~120-160 tok/s | ~140-180 tok/s | ~150-200 tok/s |
| **VRAM Peak** | ~3.5-4.0 GB | ~3.8-4.2 GB | ~4.0-4.5 GB |

*Note: Actual results depend on GPU model (T4 vs A10G) and current load*

---

## Troubleshooting

### Server won't start

**Check logs**:
```bash
docker compose logs -f
```

**Common issues**:
- Out of memory → Reduce `MAX_MODEL_LEN` in `.env`
- CUDA error → Restart Studio
- Port conflict → Change port in `docker-compose.yml`

### Benchmark fails

**Check server is running**:
```bash
curl http://localhost:8000/v1/models
```

**If connection refused**:
```bash
# Restart server
docker compose down
docker compose up -d

# Wait 1-2 minutes
sleep 120

# Try again
python -m bench.benchmark --profile quick
```

### Out of disk space

**Clean up**:
```bash
# Remove old results
rm results/run_*.json results/run_*.csv

# Clean Docker
docker system prune -a
```

---

## Saving Results

### Method 1: Download via UI

1. In Lightning.ai file browser
2. Navigate to `vllm/results/`
3. Right-click files → Download

### Method 2: Copy to clipboard

```bash
# Copy CSV content
cat results/run_*.csv

# Copy JSON content
cat results/run_*.json
```

### Method 3: Push to GitHub

```bash
# Commit results
git add results/
git commit -m "Add benchmark results from Lightning.ai T4 GPU"
git push
```

---

## Updating README with Results

After collecting results, update your README:

1. **Copy metrics** from `results/run_*.csv`
2. **Fill in the table** in README.md:

```markdown
### vLLM Performance (Lightning.ai T4 GPU, Qwen2.5-0.5B)

| Metric | Value |
|--------|-------|
| **TTFT Mean** | 45.2ms |
| **TTFT P95** | 78.5ms |
| **Latency P50** | 1.23s |
| **Latency P95** | 2.45s |
| **Throughput** | 156.3 tokens/s |
| **VRAM Peak** | 3.8GB |
| **Concurrency** | 4 |
```

3. **Add comparison** with baseline (if you ran it):

```markdown
### vLLM vs HF Baseline

| Backend | Throughput (tokens/s) | Latency P95 (s) | VRAM (GB) |
|---------|----------------------|-----------------|-----------|
| **vLLM** | 156.3 | 2.45 | 3.8 |
| **HF Transformers** | 52.1 | 4.82 | 2.9 |
| **Speedup** | **3.0x** | **2.0x faster** | +31% VRAM |
```

---

## Running Baseline Comparison

To compare vLLM with HuggingFace baseline:

```bash
# Run HF baseline
python -m baseline.hf_inference \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --num-requests 10 \
  --prompt-tokens 128 \
  --max-new-tokens 64 \
  --device cuda

# Results saved to results/baseline_hf_*.json
```

---

## Stopping Server

When done:

```bash
# Stop server
docker compose down

# Stop Lightning.ai Studio
# (Click "Stop" in Lightning.ai UI to save credits)
```

---

## Cost

**Lightning.ai Free Tier**:
- ✅ Limited free GPU hours per month
- ✅ Enough for this project (~30 minutes total)
- ✅ No credit card required

**Paid Tier** (if needed):
- ~$0.50-1.00/hour for T4 GPU
- ~$1.50-2.50/hour for A10G GPU

---

## Tips

1. **Run all benchmarks in one session** to save GPU time
2. **Download results immediately** before stopping Studio
3. **Take screenshots** of terminal output for walkthrough
4. **Commit to GitHub** from Lightning.ai to save work

---

## Next Steps

After getting results:

1. ✅ Update README with real metrics
2. ✅ Add screenshots to walkthrough
3. ✅ Push to GitHub
4. ✅ Add to CV/portfolio

**Your project is now complete with real benchmark data!** 🎉
