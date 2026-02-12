# Lightning.ai Quick Reference

## 🚀 One-Command Setup

```bash
# In Lightning.ai Studio terminal
git clone https://github.com/YOUR_USERNAME/vllm.git
cd vllm
bash scripts/lightning_setup.sh
```

## 📊 Run Benchmarks

```bash
# Quick test (10 requests, ~1 min)
python -m bench.benchmark --profile quick

# Light load (50 requests, ~3 min)
python -m bench.benchmark --profile light

# Medium load (100 requests, ~5 min)
python -m bench.benchmark --profile medium
```

## 📁 Get Results

```bash
# View results
ls -lh results/
cat results/run_*.csv

# Download via Lightning.ai file browser:
# vllm/results/ → right-click → Download
```

## 🔧 Troubleshooting

```bash
# Check server status
curl http://localhost:8000/v1/models

# View logs
docker compose logs -f

# Restart server
docker compose down && docker compose up -d

# Check GPU
nvidia-smi
```

## 💾 Save Work

```bash
# Commit to GitHub
git add results/
git commit -m "Add benchmark results from Lightning.ai"
git push
```

## ⏱️ Time Estimates

- Setup: ~5-8 minutes
- Quick benchmark: ~1 minute
- Light benchmark: ~3 minutes
- Medium benchmark: ~5 minutes
- **Total**: ~15 minutes

## 💰 Cost

- **Free tier**: Enough for this project
- **Paid**: ~$0.50-1.00/hour (T4 GPU)

## 📝 Checklist

- [ ] Create Lightning.ai account
- [ ] Launch GPU Studio (T4 or A10G)
- [ ] Clone repo
- [ ] Run setup script
- [ ] Run quick benchmark
- [ ] Run light benchmark
- [ ] Run medium benchmark
- [ ] Download results
- [ ] Update README with metrics
- [ ] Push to GitHub
- [ ] Stop Studio (save credits)

---

**Full guide**: [docs/LIGHTNING_AI_GUIDE.md](../docs/LIGHTNING_AI_GUIDE.md)
