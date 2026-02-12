# Model Selection Guide

## Recommended Models for Different VRAM Constraints

### 4GB VRAM (GTX 1650, RTX 3050)

**Priority: Quantized small models**

#### Option 1: Qwen2.5 0.5B (Safest)
```bash
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
QUANTIZATION=
```
- ✅ Fits easily in 4GB
- ✅ Good quality for size
- ✅ Fast inference
- ❌ Limited capabilities

#### Option 2: Qwen2.5 1.5B AWQ
```bash
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct-AWQ
QUANTIZATION=awq
```
- ✅ Better quality than 0.5B
- ✅ Still fits in 4GB
- ⚠️ Requires AWQ support
- ⚠️ Check model availability on HF

#### Option 3: TinyLlama 1.1B
```bash
MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0
QUANTIZATION=
```
- ✅ Widely available
- ✅ Good community support
- ❌ Lower quality than Qwen

### 8GB VRAM (RTX 3060, RTX 4060)

**Priority: 3B-7B quantized models**

#### Option 1: Qwen2.5 3B AWQ
```bash
MODEL_ID=Qwen/Qwen2.5-3B-Instruct-AWQ
QUANTIZATION=awq
```
- ✅ Excellent quality/size ratio
- ✅ Fits comfortably
- ✅ Good throughput

#### Option 2: Qwen2.5 7B AWQ
```bash
MODEL_ID=Qwen/Qwen2.5-7B-Instruct-AWQ
QUANTIZATION=awq
```
- ✅ High quality
- ⚠️ Tighter VRAM fit
- ⚠️ Reduce MAX_MODEL_LEN to 2048

#### Option 3: Llama 3.2 3B
```bash
MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
QUANTIZATION=
```
- ✅ Meta's official model
- ✅ Good quality
- ⚠️ Requires HF token

### 16GB+ VRAM (RTX 4080, RTX 4090, A100)

**Priority: 7B-13B unquantized or 30B+ quantized**

#### Option 1: Qwen2.5 7B
```bash
MODEL_ID=Qwen/Qwen2.5-7B-Instruct
QUANTIZATION=
```
- ✅ Full precision
- ✅ Excellent quality
- ✅ Long context support

#### Option 2: Llama 3.1 8B
```bash
MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
QUANTIZATION=
```
- ✅ State-of-the-art quality
- ✅ 128k context support
- ⚠️ Requires HF token

#### Option 3: Mixtral 8x7B AWQ
```bash
MODEL_ID=TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ
QUANTIZATION=awq
```
- ✅ MoE architecture
- ✅ Very high quality
- ⚠️ Slower inference

## How to Find Models

### HuggingFace Hub Search

1. Go to https://huggingface.co/models
2. Filter by:
   - Task: Text Generation
   - Library: transformers
   - Tags: conversational, instruct

### Quantized Models

Search for:
- `AWQ` in model name (recommended)
- `GPTQ` in model name
- Check TheBloke's profile: https://huggingface.co/TheBloke

### Verify Model Compatibility

```bash
# Check model card on HuggingFace
# Look for:
# - Architecture (should be supported by vLLM)
# - Quantization method
# - Model size (parameters)
```

## Model Size Estimation

**VRAM needed (rough estimate)**:

| Model Size | Unquantized (FP16) | AWQ/GPTQ (4-bit) |
|------------|-------------------|------------------|
| 0.5B | ~1GB | ~0.5GB |
| 1.5B | ~3GB | ~1.5GB |
| 3B | ~6GB | ~3GB |
| 7B | ~14GB | ~7GB |
| 13B | ~26GB | ~13GB |

Add overhead for KV cache: +1-4GB depending on `MAX_MODEL_LEN`

## Testing a New Model

### Step 1: Update .env

```bash
MODEL_ID=your/model-id
QUANTIZATION=awq  # or empty
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.80  # Start conservative
```

### Step 2: Start Server

```bash
make start
docker compose logs -f
```

Watch for:
- ✅ "Model loaded successfully"
- ❌ "CUDA out of memory"
- ❌ "Model not found"

### Step 3: Quick Test

```bash
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### Step 4: Benchmark

```bash
make test
```

## Gated Models (Require HF Token)

Some models require accepting terms on HuggingFace:

1. Go to model page (e.g., meta-llama/Llama-3.1-8B-Instruct)
2. Click "Agree and access repository"
3. Get HF token: https://huggingface.co/settings/tokens
4. Add to `.env`:
   ```bash
   HF_TOKEN=hf_xxxxxxxxxxxxx
   ```

## Quality vs Speed Trade-offs

| Model Type | Quality | Speed | VRAM |
|------------|---------|-------|------|
| 0.5B unquantized | ⭐⭐ | ⚡⚡⚡⚡ | 💾 |
| 1.5B AWQ | ⭐⭐⭐ | ⚡⚡⚡ | 💾💾 |
| 3B AWQ | ⭐⭐⭐⭐ | ⚡⚡⚡ | 💾💾💾 |
| 7B AWQ | ⭐⭐⭐⭐⭐ | ⚡⚡ | 💾💾💾💾 |
| 7B unquantized | ⭐⭐⭐⭐⭐ | ⚡⚡ | 💾💾💾💾💾💾 |

## Fallback Strategy

If vLLM fails with your GPU:

### Option 1: CPU Inference (Slow)

Use baseline HF script:
```bash
python -m baseline.hf_inference \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu
```

### Option 2: llama.cpp (Recommended for CPU)

See `docs/LLAMA_CPP_FALLBACK.md` (if implemented)

## References

- [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [AWQ Quantization](https://github.com/mit-han-lab/llm-awq)
