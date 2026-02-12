"""
Baseline HuggingFace Transformers inference script.

This provides a simple baseline for comparison with vLLM.
No server required - just direct inference measurement.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict
from statistics import mean, median

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFInferenceBaseline:
    """Simple HuggingFace inference baseline."""
    
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto"
    ):
        self.model_id = model_id
        self.device = device
        
        logger.info(f"Loading model {model_id}...")
        
        # Determine dtype
        if dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        device_map = "auto" if device == "auto" else device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        
        logger.info(f"Model loaded on {self.model.device}")
    
    def generate_prompt(self, num_tokens: int) -> str:
        """Generate a prompt with approximately num_tokens tokens."""
        words_needed = (num_tokens * 4) // 5
        base_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test prompt for benchmarking LLM inference performance. "
        )
        prompt = base_text * (words_needed // len(base_text.split()) + 1)
        return prompt[:num_tokens * 4]
    
    def run_inference(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict:
        """Run single inference and measure latency."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        latency = time.time() - start_time
        
        generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        
        return {
            'latency': latency,
            'tokens_generated': generated_tokens,
            'tokens_per_sec': generated_tokens / latency if latency > 0 else 0
        }
    
    def run_benchmark(
        self,
        num_requests: int,
        prompt_tokens: int,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[Dict]:
        """Run benchmark with specified parameters."""
        
        prompt = self.generate_prompt(prompt_tokens)
        results = []
        
        logger.info(f"Running {num_requests} sequential requests...")
        
        for i in tqdm(range(num_requests), desc="Requests"):
            result = self.run_inference(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            result['request_id'] = i
            results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Baseline HuggingFace Transformers inference benchmark"
    )
    
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='HuggingFace model ID')
    parser.add_argument('--num-requests', type=int, default=10,
                        help='Number of requests')
    parser.add_argument('--prompt-tokens', type=int, default=128,
                        help='Approximate prompt length')
    parser.add_argument('--max-new-tokens', type=int, default=64,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p sampling')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')
    parser.add_argument('--dtype', type=str, default='auto',
                        help='Data type (auto, float16, bfloat16, float32)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create baseline
    baseline = HFInferenceBaseline(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype
    )
    
    # Run benchmark
    start_time = time.time()
    results = baseline.run_benchmark(
        num_requests=args.num_requests,
        prompt_tokens=args.prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    total_time = time.time() - start_time
    
    # Calculate statistics
    latencies = [r['latency'] for r in results]
    tokens_generated = [r['tokens_generated'] for r in results]
    
    summary = {
        'model_id': args.model_id,
        'num_requests': args.num_requests,
        'total_time': total_time,
        'latency_mean': mean(latencies),
        'latency_median': median(latencies),
        'total_tokens': sum(tokens_generated),
        'throughput_tokens_per_sec': sum(tokens_generated) / total_time,
        'results': results
    }
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"baseline_hf_{args.model_id.replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE HF INFERENCE RESULTS")
    print("="*80)
    print(f"Model: {args.model_id}")
    print(f"Total Requests: {args.num_requests}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"\nLatency:")
    print(f"  Mean: {summary['latency_mean']:.3f}s")
    print(f"  Median: {summary['latency_median']:.3f}s")
    print(f"\nThroughput:")
    print(f"  Total Tokens: {summary['total_tokens']}")
    print(f"  Tokens/sec: {summary['throughput_tokens_per_sec']:.2f}")
    print("="*80 + "\n")
    
    logger.info(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
