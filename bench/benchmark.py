"""
Benchmark client for measuring LLM serving performance.

Metrics measured:
- TTFT (Time To First Token): streaming latency
- End-to-end latency (p50, p95, p99)
- Throughput (tokens/sec)
- VRAM usage (if nvidia-smi available)
"""

import argparse
import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from statistics import mean, median, stdev

import requests
import yaml
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: int
    ttft: float  # Time to first token (seconds)
    latency: float  # End-to-end latency (seconds)
    tokens_generated: int
    prompt_tokens: int
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    timestamp: str
    config: Dict
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # TTFT metrics
    ttft_mean: float
    ttft_median: float
    ttft_p95: float
    ttft_p99: float
    ttft_std: float
    
    # Latency metrics
    latency_mean: float
    latency_median: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_std: float
    
    # Throughput metrics
    total_tokens: int
    total_time: float
    throughput_tokens_per_sec: float
    
    # Resource metrics
    vram_peak_mb: Optional[float] = None
    
    # Raw data
    request_metrics: List[Dict] = None


class VRAMMonitor:
    """Monitor GPU VRAM usage."""
    
    @staticmethod
    def get_vram_usage() -> Optional[float]:
        """Get current VRAM usage in MB."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split('\n')[0])
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            logger.debug(f"Could not read VRAM: {e}")
        return None


class BenchmarkClient:
    """Client for benchmarking LLM serving endpoints."""
    
    def __init__(
        self,
        server_url: str,
        api_endpoint: str = "/v1/completions",
        timeout: int = 120,
        retry_attempts: int = 3,
        retry_delay: float = 2.0
    ):
        self.server_url = server_url.rstrip('/')
        self.api_endpoint = api_endpoint
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.vram_monitor = VRAMMonitor()
    
    def generate_prompt(self, num_tokens: int) -> str:
        """Generate a prompt with approximately num_tokens tokens."""
        # Rough estimate: 1 token ≈ 4 characters
        words_needed = (num_tokens * 4) // 5  # average word length ~5 chars
        
        base_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test prompt for benchmarking LLM inference performance. "
        )
        
        prompt = base_text * (words_needed // len(base_text.split()) + 1)
        return prompt[:num_tokens * 4]  # Approximate token count
    
    def send_request_streaming(
        self,
        request_id: int,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> RequestMetrics:
        """Send a streaming request and measure TTFT and latency."""
        url = f"{self.server_url}{self.api_endpoint}"
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True
        }
        
        start_time = time.time()
        ttft = None
        tokens_generated = 0
        
        for attempt in range(self.retry_attempts):
            try:
                with requests.post(
                    url,
                    json=payload,
                    stream=True,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                
                                # Measure TTFT on first token
                                if ttft is None:
                                    ttft = time.time() - start_time
                                
                                # Count tokens
                                if 'choices' in data and len(data['choices']) > 0:
                                    tokens_generated += 1
                            
                            except json.JSONDecodeError:
                                continue
                    
                    end_time = time.time()
                    latency = end_time - start_time
                    
                    # If no tokens generated, TTFT = latency
                    if ttft is None:
                        ttft = latency
                    
                    return RequestMetrics(
                        request_id=request_id,
                        ttft=ttft,
                        latency=latency,
                        tokens_generated=tokens_generated,
                        prompt_tokens=len(prompt.split()),  # Rough estimate
                        success=True
                    )
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request {request_id} attempt {attempt + 1} failed: {e}")
                
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    return RequestMetrics(
                        request_id=request_id,
                        ttft=0.0,
                        latency=0.0,
                        tokens_generated=0,
                        prompt_tokens=len(prompt.split()),
                        success=False,
                        error=str(e)
                    )
    
    def run_benchmark(
        self,
        num_requests: int,
        concurrency: int,
        prompt_tokens: int,
        max_new_tokens: int,
        warmup_requests: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Tuple[List[RequestMetrics], Optional[float]]:
        """Run benchmark with specified parameters."""
        
        # Warmup
        if warmup_requests > 0:
            logger.info(f"Running {warmup_requests} warmup requests...")
            prompt = self.generate_prompt(prompt_tokens)
            for i in range(warmup_requests):
                self.send_request_streaming(
                    request_id=-i-1,
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
        
        # Monitor VRAM before benchmark
        vram_start = self.vram_monitor.get_vram_usage()
        vram_peak = vram_start
        
        # Run actual benchmark
        logger.info(f"Running {num_requests} requests with concurrency {concurrency}...")
        
        prompt = self.generate_prompt(prompt_tokens)
        metrics: List[RequestMetrics] = []
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    self.send_request_streaming,
                    i,
                    prompt,
                    max_new_tokens,
                    temperature,
                    top_p
                ): i for i in range(num_requests)
            }
            
            with tqdm(total=num_requests, desc="Requests") as pbar:
                for future in as_completed(futures):
                    metric = future.result()
                    metrics.append(metric)
                    pbar.update(1)
                    
                    # Monitor VRAM during execution
                    current_vram = self.vram_monitor.get_vram_usage()
                    if current_vram and vram_peak:
                        vram_peak = max(vram_peak, current_vram)
        
        return metrics, vram_peak


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile from a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]


def aggregate_results(
    metrics: List[RequestMetrics],
    config: Dict,
    total_time: float,
    vram_peak: Optional[float]
) -> BenchmarkResults:
    """Aggregate individual request metrics into summary statistics."""
    
    successful = [m for m in metrics if m.success]
    failed = [m for m in metrics if not m.success]
    
    if not successful:
        raise ValueError("All requests failed! Check server status and configuration.")
    
    ttfts = [m.ttft for m in successful]
    latencies = [m.latency for m in successful]
    total_tokens = sum(m.tokens_generated for m in successful)
    
    return BenchmarkResults(
        timestamp=datetime.now().isoformat(),
        config=config,
        total_requests=len(metrics),
        successful_requests=len(successful),
        failed_requests=len(failed),
        
        ttft_mean=mean(ttfts),
        ttft_median=median(ttfts),
        ttft_p95=calculate_percentile(ttfts, 95),
        ttft_p99=calculate_percentile(ttfts, 99),
        ttft_std=stdev(ttfts) if len(ttfts) > 1 else 0.0,
        
        latency_mean=mean(latencies),
        latency_median=median(latencies),
        latency_p50=calculate_percentile(latencies, 50),
        latency_p95=calculate_percentile(latencies, 95),
        latency_p99=calculate_percentile(latencies, 99),
        latency_std=stdev(latencies) if len(latencies) > 1 else 0.0,
        
        total_tokens=total_tokens,
        total_time=total_time,
        throughput_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0.0,
        
        vram_peak_mb=vram_peak,
        
        request_metrics=[asdict(m) for m in metrics]
    )


def save_results(results: BenchmarkResults, output_dir: Path):
    """Save results to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results as JSON
    json_path = output_dir / f"run_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    logger.info(f"Saved JSON results to {json_path}")
    
    # Save summary as CSV
    csv_path = output_dir / f"run_{timestamp}.csv"
    summary_data = {
        k: v for k, v in asdict(results).items()
        if k != 'request_metrics' and k != 'config'
    }
    df = pd.DataFrame([summary_data])
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV summary to {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"Total Requests: {results.total_requests}")
    print(f"Successful: {results.successful_requests} | Failed: {results.failed_requests}")
    print(f"\nTTFT (Time To First Token):")
    print(f"  Mean: {results.ttft_mean*1000:.2f}ms | Median: {results.ttft_median*1000:.2f}ms")
    print(f"  P95: {results.ttft_p95*1000:.2f}ms | P99: {results.ttft_p99*1000:.2f}ms")
    print(f"\nLatency (End-to-End):")
    print(f"  Mean: {results.latency_mean:.3f}s | Median: {results.latency_median:.3f}s")
    print(f"  P50: {results.latency_p50:.3f}s | P95: {results.latency_p95:.3f}s | P99: {results.latency_p99:.3f}s")
    print(f"\nThroughput:")
    print(f"  Total Tokens: {results.total_tokens}")
    print(f"  Tokens/sec: {results.throughput_tokens_per_sec:.2f}")
    if results.vram_peak_mb:
        print(f"\nVRAM Peak: {results.vram_peak_mb:.0f} MB")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM serving performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python -m bench.benchmark --profile quick
  
  # Custom benchmark
  python -m bench.benchmark --num-requests 100 --concurrency 4 --prompt-tokens 512 --max-new-tokens 256
  
  # Heavy load test
  python -m bench.benchmark --profile heavy --server-url http://localhost:8000
        """
    )
    
    parser.add_argument('--server-url', type=str, default='http://localhost:8000',
                        help='vLLM server URL')
    parser.add_argument('--profile', type=str, choices=['quick', 'light', 'medium', 'heavy'],
                        help='Use predefined benchmark profile')
    parser.add_argument('--num-requests', type=int, help='Number of requests to send')
    parser.add_argument('--concurrency', type=int, help='Number of concurrent requests')
    parser.add_argument('--prompt-tokens', type=int, help='Approximate prompt length in tokens')
    parser.add_argument('--max-new-tokens', type=int, help='Maximum tokens to generate')
    parser.add_argument('--warmup-requests', type=int, default=0, help='Number of warmup requests')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--config', type=str, default='configs/benchmark_config.yaml',
                        help='Path to benchmark config file')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        config = {}
    
    # Determine benchmark parameters
    if args.profile:
        profile_config = config.get(args.profile, {})
        num_requests = profile_config.get('num_requests', 10)
        concurrency = profile_config.get('concurrency', 1)
        prompt_tokens = profile_config.get('prompt_tokens', 128)
        max_new_tokens = profile_config.get('max_new_tokens', 64)
        warmup_requests = profile_config.get('warmup_requests', 0)
    else:
        num_requests = args.num_requests or 10
        concurrency = args.concurrency or 1
        prompt_tokens = args.prompt_tokens or 128
        max_new_tokens = args.max_new_tokens or 64
        warmup_requests = args.warmup_requests
    
    # Get default settings
    defaults = config.get('default', {})
    server_url = args.server_url or defaults.get('server_url', 'http://localhost:8000')
    
    # Create client
    client = BenchmarkClient(
        server_url=server_url,
        api_endpoint=defaults.get('api_endpoint', '/v1/completions'),
        timeout=defaults.get('timeout', 120),
        retry_attempts=defaults.get('retry_attempts', 3),
        retry_delay=defaults.get('retry_delay', 2.0)
    )
    
    # Run benchmark
    benchmark_config = {
        'num_requests': num_requests,
        'concurrency': concurrency,
        'prompt_tokens': prompt_tokens,
        'max_new_tokens': max_new_tokens,
        'warmup_requests': warmup_requests,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'server_url': server_url
    }
    
    logger.info(f"Starting benchmark with config: {benchmark_config}")
    
    start_time = time.time()
    try:
        metrics, vram_peak = client.run_benchmark(
            num_requests=num_requests,
            concurrency=concurrency,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            warmup_requests=warmup_requests,
            temperature=args.temperature,
            top_p=args.top_p
        )
        total_time = time.time() - start_time
        
        # Aggregate and save results
        results = aggregate_results(metrics, benchmark_config, total_time, vram_peak)
        save_results(results, Path(args.output_dir))
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == '__main__':
    main()
