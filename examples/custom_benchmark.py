"""
Example script showing how to use the benchmark client programmatically.
"""

from pathlib import Path
from bench.benchmark import BenchmarkClient, aggregate_results, save_results
import time

def main():
    # Create client
    client = BenchmarkClient(
        server_url="http://localhost:8000",
        api_endpoint="/v1/completions",
        timeout=120
    )
    
    # Define benchmark parameters
    config = {
        'num_requests': 20,
        'concurrency': 2,
        'prompt_tokens': 256,
        'max_new_tokens': 128,
        'warmup_requests': 3,
        'temperature': 0.7,
        'top_p': 0.9,
        'server_url': 'http://localhost:8000'
    }
    
    print("Running custom benchmark...")
    print(f"Config: {config}")
    
    # Run benchmark
    start_time = time.time()
    metrics, vram_peak = client.run_benchmark(
        num_requests=config['num_requests'],
        concurrency=config['concurrency'],
        prompt_tokens=config['prompt_tokens'],
        max_new_tokens=config['max_new_tokens'],
        warmup_requests=config['warmup_requests'],
        temperature=config['temperature'],
        top_p=config['top_p']
    )
    total_time = time.time() - start_time
    
    # Aggregate results
    results = aggregate_results(metrics, config, total_time, vram_peak)
    
    # Save results
    save_results(results, Path('results'))
    
    print("\nBenchmark completed!")

if __name__ == '__main__':
    main()
