import functools

import pytest
from memory_profiler import memory_usage


def memory_benchmark(func):
    """Decorator to measure memory usage of a function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mem_usage = memory_usage((func, args, kwargs), interval=0.1)
        return {
            "peak_memory": max(mem_usage),
            "avg_memory": sum(mem_usage) / len(mem_usage),
            "baseline_memory": mem_usage[0],
            "result": func(*args, **kwargs),
        }

    return wrapper


# Example function to benchmark
@memory_benchmark
def process_data(data):
    # Your processing logic here
    processed = [x * 2 for x in data]
    return processed


# Pytest benchmark fixtures and tests
@pytest.fixture
def benchmark_data():
    return list(range(1000000))


def test_memory_usage(benchmark, benchmark_data):
    def run_benchmark():
        result = process_data(benchmark_data)
        return result

    stats = benchmark(run_benchmark)

    # Access memory metrics
    mem_stats = stats.stats
    print(f"\nMemory Usage Stats:")
    print(f"Peak Memory: {mem_stats['peak_memory']:.2f} MiB")
    print(f"Average Memory: {mem_stats['avg_memory']:.2f} MiB")
    print(f"Baseline Memory: {mem_stats['baseline_memory']:.2f} MiB")


# Run with: pytest --benchmark-only test_memory.py
