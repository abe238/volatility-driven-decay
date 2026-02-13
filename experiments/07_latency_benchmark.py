#!/usr/bin/env python3
"""
Experiment 07: Latency Benchmarks

Benchmark computational overhead of VDD components.
Target: total overhead < 50ms per query.
"""

import time
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import EmbeddingDistance, ADWIN, PageHinkley
from vdd.memory import VDDMemoryBank, StaticDecayMemory


def benchmark_drift_detection(embedding_dim: int = 384, iterations: int = 1000):
    """Benchmark drift detection latency."""
    results = {}

    detectors = {
        "ADWIN": ADWIN(),
        "PageHinkley": PageHinkley(),
        "EmbeddingDistance": EmbeddingDistance(),
    }

    for name, detector in detectors.items():
        embeddings = [np.random.randn(embedding_dim) for _ in range(iterations)]

        start = time.perf_counter()
        for emb in embeddings:
            detector.update(emb)
        elapsed = time.perf_counter() - start

        results[name] = {
            "total_ms": elapsed * 1000,
            "per_call_ms": elapsed * 1000 / iterations,
            "calls": iterations,
        }
        detector.reset()

    return results


def benchmark_memory_operations(embedding_dim: int = 384, memory_sizes: list = None):
    """Benchmark memory bank operations."""
    if memory_sizes is None:
        memory_sizes = [100, 1000, 10000]

    results = {}

    for n in memory_sizes:
        detector = EmbeddingDistance()
        bank = VDDMemoryBank(drift_detector=detector)

        # Add memories
        add_times = []
        for i in range(n):
            emb = np.random.randn(embedding_dim)
            start = time.perf_counter()
            bank.add(emb, f"content_{i}")
            add_times.append(time.perf_counter() - start)

        # Benchmark retrieval
        query = np.random.randn(embedding_dim)
        retrieve_times = []
        for _ in range(100):
            start = time.perf_counter()
            bank.retrieve(query, k=5)
            retrieve_times.append(time.perf_counter() - start)

        # Benchmark decay
        decay_times = []
        for _ in range(100):
            start = time.perf_counter()
            bank.decay(0.1)
            decay_times.append(time.perf_counter() - start)

        results[n] = {
            "add_mean_ms": np.mean(add_times) * 1000,
            "retrieve_mean_ms": np.mean(retrieve_times) * 1000,
            "decay_mean_ms": np.mean(decay_times) * 1000,
            "total_overhead_ms": (np.mean(retrieve_times) + np.mean(decay_times)) * 1000,
        }

    return results


def benchmark_full_pipeline(embedding_dim: int = 384, iterations: int = 100):
    """Benchmark full VDD pipeline (detect + retrieve + decay)."""
    detector = EmbeddingDistance()
    bank = VDDMemoryBank(drift_detector=detector)

    # Pre-populate with 1000 memories
    for i in range(1000):
        emb = np.random.randn(embedding_dim)
        bank.add(emb, f"content_{i}")

    # Benchmark full pipeline
    pipeline_times = []
    for _ in range(iterations):
        query = np.random.randn(embedding_dim)

        start = time.perf_counter()
        # Drift detection (happens in retrieve)
        results = bank.retrieve(query, k=5)
        # Decay
        bank.decay()
        elapsed = time.perf_counter() - start

        pipeline_times.append(elapsed)

    return {
        "mean_ms": np.mean(pipeline_times) * 1000,
        "std_ms": np.std(pipeline_times) * 1000,
        "p50_ms": np.percentile(pipeline_times, 50) * 1000,
        "p95_ms": np.percentile(pipeline_times, 95) * 1000,
        "p99_ms": np.percentile(pipeline_times, 99) * 1000,
    }


def main():
    """Main entry point."""
    print("=" * 60)
    print("Experiment 07: Latency Benchmarks")
    print("=" * 60)

    # Drift detection benchmarks
    print("\n1. DRIFT DETECTION LATENCY")
    print("-" * 40)
    drift_results = benchmark_drift_detection()
    for name, metrics in drift_results.items():
        print(f"  {name}: {metrics['per_call_ms']:.4f} ms/call")

    # Memory operations benchmarks
    print("\n2. MEMORY OPERATIONS LATENCY (by memory size)")
    print("-" * 40)
    memory_results = benchmark_memory_operations()
    print(f"  {'Size':<10} {'Add':<12} {'Retrieve':<12} {'Decay':<12} {'Total':<12}")
    for n, metrics in memory_results.items():
        print(f"  {n:<10} {metrics['add_mean_ms']:<12.4f} {metrics['retrieve_mean_ms']:<12.4f} "
              f"{metrics['decay_mean_ms']:<12.4f} {metrics['total_overhead_ms']:<12.4f}")

    # Full pipeline benchmark
    print("\n3. FULL PIPELINE LATENCY (1000 memories)")
    print("-" * 40)
    pipeline_results = benchmark_full_pipeline()
    print(f"  Mean: {pipeline_results['mean_ms']:.4f} ms")
    print(f"  P50:  {pipeline_results['p50_ms']:.4f} ms")
    print(f"  P95:  {pipeline_results['p95_ms']:.4f} ms")
    print(f"  P99:  {pipeline_results['p99_ms']:.4f} ms")

    # Assessment
    print(f"\n{'='*60}")
    print("LATENCY ASSESSMENT")
    print(f"{'='*60}")
    target_ms = 50
    actual_ms = pipeline_results['mean_ms']
    if actual_ms < target_ms:
        print(f"✅ PASSED: {actual_ms:.2f}ms < {target_ms}ms target")
    else:
        print(f"❌ FAILED: {actual_ms:.2f}ms > {target_ms}ms target")

    # Save summary
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "07_latency_results.txt", "w") as f:
        f.write("VDD Latency Benchmark Results\n")
        f.write("=" * 40 + "\n\n")
        f.write("Drift Detection (per call):\n")
        for name, metrics in drift_results.items():
            f.write(f"  {name}: {metrics['per_call_ms']:.4f} ms\n")
        f.write(f"\nFull Pipeline (1000 memories):\n")
        f.write(f"  Mean: {pipeline_results['mean_ms']:.4f} ms\n")
        f.write(f"  P95:  {pipeline_results['p95_ms']:.4f} ms\n")
        f.write(f"  P99:  {pipeline_results['p99_ms']:.4f} ms\n")
        f.write(f"\nTarget: {target_ms}ms\n")
        f.write(f"Status: {'PASSED' if actual_ms < target_ms else 'FAILED'}\n")

    print(f"\nSaved results to {results_dir / '07_latency_results.txt'}")

    return {
        "drift": drift_results,
        "memory": memory_results,
        "pipeline": pipeline_results,
    }


if __name__ == "__main__":
    main()
