#!/usr/bin/env python3
"""
Experiment 01: Drift Detection Comparison

Compares different drift detection algorithms on synthetic
data with known regime shifts. Evaluates:
- Detection latency (how quickly drift is detected)
- False positive rate
- False negative rate
- Volatility signal quality

This experiment validates that we can replace the oracle
volatility signal with real drift detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import ADWIN, EmbeddingDistance, PageHinkley
from vdd.drift_detection.base import DriftDetector


@dataclass
class ExperimentConfig:
    """Configuration for drift detection experiment."""
    steps: int = 1000
    regime_shifts: tuple = (200, 500, 800)
    noise_level: float = 0.1
    embedding_dim: int = 64
    seed: int = 42
    detection_window: int = 10  # Steps after shift to count as detected


def generate_regime_data(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with regime shifts.

    Returns:
        observations: Array of observations (embeddings)
        ground_truth_volatility: Oracle volatility signal
    """
    np.random.seed(config.seed)

    observations = []
    ground_truth = np.zeros(config.steps)

    # Current regime centroid
    centroid = np.random.randn(config.embedding_dim)
    centroid = centroid / np.linalg.norm(centroid)

    for t in range(config.steps):
        # Check for regime shift
        if t in config.regime_shifts:
            # Generate new centroid (orthogonal shift)
            shift = np.random.randn(config.embedding_dim)
            shift = shift / np.linalg.norm(shift)
            centroid = 0.3 * centroid + 0.7 * shift
            centroid = centroid / np.linalg.norm(centroid)

            # Mark volatility for detection window
            end_t = min(t + config.detection_window, config.steps)
            ground_truth[t:end_t] = 1.0

        # Generate observation with noise
        noise = np.random.randn(config.embedding_dim) * config.noise_level
        obs = centroid + noise
        obs = obs / np.linalg.norm(obs)
        observations.append(obs)

    return np.array(observations), ground_truth


def evaluate_detector(
    detector: DriftDetector,
    observations: np.ndarray,
    ground_truth: np.ndarray,
    config: ExperimentConfig,
) -> dict:
    """
    Evaluate a drift detector.

    Returns:
        Dictionary with evaluation metrics
    """
    detector.reset()

    detected_volatility = np.zeros(len(observations))
    drift_times = []

    for t, obs in enumerate(observations):
        result = detector.update(obs)
        detected_volatility[t] = result.volatility

        if result.detected:
            drift_times.append(t)

    # Compute metrics
    metrics = {
        "name": detector.name,
        "drift_count": len(drift_times),
        "drift_times": drift_times,
        "volatility": detected_volatility,
    }

    # Detection analysis for each regime shift
    true_shifts = config.regime_shifts
    detected_shifts = []
    detection_latencies = []
    false_positives = 0

    for dt in drift_times:
        # Check if this detection corresponds to a true shift
        matched = False
        for ts in true_shifts:
            if ts <= dt < ts + config.detection_window:
                if ts not in detected_shifts:
                    detected_shifts.append(ts)
                    detection_latencies.append(dt - ts)
                matched = True
                break

        if not matched:
            false_positives += 1

    # Metrics
    true_positives = len(detected_shifts)
    false_negatives = len(true_shifts) - true_positives

    metrics["true_positives"] = true_positives
    metrics["false_positives"] = false_positives
    metrics["false_negatives"] = false_negatives
    metrics["detection_latencies"] = detection_latencies
    metrics["mean_latency"] = np.mean(detection_latencies) if detection_latencies else float("inf")

    # Precision and Recall
    if true_positives + false_positives > 0:
        metrics["precision"] = true_positives / (true_positives + false_positives)
    else:
        metrics["precision"] = 0.0

    if true_positives + false_negatives > 0:
        metrics["recall"] = true_positives / (true_positives + false_negatives)
    else:
        metrics["recall"] = 0.0

    # F1
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (
            metrics["precision"] + metrics["recall"]
        )
    else:
        metrics["f1"] = 0.0

    return metrics


def run_experiment(config: ExperimentConfig) -> dict:
    """Run the full experiment."""
    print("=" * 60)
    print("Experiment 01: Drift Detection Comparison")
    print("=" * 60)
    print(f"Steps: {config.steps}")
    print(f"Regime shifts at: {config.regime_shifts}")
    print(f"Noise level: {config.noise_level}")
    print()

    # Generate data
    print("Generating synthetic data...")
    observations, ground_truth = generate_regime_data(config)
    print(f"Generated {len(observations)} observations")
    print()

    # Initialize detectors
    detectors = [
        ADWIN(delta=0.002, min_window=10),
        PageHinkley(delta=0.005, threshold=50.0),
        EmbeddingDistance(curr_window=10, arch_window=200, drift_threshold=0.3),
    ]

    # Evaluate each detector
    results = {}
    for detector in detectors:
        print(f"Evaluating {detector.name}...")
        metrics = evaluate_detector(detector, observations, ground_truth, config)
        results[detector.name] = metrics

        print(f"  True Positives: {metrics['true_positives']}/{len(config.regime_shifts)}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")
        print(f"  Mean Detection Latency: {metrics['mean_latency']:.1f} steps")
        print()

    return {
        "config": config,
        "observations": observations,
        "ground_truth": ground_truth,
        "results": results,
    }


def plot_results(experiment_results: dict, save_path: str = None):
    """Plot experiment results."""
    config = experiment_results["config"]
    ground_truth = experiment_results["ground_truth"]
    results = experiment_results["results"]

    fig, axes = plt.subplots(len(results) + 1, 1, figsize=(12, 3 * (len(results) + 1)))

    # Plot ground truth
    ax = axes[0]
    ax.fill_between(range(len(ground_truth)), ground_truth, alpha=0.3, color="gray", label="True Drift Window")
    for shift in config.regime_shifts:
        ax.axvline(x=shift, color="red", linestyle="--", alpha=0.7)
    ax.set_ylabel("Ground Truth")
    ax.set_title("Drift Detection Comparison")
    ax.legend(loc="upper right")
    ax.set_xlim(0, len(ground_truth))

    # Plot each detector
    colors = ["blue", "green", "orange", "purple"]
    for idx, (name, metrics) in enumerate(results.items()):
        ax = axes[idx + 1]
        volatility = metrics["volatility"]

        ax.plot(volatility, color=colors[idx % len(colors)], label=f"{name} Volatility")
        ax.fill_between(range(len(volatility)), volatility, alpha=0.2, color=colors[idx % len(colors)])

        # Mark detections
        for dt in metrics["drift_times"]:
            ax.axvline(x=dt, color=colors[idx % len(colors)], linestyle=":", alpha=0.5)

        # Mark true shifts
        for shift in config.regime_shifts:
            ax.axvline(x=shift, color="red", linestyle="--", alpha=0.3)

        ax.set_ylabel(f"{name}\nVolatility")
        ax.set_xlim(0, len(volatility))
        ax.legend(loc="upper right")

        # Add metrics text
        metrics_text = f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}"
        ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    axes[-1].set_xlabel("Time Steps")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def main():
    """Main entry point."""
    config = ExperimentConfig(
        steps=1000,
        regime_shifts=(200, 500, 800),
        noise_level=0.1,
        embedding_dim=64,
        seed=42,
    )

    results = run_experiment(config)

    # Create results directory
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Plot and save
    plot_results(results, save_path=str(results_dir / "01_drift_detection.png"))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nBest detector by F1 score:")
    best = max(results["results"].items(), key=lambda x: x[1]["f1"])
    print(f"  {best[0]}: F1 = {best[1]['f1']:.3f}")

    print("\nBest detector by latency (among those with recall > 0):")
    valid = [(k, v) for k, v in results["results"].items() if v["recall"] > 0]
    if valid:
        best_latency = min(valid, key=lambda x: x[1]["mean_latency"])
        print(f"  {best_latency[0]}: {best_latency[1]['mean_latency']:.1f} steps")


if __name__ == "__main__":
    main()
