#!/usr/bin/env python3
"""
Experiment 13: ADWIN vs VDD Comparison

Compare VDD's sigmoid modulation against ADWIN's binary change detection.
Hypothesis: VDD's smooth sigmoid should produce less jitter than ADWIN's binary switching.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import EmbeddingDistance
from vdd.drift_detection.adwin_simple import SimpleADWIN


class ADWINDecay:
    """Adaptive decay using ADWIN change detection."""

    def __init__(self, lambda_low: float = 0.1, lambda_high: float = 0.9, cooldown: int = 20):
        # Very sensitive ADWIN (higher delta = more sensitive)
        self.adwin = SimpleADWIN(delta=0.5, min_window=3, max_window=30)
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        self.cooldown_max = cooldown
        self.cooldown = 0

    def update(self, embedding) -> float:
        """Update and return current lambda."""
        result = self.adwin.update(embedding)

        if result.detected:
            self.cooldown = self.cooldown_max

        if self.cooldown > 0:
            self.cooldown -= 1
            return self.lambda_high
        else:
            return self.lambda_low


def generate_data(steps: int, regime_shifts: tuple, embedding_dim: int, seed: int):
    """Generate synthetic data with regime shifts."""
    np.random.seed(seed)

    truth = np.zeros(steps)
    current_val = 0
    embeddings = []
    centroid = np.random.randn(embedding_dim)
    centroid = centroid / np.linalg.norm(centroid)

    for t in range(steps):
        if t in regime_shifts:
            current_val += np.random.choice([-5, 5])
            new_dir = np.random.randn(embedding_dim)
            centroid = 0.3 * centroid + 0.7 * new_dir
            centroid = centroid / np.linalg.norm(centroid)
        else:
            current_val += np.random.normal(0, 0.1)
        truth[t] = current_val

        emb = centroid + np.random.randn(embedding_dim) * 0.1
        embeddings.append(emb)

    return truth, embeddings


def run_vdd(truth: np.ndarray, embeddings: list):
    """Run VDD method."""
    steps = len(truth)
    memory = np.zeros(steps)
    stored = 0.0
    lambdas = []

    detector = EmbeddingDistance(curr_window=10, arch_window=100, drift_threshold=0.4)
    lambda_base, lambda_max = 0.2, 0.9

    for t in range(1, steps):
        result = detector.update(embeddings[t])
        v = result.volatility
        current_lambda = lambda_base + (lambda_max - lambda_base) * v
        lambdas.append(current_lambda)

        stored = (1 - current_lambda) * stored + current_lambda * truth[t]
        memory[t] = stored

    error = np.abs(truth - memory)
    return {
        "iae": np.sum(error),
        "error": error,
        "memory": memory,
        "lambdas": np.array(lambdas),
    }


def run_adwin(truth: np.ndarray, embeddings: list):
    """Run ADWIN-based decay."""
    steps = len(truth)
    memory = np.zeros(steps)
    stored = 0.0
    lambdas = []

    adwin_decay = ADWINDecay(lambda_low=0.1, lambda_high=0.9, cooldown=20)

    for t in range(1, steps):
        current_lambda = adwin_decay.update(embeddings[t])
        lambdas.append(current_lambda)

        stored = (1 - current_lambda) * stored + current_lambda * truth[t]
        memory[t] = stored

    error = np.abs(truth - memory)
    return {
        "iae": np.sum(error),
        "error": error,
        "memory": memory,
        "lambdas": np.array(lambdas),
    }


def run_recency(truth: np.ndarray):
    """Run recency baseline."""
    steps = len(truth)
    memory = np.zeros(steps)
    stored = 0.0
    current_lambda = 0.5

    for t in range(1, steps):
        stored = (1 - current_lambda) * stored + current_lambda * truth[t]
        memory[t] = stored

    error = np.abs(truth - memory)
    return {"iae": np.sum(error), "error": error, "memory": memory}


def run_static(truth: np.ndarray):
    """Run static baseline."""
    steps = len(truth)
    memory = np.zeros(steps)
    stored = 0.0
    current_lambda = 0.1

    for t in range(1, steps):
        stored = (1 - current_lambda) * stored + current_lambda * truth[t]
        memory[t] = stored

    error = np.abs(truth - memory)
    return {"iae": np.sum(error), "error": error, "memory": memory}


def compute_jitter(lambdas: np.ndarray) -> float:
    """Compute lambda jitter (how often it changes direction)."""
    if len(lambdas) < 3:
        return 0.0

    diffs = np.diff(lambdas)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    return sign_changes / len(lambdas)


def run_experiment():
    """Run ADWIN vs VDD comparison."""
    print("=" * 60)
    print("Experiment 13: ADWIN vs VDD Comparison")
    print("=" * 60)

    steps = 200
    regime_shifts = (50, 100, 150)
    embedding_dim = 32
    n_runs = 10

    results = {
        "vdd": {"iae": [], "jitter": []},
        "adwin": {"iae": [], "jitter": []},
        "recency": {"iae": []},
        "static": {"iae": []},
    }

    print(f"\nRunning {n_runs} trials...")
    for seed in range(42, 42 + n_runs):
        truth, embeddings = generate_data(steps, regime_shifts, embedding_dim, seed)

        # VDD
        vdd_res = run_vdd(truth, embeddings)
        results["vdd"]["iae"].append(vdd_res["iae"])
        results["vdd"]["jitter"].append(compute_jitter(vdd_res["lambdas"]))

        # ADWIN
        adwin_res = run_adwin(truth, embeddings)
        results["adwin"]["iae"].append(adwin_res["iae"])
        results["adwin"]["jitter"].append(compute_jitter(adwin_res["lambdas"]))

        # Baselines
        recency_res = run_recency(truth)
        results["recency"]["iae"].append(recency_res["iae"])

        static_res = run_static(truth)
        results["static"]["iae"].append(static_res["iae"])

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS (mean ± std)")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'IAE':<20} {'Jitter':<15}")
    print("-" * 50)

    for method in ["vdd", "adwin", "recency", "static"]:
        iae_mean = np.mean(results[method]["iae"])
        iae_std = np.std(results[method]["iae"])
        if "jitter" in results[method]:
            jitter = np.mean(results[method]["jitter"])
            print(f"{method:<15} {iae_mean:>8.2f} ± {iae_std:<8.2f} {jitter:.4f}")
        else:
            print(f"{method:<15} {iae_mean:>8.2f} ± {iae_std:<8.2f} N/A")

    # Statistical comparison
    print(f"\n{'='*60}")
    print("STATISTICAL TESTS")
    print(f"{'='*60}")

    vdd_iae = results["vdd"]["iae"]
    adwin_iae = results["adwin"]["iae"]
    t_stat, p_value = stats.ttest_ind(vdd_iae, adwin_iae)
    winner = "VDD" if np.mean(vdd_iae) < np.mean(adwin_iae) else "ADWIN"
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"VDD vs ADWIN: t={t_stat:.3f}, p={p_value:.4f} {sig} → {winner} wins")

    # Jitter comparison
    vdd_jitter = np.mean(results["vdd"]["jitter"])
    adwin_jitter = np.mean(results["adwin"]["jitter"])
    print(f"\nJitter (lower = smoother):")
    print(f"  VDD:   {vdd_jitter:.4f}")
    print(f"  ADWIN: {adwin_jitter:.4f}")
    print(f"  → {'VDD' if vdd_jitter < adwin_jitter else 'ADWIN'} is smoother")

    return results


def plot_comparison(results: dict):
    """Visualize comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = ["vdd", "adwin", "recency", "static"]
    colors = {"vdd": "green", "adwin": "orange", "recency": "red", "static": "blue"}
    n = 10

    # Plot 1: IAE comparison bar chart
    ax = axes[0, 0]
    means = [np.mean(results[m]["iae"]) for m in methods]
    stds = [np.std(results[m]["iae"]) for m in methods]
    cis = [1.96 * s / np.sqrt(n) for s in stds]

    bars = ax.bar(methods, means, yerr=cis, capsize=6,
                  color=[colors[m] for m in methods], alpha=0.8, edgecolor='black')
    ax.set_ylabel("IAE")
    ax.set_title("Error Comparison\n(Error bars = 95% CI)")

    for bar, mean, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Jitter comparison
    ax = axes[0, 1]
    jitter_methods = ["vdd", "adwin"]
    jitters = [np.mean(results[m]["jitter"]) for m in jitter_methods]
    jitter_stds = [np.std(results[m]["jitter"]) for m in jitter_methods]

    bars = ax.bar(jitter_methods, jitters, color=[colors[m] for m in jitter_methods],
                  alpha=0.8, edgecolor='black')
    ax.set_ylabel("Jitter (direction changes / step)")
    ax.set_title("Lambda Smoothness\n(Lower = Smoother)")

    for bar, j in zip(bars, jitters):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{j:.4f}', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Single run lambda trajectories
    ax = axes[1, 0]
    seed = 42
    steps = 200
    regime_shifts = (50, 100, 150)
    truth, embeddings = generate_data(steps, regime_shifts, 32, seed)

    vdd_res = run_vdd(truth, embeddings)
    adwin_res = run_adwin(truth, embeddings)

    ax.plot(vdd_res["lambdas"], 'g-', label='VDD (sigmoid)', lw=1.5, alpha=0.8)
    ax.plot(adwin_res["lambdas"], 'orange', label='ADWIN (binary)', lw=1.5, alpha=0.8)

    for shift in regime_shifts:
        ax.axvline(x=shift, color='red', linestyle='--', alpha=0.5,
                   label='Regime Shift' if shift == regime_shifts[0] else '')

    ax.set_ylabel("λ(t)")
    ax.set_xlabel("Time Steps")
    ax.set_title("Lambda Trajectory Comparison\nVDD: smooth sigmoid | ADWIN: binary switching")
    ax.legend()
    ax.set_ylim(0, 1)

    # Plot 4: Memory tracking comparison
    ax = axes[1, 1]
    ax.plot(truth, 'k--', alpha=0.5, label='Ground Truth', lw=1)
    ax.plot(vdd_res["memory"], 'g-', label=f'VDD (IAE={vdd_res["iae"]:.1f})', lw=1.5)
    ax.plot(adwin_res["memory"], 'orange', label=f'ADWIN (IAE={adwin_res["iae"]:.1f})', lw=1.5, alpha=0.8)

    for shift in regime_shifts:
        ax.axvline(x=shift, color='red', linestyle='--', alpha=0.3)

    ax.set_ylabel("Value")
    ax.set_xlabel("Time Steps")
    ax.set_title("Memory Tracking\n(Red dashes = regime shifts)")
    ax.legend()

    plt.tight_layout()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "13_adwin_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {results_dir / '13_adwin_comparison.png'}")
    plt.close()


def main():
    results = run_experiment()
    plot_comparison(results)
    return results


if __name__ == "__main__":
    main()
