#!/usr/bin/env python3
"""
Experiment 11: Drift Detection Method Comparison

Compare Option A (prediction error) vs Option B (semantic distance).

Key hypothesis:
- Option A should have higher precision (fewer false positives on topic switches)
- Option B should have higher recall (catches all semantic changes)

For VDD, we care most about avoiding unnecessary high-λ periods.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vdd.drift_detection import EmbeddingDistance
from vdd.drift_detection.prediction_error import PredictionErrorDetector


def generate_data_with_topic_switches(
    steps: int,
    embedding_dim: int,
    regime_shifts: tuple,
    topic_switches: tuple,
    seed: int,
):
    """
    Generate data with both regime shifts AND topic switches.

    Regime shifts: Knowledge becomes stale (VDD should respond)
    Topic switches: Just topic change, knowledge still valid (VDD should NOT respond)
    """
    np.random.seed(seed)

    truth = np.zeros(steps)
    current_val = 0
    embeddings = []
    labels = []  # 'stable', 'regime_shift', 'topic_switch'

    # Define centroids for different topics (A, B) and regimes (1, 2, 3)
    topic_A = np.random.randn(embedding_dim)
    topic_A = topic_A / np.linalg.norm(topic_A)
    topic_B = np.random.randn(embedding_dim)
    topic_B = topic_B / np.linalg.norm(topic_B)

    current_topic = topic_A
    regime_offset = 0

    for t in range(steps):
        # Check for regime shift (knowledge becomes stale)
        if t in regime_shifts:
            regime_offset += np.random.choice([-5, 5])
            # Slight centroid shift within topic
            current_topic = current_topic + np.random.randn(embedding_dim) * 0.3
            current_topic = current_topic / np.linalg.norm(current_topic)
            labels.append('regime_shift')
        # Check for topic switch (just topic change, no knowledge change)
        elif t in topic_switches:
            current_topic = topic_B if np.allclose(current_topic / np.linalg.norm(current_topic),
                                                     topic_A / np.linalg.norm(topic_A), atol=0.5) else topic_A
            labels.append('topic_switch')
        else:
            labels.append('stable')
            current_val += np.random.normal(0, 0.1)

        truth[t] = regime_offset + current_val + np.random.normal(0, 0.1)

        emb = current_topic + np.random.randn(embedding_dim) * 0.1
        embeddings.append(emb)

    return truth, embeddings, labels


def evaluate_detector(detector_class, detector_kwargs, embeddings, labels):
    """
    Evaluate detector precision/recall.

    True positive: Detected AND was regime_shift
    False positive: Detected AND was topic_switch or stable
    """
    detector = detector_class(**detector_kwargs)

    detections = []
    volatilities = []

    for emb in embeddings:
        result = detector.update(emb)
        detections.append(result.detected)
        volatilities.append(result.volatility)

    # Calculate metrics
    tp = sum(1 for d, l in zip(detections, labels) if d and l == 'regime_shift')
    fp_topic = sum(1 for d, l in zip(detections, labels) if d and l == 'topic_switch')
    fp_stable = sum(1 for d, l in zip(detections, labels) if d and l == 'stable')
    fn = sum(1 for d, l in zip(detections, labels) if not d and l == 'regime_shift')

    total_positives = tp + fp_topic + fp_stable
    total_regime_shifts = sum(1 for l in labels if l == 'regime_shift')

    precision = tp / total_positives if total_positives > 0 else 0
    recall = tp / total_regime_shifts if total_regime_shifts > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp_topic': fp_topic,
        'fp_stable': fp_stable,
        'fn': fn,
        'volatilities': np.array(volatilities),
        'detections': detections,
    }


def run_vdd_with_detector(detector_class, detector_kwargs, truth, embeddings):
    """Run VDD memory tracking with a specific detector."""
    steps = len(truth)
    memory = np.zeros(steps)
    stored = 0.0
    lambdas = []

    detector = detector_class(**detector_kwargs)
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
        'iae': np.sum(error),
        'memory': memory,
        'lambdas': np.array(lambdas),
    }


def run_experiment():
    """Run detector comparison."""
    print("=" * 60)
    print("Experiment 11: Drift Detection Method Comparison")
    print("=" * 60)
    print("\nOption A: Prediction Error (detects knowledge staleness)")
    print("Option B: Semantic Distance (detects topic changes)")

    steps = 300
    embedding_dim = 32
    regime_shifts = (50, 150, 250)  # 3 regime shifts
    topic_switches = (100, 200)  # 2 topic switches (no knowledge change)
    n_runs = 10

    option_a_kwargs = {'window': 10, 'threshold': 0.5, 'alpha': 0.2}
    option_b_kwargs = {'curr_window': 10, 'arch_window': 100, 'drift_threshold': 0.4}

    results = {
        'option_a': {'precision': [], 'recall': [], 'f1': [], 'iae': [], 'fp_topic': []},
        'option_b': {'precision': [], 'recall': [], 'f1': [], 'iae': [], 'fp_topic': []},
    }

    print(f"\nRunning {n_runs} trials...")
    for seed in range(42, 42 + n_runs):
        truth, embeddings, labels = generate_data_with_topic_switches(
            steps, embedding_dim, regime_shifts, topic_switches, seed
        )

        # Evaluate Option A (Prediction Error)
        eval_a = evaluate_detector(PredictionErrorDetector, option_a_kwargs, embeddings, labels)
        vdd_a = run_vdd_with_detector(PredictionErrorDetector, option_a_kwargs, truth, embeddings)
        results['option_a']['precision'].append(eval_a['precision'])
        results['option_a']['recall'].append(eval_a['recall'])
        results['option_a']['f1'].append(eval_a['f1'])
        results['option_a']['iae'].append(vdd_a['iae'])
        results['option_a']['fp_topic'].append(eval_a['fp_topic'])

        # Evaluate Option B (Semantic Distance)
        eval_b = evaluate_detector(EmbeddingDistance, option_b_kwargs, embeddings, labels)
        vdd_b = run_vdd_with_detector(EmbeddingDistance, option_b_kwargs, truth, embeddings)
        results['option_b']['precision'].append(eval_b['precision'])
        results['option_b']['recall'].append(eval_b['recall'])
        results['option_b']['f1'].append(eval_b['f1'])
        results['option_b']['iae'].append(vdd_b['iae'])
        results['option_b']['fp_topic'].append(eval_b['fp_topic'])

    # Print results
    print(f"\n{'='*60}")
    print("DETECTION METRICS (mean ± std)")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Option A (Pred Err)':<25} {'Option B (Semantic)':<25}")
    print("-" * 70)

    for metric in ['precision', 'recall', 'f1', 'fp_topic']:
        a_mean = np.mean(results['option_a'][metric])
        a_std = np.std(results['option_a'][metric])
        b_mean = np.mean(results['option_b'][metric])
        b_std = np.std(results['option_b'][metric])
        print(f"{metric:<20} {a_mean:>8.3f} ± {a_std:<10.3f} {b_mean:>8.3f} ± {b_std:<10.3f}")

    print(f"\n{'='*60}")
    print("VDD PERFORMANCE WITH EACH DETECTOR")
    print(f"{'='*60}")
    a_iae = np.mean(results['option_a']['iae'])
    b_iae = np.mean(results['option_b']['iae'])
    print(f"Option A (Prediction Error): IAE = {a_iae:.2f} ± {np.std(results['option_a']['iae']):.2f}")
    print(f"Option B (Semantic Distance): IAE = {b_iae:.2f} ± {np.std(results['option_b']['iae']):.2f}")

    # Statistical test
    t_stat, p_value = stats.ttest_ind(results['option_a']['iae'], results['option_b']['iae'])
    winner = "Option A" if a_iae < b_iae else "Option B"
    print(f"\nt-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"Winner: {winner}")

    # Key finding
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")

    a_fp = np.mean(results['option_a']['fp_topic'])
    b_fp = np.mean(results['option_b']['fp_topic'])
    print(f"False positives on topic switches:")
    print(f"  Option A: {a_fp:.1f} (should be LOW - topic change ≠ knowledge stale)")
    print(f"  Option B: {b_fp:.1f}")

    if a_fp < b_fp:
        print("\n✅ Option A has fewer topic-switch false positives (as hypothesized)")
    else:
        print("\n⚠️ Option A did NOT reduce topic-switch false positives")

    return results


def plot_comparison(results: dict):
    """Visualize comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Detection metrics comparison
    ax = axes[0, 0]
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.35

    a_means = [np.mean(results['option_a'][m]) for m in metrics]
    b_means = [np.mean(results['option_b'][m]) for m in metrics]
    a_stds = [np.std(results['option_a'][m]) for m in metrics]
    b_stds = [np.std(results['option_b'][m]) for m in metrics]

    ax.bar(x - width/2, a_means, width, yerr=a_stds, label='Option A (Pred Err)',
           color='blue', alpha=0.7, capsize=4)
    ax.bar(x + width/2, b_means, width, yerr=b_stds, label='Option B (Semantic)',
           color='orange', alpha=0.7, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(['Precision', 'Recall', 'F1'])
    ax.set_ylabel('Score')
    ax.set_title('Detection Metrics Comparison')
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Plot 2: False positives on topic switches
    ax = axes[0, 1]
    options = ['Option A\n(Prediction)', 'Option B\n(Semantic)']
    fp_means = [np.mean(results['option_a']['fp_topic']), np.mean(results['option_b']['fp_topic'])]
    fp_stds = [np.std(results['option_a']['fp_topic']), np.std(results['option_b']['fp_topic'])]

    bars = ax.bar(options, fp_means, yerr=fp_stds, capsize=6,
                  color=['blue', 'orange'], alpha=0.7)
    ax.set_ylabel('False Positives on Topic Switches')
    ax.set_title('Topic Switch False Positives\n(Lower = Better)')

    for bar, mean in zip(bars, fp_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mean:.1f}', ha='center', fontweight='bold')

    # Plot 3: IAE comparison
    ax = axes[1, 0]
    iae_means = [np.mean(results['option_a']['iae']), np.mean(results['option_b']['iae'])]
    iae_stds = [np.std(results['option_a']['iae']), np.std(results['option_b']['iae'])]

    bars = ax.bar(options, iae_means, yerr=iae_stds, capsize=6,
                  color=['blue', 'orange'], alpha=0.7)
    ax.set_ylabel('IAE')
    ax.set_title('VDD Memory Error\n(Lower = Better)')

    for bar, mean in zip(bars, iae_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mean:.1f}', ha='center', fontweight='bold')

    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    a_prec = np.mean(results['option_a']['precision'])
    b_prec = np.mean(results['option_b']['precision'])
    a_rec = np.mean(results['option_a']['recall'])
    b_rec = np.mean(results['option_b']['recall'])

    summary = f"""
    COMPARISON SUMMARY
    ══════════════════════════════════════

    Option A (Prediction Error):
    • Detects when predictions fail
    • Better for: Multi-topic conversations
    • Precision: {a_prec:.3f}
    • Recall: {b_rec:.3f}

    Option B (Semantic Distance):
    • Detects embedding space shifts
    • Better for: Single-topic streams
    • Precision: {b_prec:.3f}
    • Recall: {b_rec:.3f}

    ══════════════════════════════════════
    RECOMMENDATION:
    Use Option B for production (higher recall)
    Consider Option A for multi-topic RAG
    """

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "11_detection_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {results_dir / '11_detection_comparison.png'}")
    plt.close()


def main():
    results = run_experiment()
    plot_comparison(results)
    return results


if __name__ == "__main__":
    main()
