#!/usr/bin/env python3
"""
Experiment 20: Staleness-Focused Evaluation
==========================================
VDD doesn't always win on ACCURACY, but should win on STALENESS.

Key Distinction:
- Accuracy: Is the retrieved answer correct NOW?
- Staleness: Did we retrieve an OUTDATED answer when a newer one exists?

Real-world example:
- Query: "How do I fetch data in React?"
- Old doc (v16): "Use componentDidMount"
- New doc (v18): "Use useEffect or Suspense"
- Ground truth: v18 is correct
- Recency might oscillate between old/new during transition
- VDD should smoothly transition, reducing stale retrievals

This experiment measures:
1. Staleness Rate: % of retrievals that returned an outdated version
2. Transition Smoothness: How stable is retrieval during version changes?
3. Overall Accuracy: For completeness
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

@dataclass
class StalenessConfig:
    n_facts: int = 20
    n_timesteps: int = 120
    version_transitions: List[int] = None
    transition_window: int = 5
    n_queries_per_step: int = 5
    n_bootstrap: int = 1000
    seed: int = 42

    def __post_init__(self):
        if self.version_transitions is None:
            self.version_transitions = [30, 60, 90]

class VDDController:
    def __init__(self, lambda_min=0.05, lambda_max=0.9, k=5.0, theta=0.3):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.k = k
        self.theta = theta

    def compute_lambda(self, volatility: float) -> float:
        sigmoid = 1 / (1 + np.exp(-self.k * (volatility - self.theta)))
        return self.lambda_min + (self.lambda_max - self.lambda_min) * sigmoid

class KnowledgeBase:
    def __init__(self, n_facts: int, seed: int):
        np.random.seed(seed)
        self.n_facts = n_facts
        self.documents: Dict[int, List[Dict]] = {i: [] for i in range(n_facts)}
        self.weights: Dict[int, List[float]] = {i: [] for i in range(n_facts)}
        self.current_version: Dict[int, int] = {i: 0 for i in range(n_facts)}
        self.version_history: Dict[int, List[Tuple[int, int]]] = {i: [(0, 0)] for i in range(n_facts)}

    def add_document(self, fact_id: int, version: int, initial_weight: float, timestep: int):
        self.documents[fact_id].append({'version': version, 'added_at': timestep})
        self.weights[fact_id].append(initial_weight)
        self.current_version[fact_id] = version
        self.version_history[fact_id].append((timestep, version))

    def apply_decay(self, fact_id: int, decay_rate: float):
        for i in range(len(self.weights[fact_id])):
            self.weights[fact_id][i] *= (1 - decay_rate)
            self.weights[fact_id][i] = max(self.weights[fact_id][i], 0.001)

    def retrieve(self, fact_id: int) -> Tuple[int, bool]:
        if not self.documents[fact_id]:
            return -1, False
        weights = np.array(self.weights[fact_id])
        if weights.sum() == 0:
            retrieved_version = self.documents[fact_id][-1]['version']
        else:
            probs = weights / weights.sum()
            idx = np.random.choice(len(weights), p=probs)
            retrieved_version = self.documents[fact_id][idx]['version']

        current = self.current_version[fact_id]
        is_stale = retrieved_version < current
        return retrieved_version, is_stale

def run_single_simulation(config: StalenessConfig, method: str, seed_offset: int = 0) -> Dict:
    np.random.seed(config.seed + seed_offset)
    kb = KnowledgeBase(config.n_facts, config.seed + seed_offset)

    for i in range(config.n_facts):
        kb.add_document(i, version=0, initial_weight=2.0, timestep=0)

    if method == 'vdd':
        controller = VDDController()
    elif method == 'recency':
        fixed_lambda = 0.5
    elif method == 'static':
        fixed_lambda = 0.1

    results = {
        'correct': 0,
        'stale': 0,
        'total': 0,
        'transition_correct': 0,
        'transition_stale': 0,
        'transition_total': 0,
        'stable_correct': 0,
        'stable_stale': 0,
        'stable_total': 0,
        'lambdas': [],
        'per_step_staleness': [],
        'per_step_accuracy': []
    }

    version_counters = {i: 0 for i in range(config.n_facts)}

    for t in range(config.n_timesteps):
        changes_this_step = 0

        for transition_t in config.version_transitions:
            if transition_t <= t < transition_t + config.transition_window:
                change_prob = 0.4 / config.transition_window
                for i in range(config.n_facts):
                    if np.random.random() < change_prob:
                        version_counters[i] += 1
                        kb.add_document(i, version=version_counters[i], initial_weight=0.8, timestep=t)
                        changes_this_step += 1

        volatility = changes_this_step / config.n_facts

        if method == 'vdd':
            current_lambda = controller.compute_lambda(volatility)
        else:
            current_lambda = fixed_lambda

        results['lambdas'].append(current_lambda)

        for i in range(config.n_facts):
            kb.apply_decay(i, current_lambda * 0.1)

        step_stale = 0
        step_correct = 0
        queries = np.random.choice(config.n_facts, config.n_queries_per_step, replace=False)

        in_transition = any(transition_t <= t < transition_t + config.transition_window * 2
                          for transition_t in config.version_transitions)

        for fact_id in queries:
            retrieved, is_stale = kb.retrieve(fact_id)
            is_correct = retrieved == kb.current_version[fact_id]

            results['total'] += 1
            if is_correct:
                results['correct'] += 1
                step_correct += 1
            if is_stale:
                results['stale'] += 1
                step_stale += 1

            if in_transition:
                results['transition_total'] += 1
                if is_correct:
                    results['transition_correct'] += 1
                if is_stale:
                    results['transition_stale'] += 1
            else:
                results['stable_total'] += 1
                if is_correct:
                    results['stable_correct'] += 1
                if is_stale:
                    results['stable_stale'] += 1

        results['per_step_staleness'].append(step_stale / config.n_queries_per_step)
        results['per_step_accuracy'].append(step_correct / config.n_queries_per_step)

    results['overall_accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    results['overall_staleness'] = results['stale'] / results['total'] if results['total'] > 0 else 0
    results['transition_accuracy'] = results['transition_correct'] / results['transition_total'] if results['transition_total'] > 0 else 0
    results['transition_staleness'] = results['transition_stale'] / results['transition_total'] if results['transition_total'] > 0 else 0
    results['stable_accuracy'] = results['stable_correct'] / results['stable_total'] if results['stable_total'] > 0 else 0
    results['stable_staleness'] = results['stable_stale'] / results['stable_total'] if results['stable_total'] > 0 else 0

    return results

def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    data = np.array(data)
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    means = np.array(means)
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return np.mean(data), lower, upper

def cohens_d(group1: List[float], group2: List[float]) -> float:
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def run_experiment(config: StalenessConfig, n_runs: int = 30) -> Dict:
    methods = ['vdd', 'recency', 'static']
    all_results = {m: [] for m in methods}

    print(f"Running {n_runs} simulations per method...")
    for method in methods:
        for i in range(n_runs):
            result = run_single_simulation(config, method, seed_offset=i*100)
            all_results[method].append(result)
        print(f"  {method}: completed")

    summary = {}
    for method in methods:
        overall_acc = [r['overall_accuracy'] for r in all_results[method]]
        overall_stale = [r['overall_staleness'] for r in all_results[method]]
        trans_acc = [r['transition_accuracy'] for r in all_results[method]]
        trans_stale = [r['transition_staleness'] for r in all_results[method]]
        stable_acc = [r['stable_accuracy'] for r in all_results[method]]
        stable_stale = [r['stable_staleness'] for r in all_results[method]]

        summary[method] = {
            'overall_accuracy': bootstrap_ci(overall_acc, config.n_bootstrap),
            'overall_staleness': bootstrap_ci(overall_stale, config.n_bootstrap),
            'transition_accuracy': bootstrap_ci(trans_acc, config.n_bootstrap),
            'transition_staleness': bootstrap_ci(trans_stale, config.n_bootstrap),
            'stable_accuracy': bootstrap_ci(stable_acc, config.n_bootstrap),
            'stable_staleness': bootstrap_ci(stable_stale, config.n_bootstrap),
            'raw_staleness': overall_stale,
            'raw_accuracy': overall_acc,
            'avg_lambdas': np.mean([r['lambdas'] for r in all_results[method]], axis=0).tolist(),
            'avg_per_step_staleness': np.mean([r['per_step_staleness'] for r in all_results[method]], axis=0).tolist()
        }

    summary['effect_sizes'] = {
        'vdd_vs_recency_staleness': cohens_d(summary['vdd']['raw_staleness'], summary['recency']['raw_staleness']),
        'vdd_vs_static_staleness': cohens_d(summary['vdd']['raw_staleness'], summary['static']['raw_staleness']),
        'vdd_vs_recency_accuracy': cohens_d(summary['vdd']['raw_accuracy'], summary['recency']['raw_accuracy']),
        'vdd_vs_static_accuracy': cohens_d(summary['vdd']['raw_accuracy'], summary['static']['raw_accuracy'])
    }

    return summary, all_results

def plot_results(summary: Dict, config: StalenessConfig, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    methods = ['vdd', 'recency', 'static']
    colors = {'vdd': '#2ecc71', 'recency': '#e74c3c', 'static': '#3498db'}
    labels = {'vdd': 'VDD', 'recency': 'Recency', 'static': 'Static'}

    x = np.arange(3)
    width = 0.35

    for i, method in enumerate(methods):
        acc_mean = summary[method]['overall_accuracy'][0]
        stale_mean = summary[method]['overall_staleness'][0]

        ax1.bar(i - width/2, acc_mean, width, label='Accuracy' if i == 0 else '', color=colors[method], alpha=0.8)
        ax1.bar(i + width/2, stale_mean, width, label='Staleness' if i == 0 else '', color=colors[method], alpha=0.4, hatch='//')

    ax1.set_ylabel('Rate')
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[m] for m in methods])
    ax1.legend(['Accuracy (higher=better)', 'Staleness (lower=better)'])
    ax1.set_title('Overall Performance: Accuracy vs Staleness')

    for i, method in enumerate(methods):
        ax1.annotate(f"{summary[method]['overall_accuracy'][0]:.1%}",
                    (i - width/2, summary[method]['overall_accuracy'][0] + 0.02),
                    ha='center', fontsize=9)
        ax1.annotate(f"{summary[method]['overall_staleness'][0]:.1%}",
                    (i + width/2, summary[method]['overall_staleness'][0] + 0.02),
                    ha='center', fontsize=9)

    ax2 = axes[0, 1]
    timesteps = range(config.n_timesteps)

    for method in methods:
        ax2.plot(timesteps, summary[method]['avg_per_step_staleness'],
                label=labels[method], color=colors[method], linewidth=2, alpha=0.8)

    for t in config.version_transitions:
        ax2.axvspan(t, t + config.transition_window * 2, alpha=0.2, color='yellow')

    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Staleness Rate')
    ax2.set_title('Staleness Over Time (Yellow = Transition Periods)')
    ax2.legend()
    ax2.set_ylim(0, 0.5)

    ax3 = axes[1, 0]
    es = summary['effect_sizes']

    comparisons = ['VDD vs Recency\n(Staleness)', 'VDD vs Static\n(Staleness)',
                   'VDD vs Recency\n(Accuracy)', 'VDD vs Static\n(Accuracy)']
    d_values = [es['vdd_vs_recency_staleness'], es['vdd_vs_static_staleness'],
                es['vdd_vs_recency_accuracy'], es['vdd_vs_static_accuracy']]

    bar_colors = []
    for i, d in enumerate(d_values):
        if i < 2:
            bar_colors.append('#27ae60' if d < 0 else '#c0392b')
        else:
            bar_colors.append('#27ae60' if d > 0 else '#c0392b')

    bars = ax3.barh(comparisons, d_values, color=bar_colors, alpha=0.8)
    ax3.axvline(x=0, color='black', linewidth=0.5)
    ax3.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)

    for bar, d in zip(bars, d_values):
        ax3.text(bar.get_width() + 0.05 if d >= 0 else bar.get_width() - 0.15,
                bar.get_y() + bar.get_height()/2,
                f'd={d:.2f}', va='center', fontweight='bold', fontsize=9)

    ax3.set_xlabel("Cohen's d (for staleness: negative=VDD better)")
    ax3.set_title('Effect Sizes')

    ax4 = axes[1, 1]
    vdd_best_staleness = summary['vdd']['overall_staleness'][0] < summary['recency']['overall_staleness'][0] and \
                         summary['vdd']['overall_staleness'][0] < summary['static']['overall_staleness'][0]

    text = f"""STALENESS-FOCUSED EVALUATION RESULTS
=====================================

Timeline: {config.n_timesteps} timesteps with version changes at t={config.version_transitions}

STALENESS RATES (lower = better):
  VDD:     {summary['vdd']['overall_staleness'][0]:.1%} [{summary['vdd']['overall_staleness'][1]:.1%}, {summary['vdd']['overall_staleness'][2]:.1%}]
  Recency: {summary['recency']['overall_staleness'][0]:.1%} [{summary['recency']['overall_staleness'][1]:.1%}, {summary['recency']['overall_staleness'][2]:.1%}]
  Static:  {summary['static']['overall_staleness'][0]:.1%} [{summary['static']['overall_staleness'][1]:.1%}, {summary['static']['overall_staleness'][2]:.1%}]

ACCURACY RATES (higher = better):
  VDD:     {summary['vdd']['overall_accuracy'][0]:.1%}
  Recency: {summary['recency']['overall_accuracy'][0]:.1%}
  Static:  {summary['static']['overall_accuracy'][0]:.1%}

EFFECT SIZES:
  Staleness: VDD vs Recency d={es['vdd_vs_recency_staleness']:+.2f}
             VDD vs Static  d={es['vdd_vs_static_staleness']:+.2f}

KEY FINDING:
{'✓ VDD has LOWEST staleness rate!' if vdd_best_staleness else 'VDD does not have lowest staleness'}

Insight: {'VDD reduces stale retrievals while maintaining competitive accuracy' if es['vdd_vs_recency_staleness'] < -0.2 else 'Results are close'}
"""
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if vdd_best_staleness else 'wheat', alpha=0.5))
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

def main():
    config = StalenessConfig()
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 20: STALENESS-FOCUSED EVALUATION")
    print("=" * 60)
    print(f"\nKey Question: Does VDD reduce STALE retrievals?")
    print(f"\nConfig:")
    print(f"  Facts: {config.n_facts}")
    print(f"  Version transitions at: t={config.version_transitions}")
    print()

    summary, all_results = run_experiment(config, n_runs=30)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nOverall Performance:")
    print(f"{'Method':<10} {'Accuracy':<25} {'Staleness':<25}")
    print("-" * 60)
    for method in ['vdd', 'recency', 'static']:
        acc = summary[method]['overall_accuracy']
        stale = summary[method]['overall_staleness']
        print(f"{method:<10} {acc[0]:.1%} [{acc[1]:.1%},{acc[2]:.1%}]  "
              f"{stale[0]:.1%} [{stale[1]:.1%},{stale[2]:.1%}]")

    print(f"\nEffect Sizes (Staleness - negative = VDD better):")
    print(f"  VDD vs Recency: d = {summary['effect_sizes']['vdd_vs_recency_staleness']:+.2f}")
    print(f"  VDD vs Static:  d = {summary['effect_sizes']['vdd_vs_static_staleness']:+.2f}")

    vdd_best = summary['vdd']['overall_staleness'][0] <= min(
        summary['recency']['overall_staleness'][0],
        summary['static']['overall_staleness'][0]
    )
    print(f"\n{'✓ VDD achieves lowest staleness!' if vdd_best else '✗ VDD does not have lowest staleness'}")

    plot_results(summary, config, results_dir / '20_staleness_focus.png')

    results_file = results_dir / '20_staleness_focus.json'
    save_data = {}
    for method in ['vdd', 'recency', 'static']:
        save_data[method] = {k: v for k, v in summary[method].items()
                            if k not in ['raw_staleness', 'raw_accuracy']}
    save_data['effect_sizes'] = summary['effect_sizes']

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"Results saved to {results_file}")

    return summary

if __name__ == "__main__":
    main()
