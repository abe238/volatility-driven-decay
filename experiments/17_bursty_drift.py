#!/usr/bin/env python3
"""
Experiment 17: Bursty Drift with Recovery
==========================================
Tests VDD's unique advantage: adaptive memory in environments with
irregular drift patterns (long stable → burst → new stable).

Hypothesis: VDD should win on ACCURACY because:
- Static (λ=0.1): Can't adapt during burst → stale answers
- Recency (λ=0.5): Over-forgets during stable periods → lost knowledge
- VDD: Adapts λ to volatility → high accuracy in ALL phases

Timeline:
  t=0-40:   Stable Era A (low volatility)
  t=40-50:  BURST - 80% of facts change rapidly
  t=50-100: Stable Era B (new stable, low volatility)
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

@dataclass
class BurstyDriftConfig:
    n_facts: int = 20
    n_timesteps: int = 100
    era_a_end: int = 40
    burst_end: int = 50
    burst_change_rate: float = 0.8
    n_queries_per_step: int = 5
    n_bootstrap: int = 1000
    seed: int = 42

class VDDController:
    def __init__(self, lambda_min=0.05, lambda_max=0.9, k=5.0, theta=0.3):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.k = k
        self.theta = theta
        self.volatility_history = []

    def compute_lambda(self, volatility: float) -> float:
        sigmoid = 1 / (1 + np.exp(-self.k * (volatility - self.theta)))
        return self.lambda_min + (self.lambda_max - self.lambda_min) * sigmoid

    def update(self, volatility: float) -> float:
        self.volatility_history.append(volatility)
        return self.compute_lambda(volatility)

class KnowledgeBase:
    def __init__(self, n_facts: int, seed: int):
        np.random.seed(seed)
        self.n_facts = n_facts
        self.documents: Dict[int, List[Dict]] = {i: [] for i in range(n_facts)}
        self.weights: Dict[int, List[float]] = {i: [] for i in range(n_facts)}
        self.current_version: Dict[int, int] = {i: 0 for i in range(n_facts)}

    def add_document(self, fact_id: int, version: int, initial_weight: float = 1.0):
        self.documents[fact_id].append({
            'version': version,
            'content': f"fact_{fact_id}_v{version}"
        })
        self.weights[fact_id].append(initial_weight)

    def apply_decay(self, fact_id: int, decay_rate: float):
        for i in range(len(self.weights[fact_id])):
            self.weights[fact_id][i] *= (1 - decay_rate)

    def retrieve(self, fact_id: int) -> int:
        if not self.documents[fact_id]:
            return -1
        weights = self.weights[fact_id]
        if sum(weights) == 0:
            return self.documents[fact_id][-1]['version']
        probs = np.array(weights) / sum(weights)
        idx = np.random.choice(len(weights), p=probs)
        return self.documents[fact_id][idx]['version']

    def get_ground_truth(self, fact_id: int) -> int:
        return self.current_version[fact_id]

def compute_volatility(kb: KnowledgeBase, changes_this_step: int) -> float:
    return changes_this_step / kb.n_facts

def run_single_simulation(config: BurstyDriftConfig, method: str, seed_offset: int = 0) -> Dict:
    np.random.seed(config.seed + seed_offset)
    kb = KnowledgeBase(config.n_facts, config.seed + seed_offset)

    for i in range(config.n_facts):
        kb.add_document(i, version=0, initial_weight=2.0)
        kb.current_version[i] = 0

    if method == 'vdd':
        controller = VDDController()
    elif method == 'recency':
        fixed_lambda = 0.5
    elif method == 'static':
        fixed_lambda = 0.1

    results = {
        'era_a_correct': 0, 'era_a_total': 0,
        'burst_correct': 0, 'burst_total': 0,
        'era_b_correct': 0, 'era_b_total': 0,
        'lambdas': [],
        'per_step_accuracy': []
    }

    for t in range(config.n_timesteps):
        changes_this_step = 0

        if config.era_a_end <= t < config.burst_end:
            for i in range(config.n_facts):
                if np.random.random() < config.burst_change_rate / (config.burst_end - config.era_a_end):
                    new_version = kb.current_version[i] + 1
                    kb.current_version[i] = new_version
                    kb.add_document(i, version=new_version, initial_weight=0.5)
                    changes_this_step += 1

        volatility = compute_volatility(kb, changes_this_step)

        if method == 'vdd':
            current_lambda = controller.update(volatility)
        else:
            current_lambda = fixed_lambda

        results['lambdas'].append(current_lambda)

        for i in range(config.n_facts):
            kb.apply_decay(i, current_lambda * 0.1)

        step_correct = 0
        queries = np.random.choice(config.n_facts, config.n_queries_per_step, replace=False)
        for fact_id in queries:
            retrieved = kb.retrieve(fact_id)
            ground_truth = kb.get_ground_truth(fact_id)
            is_correct = retrieved == ground_truth

            if is_correct:
                step_correct += 1

            if t < config.era_a_end:
                results['era_a_total'] += 1
                if is_correct:
                    results['era_a_correct'] += 1
            elif t < config.burst_end:
                results['burst_total'] += 1
                if is_correct:
                    results['burst_correct'] += 1
            else:
                results['era_b_total'] += 1
                if is_correct:
                    results['era_b_correct'] += 1

        results['per_step_accuracy'].append(step_correct / config.n_queries_per_step)

    results['era_a_accuracy'] = results['era_a_correct'] / results['era_a_total'] if results['era_a_total'] > 0 else 0
    results['burst_accuracy'] = results['burst_correct'] / results['burst_total'] if results['burst_total'] > 0 else 0
    results['era_b_accuracy'] = results['era_b_correct'] / results['era_b_total'] if results['era_b_total'] > 0 else 0
    results['cumulative_accuracy'] = (results['era_a_correct'] + results['burst_correct'] + results['era_b_correct']) / \
                                     (results['era_a_total'] + results['burst_total'] + results['era_b_total'])

    return results

def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    data = np.array(data)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
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

def run_experiment(config: BurstyDriftConfig, n_runs: int = 30) -> Dict:
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
        cumulative_accs = [r['cumulative_accuracy'] for r in all_results[method]]
        era_a_accs = [r['era_a_accuracy'] for r in all_results[method]]
        burst_accs = [r['burst_accuracy'] for r in all_results[method]]
        era_b_accs = [r['era_b_accuracy'] for r in all_results[method]]

        summary[method] = {
            'cumulative': bootstrap_ci(cumulative_accs, config.n_bootstrap),
            'era_a': bootstrap_ci(era_a_accs, config.n_bootstrap),
            'burst': bootstrap_ci(burst_accs, config.n_bootstrap),
            'era_b': bootstrap_ci(era_b_accs, config.n_bootstrap),
            'raw_cumulative': cumulative_accs,
            'avg_lambdas': np.mean([r['lambdas'] for r in all_results[method]], axis=0).tolist()
        }

    summary['effect_sizes'] = {
        'vdd_vs_recency': cohens_d(
            summary['vdd']['raw_cumulative'],
            summary['recency']['raw_cumulative']
        ),
        'vdd_vs_static': cohens_d(
            summary['vdd']['raw_cumulative'],
            summary['static']['raw_cumulative']
        )
    }

    return summary, all_results

def plot_results(summary: Dict, config: BurstyDriftConfig, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    methods = ['vdd', 'recency', 'static']
    colors = {'vdd': '#2ecc71', 'recency': '#e74c3c', 'static': '#3498db'}
    labels = {'vdd': 'VDD (adaptive)', 'recency': 'Recency (λ=0.5)', 'static': 'Static (λ=0.1)'}

    x = np.arange(4)
    width = 0.25

    for i, method in enumerate(methods):
        means = [summary[method]['era_a'][0], summary[method]['burst'][0],
                 summary[method]['era_b'][0], summary[method]['cumulative'][0]]
        lowers = [summary[method]['era_a'][1], summary[method]['burst'][1],
                  summary[method]['era_b'][1], summary[method]['cumulative'][1]]
        uppers = [summary[method]['era_a'][2], summary[method]['burst'][2],
                  summary[method]['era_b'][2], summary[method]['cumulative'][2]]
        errors = [[m-l for m, l in zip(means, lowers)], [u-m for m, u in zip(means, uppers)]]

        ax1.bar(x + i*width, means, width, label=labels[method], color=colors[method],
                yerr=errors, capsize=3, alpha=0.8)

    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Phase')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(['Era A\n(Stable)', 'Burst\n(High Drift)', 'Era B\n(New Stable)', 'Cumulative'])
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1.1)
    ax1.set_title('Accuracy by Phase (95% CI)')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    ax2 = axes[0, 1]
    timesteps = range(config.n_timesteps)
    for method in methods:
        ax2.plot(timesteps, summary[method]['avg_lambdas'], label=labels[method],
                 color=colors[method], linewidth=2)

    ax2.axvspan(0, config.era_a_end, alpha=0.1, color='green', label='Era A')
    ax2.axvspan(config.era_a_end, config.burst_end, alpha=0.2, color='red', label='Burst')
    ax2.axvspan(config.burst_end, config.n_timesteps, alpha=0.1, color='blue', label='Era B')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Decay Rate (λ)')
    ax2.set_title('Decay Rate Over Time')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 0.6)

    ax3 = axes[1, 0]
    effect_sizes = summary['effect_sizes']
    comparisons = ['VDD vs Recency', 'VDD vs Static']
    d_values = [effect_sizes['vdd_vs_recency'], effect_sizes['vdd_vs_static']]
    bar_colors = ['#27ae60' if d > 0 else '#c0392b' for d in d_values]

    bars = ax3.barh(comparisons, d_values, color=bar_colors, alpha=0.8)
    ax3.axvline(x=0, color='black', linewidth=0.5)
    ax3.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
    ax3.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium (0.5)')
    ax3.axvline(x=0.8, color='gray', linestyle='-.', alpha=0.5, label='Large (0.8)')
    ax3.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=-0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=-0.8, color='gray', linestyle='-.', alpha=0.5)

    for bar, d in zip(bars, d_values):
        ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'd={d:.2f}', va='center', fontweight='bold')

    ax3.set_xlabel("Cohen's d (positive = VDD better)")
    ax3.set_title('Effect Sizes: Cumulative Accuracy')
    ax3.set_xlim(-1.5, 1.5)

    ax4 = axes[1, 1]
    vdd_wins = summary['vdd']['cumulative'][0] > summary['recency']['cumulative'][0] and \
               summary['vdd']['cumulative'][0] > summary['static']['cumulative'][0]

    text = f"""BURSTY DRIFT EXPERIMENT RESULTS
================================

Timeline:
  Era A (t=0-{config.era_a_end}): Stable period
  Burst (t={config.era_a_end}-{config.burst_end}): {int(config.burst_change_rate*100)}% facts change
  Era B (t={config.burst_end}-{config.n_timesteps}): New stable period

Cumulative Accuracy (95% CI):
  VDD:     {summary['vdd']['cumulative'][0]:.1%} [{summary['vdd']['cumulative'][1]:.1%}, {summary['vdd']['cumulative'][2]:.1%}]
  Recency: {summary['recency']['cumulative'][0]:.1%} [{summary['recency']['cumulative'][1]:.1%}, {summary['recency']['cumulative'][2]:.1%}]
  Static:  {summary['static']['cumulative'][0]:.1%} [{summary['static']['cumulative'][1]:.1%}, {summary['static']['cumulative'][2]:.1%}]

Effect Sizes:
  VDD vs Recency: d = {effect_sizes['vdd_vs_recency']:+.2f}
  VDD vs Static:  d = {effect_sizes['vdd_vs_static']:+.2f}

Result: {'✓ VDD WINS on cumulative accuracy' if vdd_wins else '✗ VDD does not win'}
"""
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

def main():
    config = BurstyDriftConfig()
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 17: BURSTY DRIFT WITH RECOVERY")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Facts: {config.n_facts}")
    print(f"  Timeline: Era A (0-{config.era_a_end}) → Burst ({config.era_a_end}-{config.burst_end}) → Era B ({config.burst_end}-{config.n_timesteps})")
    print(f"  Burst change rate: {config.burst_change_rate:.0%}")
    print()

    summary, all_results = run_experiment(config, n_runs=30)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nAccuracy by Phase:")
    print(f"{'Method':<10} {'Era A':<20} {'Burst':<20} {'Era B':<20} {'Cumulative':<20}")
    print("-" * 90)
    for method in ['vdd', 'recency', 'static']:
        ea = summary[method]['era_a']
        bu = summary[method]['burst']
        eb = summary[method]['era_b']
        cu = summary[method]['cumulative']
        print(f"{method:<10} {ea[0]:.1%} [{ea[1]:.1%},{ea[2]:.1%}]  "
              f"{bu[0]:.1%} [{bu[1]:.1%},{bu[2]:.1%}]  "
              f"{eb[0]:.1%} [{eb[1]:.1%},{eb[2]:.1%}]  "
              f"{cu[0]:.1%} [{cu[1]:.1%},{cu[2]:.1%}]")

    print(f"\nEffect Sizes (Cumulative Accuracy):")
    print(f"  VDD vs Recency: d = {summary['effect_sizes']['vdd_vs_recency']:+.2f}")
    print(f"  VDD vs Static:  d = {summary['effect_sizes']['vdd_vs_static']:+.2f}")

    vdd_wins = summary['vdd']['cumulative'][0] > summary['recency']['cumulative'][0] and \
               summary['vdd']['cumulative'][0] > summary['static']['cumulative'][0]

    print(f"\n{'✓ VDD WINS ON ACCURACY' if vdd_wins else '✗ VDD does not win on accuracy'}")

    plot_results(summary, config, results_dir / '17_bursty_drift.png')

    results_file = results_dir / '17_bursty_drift.json'
    with open(results_file, 'w') as f:
        save_summary = {k: v for k, v in summary.items() if k != 'raw_cumulative'}
        for method in ['vdd', 'recency', 'static']:
            if 'raw_cumulative' in save_summary.get(method, {}):
                del save_summary[method]['raw_cumulative']
        json.dump(save_summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"Results saved to {results_file}")

    return summary

if __name__ == "__main__":
    main()
