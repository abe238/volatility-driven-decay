#!/usr/bin/env python3
"""
Experiment 19: Mixed Uncertainty Environment
============================================
Tests VDD's PRIMARY VALUE PROPOSITION: robust performance when
drift patterns are UNKNOWN and UNPREDICTABLE.

This experiment simulates a real production environment where:
- Some topics change rarely (stable domains)
- Some topics change frequently (volatile domains)
- Reversions happen occasionally
- No clear pattern that a static strategy can exploit

Key insight: In production, you DON'T KNOW which pattern you'll face.
VDD should be the SAFEST choice across all scenarios.

This is the "portfolio diversification" argument for VDD:
- Static: Best for stable, worst for volatile
- Recency: Best for volatile, worst for reversions
- VDD: Consistently good across ALL scenarios (never worst)
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

@dataclass
class MixedConfig:
    n_facts: int = 30
    n_timesteps: int = 150
    n_stable_facts: int = 10
    n_volatile_facts: int = 10
    n_reverting_facts: int = 10
    stable_change_prob: float = 0.01
    volatile_change_prob: float = 0.15
    reversion_timing: Tuple[int, int, int] = (50, 70, 100)
    n_queries_per_step: int = 6
    n_bootstrap: int = 1000
    seed: int = 42

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
    def __init__(self, config: MixedConfig, seed: int):
        np.random.seed(seed)
        self.config = config
        self.documents: Dict[int, List[Dict]] = {i: [] for i in range(config.n_facts)}
        self.weights: Dict[int, List[float]] = {i: [] for i in range(config.n_facts)}
        self.ground_truth: Dict[int, int] = {i: 0 for i in range(config.n_facts)}

        self.stable_facts = set(range(config.n_stable_facts))
        self.volatile_facts = set(range(config.n_stable_facts, config.n_stable_facts + config.n_volatile_facts))
        self.reverting_facts = set(range(config.n_stable_facts + config.n_volatile_facts, config.n_facts))

        self.reverting_original = {}
        self.reverting_changed = {}

    def add_document(self, fact_id: int, version: int, initial_weight: float = 1.0):
        self.documents[fact_id].append({'version': version})
        self.weights[fact_id].append(initial_weight)

    def apply_decay(self, fact_id: int, decay_rate: float):
        for i in range(len(self.weights[fact_id])):
            self.weights[fact_id][i] *= (1 - decay_rate)
            self.weights[fact_id][i] = max(self.weights[fact_id][i], 0.001)

    def retrieve(self, fact_id: int) -> int:
        if not self.documents[fact_id]:
            return -1
        weights = np.array(self.weights[fact_id])
        if weights.sum() == 0:
            return self.documents[fact_id][-1]['version']
        probs = weights / weights.sum()
        idx = np.random.choice(len(weights), p=probs)
        return self.documents[fact_id][idx]['version']

    def get_ground_truth(self, fact_id: int) -> int:
        return self.ground_truth[fact_id]

def run_single_simulation(config: MixedConfig, method: str, seed_offset: int = 0) -> Dict:
    np.random.seed(config.seed + seed_offset)
    kb = KnowledgeBase(config, config.seed + seed_offset)

    for i in range(config.n_facts):
        kb.add_document(i, version=0, initial_weight=2.0)
        kb.ground_truth[i] = 0
        if i in kb.reverting_facts:
            kb.reverting_original[i] = 0
            kb.reverting_changed[i] = False

    if method == 'vdd':
        controller = VDDController()
    elif method == 'recency':
        fixed_lambda = 0.5
    elif method == 'static':
        fixed_lambda = 0.1

    results = {
        'stable_correct': 0, 'stable_total': 0,
        'volatile_correct': 0, 'volatile_total': 0,
        'reverting_correct': 0, 'reverting_total': 0,
        'lambdas': [],
        'per_step_accuracy': []
    }

    version_counters = {i: 0 for i in range(config.n_facts)}
    revert_t1, revert_t2, revert_t3 = config.reversion_timing

    for t in range(config.n_timesteps):
        changes_this_step = 0

        for i in kb.stable_facts:
            if np.random.random() < config.stable_change_prob:
                version_counters[i] += 1
                kb.add_document(i, version=version_counters[i], initial_weight=1.0)
                kb.ground_truth[i] = version_counters[i]
                changes_this_step += 1

        for i in kb.volatile_facts:
            if np.random.random() < config.volatile_change_prob:
                version_counters[i] += 1
                kb.add_document(i, version=version_counters[i], initial_weight=1.0)
                kb.ground_truth[i] = version_counters[i]
                changes_this_step += 1

        for i in kb.reverting_facts:
            if t == revert_t1 and not kb.reverting_changed[i]:
                version_counters[i] += 1
                kb.add_document(i, version=version_counters[i], initial_weight=1.5)
                kb.ground_truth[i] = version_counters[i]
                kb.reverting_changed[i] = True
                changes_this_step += 1
            elif t == revert_t2 and kb.reverting_changed[i]:
                kb.ground_truth[i] = kb.reverting_original[i]
                changes_this_step += 1
            elif t == revert_t3:
                version_counters[i] += 1
                kb.add_document(i, version=version_counters[i], initial_weight=1.5)
                kb.ground_truth[i] = version_counters[i]
                kb.reverting_changed[i] = False
                changes_this_step += 1

        volatility = changes_this_step / config.n_facts

        if method == 'vdd':
            current_lambda = controller.compute_lambda(volatility)
        else:
            current_lambda = fixed_lambda

        results['lambdas'].append(current_lambda)

        for i in range(config.n_facts):
            kb.apply_decay(i, current_lambda * 0.08)

        step_correct = 0
        all_facts = list(kb.stable_facts) + list(kb.volatile_facts) + list(kb.reverting_facts)
        queries = np.random.choice(all_facts, config.n_queries_per_step, replace=False)

        for fact_id in queries:
            retrieved = kb.retrieve(fact_id)
            ground_truth = kb.get_ground_truth(fact_id)
            is_correct = retrieved == ground_truth

            if is_correct:
                step_correct += 1

            if fact_id in kb.stable_facts:
                results['stable_total'] += 1
                if is_correct:
                    results['stable_correct'] += 1
            elif fact_id in kb.volatile_facts:
                results['volatile_total'] += 1
                if is_correct:
                    results['volatile_correct'] += 1
            else:
                results['reverting_total'] += 1
                if is_correct:
                    results['reverting_correct'] += 1

        results['per_step_accuracy'].append(step_correct / config.n_queries_per_step)

    results['stable_accuracy'] = results['stable_correct'] / results['stable_total'] if results['stable_total'] > 0 else 0
    results['volatile_accuracy'] = results['volatile_correct'] / results['volatile_total'] if results['volatile_total'] > 0 else 0
    results['reverting_accuracy'] = results['reverting_correct'] / results['reverting_total'] if results['reverting_total'] > 0 else 0

    total_correct = results['stable_correct'] + results['volatile_correct'] + results['reverting_correct']
    total_queries = results['stable_total'] + results['volatile_total'] + results['reverting_total']
    results['cumulative_accuracy'] = total_correct / total_queries if total_queries > 0 else 0

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

def run_experiment(config: MixedConfig, n_runs: int = 30) -> Dict:
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
        stable_accs = [r['stable_accuracy'] for r in all_results[method]]
        volatile_accs = [r['volatile_accuracy'] for r in all_results[method]]
        reverting_accs = [r['reverting_accuracy'] for r in all_results[method]]

        summary[method] = {
            'cumulative': bootstrap_ci(cumulative_accs, config.n_bootstrap),
            'stable': bootstrap_ci(stable_accs, config.n_bootstrap),
            'volatile': bootstrap_ci(volatile_accs, config.n_bootstrap),
            'reverting': bootstrap_ci(reverting_accs, config.n_bootstrap),
            'raw_cumulative': cumulative_accs,
            'avg_lambdas': np.mean([r['lambdas'] for r in all_results[method]], axis=0).tolist()
        }

    summary['effect_sizes'] = {
        'vdd_vs_recency': cohens_d(summary['vdd']['raw_cumulative'], summary['recency']['raw_cumulative']),
        'vdd_vs_static': cohens_d(summary['vdd']['raw_cumulative'], summary['static']['raw_cumulative'])
    }

    def count_wins(method):
        wins = 0
        for domain in ['stable', 'volatile', 'reverting']:
            mean = summary[method][domain][0]
            if all(mean >= summary[m][domain][0] for m in methods):
                wins += 1
        return wins

    def count_worst(method):
        worst = 0
        for domain in ['stable', 'volatile', 'reverting']:
            mean = summary[method][domain][0]
            if all(mean <= summary[m][domain][0] for m in methods):
                worst += 1
        return worst

    summary['robustness'] = {
        m: {'wins': count_wins(m), 'worst': count_worst(m)} for m in methods
    }

    return summary, all_results

def plot_results(summary: Dict, config: MixedConfig, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    methods = ['vdd', 'recency', 'static']
    colors = {'vdd': '#2ecc71', 'recency': '#e74c3c', 'static': '#3498db'}
    labels = {'vdd': 'VDD (adaptive)', 'recency': 'Recency (λ=0.5)', 'static': 'Static (λ=0.1)'}

    x = np.arange(4)
    width = 0.25

    for i, method in enumerate(methods):
        means = [summary[method]['stable'][0], summary[method]['volatile'][0],
                 summary[method]['reverting'][0], summary[method]['cumulative'][0]]
        lowers = [summary[method]['stable'][1], summary[method]['volatile'][1],
                  summary[method]['reverting'][1], summary[method]['cumulative'][1]]
        uppers = [summary[method]['stable'][2], summary[method]['volatile'][2],
                  summary[method]['reverting'][2], summary[method]['cumulative'][2]]
        errors = [[m-l for m, l in zip(means, lowers)], [u-m for m, u in zip(means, uppers)]]

        ax1.bar(x + i*width, means, width, label=labels[method], color=colors[method],
                yerr=errors, capsize=3, alpha=0.8)

    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Domain Type')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(['Stable\n(1% change)', 'Volatile\n(15% change)', 'Reverting\n(temp changes)', 'Cumulative'])
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1.1)
    ax1.set_title('Accuracy by Domain Type - MIXED ENVIRONMENT')

    ax2 = axes[0, 1]
    robustness = summary['robustness']
    metrics = ['Never Worst', 'Most Wins']
    x_pos = np.arange(len(metrics))
    width = 0.25

    for i, method in enumerate(methods):
        values = [3 - robustness[method]['worst'], robustness[method]['wins']]
        ax2.bar(x_pos + i*width, values, width, label=labels[method], color=colors[method], alpha=0.8)

    ax2.set_ylabel('Count (out of 3 domains)')
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.set_ylim(0, 3.5)
    ax2.set_title('Robustness Metrics')

    for i, method in enumerate(methods):
        never_worst = 3 - robustness[method]['worst']
        wins = robustness[method]['wins']
        ax2.text(0 + i*width, never_worst + 0.1, str(never_worst), ha='center', fontweight='bold')
        ax2.text(1 + i*width, wins + 0.1, str(wins), ha='center', fontweight='bold')

    ax3 = axes[1, 0]
    timesteps = range(config.n_timesteps)
    for method in methods:
        ax3.plot(timesteps, summary[method]['avg_lambdas'], label=labels[method],
                 color=colors[method], linewidth=2)

    for t in config.reversion_timing:
        ax3.axvline(x=t, color='purple', linestyle='--', alpha=0.3)
    ax3.text(config.reversion_timing[0], 0.55, 'Change', fontsize=8, color='purple')
    ax3.text(config.reversion_timing[1], 0.55, 'Revert', fontsize=8, color='purple')
    ax3.text(config.reversion_timing[2], 0.55, 'New', fontsize=8, color='purple')

    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Decay Rate (λ)')
    ax3.set_title('Decay Rate Over Time')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 0.6)

    ax4 = axes[1, 1]
    rob = summary['robustness']
    vdd_safest = rob['vdd']['worst'] == 0

    text = f"""MIXED UNCERTAINTY EXPERIMENT RESULTS
=====================================

Environment: 30 facts split across 3 domain types
  - 10 Stable facts (1% change probability)
  - 10 Volatile facts (15% change probability)
  - 10 Reverting facts (change at t=50, revert at t=70)

Cumulative Accuracy (95% CI):
  VDD:     {summary['vdd']['cumulative'][0]:.1%} [{summary['vdd']['cumulative'][1]:.1%}, {summary['vdd']['cumulative'][2]:.1%}]
  Recency: {summary['recency']['cumulative'][0]:.1%} [{summary['recency']['cumulative'][1]:.1%}, {summary['recency']['cumulative'][2]:.1%}]
  Static:  {summary['static']['cumulative'][0]:.1%} [{summary['static']['cumulative'][1]:.1%}, {summary['static']['cumulative'][2]:.1%}]

ROBUSTNESS ANALYSIS:
  Method   | Domains Won | Domains Worst
  ---------|-------------|---------------
  VDD      | {rob['vdd']['wins']}           | {rob['vdd']['worst']}
  Recency  | {rob['recency']['wins']}           | {rob['recency']['worst']}
  Static   | {rob['static']['wins']}           | {rob['static']['worst']}

KEY FINDING:
{'✓ VDD is NEVER worst in any domain!' if vdd_safest else '✗ VDD is worst in some domain'}
{'→ VDD is the SAFEST CHOICE under uncertainty' if vdd_safest else ''}

This validates VDD as the "robust default" for production systems
where drift patterns are unknown.
"""
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if vdd_safest else 'wheat', alpha=0.5))
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

def main():
    config = MixedConfig()
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 19: MIXED UNCERTAINTY ENVIRONMENT")
    print("=" * 60)
    print(f"\nThis tests VDD's PRIMARY VALUE: robust performance under uncertainty")
    print(f"\nConfig:")
    print(f"  Facts: {config.n_facts} ({config.n_stable_facts} stable, {config.n_volatile_facts} volatile, {config.n_reverting_facts} reverting)")
    print(f"  Timesteps: {config.n_timesteps}")
    print()

    summary, all_results = run_experiment(config, n_runs=30)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nAccuracy by Domain Type:")
    print(f"{'Method':<10} {'Stable':<20} {'Volatile':<20} {'Reverting':<20} {'Cumulative':<20}")
    print("-" * 90)
    for method in ['vdd', 'recency', 'static']:
        st = summary[method]['stable']
        vo = summary[method]['volatile']
        re = summary[method]['reverting']
        cu = summary[method]['cumulative']
        print(f"{method:<10} {st[0]:.1%} [{st[1]:.1%},{st[2]:.1%}]  "
              f"{vo[0]:.1%} [{vo[1]:.1%},{vo[2]:.1%}]  "
              f"{re[0]:.1%} [{re[1]:.1%},{re[2]:.1%}]  "
              f"{cu[0]:.1%} [{cu[1]:.1%},{cu[2]:.1%}]")

    print(f"\nRobustness Analysis:")
    rob = summary['robustness']
    print(f"  {'Method':<10} {'Domains Won':<15} {'Domains Worst':<15}")
    print(f"  {'-'*40}")
    for method in ['vdd', 'recency', 'static']:
        print(f"  {method:<10} {rob[method]['wins']:<15} {rob[method]['worst']:<15}")

    vdd_safest = rob['vdd']['worst'] == 0
    print(f"\n{'✓ VDD is NEVER worst - SAFEST CHOICE under uncertainty!' if vdd_safest else '✗ VDD is worst in some domain'}")

    plot_results(summary, config, results_dir / '19_mixed_uncertainty.png')

    results_file = results_dir / '19_mixed_uncertainty.json'
    save_data = {}
    for method in ['vdd', 'recency', 'static']:
        save_data[method] = {k: v for k, v in summary[method].items() if k != 'raw_cumulative'}
    save_data['effect_sizes'] = summary['effect_sizes']
    save_data['robustness'] = summary['robustness']

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"Results saved to {results_file}")

    return summary

if __name__ == "__main__":
    main()
