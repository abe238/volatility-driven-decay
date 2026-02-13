#!/usr/bin/env python3
"""
Experiment 18: Reversion Scenario (Temporary Changes)
=====================================================
Tests when VDD should win on ACCURACY: when knowledge REVERTS
to its original state after temporary changes.

Real-world examples:
- API deprecation announcement → reversal due to community backlash
- Temporary security workaround → proper fix restores original behavior
- Feature flag rollout → rollback due to bugs
- COVID-era policy changes → return to pre-COVID policies

Hypothesis: VDD wins when knowledge RETURNS to original state because:
- Recency (λ=0.5): Aggressively forgets original knowledge during change period
- VDD: Preserves original knowledge with lower decay during stable periods
- When reversion happens, VDD still has original knowledge available

Timeline:
  t=0-50:   Original state (version 0)
  t=50-60:  Temporary change (version 1)
  t=60-100: Reversion to original (version 0 is correct again)
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

@dataclass
class ReversionConfig:
    n_facts: int = 20
    n_timesteps: int = 100
    original_end: int = 50
    change_end: int = 60
    change_rate: float = 0.7
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
        self.ground_truth: Dict[int, int] = {i: 0 for i in range(n_facts)}
        self.changed_facts: set = set()

    def add_document(self, fact_id: int, version: int, initial_weight: float = 1.0):
        self.documents[fact_id].append({
            'version': version,
            'content': f"fact_{fact_id}_v{version}"
        })
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

def compute_volatility(changes_this_step: int, n_facts: int) -> float:
    return changes_this_step / n_facts

def run_single_simulation(config: ReversionConfig, method: str, seed_offset: int = 0) -> Dict:
    np.random.seed(config.seed + seed_offset)
    kb = KnowledgeBase(config.n_facts, config.seed + seed_offset)

    for i in range(config.n_facts):
        kb.add_document(i, version=0, initial_weight=2.0)
        kb.ground_truth[i] = 0

    if method == 'vdd':
        controller = VDDController()
    elif method == 'recency':
        fixed_lambda = 0.5
    elif method == 'static':
        fixed_lambda = 0.1

    results = {
        'original_correct': 0, 'original_total': 0,
        'change_correct': 0, 'change_total': 0,
        'reversion_correct': 0, 'reversion_total': 0,
        'lambdas': [],
        'per_step_accuracy': []
    }

    for t in range(config.n_timesteps):
        changes_this_step = 0

        if config.original_end <= t < config.change_end:
            for i in range(config.n_facts):
                if i not in kb.changed_facts and np.random.random() < config.change_rate / (config.change_end - config.original_end):
                    kb.changed_facts.add(i)
                    kb.add_document(i, version=1, initial_weight=1.5)
                    kb.ground_truth[i] = 1
                    changes_this_step += 1

        elif t == config.change_end:
            for i in kb.changed_facts:
                kb.ground_truth[i] = 0
            changes_this_step = len(kb.changed_facts)

        volatility = compute_volatility(changes_this_step, config.n_facts)

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

            if t < config.original_end:
                results['original_total'] += 1
                if is_correct:
                    results['original_correct'] += 1
            elif t < config.change_end:
                results['change_total'] += 1
                if is_correct:
                    results['change_correct'] += 1
            else:
                results['reversion_total'] += 1
                if is_correct:
                    results['reversion_correct'] += 1

        results['per_step_accuracy'].append(step_correct / config.n_queries_per_step)

    results['original_accuracy'] = results['original_correct'] / results['original_total'] if results['original_total'] > 0 else 0
    results['change_accuracy'] = results['change_correct'] / results['change_total'] if results['change_total'] > 0 else 0
    results['reversion_accuracy'] = results['reversion_correct'] / results['reversion_total'] if results['reversion_total'] > 0 else 0
    results['cumulative_accuracy'] = (results['original_correct'] + results['change_correct'] + results['reversion_correct']) / \
                                     (results['original_total'] + results['change_total'] + results['reversion_total'])

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

def run_experiment(config: ReversionConfig, n_runs: int = 30) -> Dict:
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
        original_accs = [r['original_accuracy'] for r in all_results[method]]
        change_accs = [r['change_accuracy'] for r in all_results[method]]
        reversion_accs = [r['reversion_accuracy'] for r in all_results[method]]

        summary[method] = {
            'cumulative': bootstrap_ci(cumulative_accs, config.n_bootstrap),
            'original': bootstrap_ci(original_accs, config.n_bootstrap),
            'change': bootstrap_ci(change_accs, config.n_bootstrap),
            'reversion': bootstrap_ci(reversion_accs, config.n_bootstrap),
            'raw_cumulative': cumulative_accs,
            'raw_reversion': reversion_accs,
            'avg_lambdas': np.mean([r['lambdas'] for r in all_results[method]], axis=0).tolist()
        }

    summary['effect_sizes'] = {
        'vdd_vs_recency_cumulative': cohens_d(
            summary['vdd']['raw_cumulative'],
            summary['recency']['raw_cumulative']
        ),
        'vdd_vs_static_cumulative': cohens_d(
            summary['vdd']['raw_cumulative'],
            summary['static']['raw_cumulative']
        ),
        'vdd_vs_recency_reversion': cohens_d(
            summary['vdd']['raw_reversion'],
            summary['recency']['raw_reversion']
        ),
        'vdd_vs_static_reversion': cohens_d(
            summary['vdd']['raw_reversion'],
            summary['static']['raw_reversion']
        )
    }

    return summary, all_results

def plot_results(summary: Dict, config: ReversionConfig, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    methods = ['vdd', 'recency', 'static']
    colors = {'vdd': '#2ecc71', 'recency': '#e74c3c', 'static': '#3498db'}
    labels = {'vdd': 'VDD (adaptive)', 'recency': 'Recency (λ=0.5)', 'static': 'Static (λ=0.1)'}

    x = np.arange(4)
    width = 0.25

    for i, method in enumerate(methods):
        means = [summary[method]['original'][0], summary[method]['change'][0],
                 summary[method]['reversion'][0], summary[method]['cumulative'][0]]
        lowers = [summary[method]['original'][1], summary[method]['change'][1],
                  summary[method]['reversion'][1], summary[method]['cumulative'][1]]
        uppers = [summary[method]['original'][2], summary[method]['change'][2],
                  summary[method]['reversion'][2], summary[method]['cumulative'][2]]
        errors = [[m-l for m, l in zip(means, lowers)], [u-m for m, u in zip(means, uppers)]]

        ax1.bar(x + i*width, means, width, label=labels[method], color=colors[method],
                yerr=errors, capsize=3, alpha=0.8)

    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Phase')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(['Original\n(v0)', 'Temporary\nChange (v1)', 'Reversion\n(back to v0)', 'Cumulative'])
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1.1)
    ax1.set_title('Accuracy by Phase (95% CI) - REVERSION SCENARIO')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    ax2 = axes[0, 1]
    timesteps = range(config.n_timesteps)
    for method in methods:
        ax2.plot(timesteps, summary[method]['avg_lambdas'], label=labels[method],
                 color=colors[method], linewidth=2)

    ax2.axvspan(0, config.original_end, alpha=0.1, color='green', label='Original (v0)')
    ax2.axvspan(config.original_end, config.change_end, alpha=0.2, color='orange', label='Temp Change (v1)')
    ax2.axvspan(config.change_end, config.n_timesteps, alpha=0.1, color='blue', label='Reversion (v0)')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Decay Rate (λ)')
    ax2.set_title('Decay Rate Over Time')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 0.6)

    ax3 = axes[1, 0]
    effect_sizes = summary['effect_sizes']
    comparisons = ['VDD vs Recency\n(Reversion Phase)', 'VDD vs Static\n(Reversion Phase)',
                   'VDD vs Recency\n(Cumulative)', 'VDD vs Static\n(Cumulative)']
    d_values = [effect_sizes['vdd_vs_recency_reversion'], effect_sizes['vdd_vs_static_reversion'],
                effect_sizes['vdd_vs_recency_cumulative'], effect_sizes['vdd_vs_static_cumulative']]
    bar_colors = ['#27ae60' if d > 0 else '#c0392b' for d in d_values]

    bars = ax3.barh(comparisons, d_values, color=bar_colors, alpha=0.8)
    ax3.axvline(x=0, color='black', linewidth=0.5)
    ax3.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=0.8, color='gray', linestyle='-.', alpha=0.5)
    ax3.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=-0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=-0.8, color='gray', linestyle='-.', alpha=0.5)

    for bar, d in zip(bars, d_values):
        ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'd={d:.2f}', va='center', fontweight='bold')

    ax3.set_xlabel("Cohen's d (positive = VDD better)")
    ax3.set_title('Effect Sizes')
    ax3.set_xlim(-2, 2)

    ax4 = axes[1, 1]
    vdd_wins_reversion = summary['vdd']['reversion'][0] > summary['recency']['reversion'][0]

    text = f"""REVERSION SCENARIO EXPERIMENT RESULTS
======================================

Timeline:
  Original (t=0-{config.original_end}): Version 0 correct
  Temp Change (t={config.original_end}-{config.change_end}): {int(config.change_rate*100)}% facts change to v1
  Reversion (t={config.change_end}-{config.n_timesteps}): ALL facts RETURN to v0

KEY METRIC - Reversion Phase Accuracy (95% CI):
  VDD:     {summary['vdd']['reversion'][0]:.1%} [{summary['vdd']['reversion'][1]:.1%}, {summary['vdd']['reversion'][2]:.1%}]
  Recency: {summary['recency']['reversion'][0]:.1%} [{summary['recency']['reversion'][1]:.1%}, {summary['recency']['reversion'][2]:.1%}]
  Static:  {summary['static']['reversion'][0]:.1%} [{summary['static']['reversion'][1]:.1%}, {summary['static']['reversion'][2]:.1%}]

Reversion Phase Effect Sizes:
  VDD vs Recency: d = {effect_sizes['vdd_vs_recency_reversion']:+.2f}
  VDD vs Static:  d = {effect_sizes['vdd_vs_static_reversion']:+.2f}

Result: {'✓ VDD WINS on reversion accuracy!' if vdd_wins_reversion else '✗ VDD does not win'}

Insight: {'VDD preserved original knowledge better during temp change' if vdd_wins_reversion else 'Recency adapted faster to reversion'}
"""
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

def main():
    config = ReversionConfig()
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 18: REVERSION SCENARIO")
    print("=" * 60)
    print(f"\nHypothesis: VDD wins when knowledge REVERTS to original state")
    print(f"\nConfig:")
    print(f"  Facts: {config.n_facts}")
    print(f"  Timeline: Original (0-{config.original_end}) → Change ({config.original_end}-{config.change_end}) → Reversion ({config.change_end}-{config.n_timesteps})")
    print(f"  Change rate: {config.change_rate:.0%}")
    print()

    summary, all_results = run_experiment(config, n_runs=30)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nAccuracy by Phase:")
    print(f"{'Method':<10} {'Original':<20} {'Change':<20} {'Reversion':<20} {'Cumulative':<20}")
    print("-" * 90)
    for method in ['vdd', 'recency', 'static']:
        orig = summary[method]['original']
        chg = summary[method]['change']
        rev = summary[method]['reversion']
        cu = summary[method]['cumulative']
        print(f"{method:<10} {orig[0]:.1%} [{orig[1]:.1%},{orig[2]:.1%}]  "
              f"{chg[0]:.1%} [{chg[1]:.1%},{chg[2]:.1%}]  "
              f"{rev[0]:.1%} [{rev[1]:.1%},{rev[2]:.1%}]  "
              f"{cu[0]:.1%} [{cu[1]:.1%},{cu[2]:.1%}]")

    print(f"\nEffect Sizes:")
    print(f"  Reversion Phase:")
    print(f"    VDD vs Recency: d = {summary['effect_sizes']['vdd_vs_recency_reversion']:+.2f}")
    print(f"    VDD vs Static:  d = {summary['effect_sizes']['vdd_vs_static_reversion']:+.2f}")
    print(f"  Cumulative:")
    print(f"    VDD vs Recency: d = {summary['effect_sizes']['vdd_vs_recency_cumulative']:+.2f}")
    print(f"    VDD vs Static:  d = {summary['effect_sizes']['vdd_vs_static_cumulative']:+.2f}")

    vdd_wins_reversion = summary['vdd']['reversion'][0] > summary['recency']['reversion'][0]
    print(f"\n{'✓ VDD WINS ON REVERSION ACCURACY' if vdd_wins_reversion else '✗ VDD does not win on reversion accuracy'}")

    plot_results(summary, config, results_dir / '18_reversion.png')

    results_file = results_dir / '18_reversion.json'
    save_data = {}
    for method in ['vdd', 'recency', 'static']:
        save_data[method] = {k: v for k, v in summary[method].items()
                            if k not in ['raw_cumulative', 'raw_reversion']}
    save_data['effect_sizes'] = summary['effect_sizes']

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"Results saved to {results_file}")

    return summary

if __name__ == "__main__":
    main()
