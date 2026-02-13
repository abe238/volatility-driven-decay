#!/usr/bin/env python3
"""
Experiment 26: Precision Analysis and Mitigation

Characterizes the 1% precision issue and tests mitigations:
1. Baseline: confirms ~1% precision with default parameters
2. Threshold tuning: higher drift_threshold reduces false activations
3. Window tuning: larger arch_window smooths archive centroid
4. Combined: threshold + window + external smoothing

Key insight: the 1% precision is structurally expected because
post-drift periods SHOULD show elevated volatility (cleaning up
stale documents). The experiment separates "true false positives"
(noise during genuinely stable periods) from "expected elevated
activity" (post-drift cleanup).
"""

import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vdd.drift_detection import EmbeddingDistance
from utils.statistics import bootstrap_ci

EMBEDDING_DIM = 64


def generate_scenario(n_steps: int = 600, seed: int = 42):
    np.random.seed(seed)
    drift_points = [200, 400]
    drift_window = 5
    cleanup_window = 50

    labels = []
    for t in range(n_steps):
        in_drift = any(dp <= t < dp + drift_window for dp in drift_points)
        in_cleanup = any(dp + drift_window <= t < dp + cleanup_window for dp in drift_points)
        if in_drift:
            labels.append("drift")
        elif in_cleanup:
            labels.append("cleanup")
        else:
            labels.append("stable")

    current_centroid = np.random.randn(EMBEDDING_DIM)
    current_centroid /= np.linalg.norm(current_centroid)

    embeddings = []
    for t in range(n_steps):
        if t in drift_points:
            shift = np.random.randn(EMBEDDING_DIM) * 0.8
            current_centroid = current_centroid + shift
            current_centroid /= np.linalg.norm(current_centroid)

        noise = np.random.randn(EMBEDDING_DIM) * 0.05
        emb = current_centroid + noise
        emb /= np.linalg.norm(emb)
        embeddings.append(emb)

    return embeddings, labels, drift_points


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def run_config(embeddings, labels, curr_window=10, arch_window=100,
               drift_threshold=0.3, ext_alpha=1.0,
               lambda_base=0.15, lambda_max=0.85):
    detector = EmbeddingDistance(
        curr_window=curr_window, arch_window=arch_window,
        drift_threshold=drift_threshold
    )

    smoothed_vol = 0.0
    margin = 0.05
    lambdas = []

    for emb in embeddings:
        result = detector.update(emb)
        raw_vol = result.volatility

        if ext_alpha >= 1.0:
            smoothed_vol = raw_vol
        else:
            smoothed_vol = ext_alpha * raw_vol + (1 - ext_alpha) * smoothed_vol

        lam = lambda_base + (lambda_max - lambda_base) * sigmoid(10 * (smoothed_vol - 0.5))
        lambdas.append(lam)

    lambdas = np.array(lambdas)
    labels_arr = np.array(labels)

    elevated = lambdas > (lambda_base + margin)

    stable_mask = labels_arr == "stable"
    drift_mask = labels_arr == "drift"
    cleanup_mask = labels_arr == "cleanup"

    stable_false_act = np.mean(elevated[stable_mask]) if stable_mask.any() else 0
    cleanup_elevated = np.mean(elevated[cleanup_mask]) if cleanup_mask.any() else 0
    drift_elevated = np.mean(elevated[drift_mask]) if drift_mask.any() else 0

    mean_stable_lam = float(np.mean(lambdas[stable_mask])) if stable_mask.any() else 0
    mean_cleanup_lam = float(np.mean(lambdas[cleanup_mask])) if cleanup_mask.any() else 0
    mean_drift_lam = float(np.mean(lambdas[drift_mask])) if drift_mask.any() else 0

    tp = np.sum(elevated & (drift_mask | cleanup_mask))
    fp = np.sum(elevated & stable_mask)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        "stable_false_act": float(stable_false_act),
        "cleanup_elevated": float(cleanup_elevated),
        "drift_elevated": float(drift_elevated),
        "precision": float(precision),
        "mean_stable_lambda": mean_stable_lam,
        "mean_cleanup_lambda": mean_cleanup_lam,
        "mean_drift_lambda": mean_drift_lam,
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 26: PRECISION ANALYSIS AND MITIGATION")
    print("Separating true false positives from expected post-drift activity")
    print("=" * 70)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    configs = {
        "baseline":      {"curr_window": 10, "arch_window": 100, "drift_threshold": 0.3, "ext_alpha": 1.0},
        "high_thresh":   {"curr_window": 10, "arch_window": 100, "drift_threshold": 0.5, "ext_alpha": 1.0},
        "large_archive": {"curr_window": 10, "arch_window": 500, "drift_threshold": 0.3, "ext_alpha": 1.0},
        "smoothed":      {"curr_window": 10, "arch_window": 100, "drift_threshold": 0.3, "ext_alpha": 0.3},
        "tuned":         {"curr_window": 5,  "arch_window": 200, "drift_threshold": 0.4, "ext_alpha": 1.0},
        "full_combo":    {"curr_window": 5,  "arch_window": 200, "drift_threshold": 0.4, "ext_alpha": 0.3},
    }

    n_runs = 30
    results = {name: {
        "stable_false_act": [], "cleanup_elevated": [], "drift_elevated": [],
        "precision": [], "mean_stable_lambda": [], "mean_cleanup_lambda": [],
        "mean_drift_lambda": [],
    } for name in configs}

    for seed in range(100, 100 + n_runs):
        embeddings, labels, drift_pts = generate_scenario(n_steps=600, seed=seed)
        for name, cfg in configs.items():
            trial = run_config(embeddings, labels, **cfg)
            for k in trial:
                results[name][k].append(trial[k])

    print(f"\n{'Config':<16} {'StableFP%':<11} {'Cleanup%':<10} {'Drift%':<9} "
          f"{'Precision':<11} {'λ_stable':<10} {'λ_cleanup':<10} {'λ_drift':<10}")
    print("-" * 97)

    output = {}
    for name in configs:
        d = results[name]
        sfp = np.mean(d["stable_false_act"])
        ce = np.mean(d["cleanup_elevated"])
        de = np.mean(d["drift_elevated"])
        p = np.mean(d["precision"])
        ls = np.mean(d["mean_stable_lambda"])
        lc = np.mean(d["mean_cleanup_lambda"])
        ld = np.mean(d["mean_drift_lambda"])

        print(f"{name:<16} {sfp*100:>6.1f}%    {ce*100:>6.1f}%   {de*100:>5.1f}%   "
              f"{p:.3f}      {ls:.3f}     {lc:.3f}     {ld:.3f}")

        output[name] = {
            "config": configs[name],
            "stable_false_act_mean": round(sfp, 4),
            "stable_false_act_std": round(float(np.std(d["stable_false_act"])), 4),
            "cleanup_elevated_mean": round(ce, 4),
            "drift_elevated_mean": round(de, 4),
            "precision_mean": round(p, 4),
            "precision_std": round(float(np.std(d["precision"])), 4),
            "precision_ci": [round(x, 4) for x in bootstrap_ci(np.array(d["precision"]))],
            "mean_stable_lambda": round(ls, 4),
            "mean_cleanup_lambda": round(lc, 4),
            "mean_drift_lambda": round(ld, 4),
        }

    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    baseline_fp = np.mean(results["baseline"]["stable_false_act"])
    baseline_p = np.mean(results["baseline"]["precision"])
    combo_fp = np.mean(results["full_combo"]["stable_false_act"])
    combo_p = np.mean(results["full_combo"]["precision"])
    tuned_fp = np.mean(results["tuned"]["stable_false_act"])
    tuned_p = np.mean(results["tuned"]["precision"])

    print(f"  Baseline: {baseline_fp*100:.1f}% stable false activations, precision={baseline_p:.3f}")
    print(f"  Tuned:    {tuned_fp*100:.1f}% stable false activations, precision={tuned_p:.3f}")
    print(f"  Combined: {combo_fp*100:.1f}% stable false activations, precision={combo_p:.3f}")

    if baseline_fp > 0:
        fp_reduction = (baseline_fp - combo_fp) / baseline_fp * 100
        print(f"  False positive reduction: {fp_reduction:.0f}%")
    if combo_p > baseline_p and baseline_p > 0:
        p_improvement = (combo_p - baseline_p) / baseline_p * 100
        print(f"  Precision improvement: {p_improvement:.0f}%")

    cleanup_baseline = np.mean(results["baseline"]["cleanup_elevated"])
    cleanup_combo = np.mean(results["full_combo"]["cleanup_elevated"])
    print(f"\n  Post-drift cleanup: baseline={cleanup_baseline*100:.1f}% vs combo={cleanup_combo*100:.1f}% elevated")
    print(f"  (Elevated lambda during cleanup is DESIRABLE — removes stale docs)")

    output["_metadata"] = {
        "n_runs": n_runs, "n_steps": 600, "drift_points": [200, 400],
        "drift_window": 5, "cleanup_window": 50, "seeds": "100-129",
    }

    with open(results_dir / "26_smoothing_precision.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/26_smoothing_precision.json")

    plot_results(output, configs, results_dir)

    # Generate example traces
    embeddings, labels, drift_pts = generate_scenario(n_steps=600, seed=42)
    plot_traces(embeddings, labels, drift_pts, configs, results_dir)

    return output


def plot_results(output, configs, results_dir):
    names = list(configs.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']

    fp_vals = [output[n]["stable_false_act_mean"] * 100 for n in names]
    axes[0].barh(names, fp_vals, color=colors, alpha=0.8)
    axes[0].set_xlabel("Stable False Activation (%)")
    axes[0].set_title("False Positives During Stable Periods\n(Lower = Better)")
    for i, v in enumerate(fp_vals):
        axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

    p_vals = [output[n]["precision_mean"] for n in names]
    axes[1].barh(names, p_vals, color=colors, alpha=0.8)
    axes[1].set_xlabel("Precision")
    axes[1].set_title("Drift+Cleanup Precision\n(Higher = Better)")
    for i, v in enumerate(p_vals):
        axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

    lambda_data = {
        'Stable': [output[n]["mean_stable_lambda"] for n in names],
        'Cleanup': [output[n]["mean_cleanup_lambda"] for n in names],
        'Drift': [output[n]["mean_drift_lambda"] for n in names],
    }
    x = np.arange(len(names))
    w = 0.25
    for i, (label, vals) in enumerate(lambda_data.items()):
        axes[2].bar(x + (i - 1) * w, vals, w, label=label, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[2].set_ylabel("Mean λ")
    axes[2].set_title("Lambda by Period Type")
    axes[2].legend()

    plt.suptitle("Experiment 26: Precision Analysis and Mitigation Strategies",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "26_smoothing_precision.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to results/26_smoothing_precision.png")
    plt.close()


def plot_traces(embeddings, labels, drift_pts, configs, results_dir):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    trace_configs = {
        "baseline": configs["baseline"],
        "tuned": configs["tuned"],
        "full_combo": configs["full_combo"],
    }
    trace_colors = {"baseline": "#e74c3c", "tuned": "#3498db", "full_combo": "#2ecc71"}

    for name, cfg in trace_configs.items():
        detector = EmbeddingDistance(
            curr_window=cfg["curr_window"], arch_window=cfg["arch_window"],
            drift_threshold=cfg["drift_threshold"]
        )
        ext_alpha = cfg["ext_alpha"]
        smoothed_vol = 0.0
        vols = []
        lams = []

        for emb in embeddings:
            result = detector.update(emb)
            raw_vol = result.volatility
            if ext_alpha >= 1.0:
                smoothed_vol = raw_vol
            else:
                smoothed_vol = ext_alpha * raw_vol + (1 - ext_alpha) * smoothed_vol
            vols.append(smoothed_vol)
            lam = 0.15 + 0.70 * sigmoid(10 * (smoothed_vol - 0.5))
            lams.append(lam)

        t = range(len(embeddings))
        axes[0].plot(t, vols, label=name, color=trace_colors[name], alpha=0.7, linewidth=1)
        axes[1].plot(t, lams, label=name, color=trace_colors[name], alpha=0.7, linewidth=1)

    for dp in drift_pts:
        for ax in axes[:2]:
            ax.axvline(dp, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axvspan(dp, dp + 50, alpha=0.1, color='orange')

    axes[0].set_ylabel("Volatility V(t)")
    axes[0].set_title("Volatility Signal")
    axes[0].legend(fontsize=9)

    axes[1].set_ylabel("Decay Rate λ(t)")
    axes[1].set_title("VDD Lambda")
    axes[1].axhline(0.15, color='black', linestyle=':', alpha=0.3, label='λ_base')
    axes[1].legend(fontsize=9)

    label_colors = {"stable": "#2ecc71", "cleanup": "#f39c12", "drift": "#e74c3c"}
    label_vals = [label_colors.get(l, "#ccc") for l in labels]
    for t in range(len(labels)):
        axes[2].axvspan(t, t+1, alpha=0.3, color=label_vals[t], linewidth=0)
    axes[2].set_ylabel("Period")
    axes[2].set_xlabel("Timestep")
    axes[2].set_title("Ground Truth (green=stable, orange=cleanup, red=drift)")
    axes[2].set_yticks([])

    plt.suptitle("Exp 26: Lambda Traces — Baseline vs Tuned vs Combined",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "26_smoothing_traces.png", dpi=150, bbox_inches='tight')
    print(f"Saved trace plot to results/26_smoothing_traces.png")
    plt.close()


if __name__ == "__main__":
    main()
