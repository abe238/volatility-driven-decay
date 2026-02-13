#!/usr/bin/env python3
"""
Run all VDD experiments for reproducibility.

Usage:
    python run_experiments.py --all           # Run all experiments
    python run_experiments.py --core          # Run core experiments (1-8)
    python run_experiments.py --extended      # Run extended experiments (9-15)
    python run_experiments.py --realworld     # Run real-world experiments (16-20)
    python run_experiments.py -e 2 5 16       # Run specific experiments
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Experiment definitions
EXPERIMENTS = {
    # Core validation (1-8)
    1: ("01_drift_detection_comparison.py", "Drift Detection Comparison", 30),
    2: ("02_scalar_simulation_fixed.py", "VDD with Real Detection", 20),
    3: ("03_vector_memory_test.py", "Vector Memory Bank", 45),
    4: ("04_mixed_drift.py", "Mixed-Drift Scenario", 60),
    5: ("05_ablation_study.py", "Ablation Study", 120),
    6: ("06_stability_analysis.py", "Stability Analysis (10K)", 90),
    7: ("07_latency_benchmark.py", "Latency Benchmark", 30),
    8: ("08_baseline_comparison.py", "Baseline Comparison", 60),

    # Extended validation (9-15)
    9: ("09_precision_analysis.py", "Precision Analysis", 45),
    10: ("10_detection_comparison.py", "Detection Method Comparison", 60),
    12: ("12_gradual_drift.py", "Gradual Drift Patterns", 45),
    13: ("13_adwin_comparison.py", "ADWIN Comparison", 60),
    14: ("14_scaling_analysis.py", "Computational Scaling", 90),
    15: ("15_statistical_validation.py", "Statistical Validation (20-fold)", 120),

    # Real-world validation (16-20)
    16: ("16_real_rag.py", "Real-World RAG (React Docs)", 180),
    17: ("17_bursty_drift.py", "Bursty Drift Patterns", 60),
    18: ("18_reversion_scenario.py", "Reversion Scenarios", 60),
    19: ("19_mixed_uncertainty.py", "Mixed Uncertainty", 60),
    20: ("20_staleness_focus.py", "Staleness-Focused Evaluation", 60),
}

CORE = [1, 2, 3, 4, 5, 6, 7, 8]
EXTENDED = [9, 10, 12, 13, 14, 15]
REALWORLD = [16, 17, 18, 19, 20]


def run_experiment(exp_num: int, exp_dir: Path) -> tuple[bool, float]:
    """Run a single experiment and return (success, duration)."""
    if exp_num not in EXPERIMENTS:
        print(f"  ❌ Experiment {exp_num} not found")
        return False, 0.0

    filename, description, est_time = EXPERIMENTS[exp_num]
    exp_path = exp_dir / filename

    if not exp_path.exists():
        print(f"  ❌ File not found: {exp_path}")
        return False, 0.0

    print(f"\n{'='*60}")
    print(f"Experiment {exp_num}: {description}")
    print(f"File: {filename}")
    print(f"Estimated time: {est_time}s")
    print('='*60)

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(exp_path)],
            capture_output=False,
            text=True,
            timeout=est_time * 3  # 3x timeout buffer
        )
        duration = time.time() - start

        if result.returncode == 0:
            print(f"  ✅ Completed in {duration:.1f}s")
            return True, duration
        else:
            print(f"  ❌ Failed with exit code {result.returncode}")
            return False, duration

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        print(f"  ⏰ Timeout after {duration:.1f}s")
        return False, duration
    except Exception as e:
        duration = time.time() - start
        print(f"  ❌ Error: {e}")
        return False, duration


def main():
    parser = argparse.ArgumentParser(
        description="Run VDD experiments for reproducibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_experiments.py --all           # Run all 19 experiments
    python run_experiments.py --core          # Run core experiments (1-8)
    python run_experiments.py --extended      # Run extended experiments (9-15)
    python run_experiments.py --realworld     # Run real-world experiments (16-20)
    python run_experiments.py -e 2 5 16       # Run specific experiments
    python run_experiments.py --list          # List all experiments
        """
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--core", action="store_true", help="Run core experiments (1-8)")
    parser.add_argument("--extended", action="store_true", help="Run extended experiments (9-15)")
    parser.add_argument("--realworld", action="store_true", help="Run real-world experiments (16-20)")
    parser.add_argument("-e", "--experiments", type=int, nargs="+", help="Specific experiment numbers")
    parser.add_argument("--list", action="store_true", help="List all experiments")

    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\nAvailable Experiments:")
        print("-" * 70)
        for num, (filename, desc, est) in sorted(EXPERIMENTS.items()):
            print(f"  {num:2d}. {desc:<40} ({est}s est.)")
        print("-" * 70)
        print(f"\nCore (1-8): {sum(EXPERIMENTS[n][2] for n in CORE)}s total")
        print(f"Extended (9-15): {sum(EXPERIMENTS[n][2] for n in EXTENDED)}s total")
        print(f"Real-world (16-20): {sum(EXPERIMENTS[n][2] for n in REALWORLD)}s total")
        print(f"All: {sum(e[2] for e in EXPERIMENTS.values())}s total (~{sum(e[2] for e in EXPERIMENTS.values())//60} min)")
        return

    # Determine which experiments to run
    to_run = []
    if args.all:
        to_run = sorted(EXPERIMENTS.keys())
    elif args.core:
        to_run = CORE
    elif args.extended:
        to_run = EXTENDED
    elif args.realworld:
        to_run = REALWORLD
    elif args.experiments:
        to_run = args.experiments
    else:
        parser.print_help()
        return

    # Find experiments directory
    exp_dir = Path(__file__).parent / "experiments"
    if not exp_dir.exists():
        print(f"Error: experiments directory not found at {exp_dir}")
        sys.exit(1)

    # Run experiments
    print(f"\n🔬 VDD Experiment Runner")
    print(f"Running {len(to_run)} experiments: {to_run}")

    total_start = time.time()
    results = []

    for exp_num in to_run:
        success, duration = run_experiment(exp_num, exp_dir)
        results.append((exp_num, success, duration))

    # Summary
    total_duration = time.time() - total_start
    successful = sum(1 for _, s, _ in results if s)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for exp_num, success, duration in results:
        status = "✅" if success else "❌"
        desc = EXPERIMENTS[exp_num][1] if exp_num in EXPERIMENTS else "Unknown"
        print(f"  {status} Exp {exp_num:2d}: {desc:<35} ({duration:.1f}s)")

    print('='*60)
    print(f"Total: {successful}/{len(results)} successful in {total_duration:.1f}s")

    if successful < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
