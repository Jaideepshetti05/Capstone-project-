"""
Script to execute the 6 production robustness and Byzantine poisoning experiments:
  1. attack_labelflip_fedavg
  2. attack_labelflip_trimmed_mean
  3. attack_labelflip_median
  4. attack_weightpoison_fedavg
  5. attack_weightpoison_trimmed_mean
  6. attack_weightpoison_median

Evaluates 10 clients, 20% malicious (2 clients), 50 global communication rounds,
measuring test accuracy, Macro F1, ROC-AUC, clean vs attacked degradation, and defense gains.
"""

import sys
import time
import pathlib
import subprocess

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

EXPERIMENTS = [
    # Label Flipping Attack
    "attack_labelflip_fedavg",
    "attack_labelflip_trimmed_mean",
    "attack_labelflip_median",
    # Weight Poisoning Attack
    "attack_weightpoison_fedavg",
    "attack_weightpoison_trimmed_mean",
    "attack_weightpoison_median",
]


def run_all(num_rounds: int = 50):
    total_start = time.time()
    results_summary = []

    print("=" * 80)
    print(f"Starting Phase C: 6 Robustness & Byzantine Poisoning Experiments ({num_rounds} rounds each)")
    print("=" * 80)

    for idx, exp_name in enumerate(EXPERIMENTS, start=1):
        print(f"\n[{idx}/6] Executing experiment: {exp_name} ({num_rounds} rounds)...")
        exp_start = time.time()

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "run_federated.py"),
            "--experiment", exp_name,
            "--num_rounds", str(num_rounds),
        ]

        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

        elapsed = time.time() - exp_start
        if proc.returncode != 0:
            print(f"FAILED: {exp_name} after {elapsed:.2f}s")
            print("STDERR:")
            print(proc.stderr[-500:])
            raise RuntimeError(f"Experiment {exp_name} failed with returncode {proc.returncode}")

        print(f"SUCCESS: {exp_name} completed in {elapsed:.2f}s")
        results_summary.append({"experiment": exp_name, "duration_sec": round(elapsed, 2), "status": "SUCCESS"})

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"All 6 Robustness Experiments Completed Successfully in {total_elapsed:.2f}s!")
    print("=" * 80)
    return results_summary


if __name__ == "__main__":
    rounds = 50
    if len(sys.argv) > 1:
        rounds = int(sys.argv[1])
    run_all(num_rounds=rounds)
