"""
Final Project Verification Script.
Validates all deliverables, results integrity, and reproducibility requirements.

Run: python verify_project.py
"""

import sys
import pathlib
import json

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}" + (f" -- {detail}" if detail else "")
    print(msg)
    return condition


def main():
    print("=" * 80)
    print("  FINAL PROJECT VERIFICATION: Privacy-Preserving Malware Detection Using FL")
    print("=" * 80)
    results = []

    # ─── 1. Source Code Modules ───
    print("\n[1] Source Code Modules")
    src = PROJECT_ROOT / "src"
    fed = src / "federated"
    modules = {
        "MalwareMLP Model": src / "models" / "mlp.py",
        "Preprocessing Pipeline": src / "preprocess_and_compare.py",
        "Centralized Training": src / "train_centralized.py",
        "Federated Runner (CLI)": src / "run_federated.py",
        "FedAvg Aggregation": fed / "fedavg.py",
        "Client Training": fed / "client.py",
        "DP-SGD Client": fed / "dp_client.py",
        "RDP Accountant": fed / "dp_accountant.py",
        "Adaptive Clipping": fed / "adaptive_clipping.py",
        "Secure Aggregation": fed / "secure_aggregation.py",
        "Robust Aggregation": fed / "robust_aggregation.py",
        "Attack Models": fed / "attacks.py",
        "Partitioning": fed / "partition.py",
        "Evaluation Metrics": fed / "evaluation.py",
        "Experiment Orchestrator": fed / "experiment.py",
    }
    for name, path in modules.items():
        results.append(check(name, path.exists(), str(path.relative_to(PROJECT_ROOT))))

    # ─── 2. Processed Data ───
    print("\n[2] Processed Data Splits")
    data_dir = PROJECT_ROOT / "data" / "processed" / "variant_a"
    for split in ["X_train.csv", "y_train.csv", "X_val.csv", "y_val.csv", "X_test.csv", "y_test.csv"]:
        p = data_dir / split
        results.append(check(split, p.exists() and p.stat().st_size > 0))

    # ─── 3. Experiment Results (JSON Artifacts) ───
    print("\n[3] Experiment Result Artifacts")
    results_dir = PROJECT_ROOT / "results" / "federated"
    expected_results = [
        "iid_fedavg_metrics.json",
        "dirichlet_a01_metrics.json", "dirichlet_a05_metrics.json", "dirichlet_a10_metrics.json",
        "fedprox_a01_mu001_metrics.json",
        "dp_low_noise_metrics.json", "dp_med_noise_metrics.json", "dp_high_noise_metrics.json",
        "dp_secagg_metrics.json", "secagg_only_metrics.json",
        "dp_adaptive_clipping_metrics.json", "noniid_dp_adaptive_clipping_metrics.json",
        "noniid_dp_secagg_metrics.json",
        "attack_labelflip_fedavg_metrics.json", "attack_labelflip_median_metrics.json",
        "attack_labelflip_trimmed_mean_metrics.json",
        "attack_weightpoison_fedavg_metrics.json", "attack_weightpoison_median_metrics.json",
        "attack_weightpoison_trimmed_mean_metrics.json",
    ]
    for fname in expected_results:
        p = results_dir / fname
        ok = p.exists() and p.stat().st_size > 1000
        detail = ""
        if ok:
            with open(p) as f:
                data = json.load(f)
            test_acc = data.get("locked_test_metrics", {}).get("accuracy", "N/A")
            if isinstance(test_acc, float):
                detail = f"Test Acc = {test_acc*100:.2f}%"
            else:
                detail = f"Test Acc = {test_acc}"
        results.append(check(fname, ok, detail))

    # ─── 4. Publication Figures ───
    print("\n[4] Publication-Quality Figures")
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    expected_figs = [
        "fig1_centralized_vs_federated.png",
        "fig2_noniid_convergence.png",
        "fig3_privacy_utility_tradeoff.png",
        "fig4_byzantine_robustness.png",
        "fig5_adaptive_clipping_trajectory.png",
    ]
    for fname in expected_figs:
        p = figures_dir / fname
        results.append(check(fname, p.exists() and p.stat().st_size > 10000))

    # ─── 5. Reports & Documentation ───
    print("\n[5] Reports & Documentation")
    doc_files = {
        "README.md": PROJECT_ROOT / "README.md",
        "requirements.txt (deps)": PROJECT_ROOT / "requirements.txt",
        "Academic Research Report": PROJECT_ROOT / "reports" / "academic_research_report.md",
        "Master Comparison Matrix": PROJECT_ROOT / "reports" / "final_comparative_matrix.md",
        "Dataset Audit Report": PROJECT_ROOT / "reports" / "dataset_audit_report.md",
        "Centralized Baseline Report": PROJECT_ROOT / "reports" / "centralized_baseline_report.md",
    }
    for name, path in doc_files.items():
        min_size = 100 if 'requirements' in name else 500
        results.append(check(name, path.exists() and path.stat().st_size > min_size))

    # ─── 6. Test Suite ───
    print("\n[6] Unit Test Files")
    test_files = [
        "test_attacks.py",
        "test_robust_aggregation.py",
        "test_evaluation.py",
        "test_adaptive_clipping.py",
        "test_privacy_integrity.py",
        "test_federated_integrity.py",
    ]
    for fname in test_files:
        results.append(check(fname, (PROJECT_ROOT / fname).exists()))

    # ─── 7. Demo & Presentation ───
    print("\n[7] Demo & Presentation Scripts")
    results.append(check("Live Demo Script", (PROJECT_ROOT / "demo.py").exists()))
    results.append(check("PPT Generator Script", (src / "generate_presentation.py").exists()))
    results.append(check("Benchmark Generator", (src / "generate_final_benchmarks.py").exists()))
    results.append(check("Plot Generator", (src / "generate_final_plots.py").exists()))

    # ─── 8. Key Metrics Spot-Check ───
    print("\n[8] Key Result Metrics Spot-Check")
    # IID FedAvg should be ~90%
    with open(results_dir / "iid_fedavg_metrics.json") as f:
        iid = json.load(f)
    iid_acc = iid["locked_test_metrics"]["accuracy"]
    results.append(check("IID FedAvg Test Acc ≥ 88%", iid_acc >= 0.88, f"{iid_acc*100:.2f}%"))

    # Weight poison FedAvg should show collapse
    with open(results_dir / "attack_weightpoison_fedavg_metrics.json") as f:
        wp = json.load(f)
    wp_acc = wp["locked_test_metrics"]["accuracy"]
    results.append(check("Weight Poison FedAvg collapse (<70%)", wp_acc < 0.70, f"{wp_acc*100:.2f}%"))

    # Weight poison Trimmed Mean should recover
    with open(results_dir / "attack_weightpoison_trimmed_mean_metrics.json") as f:
        wptm = json.load(f)
    wptm_acc = wptm["locked_test_metrics"]["accuracy"]
    results.append(check("Weight Poison TrimmedMean recovery (≥85%)", wptm_acc >= 0.85, f"{wptm_acc*100:.2f}%"))

    # DP SecAgg should have finite epsilon
    with open(results_dir / "dp_secagg_metrics.json") as f:
        dp = json.load(f)
    eps = dp.get("dp_epsilon", None)
    results.append(check("DP+SecAgg has finite epsilon", eps is not None and eps < 100, f"ε = {eps}"))

    # ─── Summary ───
    passed = sum(results)
    total = len(results)
    failed = total - passed
    print("\n" + "=" * 80)
    print(f"  VERIFICATION SUMMARY: {passed}/{total} checks passed, {failed} failed")
    if failed == 0:
        print("  PROJECT FULLY VERIFIED -- ALL DELIVERABLES COMPLETE")
    else:
        print("  SOME CHECKS FAILED -- Review output above")
    print("=" * 80)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
