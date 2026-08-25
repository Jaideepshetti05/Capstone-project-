"""
CICMalDroid 2020 Dataset Comprehensive Audit Script
Privacy-Preserving Malware Detection Using Federated Learning

This script performs a full, non-destructive, independent audit of all dataset CSV files:
  - feature_vectors_syscallsbinders_frequency_5_Cat.csv  (PRIMARY)
  - feature_vectors_syscalls_frequency_5_Cat.csv         (SECONDARY)
  - feature_vectors_static.csv
  - syscall_unique.csv

Outputs:
  - reports/dataset_audit_summary.json
  - reports/dataset_audit_report.md

IMPORTANT: This script does NOT modify any dataset file.
"""

import os
import sys
import json
import time
import math
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Paths ---
BASE_DIR    = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "primary_dynamic"   : BASE_DIR / "feature_vectors_syscallsbinders_frequency_5_Cat.csv",
    "secondary_dynamic" : BASE_DIR / "feature_vectors_syscalls_frequency_5_Cat.csv",
    "static"            : BASE_DIR / "feature_vectors_static.csv",
    "syscall_unique"    : BASE_DIR / "syscall_unique.csv",
}

SEP = "=" * 70


# --- Helpers ---

def human_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def safe_json(obj):
    """Recursively convert numpy/pandas types to native Python for JSON."""
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def file_meta(fp):
    if not fp.exists():
        return {"exists": False, "filename": fp.name}
    s = fp.stat()
    return {
        "exists"     : True,
        "filename"   : fp.name,
        "size_bytes" : s.st_size,
        "size_human" : human_size(s.st_size),
    }


# --- Syscall-unique audit ---

def audit_syscall_unique(fp):
    meta = file_meta(fp)
    print(f"\n{SEP}\nAuditing: {fp.name}\n{SEP}")
    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
        lines = [l.strip() for l in fh if l.strip()]
    unique_set = set(lines)
    duplicates = len(lines) - len(unique_set)
    result = {
        **meta,
        "num_lines"            : len(lines),
        "unique_syscall_count" : len(unique_set),
        "duplicate_lines"      : duplicates,
        "sample_entries"       : lines[:20],
        "note"                 : "Reference list of system-call names; not a feature matrix.",
    }
    print(f"  Lines: {len(lines)} | Unique: {len(unique_set)} | Duplicates: {duplicates}")
    return result


# --- Tabular CSV audit ---

TARGET_CANDIDATES = ["Class", "class", "Label", "label", "Category",
                     "category", "Family", "family", "Target", "target"]


def detect_target(df):
    for cand in TARGET_CANDIDATES:
        if cand in df.columns:
            return cand
    return df.columns[-1]


def audit_tabular_csv(fp, role="generic"):
    meta = file_meta(fp)
    if not meta["exists"]:
        print(f"  FILE NOT FOUND: {fp}")
        return meta

    is_primary = (role == "primary")

    print(f"\n{SEP}")
    print(f"Auditing [{role.upper()}]: {fp.name}  ({meta['size_human']})")
    print(SEP)

    t0 = time.time()
    df = pd.read_csv(fp, low_memory=False)
    load_sec = round(time.time() - t0, 2)
    print(f"  Loaded {df.shape[0]:,} rows x {df.shape[1]:,} columns in {load_sec}s")

    num_rows, num_cols = df.shape
    memory_bytes = int(df.memory_usage(deep=True).sum())

    # Target column
    target_col   = detect_target(df)
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"  Target column detected: '{target_col}'")

    # Class distribution
    target_series  = df[target_col]
    vc             = target_series.value_counts(dropna=False)
    unique_classes = vc.index.tolist()
    class_counts   = {str(k): int(v) for k, v in vc.items()}
    class_pct      = {str(k): round(v / num_rows * 100, 4) for k, v in vc.items()}
    num_classes    = len(unique_classes)
    print(f"  Classes ({num_classes}):")
    for cls, cnt in class_counts.items():
        print(f"    {str(cls):<40s} {cnt:6,d}  ({class_pct[str(cls)]:.2f}%)")

    # Data types
    dtype_breakdown = {str(k): int(v) for k, v in df.dtypes.value_counts().items()}

    # Missing / null values
    null_per_col    = df.isnull().sum()
    total_nulls     = int(null_per_col.sum())
    cols_with_nulls = {c: int(null_per_col[c]) for c in df.columns if null_per_col[c] > 0}
    print(f"  Missing values: {total_nulls}")

    # Duplicate rows
    exact_dups      = int(df.duplicated().sum())
    feat_dups       = int(df.duplicated(subset=feature_cols).sum())
    label_conflicts = max(0, feat_dups - exact_dups)
    print(f"  Exact duplicates: {exact_dups:,} | Feature-only dups: {feat_dups:,} | Label conflicts: {label_conflicts:,}")

    # Duplicate column names
    col_name_counts = {}
    for c in df.columns:
        col_name_counts[c] = col_name_counts.get(c, 0) + 1
    dup_col_names = [c for c, cnt in col_name_counts.items() if cnt > 1]

    # Numeric vs non-numeric features
    feat_df          = df[feature_cols]
    num_feat_df      = feat_df.select_dtypes(include=[np.number])
    non_numeric_cols = [c for c in feature_cols if c not in num_feat_df.columns]
    print(f"  Numeric features: {len(num_feat_df.columns):,} | Non-numeric: {len(non_numeric_cols)}")

    # Inf / Negative
    inf_count = 0
    neg_count = 0
    if not num_feat_df.empty:
        arr       = num_feat_df.values
        inf_count = int(np.isinf(arr).sum())
        neg_count = int((arr < 0).sum())
    print(f"  Infinite values: {inf_count} | Negative values: {neg_count}")

    # Zero / near-zero variance & high sparsity
    zero_var_cols      = []
    near_zero_var_cols = []
    high_sparsity_cols = []
    variances          = None

    if not num_feat_df.empty:
        variances          = num_feat_df.var()
        zero_var_cols      = variances[variances == 0].index.tolist()
        near_zero_var_cols = variances[(variances > 0) & (variances < 1e-5)].index.tolist()
        zero_counts        = (num_feat_df == 0).sum()
        high_sparsity_cols = zero_counts[zero_counts / num_rows > 0.99].index.tolist()

    print(f"  Zero-variance: {len(zero_var_cols):,} | Near-zero: {len(near_zero_var_cols):,} | High-sparsity: {len(high_sparsity_cols):,}")

    col_list   = list(df.columns)
    sample_cols = col_list[:10] + (["..."] if num_cols > 20 else []) + col_list[-5:]

    result = {
        **meta,
        "role"                              : role,
        "load_time_seconds"                 : load_sec,
        "rows"                              : num_rows,
        "columns"                           : num_cols,
        "memory_bytes"                      : memory_bytes,
        "memory_human"                      : human_size(memory_bytes),
        "column_names_sample"               : sample_cols,
        "all_column_names"                  : col_list,
        "dtype_breakdown"                   : dtype_breakdown,
        "target_column"                     : target_col,
        "num_classes"                       : num_classes,
        "class_labels"                      : [str(c) for c in unique_classes],
        "class_counts"                      : class_counts,
        "class_percentages"                 : class_pct,
        "total_missing_values"              : total_nulls,
        "columns_with_missing"              : cols_with_nulls,
        "exact_duplicate_rows"              : exact_dups,
        "feature_only_duplicate_rows"       : feat_dups,
        "label_conflict_duplicate_rows"     : label_conflicts,
        "duplicate_column_names"            : dup_col_names,
        "num_numeric_features"              : len(num_feat_df.columns),
        "num_non_numeric_features"          : len(non_numeric_cols),
        "non_numeric_feature_names"         : non_numeric_cols[:30],
        "infinite_values_count"             : inf_count,
        "negative_values_count"             : neg_count,
        "zero_variance_feature_count"       : len(zero_var_cols),
        "zero_variance_features"            : zero_var_cols,
        "near_zero_variance_feature_count"  : len(near_zero_var_cols),
        "near_zero_variance_features_sample": near_zero_var_cols[:20],
        "high_sparsity_feature_count"       : len(high_sparsity_cols),
        "high_sparsity_features_sample"     : high_sparsity_cols[:20],
    }

    # Primary-only deep analysis
    if is_primary and not num_feat_df.empty and variances is not None:
        print("  Running deep statistical analysis (primary dataset)...")
        non_zero_feats = num_feat_df.drop(columns=zero_var_cols, errors="ignore")

        arr    = non_zero_feats.values.astype(float)
        g_min  = float(np.nanmin(arr))
        g_max  = float(np.nanmax(arr))
        g_mean = float(np.nanmean(arr))
        g_std  = float(np.nanstd(arr))

        desc           = non_zero_feats.describe().T
        top_mean_feats = {str(k): float(v) for k, v in desc["mean"].sort_values(ascending=False).head(20).items()}
        top_max_feats  = {str(k): float(v) for k, v in desc["max"].sort_values(ascending=False).head(20).items()}
        top_var_feats  = {str(k): float(v) for k, v in variances.drop(labels=zero_var_cols, errors="ignore").sort_values(ascending=False).head(20).items()}

        print("  Computing correlation matrix...")
        t_c  = time.time()
        corr = non_zero_feats.corr().abs()
        print(f"  Correlation computed in {time.time()-t_c:.1f}s")

        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs_95, pairs_99, pairs_100 = [], [], []

        for col in upper.columns:
            hi95 = upper.index[upper[col] > 0.95].tolist()
            for idx in hi95:
                val   = float(upper.loc[idx, col])
                entry = {"f1": str(idx), "f2": str(col), "r": round(val, 6)}
                pairs_95.append(entry)
                if val > 0.99:
                    pairs_99.append(entry)
                if np.isclose(val, 1.0, atol=1e-6):
                    pairs_100.append(entry)

        print(f"  |r|>0.95: {len(pairs_95)} | |r|>0.99: {len(pairs_99)} | r=1.0: {len(pairs_100)}")

        binder_cols  = [c for c in feature_cols if
                        "binder" in c.lower() or
                        "android.app" in c.lower() or
                        "android.content" in c.lower() or
                        "android.os" in c.lower() or
                        c.startswith("android.") or
                        "/" in c or ":" in c]
        syscall_only = [c for c in feature_cols if c not in binder_cols]

        result["deep_statistics"] = {
            "global_min"                       : g_min,
            "global_max"                       : g_max,
            "global_mean"                      : g_mean,
            "global_std"                       : g_std,
            "top_20_highest_mean_features"     : top_mean_feats,
            "top_20_highest_max_features"      : top_max_feats,
            "top_20_highest_variance_features" : top_var_feats,
            "corr_pairs_above_0.95_count"      : len(pairs_95),
            "corr_pairs_above_0.99_count"      : len(pairs_99),
            "corr_pairs_exactly_1.0_count"     : len(pairs_100),
            "top_20_correlated_pairs"          : sorted(pairs_95, key=lambda x: x["r"], reverse=True)[:20],
            "perfectly_correlated_pairs"       : pairs_100[:20],
        }
        result["feature_type_breakdown"] = {
            "total_feature_columns"  : len(feature_cols),
            "binder_ipc_features"    : len(binder_cols),
            "syscall_only_features"  : len(syscall_only),
            "sample_binder_features" : binder_cols[:10],
            "sample_syscall_features": syscall_only[:10],
        }

    return result


# --- Cross-dataset comparison ---

def compare_dynamic(primary, secondary):
    p_rows    = primary.get("rows", 0)
    s_rows    = secondary.get("rows", 0)
    p_classes = primary.get("class_counts", {})
    s_classes = secondary.get("class_counts", {})
    row_match = (p_rows == s_rows)
    cls_match = (p_classes == s_classes)
    return {
        "primary_rows"             : p_rows,
        "secondary_rows"           : s_rows,
        "rows_match"               : row_match,
        "row_difference"           : abs(p_rows - s_rows),
        "class_distributions_match": cls_match,
        "primary_class_counts"     : p_classes,
        "secondary_class_counts"   : s_classes,
        "primary_feature_count"    : primary.get("num_numeric_features", 0),
        "secondary_feature_count"  : secondary.get("num_numeric_features", 0),
        "assessment": (
            "Consistent: same samples and label distribution."
            if row_match and cls_match
            else "DISCREPANCY detected between the two dynamic datasets!"
        ),
    }


# --- Markdown report builder ---

def build_markdown_report(results):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md  = []
    app = md.append

    app("# CICMalDroid 2020 - Comprehensive Dataset Audit Report\n")
    app(f"**Generated:** {ts}  ")
    app("**Project:** Privacy-Preserving Malware Detection Using Federated Learning  ")
    app("**Script:** `src/data_audit.py`\n")
    app("---\n")

    app("## 1. Dataset Files Overview\n")
    app("| File | Size | Rows | Columns | Target Column | Classes |")
    app("|------|------|------|---------|---------------|---------|")
    for key in ["primary_dynamic", "secondary_dynamic", "static", "syscall_unique"]:
        r = results.get(key, {})
        if not r.get("exists", False):
            app(f"| {key} | NOT FOUND | - | - | - | - |")
            continue
        if "num_lines" in r:  # syscall_unique reference list
            app(f"| {r['filename']} | {r['size_human']} | {r['num_lines']} lines | - | N/A (reference list) | - |")
        elif r.get("target_column") == "NO_CLASS_COLUMN":
            app(f"| {r['filename']} | {r['size_human']} | {r['rows']:,} | {r['columns']:,} | N/A (no class col) | N/A |")
        else:
            app(f"| {r['filename']} | {r['size_human']} | {r['rows']:,} | {r['columns']:,} | `{r['target_column']}` | {r['num_classes']} |")

    tabular_keys = [
        ("primary_dynamic",   "PRIMARY - feature_vectors_syscallsbinders_frequency_5_Cat.csv"),
        ("secondary_dynamic", "SECONDARY - feature_vectors_syscalls_frequency_5_Cat.csv"),
        ("static",            "AUXILIARY - feature_vectors_static.csv"),
    ]

    for sec_num, (key, title) in enumerate(tabular_keys, start=2):
        r = results.get(key, {})
        if not r or not r.get("exists", False):
            continue

        app(f"\n---\n\n## {sec_num}. {title}\n")
        app(f"- **File size:** {r['size_human']}")
        app(f"- **Rows:** {r['rows']:,}")
        app(f"- **Columns:** {r['columns']:,}")
        app(f"- **In-RAM memory:** {r['memory_human']}")
        app(f"- **Load time:** {r['load_time_seconds']}s")
        app(f"- **Target column:** `{r['target_column']}`")
        app(f"- **Number of classes:** {r['num_classes']}")

        app(f"\n### {sec_num}.1 Class Distribution\n")
        app("| Class Label | Count | Percentage |")
        app("|-------------|-------|-----------|")
        for cls, cnt in r["class_counts"].items():
            pct = r["class_percentages"].get(cls, 0)
            app(f"| {cls} | {cnt:,} | {pct:.2f}% |")

        app(f"\n### {sec_num}.2 Data Quality\n")
        app("| Metric | Value |")
        app("|--------|-------|")
        app(f"| Missing values | {r['total_missing_values']:,} |")
        app(f"| Exact duplicate rows | {r['exact_duplicate_rows']:,} |")
        app(f"| Feature-only duplicate rows | {r['feature_only_duplicate_rows']:,} |")
        app(f"| Label-conflict duplicates | {r['label_conflict_duplicate_rows']:,} |")
        app(f"| Duplicate column names | {len(r['duplicate_column_names'])} |")
        app(f"| Numeric features | {r['num_numeric_features']:,} |")
        app(f"| Non-numeric features | {r['num_non_numeric_features']:,} |")
        app(f"| Infinite values | {r['infinite_values_count']:,} |")
        app(f"| Negative values | {r['negative_values_count']:,} |")
        app(f"| Zero-variance features | {r['zero_variance_feature_count']:,} |")
        app(f"| Near-zero-variance features | {r['near_zero_variance_feature_count']:,} |")
        app(f"| High-sparsity features (>99% zeros) | {r['high_sparsity_feature_count']:,} |")

        if key == "primary_dynamic" and "deep_statistics" in r:
            ds = r["deep_statistics"]
            app(f"\n### {sec_num}.3 Deep Statistics\n")
            app("| Stat | Value |")
            app("|------|-------|")
            app(f"| Global min | {ds['global_min']} |")
            app(f"| Global max | {ds['global_max']} |")
            app(f"| Global mean | {ds['global_mean']:.6f} |")
            app(f"| Global std | {ds['global_std']:.6f} |")
            app(f"| Correlated pairs |r|>0.95 | {ds['corr_pairs_above_0.95_count']:,} |")
            app(f"| Correlated pairs |r|>0.99 | {ds['corr_pairs_above_0.99_count']:,} |")
            app(f"| Perfectly correlated pairs r=1.0 | {ds['corr_pairs_exactly_1.0_count']:,} |")

            if ds.get("top_20_correlated_pairs"):
                app(f"\n#### Top Correlated Feature Pairs (|r| > 0.95)\n")
                app("| Feature 1 | Feature 2 | Pearson r |")
                app("|-----------|-----------|-----------|")
                for p in ds["top_20_correlated_pairs"][:15]:
                    app(f"| `{p['f1']}` | `{p['f2']}` | {p['r']:.4f} |")

        if key == "primary_dynamic" and "feature_type_breakdown" in r:
            fb = r["feature_type_breakdown"]
            app(f"\n### {sec_num}.4 Feature Composition\n")
            app("| Category | Count |")
            app("|----------|-------|")
            app(f"| Total feature columns | {fb['total_feature_columns']:,} |")
            app(f"| Binder/IPC features | {fb['binder_ipc_features']:,} |")
            app(f"| Syscall-only features | {fb['syscall_only_features']:,} |")

    if "dynamic_comparison" in results:
        dc = results["dynamic_comparison"]
        n  = len(tabular_keys) + 2
        app(f"\n---\n\n## {n}. Dynamic Dataset Consistency Check\n")
        app("| Metric | Value |")
        app("|--------|-------|")
        app(f"| Primary rows | {dc['primary_rows']:,} |")
        app(f"| Secondary rows | {dc['secondary_rows']:,} |")
        app(f"| Row counts match | {'YES' if dc['rows_match'] else 'NO'} |")
        app(f"| Row difference | {dc['row_difference']:,} |")
        app(f"| Class distributions match | {'YES' if dc['class_distributions_match'] else 'NO'} |")
        app(f"| Primary feature count | {dc['primary_feature_count']:,} |")
        app(f"| Secondary feature count | {dc['secondary_feature_count']:,} |")
        app(f"\n**Assessment:** {dc['assessment']}")

    app("\n---\n\n## Final Audit Summary and Recommendations\n")

    pr  = results.get("primary_dynamic", {})
    cc  = pr.get("class_counts", {})
    pct = pr.get("class_percentages", {})

    app("### A. Dataset Summary\n")
    app(f"The **primary dataset** (`feature_vectors_syscallsbinders_frequency_5_Cat.csv`) contains "
        f"**{pr.get('rows', '?'):,} samples** across **{pr.get('num_classes', '?')} malware categories** "
        f"using **{pr.get('num_numeric_features', '?'):,} numeric features** derived from "
        f"dynamic analysis (system calls + binder IPC).")

    app("\n### B. Recommended Primary Dataset\n")
    app("**`feature_vectors_syscallsbinders_frequency_5_Cat.csv`** - combines syscall "
        "frequency features AND Android binder/IPC calls, providing a richer behavioral "
        "fingerprint. Preferred over the syscalls-only secondary dataset.")

    app("\n### C. Data-Quality Problems Found\n")
    issues = []
    if pr.get("exact_duplicate_rows", 0) > 0:
        issues.append(f"- WARNING: **{pr['exact_duplicate_rows']:,} exact duplicate rows** in primary dataset.")
    else:
        issues.append("- OK: No exact duplicate rows.")
    if pr.get("label_conflict_duplicate_rows", 0) > 0:
        issues.append(f"- WARNING: **{pr['label_conflict_duplicate_rows']:,} label-conflicting duplicates** "
                      f"(same features, different label).")
    if pr.get("zero_variance_feature_count", 0) > 0:
        issues.append(f"- CRITICAL: **{pr['zero_variance_feature_count']:,} zero-variance features** must be dropped.")
    if pr.get("high_sparsity_feature_count", 0) > 0:
        issues.append(f"- WARNING: **{pr['high_sparsity_feature_count']:,} high-sparsity features** (>99% zeros).")
    if pr.get("total_missing_values", 0) == 0:
        issues.append("- OK: No missing values detected.")
    else:
        issues.append(f"- CRITICAL: **{pr['total_missing_values']:,} missing values** detected.")
    if pr.get("infinite_values_count", 0) > 0:
        issues.append(f"- CRITICAL: **{pr['infinite_values_count']:,} infinite values** detected.")
    if pr.get("negative_values_count", 0) > 0:
        issues.append(f"- WARNING: **{pr['negative_values_count']:,} negative values** in frequency features.")

    if cc and pct:
        max_pct   = max(pct.values())
        min_pct   = min(pct.values())
        imb_ratio = max_pct / min_pct if min_pct > 0 else float("inf")
        if imb_ratio > 3:
            issues.append(f"- WARNING: **Class imbalance** - largest class is {imb_ratio:.1f}x larger than smallest.")
        else:
            issues.append(f"- OK: Class imbalance ratio ~{imb_ratio:.2f}x (moderate).")

    ds_corr = pr.get("deep_statistics", {})
    if ds_corr.get("corr_pairs_exactly_1.0_count", 0) > 0:
        issues.append(f"- WARNING: **{ds_corr['corr_pairs_exactly_1.0_count']} perfectly correlated feature pairs**.")

    for issue in issues:
        app(issue)

    app("\n### D. Potential Data Leakage Concerns\n")
    app("- Dataset does NOT contain explicit APK hashes or timestamps that would directly leak train/test membership.")
    app("- Duplicate samples MUST be removed **before** splitting - identical samples in both train and test cause optimistic evaluation.")
    app("- Perfectly correlated features may indicate redundant derived features inflating model performance.")
    app("- For centralized vs federated comparison: use the SAME de-duplicated dataset as baseline for BOTH experiments.")

    app("\n### E. Recommended Preprocessing Steps\n")
    app("1. Remove exact duplicate rows (before any split).")
    app("2. Drop zero-variance features.")
    app("3. Optionally drop near-zero-variance and >99%-sparse features.")
    app("4. Encode labels to integer indices.")
    app("5. Normalize features: StandardScaler or MinMaxScaler (fit on train only).")
    app("6. Handle class imbalance via stratified splitting; optionally use class-weighted loss.")
    app("7. Remove or merge perfectly correlated features.")

    app("\n### F. Recommended Train / Validation / Test Strategy\n")
    app("| Split | Ratio | Notes |")
    app("|-------|-------|-------|")
    app("| Training | 70% | Model and FL client training |")
    app("| Validation | 10% | Hyperparameter tuning |")
    app("| Test | 20% | Final evaluation; held out from all experiments |")
    app("")
    app("- Use **stratified splits** to preserve class ratios.")
    app("- For FL: partition the training set into N client shards.")
    app("  - **IID partition:** random shuffle and equal split across clients.")
    app("  - **Non-IID partition:** Dirichlet(alpha) allocation - lower alpha = stronger non-IID.")
    app("- The global test set remains centralized for fair comparison.")

    app("\n### G. Readiness for Next Phase\n")
    no_nulls = pr.get("total_missing_values", 1) == 0
    no_inf   = pr.get("infinite_values_count", 1) == 0
    exists   = pr.get("exists", False)
    if no_nulls and no_inf and exists:
        app("**READY** - Dataset is suitable for preprocessing and model development, "
            "pending the cleaning steps above (duplicate removal, zero-variance drop, normalization).")
    else:
        app("**NOT READY** - Additional data cleaning required before model training.")

    app("\n---\n*End of Audit Report*")
    return "\n".join(md)


# --- Python-csv audit for ultra-wide CSVs (50k+ columns) that OOM pandas ---

def audit_static_csv_python(fp):
    """
    Audit the ultra-wide static features CSV (50k+ columns) using Python's
    built-in csv module to avoid pandas C-parser OOM errors.
    Streams row-by-row: no full matrix ever loaded into RAM.
    Variance/correlation analysis is skipped due to extreme dimensionality.
    """
    import csv as _csv

    meta = file_meta(fp)
    if not meta["exists"]:
        return meta

    print(f"\n{SEP}")
    print(f"Auditing [STATIC/PYTHON-CSV]: {fp.name}  ({meta['size_human']})")
    print(f"  Using Python csv module (no C-parser memory limits)")
    print(SEP)

    t0 = time.time()

    # Step 1: Read header only
    with open(fp, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader   = _csv.reader(fh)
        col_list = [c.strip() for c in next(reader)]

    num_cols = len(col_list)
    print(f"  Header read: {num_cols:,} columns")

    # Step 2: Detect target column
    # The static features CSV has NO class/label column.
    # Column [0] is an unnamed row-index column ('0','1','2'...).
    # All other columns are static feature names (intents, permissions, APIs).
    target_col = None
    target_idx = None
    for cand in TARGET_CANDIDATES:
        for i, c in enumerate(col_list):
            if c == cand:
                target_col = c
                target_idx = i
                break
        if target_col:
            break

    no_class_col = False
    if target_col is None:
        # No standard label column found. Column[0] is a row index.
        # Mark as no-class-column file.
        no_class_col = True
        target_col = "NO_CLASS_COLUMN"
        target_idx  = 0   # treat row-index col as pseudo-target for hashing
        print(f"  INFO: No standard class/label column found in this file.")
        print(f"  Column[0] appears to be a row index. This is a FEATURES-ONLY file.")
        print(f"  Labels must come from the dynamic dataset joined by row index.")
    else:
        print(f"  Target column [idx={target_idx}]: '{target_col}'")

    feature_cols = [c for c in col_list if c != target_col]
    print(f"  Feature columns: {len(feature_cols):,}")

    # Step 3: Stream all rows with Python csv module
    total_rows    = 0
    total_nulls   = 0
    class_counter = {}
    exact_dup_set = set()
    feat_dup_set  = set()
    exact_dups    = 0
    feat_dups     = 0
    neg_count     = 0

    # Track sparsity on first SPARSITY_COLS feature columns (sample)
    SPARSITY_COLS  = min(2000, len(feature_cols))
    sparsity_zeros = [0] * SPARSITY_COLS
    # Indices in row corresponding to sampled feature columns (skip target)
    feat_indices     = [i for i in range(num_cols) if i != target_idx]
    sparsity_indices = feat_indices[:SPARSITY_COLS]

    with open(fp, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = _csv.reader(fh)
        next(reader)  # skip header

        for row in reader:
            if not row:
                continue
            total_rows += 1

            if total_rows % 500 == 0:
                print(f"  ... {total_rows:,} rows processed ...")

            # Null check
            total_nulls += sum(
                1 for v in row
                if v.strip() == "" or v.strip().lower() == "nan"
            )

            # Class / target value
            cls_val = row[target_idx].strip() if target_idx < len(row) else "?"
            if cls_val == "" or cls_val.lower() == "nan":
                cls_val = "NaN"
            class_counter[cls_val] = class_counter.get(cls_val, 0) + 1

            # Duplicate detection (full row)
            full_hash = hash(tuple(row))
            if full_hash in exact_dup_set:
                exact_dups += 1
            else:
                exact_dup_set.add(full_hash)

            # Feature-only hash
            feat_hash = hash(tuple(row[i] for i in range(len(row)) if i != target_idx))
            if feat_hash in feat_dup_set:
                feat_dups += 1
            else:
                feat_dup_set.add(feat_hash)

            # Sparsity on sampled columns + negative check
            for si, col_i in enumerate(sparsity_indices):
                if col_i >= len(row):
                    continue
                v = row[col_i].strip()
                try:
                    fv = float(v)
                    if fv == 0.0:
                        sparsity_zeros[si] += 1
                    elif fv < 0:
                        neg_count += 1
                except (ValueError, TypeError):
                    pass

    load_sec = round(time.time() - t0, 2)
    print(f"  Finished: {total_rows:,} rows in {load_sec}s")

    # Compute summary stats
    label_conflicts = max(0, feat_dups - exact_dups)

    if no_class_col:
        # No class column — skip class distribution output
        class_counter = {}
        class_pct     = {}
        print(f"  NOTE: No class column — class distribution not applicable.")
    else:
        class_pct = {k: round(v / total_rows * 100, 4) for k, v in class_counter.items()}
        print(f"  Classes ({len(class_counter)}):")
        for cls in sorted(class_counter.keys(), key=str):
            cnt = class_counter[cls]
            print(f"    {str(cls):<40s} {cnt:6,d}  ({class_pct[cls]:.2f}%)")

    sampled_high_sparsity = sum(
        1 for z in sparsity_zeros
        if total_rows > 0 and z / total_rows > 0.99
    )
    sparsity_ratio          = sampled_high_sparsity / SPARSITY_COLS if SPARSITY_COLS > 0 else 0
    est_high_sparsity_total = int(sparsity_ratio * len(feature_cols))

    print(f"  Missing: {total_nulls:,} | Exact dups: {exact_dups:,} | Feat dups: {feat_dups:,}")
    print(f"  High-sparsity (sampled {SPARSITY_COLS:,} cols): {sampled_high_sparsity}/{SPARSITY_COLS}")
    print(f"  Estimated high-sparsity across all {len(feature_cols):,} features: ~{est_high_sparsity_total:,}")

    sample_cols = col_list[:10] + (["..."] if num_cols > 20 else []) + col_list[-5:]

    note_text = (
        f"Audited via Python csv module (50k+ columns too wide for pandas C-parser). "
        f"Variance/correlation analysis skipped. "
        f"Sparsity estimated from first {SPARSITY_COLS} feature columns. "
    )
    if no_class_col:
        note_text += (
            "NO CLASS/LABEL COLUMN FOUND in this file. "
            "Column[0] is a row index. Labels must be joined from the dynamic datasets by row position."
        )

    return {
        **meta,
        "role"                              : "static_python_csv",
        "load_time_seconds"                 : load_sec,
        "rows"                              : total_rows,
        "columns"                           : num_cols,
        "memory_bytes"                      : None,
        "memory_human"                      : "N/A (streamed row-by-row, not loaded into RAM)",
        "column_names_sample"               : sample_cols,
        "dtype_breakdown"                   : {"binary/numeric (estimated)": len(feature_cols)},
        "target_column"                     : target_col,
        "no_class_column"                   : no_class_col,
        "num_classes"                       : len(class_counter),
        "class_labels"                      : sorted(class_counter.keys(), key=str),
        "class_counts"                      : class_counter,
        "class_percentages"                 : class_pct,
        "total_missing_values"              : total_nulls,
        "columns_with_missing"              : {},
        "exact_duplicate_rows"              : exact_dups,
        "feature_only_duplicate_rows"       : feat_dups,
        "label_conflict_duplicate_rows"     : label_conflicts,
        "duplicate_column_names"            : [],
        "num_numeric_features"              : len(feature_cols),
        "num_non_numeric_features"          : 0,
        "non_numeric_feature_names"         : [],
        "infinite_values_count"             : 0,
        "negative_values_count"             : neg_count,
        "zero_variance_feature_count"       : 0,
        "zero_variance_features"            : [],
        "near_zero_variance_feature_count"  : 0,
        "near_zero_variance_features_sample": [],
        "high_sparsity_feature_count"       : est_high_sparsity_total,
        "high_sparsity_features_sample"     : [],
        "sparsity_sample_cols_checked"      : SPARSITY_COLS,
        "sparsity_sampled_high_sparsity"    : sampled_high_sparsity,
        "note"                              : note_text,
    }


# --- Entry point ---

def run_full_audit():
    print(f"\n{'#'*70}")
    print("  CICMalDroid 2020 - Full Dataset Audit")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    results = {}

    if FILES["syscall_unique"].exists():
        results["syscall_unique"] = audit_syscall_unique(FILES["syscall_unique"])

    if FILES["secondary_dynamic"].exists():
        results["secondary_dynamic"] = audit_tabular_csv(FILES["secondary_dynamic"], role="secondary")

    if FILES["primary_dynamic"].exists():
        results["primary_dynamic"] = audit_tabular_csv(FILES["primary_dynamic"], role="primary")

    if FILES["static"].exists():
        sz = FILES["static"].stat().st_size
        print(f"\n{SEP}")
        print(f"  NOTE: feature_vectors_static.csv is {human_size(sz)}.")
        print(f"  This file has ~50k columns — using Python csv streaming (no pandas).")
        results["static"] = audit_static_csv_python(FILES["static"])

    if "primary_dynamic" in results and "secondary_dynamic" in results:
        results["dynamic_comparison"] = compare_dynamic(
            results["primary_dynamic"], results["secondary_dynamic"]
        )

    # Save JSON
    json_path    = REPORTS_DIR / "dataset_audit_summary.json"
    safe_results = safe_json(results)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(safe_results, fh, indent=2)
    print(f"\nSaved JSON summary: {json_path}")

    # Save Markdown report
    md_text  = build_markdown_report(safe_results)
    md_path  = REPORTS_DIR / "dataset_audit_report.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_text)
    print(f"Saved Markdown report: {md_path}")

    # Quick console summary
    print(f"\n{'='*70}")
    print("QUICK CONSOLE SUMMARY")
    print(f"{'='*70}")
    for key in ["primary_dynamic", "secondary_dynamic", "static"]:
        r = results.get(key, {})
        if not r or not r.get("exists", False):
            continue
        print(f"\n[{key.upper()}] {r['filename']}")
        print(f"  Rows: {r['rows']:,} | Cols: {r['columns']:,} | Target: '{r['target_column']}'")
        for cls, cnt in r["class_counts"].items():
            pct = r["class_percentages"].get(str(cls), 0)
            print(f"    {str(cls):<40s} {cnt:6,d}  ({pct:.2f}%)")
        print(f"  Duplicates: {r['exact_duplicate_rows']:,} | Zero-var: {r['zero_variance_feature_count']:,}")
        print(f"  Nulls: {r['total_missing_values']:,} | Inf: {r['infinite_values_count']:,} | Neg: {r['negative_values_count']:,}")

    if "dynamic_comparison" in results:
        dc = results["dynamic_comparison"]
        print(f"\n[COMPARISON] {dc['assessment']}")

    print(f"\n{'='*70}")
    print("Audit complete. See reports/ for detailed output.")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    run_full_audit()
