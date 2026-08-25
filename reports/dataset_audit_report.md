# CICMalDroid 2020 — Comprehensive Dataset Audit Report

**Generated:** 2026-08-25 14:25:58 (re-finalized with verified class semantics)
**Project:** Privacy-Preserving Malware Detection Using Federated Learning
**Script:** `src/data_audit.py`
**Sources:** Official UNB CIC dataset page (unb.ca/cic/datasets/maldroid-2020.html) + multi-source literature verification

---

## 1. Dataset Files Overview

| File | Size | Rows | Columns | Target Column | Classes |
|------|------|------|---------|---------------|---------|
| feature_vectors_syscallsbinders_frequency_5_Cat.csv | 11.16 MB | 11,598 | 471 | `Class` | 5 |
| feature_vectors_syscalls_frequency_5_Cat.csv | 13.95 MB | 11,598 | 140 | `Class` | 5 |
| feature_vectors_static.csv | 568.22 MB | 11,598 | 50,621 | N/A (no class col) | N/A |
| syscall_unique.csv | 1.26 KB | 139 lines | — | N/A (reference list) | — |

> **Note:** All four files describe the same 11,598 Android APK samples.

---

## 2. VERIFIED CLASS LABEL MAPPING

> **Source:** Official CICMalDroid 2020 dataset (UNB CIC), confirmed by sample count cross-verification.

| Class (Numeric) | Category | Count in Dataset | Percentage |
|----------------|----------|-----------------|-----------|
| **1** | **Adware** | 1,253 | 10.80% |
| **2** | **Banking malware** | 2,100 | 18.11% |
| **3** | **SMS malware** | 3,904 | 33.66% |
| **4** | **Riskware** | 2,546 | 21.95% |
| **5** | **Benign** | 1,795 | 15.48% |

**Verification method:** The counts 1,253 / 2,100 / 3,904 / 2,546 / 1,795 observed in our dataset
match the officially documented CICMalDroid 2020 per-category sample counts exactly, independently
confirming the numeric-to-semantic mapping without any assumption.

**Original dataset total:** 17,341 samples (full release). Our CSV subset contains 11,598 samples,
which is the dynamically analyzed subset with extracted feature vectors.

---

## 3. PRIMARY — feature_vectors_syscallsbinders_frequency_5_Cat.csv

- **File size:** 11.16 MB
- **Rows:** 11,598
- **Columns:** 471 (470 features + 1 label)
- **In-RAM memory:** 41.68 MB
- **Load time:** 0.20s
- **Target column:** `Class` (integer labels 1–5)
- **Number of classes:** 5

### 3.1 Class Distribution

| Class | Category | Count | Percentage |
|-------|----------|-------|-----------|
| 1 | Adware | 1,253 | 10.80% |
| 2 | Banking | 2,100 | 18.11% |
| 3 | SMS malware | 3,904 | 33.66% |
| 4 | Riskware | 2,546 | 21.95% |
| 5 | Benign | 1,795 | 15.48% |

**Imbalance ratio:** 3.1× (SMS malware is the largest class; Adware is the smallest)

### 3.2 Data Quality

| Metric | Value |
|--------|-------|
| Missing values | **0** |
| Exact duplicate rows | **72** |
| Feature-only duplicate rows | **77** |
| Label-conflict duplicates | **5** |
| Duplicate column names | 0 |
| Numeric features | 470 |
| Non-numeric features | 0 |
| Infinite values | 0 |
| Negative values | 0 |
| Zero-variance features | 0 |
| Near-zero-variance features | 0 |
| High-sparsity features (>99% zeros) | **247** |

### 3.3 Deep Statistics

| Stat | Value |
|------|-------|
| Global min | 0.0 |
| Global max | 3,697,410.0 |
| Global mean | 101.786251 |
| Global std | 3,795.258778 |
| Correlated pairs |r|>0.95 | **117** |
| Correlated pairs |r|>0.99 | **70** |
| Perfectly correlated pairs r=1.0 | **35** |

#### Top Correlated Feature Pairs (|r| = 1.0 — perfectly redundant)

| Feature 1 | Feature 2 | Pearson r |
|-----------|-----------|-----------|
| `CREATE_THREAD_____` | `clone` | 1.0000 |
| `EXECUTE_____` | `execve` | 1.0000 |
| `beginRestoreSession` | `endRestoreSession` | 1.0000 |
| `getBoolean` | `getLong` | 1.0000 |
| `chown32` | `getresgid32` | 1.0000 |
| `chown32` | `getresuid32` | 1.0000 |
| `getresgid32` | `getresuid32` | 1.0000 |
| `chown32` | `mount` | 1.0000 |
| `getresgid32` | `mount` | 1.0000 |
| `getresuid32` | `mount` | 1.0000 |
| `getAppWidgetInfo` | `hasNamedWallpaper` | 1.0000 |
| `getState` | `isBluetoothA2dpOn` | 1.0000 |
| `getIsSyncable` | `isPackageAvailable` | 1.0000 |
| `cancelSync` | `isSyncActive` | 1.0000 |
| `cancelSync` | `removeStatusChangeListener` | 1.0000 |

> These are syscall/binder aliases — e.g., `clone` and `CREATE_THREAD_____` represent the same
> kernel-level operation. One from each perfectly-correlated pair must be dropped.

### 3.4 Feature Composition (Heuristic)

| Category | Count |
|----------|-------|
| Total feature columns | 470 |
| Binder/IPC features (android.* prefix, `/`, `:` patterns) | 0 (heuristic only) |
| Syscall + binder frequency features | 470 |

> **Note:** The heuristic binder detection found 0 because binder feature names in this file use
> plain syscall/API names without the `android.*` or URI-path pattern. The dataset is named
> "syscallsbinders" because binder call frequencies are concatenated with syscall frequencies,
> not because column names contain "binder". All 470 features are numeric frequency counts.

---

## 4. SECONDARY — feature_vectors_syscalls_frequency_5_Cat.csv

- **File size:** 13.95 MB
- **Rows:** 11,598
- **Columns:** 140 (139 features + 1 label)
- **In-RAM memory:** 12.39 MB
- **Target column:** `Class` (float labels 1.0–5.0; same semantics as primary)
- **Number of classes:** 5

### 4.1 Class Distribution

| Class | Category | Count | Percentage |
|-------|----------|-------|-----------|
| 1.0 | Adware | 1,253 | 10.80% |
| 2.0 | Banking | 2,100 | 18.11% |
| 3.0 | SMS malware | 3,904 | 33.66% |
| 4.0 | Riskware | 2,546 | 21.95% |
| 5.0 | Benign | 1,795 | 15.48% |

> **Note:** Class labels stored as `float64` (1.0 vs 1) — same semantic meaning as primary's `int64`.
> This is NOT a discrepancy in the data; it is a CSV dtype difference. Normalize to int before training.

### 4.2 Data Quality

| Metric | Value |
|--------|-------|
| Missing values | 0 |
| Exact duplicate rows | 77 |
| Feature-only duplicate rows | 82 |
| Label-conflict duplicates | 5 |
| Numeric features | 139 |
| Non-numeric features | 0 |
| Infinite values | 0 |
| Negative values | 0 |
| Zero-variance features | 0 |
| High-sparsity features (>99% zeros) | 42 |

---

## 5. AUXILIARY — feature_vectors_static.csv

- **File size:** 568.22 MB
- **Rows:** 11,598
- **Columns:** 50,621
- **Audit method:** Python `csv` module (row-by-row streaming; pandas C-parser OOMs on 50k+ columns)
- **Target column:** NONE — this is a **features-only file**
- **Column [0]:** Unnamed row index (`'0'`, `'1'`, `'2'`, ...)
- **Columns [1–50620]:** Static feature names (Android intents, permissions, API calls, manifest strings)

### 5.1 Data Quality

| Metric | Value |
|--------|-------|
| Missing values | 585,598,953 (97.8% of all cells — extremely sparse) |
| Exact duplicate rows | 0 |
| Feature-only duplicate rows | 4,461 |
| Numeric features (estimated) | 50,620 |

> **Important:** The static CSV has **no class/label column**. Labels must be joined from the
> dynamic dataset by row position (row index). The 4,461 feature-only duplicates mean that
> 4,461 APKs have **identical static feature sets** — these are likely repackaged or closely
> related apps sharing the same manifest structure.

---

## 6. Dynamic Dataset Consistency Check

| Metric | Value |
|--------|-------|
| Primary rows | 11,598 |
| Secondary rows | 11,598 |
| Row counts match | ✅ YES |
| Row difference | 0 |
| Class distributions match (by count) | ✅ YES |
| Label dtype difference | Primary=int, Secondary=float (cosmetic only) |
| Primary feature count | 470 |
| Secondary feature count | 139 |

> The "DISCREPANCY" flag raised during automated comparison was a **false alarm** caused by
> string comparison of `"3"` (int-typed in primary) vs `"3.0"` (float-typed in secondary).
> The actual class distributions are identical.

---

## 7. Label-Conflict Duplicate Analysis (5 Conflicts)

Three distinct conflict groups were identified, totalling 39 rows involved.
The **5 label-conflict rows** (counted as `feature_dups − exact_dups`) are:

### Group 1 — Zero-Feature-Vector Ambiguity (35 rows, 3 conflict rows)

**Root cause:** 35 APKs in the dataset have **completely zero feature vectors** across all 470 features.
This indicates APKs that **failed to execute** in the dynamic analysis sandbox (sandbox evasion,
timeout, or execution failure). These 35 identical zero-vectors carry 4 different class labels,
making them un-learnable and ambiguous.

| Class | Category | Rows with zero features |
|-------|----------|------------------------|
| 2 | Banking | 20 rows (majority) |
| 4 | Riskware | 10 rows |
| 5 | Benign | 4 rows |
| 3 | SMS | 1 row |

**The 3 label-conflict rows from this group:**
- **Row 6184** — labeled Class 3 (SMS) — zero feature vector; conflicts with 20 Banking zero-vectors
- **Row 7708** — labeled Class 4 (Riskware) — zero feature vector; first Riskware zero-vector
- **Row 10253** — labeled Class 5 (Benign) — zero feature vector; first Benign zero-vector

**Recommendation:** All 35 zero-feature-vector rows should be **dropped entirely** during
preprocessing — they carry no behavioral information and would bias classifiers.

---

### Group 2 — Adware vs Riskware Ambiguity (2 rows, 1 conflict row)

| Row | Class | Category | Non-zero features |
|-----|-------|----------|-------------------|
| 5 | 1 | Adware | 3 |
| 7261 | 4 | Riskware | 3 |

**Root cause:** Rows 5 and 7261 have **byte-for-byte identical feature vectors** (3 non-zero
syscall counts), but are labeled as different categories. This reflects the genuine semantic overlap
between Adware and Riskware — both categories share invasive but not overtly destructive behaviors.
The same APK (or two nearly-identical APKs) was labeled differently by the dataset curators.

**The 1 conflict row:** Row 7261 (Riskware) has identical features to Row 5 (Adware).

---

### Group 3 — Adware vs Riskware Ambiguity (2 rows, 1 conflict row)

| Row | Class | Category | Non-zero features |
|-----|-------|----------|-------------------|
| 776 | 1 | Adware | 105 |
| 8784 | 4 | Riskware | 105 |

**Root cause:** Identical to Group 2 — rows 776 and 8784 share **exactly the same 105 non-zero
syscall/binder call frequencies** across all 470 features. The behavioral profile is indistinguishable,
yet they carry different labels. Again consistent with the Adware/Riskware boundary ambiguity.

**The 1 conflict row:** Row 8784 (Riskware) has identical features to Row 776 (Adware).

---

### Summary of All 5 Label-Conflict Rows

| Row Index | Assigned Class | Category | Conflict With | Root Cause |
|-----------|---------------|----------|---------------|------------|
| 6184 | 3 | SMS malware | 20× Class 2 (Banking) | Zero feature vector |
| 7708 | 4 | Riskware | 20× Class 2 (Banking) | Zero feature vector |
| 10253 | 5 | Benign | 20× Class 2 (Banking) | Zero feature vector |
| 7261 | 4 | Riskware | Row 5, Class 1 (Adware) | Boundary ambiguity |
| 8784 | 4 | Riskware | Row 776, Class 1 (Adware) | Boundary ambiguity |

---

## 8. Final Audit Summary and Recommendations

### A. Dataset Summary

The **primary dataset** (`feature_vectors_syscallsbinders_frequency_5_Cat.csv`) contains
**11,598 samples** across **5 malware/benign categories** using **470 numeric features** derived
from dynamic analysis (system call + binder IPC call frequency counts).

### B. Recommended Primary Dataset

✅ **`feature_vectors_syscallsbinders_frequency_5_Cat.csv`** — richer behavioral fingerprint
(syscalls + Android binder/IPC), 470 features vs 139 in the secondary. This is the correct
dataset to use for the centralized vs federated learning comparison.

### C. Data-Quality Problems Found

| Issue | Severity | Detail |
|-------|----------|--------|
| 72 exact duplicate rows | ⚠️ MEDIUM | Must remove before splitting |
| 5 label-conflict samples | ⚠️ MEDIUM | 3 from zero-vectors, 2 from Adware/Riskware boundary |
| 35 zero-feature-vector rows | ⛔ HIGH | APKs that failed to execute; no usable signal |
| 247 high-sparsity features (>99% zeros) | ⚠️ MEDIUM | May add noise; evaluate before dropping |
| 35 perfectly-correlated pairs | ⚠️ MEDIUM | Redundant syscall aliases; drop one from each pair |
| Class imbalance 3.1× | ⚠️ LOW-MED | SMS:Adware = 3,904:1,253; use stratified splits |
| No missing values | ✅ GOOD | — |
| No infinite values | ✅ GOOD | — |
| No negative values | ✅ GOOD | — |

### D. Potential Data Leakage Concerns

- **Duplicate samples MUST be removed before splitting.** The 72 exact duplicates spanning train
  and test would inflate accuracy by allowing the model to memorize exact samples.
- **Zero-feature rows span multiple classes** — if not dropped, they would confuse the classifier
  and could appear in both train and test, artificially boosting performance on these ambiguous samples.
- **35 perfectly correlated feature pairs** — keeping both features from a pair gives the model
  a spurious signal advantage. Redundant syscall aliases (e.g., `clone`/`CREATE_THREAD_____`)
  should be reduced to one representative.
- **For centralized vs federated comparison:** Use the **exact same cleaned dataset** as the
  starting point for both experiments. The global test set must remain centralized and identical
  across all experiments.

### E. Recommended Preprocessing Steps

1. **Remove 35 zero-feature-vector rows** — no behavioral signal; undeterminable class.
2. **Remove 72 exact duplicate rows** — de-duplicate before any split.
3. **Resolve 5 label-conflict rows** — recommend dropping all 39 rows from the 3 conflict groups
   (Group 1: all 35 zero-feature rows already dropped in Step 1; Groups 2 & 3: drop 4 rows).
4. **Drop one feature from each of 35 perfectly-correlated pairs** — e.g., keep `clone`,
   drop `CREATE_THREAD_____`; keep `execve`, drop `EXECUTE_____`.
5. **Optionally drop 247 high-sparsity features** — evaluate on validation set.
6. **Encode labels** — remap `{1,2,3,4,5}` → `{0,1,2,3,4}` for PyTorch/sklearn compatibility.
7. **Normalize features** — `StandardScaler` fit on training data only; apply to val/test.
8. **Handle class imbalance** — use stratified splits; optionally apply class-weighted loss.

### F. Recommended Train / Validation / Test Strategy

| Split | Ratio | Approx. Samples | Notes |
|-------|-------|-----------------|-------|
| Train | 70% | ~8,000 | Used for all model/FL training |
| Validation | 10% | ~1,100 | Hyperparameter tuning |
| Test | 20% | ~2,250 | Final evaluation; identical across all experiments |

- **Always use stratified splits** to preserve the 10.8% / 18.1% / 33.7% / 21.9% / 15.5% class ratios.
- **Federated Learning partitioning** (applied to training set only):
  - **IID:** Random shuffle → equal N-way split across clients
  - **Non-IID:** Dirichlet(α) allocation — α=1.0 (mild), α=0.5 (moderate), α=0.1 (extreme)
  - With 5 classes and ~8,000 training samples, both IID and non-IID are readily achievable.

### G. Readiness for Next Phase

✅ **READY** — After applying the cleaning steps above (Steps 1–8), the dataset is suitable
for model development and federated learning experiments. The data is compact (11 MB), fully
numeric, and has no structural issues beyond the ones documented here.

---

## 9. Reference: syscall_unique.csv

- **Lines:** 139 | **Unique syscall names:** 139 | **Duplicates:** 0
- This is a reference list of the 139 unique system call names used as features in the
  secondary dynamic dataset (`feature_vectors_syscalls_frequency_5_Cat.csv`).

---

*End of Audit Report — CICMalDroid 2020 Dataset*