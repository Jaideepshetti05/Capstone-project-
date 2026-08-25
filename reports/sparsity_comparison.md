# CICMalDroid 2020 — Feature Sparsity Comparison Report

**Generated:** 2026-08-25  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Dataset:** `feature_vectors_syscallsbinders_frequency_5_Cat.csv`  
**Evaluation Scope:** Validation set ONLY (Test set remains strictly untouched)  

---

## 1. Executive Summary & Recommendation

| Variant | Features | Validation Accuracy | Validation Macro F1 | Validation Weighted F1 | Recommendation |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Variant A (Keep Sparse)** | **443** | **86.37%** | **0.8450** | **0.8624** | **RECOMMENDED (Official)** |
| **Variant B (Drop Sparse)** | **221** | 85.59% | 0.8390 | 0.8546 | Rejected |
| **Difference (A − B)** | **+222** | **+0.78%** | **+0.0060** | **+0.0078** | *Variant A is superior* |

### Final Recommendation
**Variant A (443 features) should become our official dataset for the modeling and federated learning phase.**

**Key Findings:**
1. **Higher Overall Performance:** Variant A achieves **86.37% accuracy** and **0.8450 Macro F1**, outperforming Variant B (85.59% accuracy, 0.8390 Macro F1).
2. **Critical Class-Discriminative Signals Preserved:** Dropping the >99% sparse features degraded detection performance on **Banking malware** (F1 dropped from 0.7981 to 0.7818, −1.63%) and **Riskware** (F1 dropped from 0.8536 to 0.8299, −2.37%).
3. **Sparse Signals are Meaningful:** As demonstrated in the research review, features like `getUserIcon`, `getUsers`, `getDeviceOwner`, and `setRingerMode` appear infrequently but are heavily concentrated in specific malware families. Retaining them provides indispensable discriminative signal.

---

## 2. Preprocessing & Data Cleaning Summary

All cleaning steps followed the approved research decisions:

| Stage | Action | Rows / Features | Notes |
|:---|:---|:---:|:---|
| **Raw Input** | Load original dataset | 11,598 rows x 471 cols | Raw CSV completely unchanged |
| **Quality Filter 1** | Drop all-zero feature rows | −35 rows | Sandbox execution failures (span 4 classes) |
| **Quality Filter 2** | Drop conflict pairs (Groups 2 & 3) | −4 rows | Rows 5, 776, 7261, 8784 (Adware/Riskware ambiguity) |
| **Quality Filter 3** | Drop exact duplicate rows | −41 rows | Retained 1st instance |
| **Cleaned Dataset** | Zero remaining duplicates/conflicts | **11,518 rows** | Total removed: 80 rows (0.69%) |
| **Correlation Drop** | Drop redundant features | −27 features | Derived from 35 perfect-correlation pairs |
| **Variant A Features**| Full approved feature set | **443 features** | Saved to `data/processed/variant_a/` |
| **Variant B Features**| Drop >99% sparse in training | **221 features** | 222 sparse features removed |

---

## 3. Data Splitting & Scaling Protocol

A strict, leak-free protocol was implemented:

1. **Split Strategy:** 70% Train / 10% Validation / 20% Test, stratified on target labels with `random_state=42`.
2. **Split Sizes:**
   - **Train:** 8,062 samples (70.00%)
   - **Validation:** 1,152 samples (10.00%)
   - **Test:** 2,304 samples (20.00%) — **Untouched**
3. **Class Balance Across Splits:**

| Class Label | Category | Cleaned Total | Train (70%) | Validation (10%) | Test (20%) |
|:---|:---|:---:|:---:|:---:|:---:|
| 1 (Encoded 0) | Adware | 1,251 (10.86%) | 875 | 125 | 251 |
| 2 (Encoded 1) | Banking malware | 2,043 (17.74%) | 1,430 | 205 | 408 |
| 3 (Encoded 2) | SMS malware | 3,902 (33.88%) | 2,731 | 390 | 781 |
| 4 (Encoded 3) | Riskware | 2,531 (21.97%) | 1,772 | 253 | 506 |
| 5 (Encoded 4) | Benign | 1,791 (15.55%) | 1,254 | 179 | 358 |
| **Total** | | **11,518** | **8,062** | **1,152** | **2,304** |

4. **StandardScaler Fitting:**
   - Fit `StandardScaler` **strictly on `X_train`**.
   - Scaler applied to transform `X_val` and `X_test`.
   - Zero test/validation distribution information leaked into preprocessing.

---

## 4. Detailed Validation Results Comparison

Baseline model: `LogisticRegression(random_state=42, max_iter=2000)` evaluated on the centralized validation set.

### 4.1 Aggregate Metrics

| Metric | Variant A (443 Features) | Variant B (221 Features) | Delta (A − B) |
|:---|:---:|:---:|:---:|
| **Accuracy** | **0.8637** | 0.8559 | **+0.0078 (+0.78%)** |
| **Macro F1** | **0.8450** | 0.8390 | **+0.0060** |
| **Weighted F1** | **0.8624** | 0.8546 | **+0.0078** |
| **Macro Precision** | **0.8552** | 0.8476 | **+0.0076** |
| **Macro Recall** | **0.8382** | 0.8333 | **+0.0049** |

### 4.2 Per-Class Performance Breakdown

| Class | Variant A Precision | Variant A Recall | Variant A F1 | Variant B Precision | Variant B Recall | Variant B F1 | F1 Delta (A − B) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Adware (0)** | 0.7874 | 0.8000 | 0.7937 | 0.8095 | 0.8160 | **0.8127** | −0.0190 |
| **Banking (1)** | **0.7961** | **0.8000** | **0.7981** | 0.7689 | 0.7951 | 0.7818 | **+0.0163** |
| **SMS (2)** | 0.8843 | **0.9795** | **0.9294** | **0.8876** | 0.9718 | 0.9278 | **+0.0016** |
| **Riskware (3)** | **0.8922** | **0.8182** | **0.8536** | 0.8734 | 0.7905 | 0.8299 | **+0.0237** |
| **Benign (4)** | **0.9161** | 0.7933 | **0.8503** | 0.8987 | 0.7933 | 0.8427 | **+0.0076** |

### 4.3 Validation Confusion Matrices

#### Variant A (443 features)
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware       100       12       7         5        1 |   125
Banking        6      164      17         9        9 |   205
SMS            1        6     382         1        0 |   390
Riskware      12       19      12       207        3 |   253
Benign         8        5      14        10      142 |   179
```

#### Variant B (221 features)
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware       102       12       5         6        0 |   125
Banking        7      163      18        10        7 |   205
SMS            0        8     379         3        0 |   390
Riskware      10       22      12       200        9 |   253
Benign         7        7      13        10      142 |   179
```

---

## 5. Artifact Directory Structure

All processed artifacts are saved under `data/processed/`:

```
data/processed/
├── cleaned_dataset.csv             # 11,518 rows x 444 columns (443 features + Class)
├── label_mapping.json              # Integer encoding and category metadata
├── split_indices.json              # Exact row indices for Train / Val / Test
├── variant_a/                      # OFFICIAL RECOMMENDED DATASET
│   ├── X_train.csv                 # 8,062 rows x 443 features (StandardScaled)
│   ├── X_val.csv                   # 1,152 rows x 443 features (StandardScaled)
│   ├── X_test.csv                  # 2,304 rows x 443 features (StandardScaled, UNTOUCHED)
│   ├── y_train.csv                 # 8,062 labels (0-4)
│   ├── y_val.csv                   # 1,152 labels (0-4)
│   ├── y_test.csv                  # 2,304 labels (0-4)
│   ├── scaler_a.joblib             # Fitted StandardScaler object
│   └── features.json               # 443 feature column names
└── variant_b/                      # COMPARISON DATASET
    ├── X_train.csv                 # 8,062 rows x 221 features (StandardScaled)
    ├── X_val.csv                   # 1,152 rows x 221 features (StandardScaled)
    ├── X_test.csv                  # 2,304 rows x 221 features (StandardScaled, UNTOUCHED)
    ├── y_train.csv                 # 8,062 labels (0-4)
    ├── y_val.csv                   # 1,152 labels (0-4)
    ├── y_test.csv                  # 2,304 labels (0-4)
    ├── scaler_b.joblib             # Fitted StandardScaler object
    ├── features.json               # 221 feature column names
    └── dropped_sparse_features.json # 222 dropped feature column names
```

---

## 6. Readiness for Centralized & Federated Learning

1. **Clean Baseline Established:** Dataset is 100% clean, scaled, encoded, and split.
2. **Strict Test Set Integrity:** The test set of 2,304 samples has remained completely untouched.
3. **Next Steps (Pending User Direction):**
   - Implement Centralized Baseline Models (e.g. MLP / Neural Network, Random Forest) on Variant A.
   - Implement Federated Learning partitioning (IID and Non-IID Dirichlet) on `X_train` / `y_train` from Variant A.
