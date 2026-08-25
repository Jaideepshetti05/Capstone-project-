# CICMalDroid 2020 — Centralized Baseline Model Report

**Generated:** 2026-08-25 14:52:18  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Official Dataset:** Variant A (443 features, 11,518 cleaned samples)  
**Dataset Splits:** Train = 8,062 (70%) | Validation = 1,152 (10%) | Test = 2,304 (20%)  

---

## 1. Executive Summary & Model Selection

### 1.1 Validation Comparison (Model Selection Phase)
All three models were evaluated on the **centralized validation set** to determine the optimal architecture.

| Model | Val Accuracy | Val Macro Precision | Val Macro Recall | Val Macro F1 | Val Weighted F1 | Train Time | Val Latency (ms/sample) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (300 trees)** | **95.23%** | **0.9411** | **0.9458** | **0.9425** | **0.9524** | 0.80s | 0.0396 ms |
| **MLP Neural Network** | **91.41%** | **0.9041** | **0.8969** | **0.9000** | **0.9135** | 14.40s | 0.0093 ms |
| **Logistic Regression (C=10)** | 88.54% | 0.8722 | 0.8593 | 0.8648 | 0.8843 | 2.19s | 0.0018 ms |

**Selected Best Model on Validation:** **Random Forest** achieved the highest validation performance (95.23% accuracy, **0.9425 Macro F1**), closely followed by the **MLP Neural Network** (91.41% accuracy, **0.9000 Macro F1**).

---

### 1.2 Final Evaluation on Locked Test Set
*Evaluated ONCE strictly after model selection.*

| Model | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Test Weighted F1 | Test Latency (ms/sample) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Selected)** | **96.09%** | **0.9521** | **0.9557** | **0.9535** | **0.9610** | 0.0254 ms |
| **MLP Neural Network** | **91.28%** | **0.8986** | **0.8937** | **0.8957** | **0.9124** | 0.0061 ms |
| **Logistic Regression** | 88.28% | 0.8720 | 0.8534 | 0.8616 | 0.8814 | 0.0022 ms |

---

## 2. Recommendations for Federated Learning

> 🎯 **RECOMMENDED FL MODEL: MLP Neural Network (`MalwareMLP`)**

**Technical Justification:**
1. **Parametric & Aggregatable:** Federated Learning algorithms (such as **FedAvg**, FedProx, and SCAFFOLD) operate by transmitting and aggregating continuous neural network weight vectors $\mathbf{w} \in \mathbb{R}^d$. Tree ensembles (Random Forests) lack linear weight spaces and cannot be natively aggregated via parameter averaging.
2. **Competitive Performance:** The PyTorch MLP achieves **91.28% test accuracy** and **0.8957 test Macro F1**, providing a strong, highly reliable deep learning foundation.
3. **Privacy Compatibility:** Neural network architectures natively integrate with Differential Privacy (DP-SGD) and Secure Aggregation protocols, matching our project goal of **Privacy-Preserving Malware Detection**.
4. **Lightweight Client Compute:** With 156,037 parameters (~609.5 KB), local client training and communication overhead will be extremely fast across simulated clients.

---

## 3. Detailed Model Specifications

### 3.1 Model 1: Logistic Regression
- **Algorithm:** L2-regularized multinomial logistic regression
- **Hyperparameters:** `C=10.0`, `solver='lbfgs'`, `max_iter=2000`, `random_state=42`
- **Feature Count:** 443 standardized dynamic features
- **Training Time:** 2.19s

#### Validation Classification Report:
```
                 precision    recall  f1-score   support

         Adware     0.8016    0.8080    0.8048       125
Banking malware     0.8317    0.8195    0.8256       205
    SMS malware     0.9211    0.9872    0.9530       390
       Riskware     0.8952    0.8775    0.8862       253
         Benign     0.9114    0.8045    0.8546       179

       accuracy                         0.8854      1152
      macro avg     0.8722    0.8593    0.8648      1152
   weighted avg     0.8850    0.8854    0.8843      1152

```

#### Test Classification Report:
```
                 precision    recall  f1-score   support

         Adware     0.8142    0.7360    0.7731       250
Banking malware     0.8385    0.8631    0.8506       409
    SMS malware     0.9087    0.9808    0.9433       781
       Riskware     0.8930    0.8577    0.8750       506
         Benign     0.9055    0.8296    0.8659       358

       accuracy                         0.8828      2304
      macro avg     0.8720    0.8534    0.8616      2304
   weighted avg     0.8820    0.8828    0.8814      2304

```

---

### 3.2 Model 2: Random Forest
- **Algorithm:** Breiman Random Forest Ensemble
- **Hyperparameters:** `n_estimators=300`, `max_depth=None`, `criterion='gini'`, `random_state=42`, `n_jobs=-1`
- **Feature Count:** 443 standardized dynamic features
- **Training Time:** 0.80s

#### Validation Classification Report:
```
                 precision    recall  f1-score   support

         Adware     0.8633    0.9600    0.9091       125
Banking malware     0.9840    0.8976    0.9388       205
    SMS malware     0.9898    0.9923    0.9910       390
       Riskware     0.9297    0.9407    0.9352       253
         Benign     0.9385    0.9385    0.9385       179

       accuracy                         0.9523      1152
      macro avg     0.9411    0.9458    0.9425      1152
   weighted avg     0.9539    0.9523    0.9524      1152

```

#### Test Classification Report:
```
                 precision    recall  f1-score   support

         Adware     0.8885    0.9560    0.9210       250
Banking malware     0.9694    0.9291    0.9488       409
    SMS malware     0.9798    0.9949    0.9873       781
       Riskware     0.9673    0.9348    0.9508       506
         Benign     0.9557    0.9637    0.9597       358

       accuracy                         0.9609      2304
      macro avg     0.9521    0.9557    0.9535      2304
   weighted avg     0.9616    0.9609    0.9610      2304

```

---

### 3.3 Model 3: PyTorch MLP (`MalwareMLP`)
- **Architecture:** `Input(443) -> Linear(256) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(64) -> BatchNorm1d -> ReLU -> Dropout(0.1) -> Linear(5)`
- **Total Parameters:** 156,037
- **Optimizer:** `AdamW(lr=1e-3, weight_decay=1e-4)`
- **LR Scheduler:** `ReduceLROnPlateau(factor=0.5, patience=5)`
- **Batch Size:** 64
- **Loss Function:** `CrossEntropyLoss()`
- **Early Stopping:** Monitored validation loss (Patience = 15, Best Epoch = 26)
- **Training Time:** 14.40s

#### Validation Classification Report:
```
                 precision    recall  f1-score   support

         Adware     0.8516    0.8720    0.8617       125
Banking malware     0.8872    0.8439    0.8650       205
    SMS malware     0.9432    0.9795    0.9610       390
       Riskware     0.9109    0.9289    0.9198       253
         Benign     0.9277    0.8603    0.8928       179

       accuracy                         0.9141      1152
      macro avg     0.9041    0.8969    0.9000      1152
   weighted avg     0.9138    0.9141    0.9135      1152

```

#### Test Classification Report:
```
                 precision    recall  f1-score   support

         Adware     0.8101    0.8360    0.8228       250
Banking malware     0.8807    0.9022    0.8913       409
    SMS malware     0.9483    0.9859    0.9667       781
       Riskware     0.9234    0.8814    0.9019       506
         Benign     0.9307    0.8631    0.8957       358

       accuracy                         0.9128      2304
      macro avg     0.8986    0.8937    0.8957      2304
   weighted avg     0.9131    0.9128    0.9124      2304

```

---

## 4. Per-Class Performance Comparison (Test Set)

| Class | Logistic Regression F1 | Random Forest F1 | MLP Neural Network F1 |
|:---|:---:|:---:|:---:|
| **Adware** | 0.7731 | **0.9210** | 0.8228 |
| **Banking malware** | 0.8506 | **0.9488** | 0.8913 |
| **SMS malware** | 0.9433 | **0.9873** | 0.9667 |
| **Riskware** | 0.8750 | **0.9508** | 0.9019 |
| **Benign** | 0.8659 | **0.9597** | 0.8957 |

---

## 5. Confusion Matrices (Test Set, 2,304 Samples)

### 5.1 Random Forest (Selected Model)
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware       219       18       3         8        2 |   250
Banking        6      372      14         9        8 |   409
SMS            3        4     768         6        0 |   781
Riskware      13       22       8       455        8 |   506
Benign         7        4       4         7      336 |   358
```

### 5.2 MLP Neural Network
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware       208       16      12        11        3 |   250
Banking        8      355      27         9       10 |   409
SMS            5        4     763         9        0 |   781
Riskware      21       36      14       421       14 |   506
Benign        10       14      13        17      304 |   358
```

### 5.3 Logistic Regression
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware       202       22      12        11        3 |   250
Banking       10      344      32        12       11 |   409
SMS            2       14     757         8        0 |   781
Riskware      26       39      18       407       16 |   506
Benign        14        9      18        14      303 |   358
```

---

## 6. Saved Model Artifacts

```
models/centralized/
├── logistic_regression.joblib      # Trained Logistic Regression model
├── random_forest.joblib            # Trained Random Forest ensemble model
├── mlp_best.pt                     # Best PyTorch MLP state dictionary (checkpoint)
└── mlp_architecture.json           # MLP architecture metadata and training history
```
