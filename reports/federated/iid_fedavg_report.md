# CICMalDroid 2020 — Federated Learning (IID FedAvg) Report

**Generated:** 2026-08-25 15:13:15  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** Federated Averaging (FedAvg, McMahan et al., 2017)  
**Partition Strategy:** Stratified IID across 10 clients  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary & Benchmark Comparison

| Experiment Setup | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Test Weighted F1 | Total Communication |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Centralized MLP Baseline** | **91.28%** | **0.8986** | **0.8937** | **0.8957** | **0.9124** | 0.0 MB (Local) |
| **FedAvg (IID, 10 Clients, 50 Rds)** | **90.19%** | **0.8959** | **0.8776** | **0.8853** | **0.9009** | **595.23 MB** |
| **Performance Difference** | **-1.09%** | **-0.0027** | **-0.0161** | **-0.0104** | **-0.0115** | — |

### Key Findings:
1. **Strong Federated Convergence:** FedAvg on stratified IID data reaches **90.19% test accuracy** and **0.8853 Macro F1**, closely tracking the centralized MLP benchmark with zero raw data sharing.
2. **Deterministic & Leak-Free:** Strict validation assertions confirmed that all 8,062 training samples were partitioned with zero duplication across clients. Validation and test sets remained completely centralized.
3. **Communication Efficiency:** Total communication across all 50 rounds was **595.23 MB** (11.9047 MB/round for all 10 clients), confirming lightweight transmission requirements.

---

## 2. Experimental Configuration

| Parameter | Value | Notes |
|:---|:---|:---|
| **Federated Algorithm** | FedAvg (Weighted Parameter Averaging) | $\mathbf{w}_{\text{global}} = \sum \frac{n_k}{N} \mathbf{w}_k$ |
| **Number of Clients ($K$)** | 10 | 10 independent simulated nodes |
| **Client Participation ($C$)** | 100% (1.0) | All 10 clients participate every round |
| **Global Communication Rounds** | 50 | Evaluated on validation set after each round |
| **Local Client Epochs ($E$)** | 2 | Local training passes per communication round |
| **Local Batch Size ($B$)** | 64 | Mini-batch SGD / AdamW |
| **Local Optimizer** | AdamW | Learning rate = 0.001, Weight decay = 0.0001 |
| **Loss Function** | CrossEntropyLoss | Multi-class categorical cross-entropy |
| **Random Seed** | 42 | Deterministic Python, NumPy, PyTorch seeds |
| **Dataset & Features** | Variant A (443 features) | 8,062 Train / 1,152 Val / 2,304 Test |
| **Total Parameters** | 156,037 | float32 (~609.52 KB payload) |

---

## 3. Client Data Distribution Table (IID Partition)

Total training samples assigned: **8,062 / 8,062** (100% coverage, 0 duplicates, 0 unassigned).

| Client ID | Total Samples | Adware (0) | Banking (1) | SMS (2) | Riskware (3) | Benign (4) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Client 00 | 809 (10.0%) | 88 | 143 | 274 | 178 | 126 |
| Client 01 | 808 (10.0%) | 88 | 143 | 273 | 178 | 126 |
| Client 02 | 807 (10.0%) | 88 | 143 | 273 | 177 | 126 |
| Client 03 | 807 (10.0%) | 88 | 143 | 273 | 177 | 126 |
| Client 04 | 806 (10.0%) | 88 | 143 | 273 | 177 | 125 |
| Client 05 | 806 (10.0%) | 88 | 143 | 273 | 177 | 125 |
| Client 06 | 805 (10.0%) | 87 | 143 | 273 | 177 | 125 |
| Client 07 | 805 (10.0%) | 87 | 143 | 273 | 177 | 125 |
| Client 08 | 805 (10.0%) | 87 | 143 | 273 | 177 | 125 |
| Client 09 | 804 (10.0%) | 87 | 142 | 273 | 177 | 125 |
| **Total Global Train** | **8,062** | **876** | **1,429** | **2,731** | **1,772** | **1,254** |

---

## 4. Round-by-Round Validation Convergence

*Validation metrics tracked on the centralized validation set (1,152 samples) at the conclusion of each communication round:*

| Global Round | Validation Loss | Validation Accuracy | Validation Macro F1 | Validation Weighted F1 | Cumulative Comm. |
|:---|:---:|:---:|:---:|:---:|:---:|
| Round 01 | 1.1715 | 61.55% | 0.5391 | 0.5842 | 11.90 MB |
| Round 05 | 0.5375 | 83.77% | 0.8143 | 0.8361 | 59.52 MB |
| Round 10 | 0.4522 | 87.15% | 0.8506 | 0.8705 | 119.05 MB |
| Round 15 | 0.4222 | 87.24% | 0.8518 | 0.8713 | 178.57 MB |
| Round 20 | 0.4086 | 88.11% | 0.8633 | 0.8800 | 238.09 MB |
| Round 25 | 0.3828 | 88.80% | 0.8710 | 0.8872 | 297.62 MB |
| Round 30 | 0.3794 | 88.37% | 0.8677 | 0.8827 | 357.14 MB |
| Round 35 | 0.3844 | 89.76% | 0.8825 | 0.8968 | 416.66 MB |
| Round 40 | 0.3765 | 89.67% | 0.8812 | 0.8962 | 476.19 MB |
| Round 45 | 0.3517 | 90.36% | 0.8880 | 0.9033 | 535.71 MB |
| Round 49 🌟 (Best) | 0.3575 | 90.45% | 0.8903 | 0.9040 | 583.33 MB |
| Round 50 | 0.3591 | 90.19% | 0.8870 | 0.9011 | 595.23 MB |

**Best Validation Round:** **Round 49** (Val Accuracy = **90.45%**, Val Macro F1 = **0.8903**, Val Loss = **0.3575**).

---

## 5. Locked Test Set Performance (2,304 Samples)

*Evaluated ONCE using the best global model checkpoint (Round 49) strictly after training completion.*

### 5.1 Per-Class Metrics

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.8546 | 0.7760 | **0.8134** | 250 |
| **Banking malware** | 0.8477 | 0.9120 | **0.8787** | 409 |
| **SMS malware** | 0.9166 | 0.9846 | **0.9494** | 781 |
| **Riskware** | 0.9278 | 0.8636 | **0.8946** | 506 |
| **Benign** | 0.9327 | 0.8520 | **0.8905** | 358 |
| **Macro Average** | **0.8959** | **0.8776** | **0.8853** | 2,304 |
| **Weighted Average** | **0.8959** | **0.8776** | **0.9009** | 2,304 |

### 5.2 Test Set Confusion Matrix

```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware        194       21      13         13        9 |   250
Banking        10      373      17          8        1 |   409
SMS             0       11     769          1        0 |   781
Riskware       12       19      26        437       12 |   506
Benign         11       16      14         12      305 |   358
```

---

## 6. Communication & Compute Accounting

- **Parameters per Model Transfer:** 156,037
- **Precision:** 32-bit floating point (4 bytes / parameter)
- **Model Payload Size:** 624,148 bytes (609.52 KB / transfer)
- **Downlink per Round:** 6,241,480 bytes (5.95 MB across 10 clients)
- **Uplink per Round:** 6,241,480 bytes (5.95 MB across 10 clients)
- **Total Round Bandwidth:** 12,482,960 bytes (11.9047 MB)
- **Total Experiment Bandwidth (50 rounds):** **624,148,000 bytes (595.23 MB)**
- **Total Wall-Clock Training Time:** **39.19 seconds** (0.78 s/round)
- **Inference Latency:** 0.0079 ms/sample

---

## 7. Artifacts Saved

- Model Checkpoints: `models/federated/iid/best_model.pt`, `models/federated/iid/final_model.pt`
- Partition Metadata: `models/federated/iid/partition.json`
- Configuration: `models/federated/iid/config.json`
- Experiment Results: `results/federated/iid_metrics.json`
