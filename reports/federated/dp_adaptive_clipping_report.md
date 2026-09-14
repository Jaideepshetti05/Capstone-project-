# Experiment Report: dp_adaptive_clipping

**Generated:** 2026-09-14 21:39:12  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: True, SecAgg: True)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **75.35%** |
| **Test Macro F1** | **0.7122** |
| **Test Weighted F1** | **0.7495** |
| **Test Macro Precision** | **0.7293** |
| **Test Macro Recall** | **0.7032** |
| **Differential Privacy Guarantee** | **ε = 25.5143, δ = 1e-05** |
| **Secure Aggregation** | **Enabled (Pairwise Additive Masking)** |
| **Best Validation Round** | **Round 46** (Val F1: 0.7317, Val Acc: 77.17%) |
| **Training Time** | **1899.89s** |
| **Total Communication** | **595.23 MB** |

> [!NOTE]
> **Privacy Definition:** DP bounds the probability ratio of outputs on adjacent datasets by exp(ε). Secure Aggregation provides cryptographic confidentiality by masking individual updates so the server only observes the aggregated sum.

---

## 2. Configuration

| Parameter | Value |
|:---|:---|
| **Algorithm** | FedAvg |
| **DP Enabled** | True |
| **Noise Multiplier (sigma)** | 1.0 |
| **Clipping Norm (C)** | 1.0 |
| **Target Delta** | 1e-05 |
| **SecAgg Enabled** | True |
| **Partition** | iid (alpha = None) |
| **Clients (K)** | 10 |
| **Rounds (T)** | 50 |
| **Local Epochs (E)** | 2 |
| **Batch Size (B)** | 64 |
| **Learning Rate** | 0.001 |
| **Optimizer** | AdamW (wd = 0.0001) |

---

## 3. Client Data Distribution

| Client ID | Total Samples | Adware (0) | Banking (1) | SMS (2) | Riskware (3) | Benign (4) | Dominant Class (% of Client) | Entropy | TVD to Prior |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---:|
| Client 00 | 809 (10.0%) | 88 | 143 | 274 | 178 | 126 | SMS malware (33.87%) | 2.22 b | 0.001 |
| Client 01 | 808 (10.0%) | 88 | 143 | 273 | 178 | 126 | SMS malware (33.79%) | 2.22 b | 0.001 |
| Client 02 | 807 (10.0%) | 88 | 143 | 273 | 177 | 126 | SMS malware (33.83%) | 2.22 b | 0.001 |
| Client 03 | 807 (10.0%) | 88 | 143 | 273 | 177 | 126 | SMS malware (33.83%) | 2.22 b | 0.001 |
| Client 04 | 806 (10.0%) | 88 | 143 | 273 | 177 | 125 | SMS malware (33.87%) | 2.22 b | 0.001 |
| Client 05 | 806 (10.0%) | 88 | 143 | 273 | 177 | 125 | SMS malware (33.87%) | 2.22 b | 0.001 |
| Client 06 | 805 (10.0%) | 87 | 143 | 273 | 177 | 125 | SMS malware (33.91%) | 2.22 b | 0.001 |
| Client 07 | 805 (10.0%) | 87 | 143 | 273 | 177 | 125 | SMS malware (33.91%) | 2.22 b | 0.001 |
| Client 08 | 805 (10.0%) | 87 | 143 | 273 | 177 | 125 | SMS malware (33.91%) | 2.22 b | 0.001 |
| Client 09 | 804 (10.0%) | 87 | 142 | 273 | 177 | 125 | SMS malware (33.96%) | 2.22 b | 0.001 |
| **Total Global Train** | **8,062** | **876** | **1,429** | **2,731** | **1,772** | **1,254** | **SMS (33.87%)** | **2.20 b** | **0.000** |

---

## 4. Round-by-Round Validation History

| Global Round | Validation Loss | Validation Accuracy | Validation Macro F1 | Validation Weighted F1 | Epsilon (ε) | Cumulative Comm. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Round 01 | 1.6534 | 31.68% | 0.2317 | 0.2859 | 4.16 | 11.90 MB |
| Round 05 | 1.3693 | 45.75% | 0.2921 | 0.3702 | 7.66 | 59.52 MB |
| Round 10 | 1.1777 | 55.47% | 0.3962 | 0.4847 | 10.79 | 119.05 MB |
| Round 15 | 1.0982 | 59.98% | 0.4549 | 0.5429 | 13.30 | 178.57 MB |
| Round 20 | 1.0411 | 62.59% | 0.4922 | 0.5777 | 15.82 | 238.09 MB |
| Round 25 | 0.9713 | 67.97% | 0.5853 | 0.6534 | 18.33 | 297.62 MB |
| Round 30 | 0.9146 | 71.70% | 0.6480 | 0.7025 | 19.91 | 357.14 MB |
| Round 35 | 0.8634 | 75.52% | 0.7086 | 0.7487 | 21.31 | 416.66 MB |
| Round 40 | 0.8274 | 74.74% | 0.7012 | 0.7422 | 22.71 | 476.19 MB |
| Round 45 | 0.8062 | 75.95% | 0.7174 | 0.7546 | 24.11 | 535.71 MB |
| Round 46 🌟 (Best) | 0.8033 | 77.17% | 0.7317 | 0.7674 | 24.39 | 547.62 MB |
| Round 50 | 0.7770 | 75.78% | 0.7170 | 0.7528 | 25.51 | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.5492 | 0.5360 | **0.5425** | 250 |
| **Banking malware** | 0.6651 | 0.6993 | **0.6818** | 409 |
| **SMS malware** | 0.8063 | 0.9437 | **0.8696** | 781 |
| **Riskware** | 0.7921 | 0.6779 | **0.7306** | 506 |
| **Benign** | 0.8339 | 0.6592 | **0.7363** | 358 |
| **Macro Average** | **0.7293** | **0.7032** | **0.7122** | 2,304 |
| **Weighted Average** | **0.7293** | **0.7032** | **0.7495** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware        134       26      30         47       13 |   250
Banking        34      286      57         14       18 |   409
SMS            10       29     737          4        1 |   781
Riskware       33       61      54        343       15 |   506
Benign         33       28      36         25      236 |   358
```
