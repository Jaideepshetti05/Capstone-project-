# Experiment Report: dp_secagg

**Generated:** 2026-08-26 20:25:41  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: True, SecAgg: True)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **74.57%** |
| **Test Macro F1** | **0.7041** |
| **Test Weighted F1** | **0.7410** |
| **Test Macro Precision** | **0.7214** |
| **Test Macro Recall** | **0.6951** |
| **Differential Privacy Guarantee** | **ε = 25.5143, δ = 1e-05** |
| **Secure Aggregation** | **Enabled (Pairwise Additive Masking)** |
| **Best Validation Round** | **Round 49** (Val F1: 0.7323, Val Acc: 77.17%) |
| **Training Time** | **1530.25s** |
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
| Round 10 | 1.1778 | 55.47% | 0.3962 | 0.4847 | 10.79 | 119.05 MB |
| Round 15 | 1.1169 | 60.07% | 0.4549 | 0.5432 | 13.30 | 178.57 MB |
| Round 20 | 1.0852 | 62.76% | 0.4932 | 0.5807 | 15.82 | 238.09 MB |
| Round 25 | 1.0265 | 67.36% | 0.5705 | 0.6436 | 18.33 | 297.62 MB |
| Round 30 | 0.9676 | 70.14% | 0.6251 | 0.6847 | 19.91 | 357.14 MB |
| Round 35 | 0.9193 | 73.18% | 0.6797 | 0.7233 | 21.31 | 416.66 MB |
| Round 40 | 0.8821 | 75.26% | 0.7072 | 0.7470 | 22.71 | 476.19 MB |
| Round 45 | 0.8637 | 75.61% | 0.7135 | 0.7503 | 24.11 | 535.71 MB |
| Round 49 🌟 (Best) | 0.8461 | 77.17% | 0.7323 | 0.7672 | 25.23 | 583.33 MB |
| Round 50 | 0.8382 | 75.78% | 0.7174 | 0.7521 | 25.51 | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.5366 | 0.5280 | **0.5323** | 250 |
| **Banking malware** | 0.6795 | 0.6895 | **0.6845** | 409 |
| **SMS malware** | 0.7916 | 0.9437 | **0.8610** | 781 |
| **Riskware** | 0.7820 | 0.6522 | **0.7112** | 506 |
| **Benign** | 0.8172 | 0.6620 | **0.7315** | 358 |
| **Macro Average** | **0.7214** | **0.6951** | **0.7041** | 2,304 |
| **Weighted Average** | **0.7214** | **0.6951** | **0.7410** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware        132       28      30         46       14 |   250
Banking        32      282      60         16       19 |   409
SMS            13       27     737          3        1 |   781
Riskware       39       55      63        330       19 |   506
Benign         30       23      41         27      237 |   358
```
