# Experiment Report: attack_weightpoison_trimmed_mean

**Generated:** 2026-09-14 22:28:38  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: False, SecAgg: False)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **90.02%** |
| **Test Macro F1** | **0.8830** |
| **Test Weighted F1** | **0.8993** |
| **Test Macro Precision** | **0.8928** |
| **Test Macro Recall** | **0.8761** |
| **Differential Privacy Guarantee** | **ε = None, δ = None** |
| **Secure Aggregation** | **Disabled** |
| **Best Validation Round** | **Round 47** (Val F1: 0.8827, Val Acc: 89.93%) |
| **Training Time** | **74.84s** |
| **Total Communication** | **595.23 MB** |

> [!NOTE]
> **Privacy Definition:** DP bounds the probability ratio of outputs on adjacent datasets by exp(ε). Secure Aggregation provides cryptographic confidentiality by masking individual updates so the server only observes the aggregated sum.

---

## 2. Configuration

| Parameter | Value |
|:---|:---|
| **Algorithm** | FedAvg |
| **DP Enabled** | False |
| **Noise Multiplier (sigma)** | None |
| **Clipping Norm (C)** | None |
| **Target Delta** | None |
| **SecAgg Enabled** | False |
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
| Round 01 | 1.1628 | 62.24% | 0.5492 | 0.5923 | N/A | 11.90 MB |
| Round 05 | 0.5354 | 83.51% | 0.8110 | 0.8340 | N/A | 59.52 MB |
| Round 10 | 0.4634 | 86.72% | 0.8455 | 0.8663 | N/A | 119.05 MB |
| Round 15 | 0.4195 | 86.46% | 0.8424 | 0.8638 | N/A | 178.57 MB |
| Round 20 | 0.4059 | 87.50% | 0.8547 | 0.8744 | N/A | 238.09 MB |
| Round 25 | 0.3966 | 88.19% | 0.8632 | 0.8813 | N/A | 297.62 MB |
| Round 30 | 0.3819 | 88.37% | 0.8655 | 0.8833 | N/A | 357.14 MB |
| Round 35 | 0.3866 | 89.41% | 0.8758 | 0.8932 | N/A | 416.66 MB |
| Round 40 | 0.3744 | 89.24% | 0.8740 | 0.8920 | N/A | 476.19 MB |
| Round 45 | 0.3733 | 90.02% | 0.8825 | 0.8995 | N/A | 535.71 MB |
| Round 47 🌟 (Best) | 0.3775 | 89.93% | 0.8827 | 0.8988 | N/A | 559.52 MB |
| Round 50 | 0.3886 | 89.58% | 0.8784 | 0.8947 | N/A | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.8553 | 0.7800 | **0.8159** | 250 |
| **Banking malware** | 0.8300 | 0.9071 | **0.8668** | 409 |
| **SMS malware** | 0.9232 | 0.9846 | **0.9529** | 781 |
| **Riskware** | 0.9315 | 0.8597 | **0.8941** | 506 |
| **Benign** | 0.9240 | 0.8492 | **0.8850** | 358 |
| **Macro Average** | **0.8928** | **0.8761** | **0.8830** | 2,304 |
| **Weighted Average** | **0.8928** | **0.8761** | **0.8993** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware        195       25      13          9        8 |   250
Banking        14      371      16          7        1 |   409
SMS             0       11     769          1        0 |   781
Riskware       12       22      21        435       16 |   506
Benign          7       18      14         15      304 |   358
```
