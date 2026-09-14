# Experiment Report: attack_labelflip_fedavg

**Generated:** 2026-09-14 22:18:57  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: False, SecAgg: False)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **89.58%** |
| **Test Macro F1** | **0.8716** |
| **Test Weighted F1** | **0.8934** |
| **Test Macro Precision** | **0.8888** |
| **Test Macro Recall** | **0.8612** |
| **Differential Privacy Guarantee** | **ε = None, δ = None** |
| **Secure Aggregation** | **Disabled** |
| **Best Validation Round** | **Round 50** (Val F1: 0.8817, Val Acc: 90.02%) |
| **Training Time** | **73.90s** |
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
| Round 01 | 1.2481 | 62.50% | 0.5477 | 0.5940 | N/A | 11.90 MB |
| Round 05 | 0.6610 | 84.03% | 0.8142 | 0.8383 | N/A | 59.52 MB |
| Round 10 | 0.6014 | 84.29% | 0.8216 | 0.8409 | N/A | 119.05 MB |
| Round 15 | 0.5601 | 88.28% | 0.8630 | 0.8815 | N/A | 178.57 MB |
| Round 20 | 0.5650 | 88.02% | 0.8615 | 0.8789 | N/A | 238.09 MB |
| Round 25 | 0.5172 | 88.98% | 0.8719 | 0.8883 | N/A | 297.62 MB |
| Round 30 | 0.5287 | 88.72% | 0.8694 | 0.8863 | N/A | 357.14 MB |
| Round 35 | 0.5181 | 86.46% | 0.8421 | 0.8633 | N/A | 416.66 MB |
| Round 40 | 0.4912 | 90.02% | 0.8813 | 0.8992 | N/A | 476.19 MB |
| Round 45 | 0.4838 | 89.41% | 0.8754 | 0.8928 | N/A | 535.71 MB |
| Round 50 🌟 (Best) | 0.4745 | 90.02% | 0.8817 | 0.8989 | N/A | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.8830 | 0.6640 | **0.7580** | 250 |
| **Banking malware** | 0.8796 | 0.8753 | **0.8775** | 409 |
| **SMS malware** | 0.9156 | 0.9859 | **0.9494** | 781 |
| **Riskware** | 0.9115 | 0.8953 | **0.9033** | 506 |
| **Benign** | 0.8544 | 0.8855 | **0.8697** | 358 |
| **Macro Average** | **0.8888** | **0.8612** | **0.8716** | 2,304 |
| **Weighted Average** | **0.8888** | **0.8612** | **0.8934** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware        166       24      19         13       28 |   250
Banking         9      358      20         14        8 |   409
SMS             0        8     770          3        0 |   781
Riskware        9        7      19        453       18 |   506
Benign          4       10      13         14      317 |   358
```
