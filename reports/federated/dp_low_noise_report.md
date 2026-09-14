# Experiment Report: dp_low_noise

**Generated:** 2026-08-26 19:10:20  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: True, SecAgg: False)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **79.99%** |
| **Test Macro F1** | **0.7718** |
| **Test Weighted F1** | **0.7981** |
| **Test Macro Precision** | **0.7827** |
| **Test Macro Recall** | **0.7661** |
| **Differential Privacy Guarantee** | **ε = 49.9928, δ = 1e-05** |
| **Secure Aggregation** | **Disabled** |
| **Best Validation Round** | **Round 49** (Val F1: 0.7913, Val Acc: 81.94%) |
| **Training Time** | **1720.01s** |
| **Total Communication** | **595.23 MB** |

> [!NOTE]
> **Privacy Definition:** DP bounds the probability ratio of outputs on adjacent datasets by exp(ε). Secure Aggregation provides cryptographic confidentiality by masking individual updates so the server only observes the aggregated sum.

---

## 2. Configuration

| Parameter | Value |
|:---|:---|
| **Algorithm** | FedAvg |
| **DP Enabled** | True |
| **Noise Multiplier (sigma)** | 0.5 |
| **Clipping Norm (C)** | 1.0 |
| **Target Delta** | 1e-05 |
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
| Round 01 | 1.5860 | 46.53% | 0.3160 | 0.4030 | 5.27 | 11.90 MB |
| Round 05 | 1.1908 | 54.95% | 0.3917 | 0.4793 | 12.21 | 59.52 MB |
| Round 10 | 1.0702 | 64.15% | 0.5127 | 0.5968 | 18.19 | 119.05 MB |
| Round 15 | 0.9721 | 70.57% | 0.6287 | 0.6868 | 23.20 | 178.57 MB |
| Round 20 | 0.8905 | 74.48% | 0.7003 | 0.7375 | 27.86 | 238.09 MB |
| Round 25 | 0.8257 | 76.30% | 0.7237 | 0.7585 | 32.53 | 297.62 MB |
| Round 30 | 0.7874 | 77.78% | 0.7415 | 0.7754 | 36.14 | 357.14 MB |
| Round 35 | 0.7599 | 78.91% | 0.7513 | 0.7861 | 39.60 | 416.66 MB |
| Round 40 | 0.7450 | 80.03% | 0.7658 | 0.7980 | 43.06 | 476.19 MB |
| Round 45 | 0.7413 | 81.16% | 0.7805 | 0.8093 | 46.53 | 535.71 MB |
| Round 49 🌟 (Best) | 0.7283 | 81.94% | 0.7913 | 0.8173 | 49.30 | 583.33 MB |
| Round 50 | 0.7191 | 81.77% | 0.7884 | 0.8156 | 49.99 | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.6590 | 0.6880 | **0.6732** | 250 |
| **Banking malware** | 0.7139 | 0.7262 | **0.7200** | 409 |
| **SMS malware** | 0.8362 | 0.9475 | **0.8884** | 781 |
| **Riskware** | 0.8665 | 0.7312 | **0.7931** | 506 |
| **Benign** | 0.8381 | 0.7374 | **0.7845** | 358 |
| **Macro Average** | **0.7827** | **0.7661** | **0.7718** | 2,304 |
| **Weighted Average** | **0.7827** | **0.7661** | **0.7981** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware        172       20      26         20       12 |   250
Banking        28      297      51         14       19 |   409
SMS            10       28     740          1        2 |   781
Riskware       31       55      32        370       18 |   506
Benign         20       16      36         22      264 |   358
```
