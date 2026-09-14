# Experiment Report: dp_high_noise

**Generated:** 2026-08-26 19:57:50  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: True, SecAgg: False)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **66.02%** |
| **Test Macro F1** | **0.5592** |
| **Test Weighted F1** | **0.6314** |
| **Test Macro Precision** | **0.6352** |
| **Test Macro Recall** | **0.5600** |
| **Differential Privacy Guarantee** | **ε = 8.706, δ = 1e-05** |
| **Secure Aggregation** | **Disabled** |
| **Best Validation Round** | **Round 49** (Val F1: 0.5844, Val Acc: 68.06%) |
| **Training Time** | **1757.87s** |
| **Total Communication** | **595.23 MB** |

> [!NOTE]
> **Privacy Definition:** DP bounds the probability ratio of outputs on adjacent datasets by exp(ε). Secure Aggregation provides cryptographic confidentiality by masking individual updates so the server only observes the aggregated sum.

---

## 2. Configuration

| Parameter | Value |
|:---|:---|
| **Algorithm** | FedAvg |
| **DP Enabled** | True |
| **Noise Multiplier (sigma)** | 2.0 |
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
| Round 01 | 1.6908 | 17.01% | 0.0966 | 0.0885 | 1.36 | 11.90 MB |
| Round 05 | 1.5415 | 44.53% | 0.2839 | 0.3615 | 2.69 | 59.52 MB |
| Round 10 | 1.3619 | 45.66% | 0.2961 | 0.3725 | 3.76 | 119.05 MB |
| Round 15 | 1.2518 | 48.52% | 0.3211 | 0.4042 | 4.61 | 178.57 MB |
| Round 20 | 1.1876 | 54.95% | 0.3901 | 0.4781 | 5.37 | 238.09 MB |
| Round 25 | 1.1457 | 56.77% | 0.4135 | 0.5023 | 6.00 | 297.62 MB |
| Round 30 | 1.1177 | 60.94% | 0.4618 | 0.5517 | 6.62 | 357.14 MB |
| Round 35 | 1.1054 | 62.07% | 0.4835 | 0.5705 | 7.24 | 416.66 MB |
| Round 40 | 1.0729 | 64.15% | 0.5264 | 0.6021 | 7.73 | 476.19 MB |
| Round 45 | 1.0531 | 65.28% | 0.5448 | 0.6181 | 8.22 | 535.71 MB |
| Round 49 🌟 (Best) | 1.0329 | 68.06% | 0.5844 | 0.6533 | 8.61 | 583.33 MB |
| Round 50 | 1.0282 | 68.14% | 0.5844 | 0.6536 | 8.71 | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.5088 | 0.1160 | **0.1889** | 250 |
| **Banking malware** | 0.5040 | 0.6210 | **0.5564** | 409 |
| **SMS malware** | 0.7302 | 0.9462 | **0.8243** | 781 |
| **Riskware** | 0.6408 | 0.6700 | **0.6551** | 506 |
| **Benign** | 0.7921 | 0.4469 | **0.5714** | 358 |
| **Macro Average** | **0.6352** | **0.5600** | **0.5592** | 2,304 |
| **Weighted Average** | **0.6352** | **0.5600** | **0.6314** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware         29       81      45         80       15 |   250
Banking        12      254      98         35       10 |   409
SMS             8       30     739          2        2 |   781
Riskware        1       77      74        339       15 |   506
Benign          7       62      56         73      160 |   358
```
