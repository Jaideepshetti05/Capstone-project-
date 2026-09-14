# Experiment Report: attack_weightpoison_fedavg

**Generated:** 2026-09-14 22:27:17  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: False, SecAgg: False)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **57.03%** |
| **Test Macro F1** | **0.4420** |
| **Test Weighted F1** | **0.5222** |
| **Test Macro Precision** | **0.4508** |
| **Test Macro Recall** | **0.4645** |
| **Differential Privacy Guarantee** | **ε = None, δ = None** |
| **Secure Aggregation** | **Disabled** |
| **Best Validation Round** | **Round 49** (Val F1: 0.4544, Val Acc: 57.90%) |
| **Training Time** | **68.32s** |
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
| Round 01 | nan | 10.85% | 0.0392 | 0.0212 | N/A | 11.90 MB |
| Round 05 | 2.8821 | 31.25% | 0.1899 | 0.2398 | N/A | 59.52 MB |
| Round 10 | 3.4197 | 43.06% | 0.2743 | 0.3354 | N/A | 119.05 MB |
| Round 15 | 3.6036 | 29.17% | 0.2526 | 0.2178 | N/A | 178.57 MB |
| Round 20 | 4.7602 | 28.99% | 0.2584 | 0.2211 | N/A | 238.09 MB |
| Round 25 | 5.3180 | 46.09% | 0.3292 | 0.4047 | N/A | 297.62 MB |
| Round 30 | 4.0410 | 31.86% | 0.3181 | 0.3342 | N/A | 357.14 MB |
| Round 35 | 6.2726 | 22.57% | 0.1441 | 0.1289 | N/A | 416.66 MB |
| Round 40 | 4.0988 | 45.92% | 0.2955 | 0.3640 | N/A | 476.19 MB |
| Round 45 | 3.7892 | 52.26% | 0.3969 | 0.4691 | N/A | 535.71 MB |
| Round 49 🌟 (Best) | 5.0041 | 57.90% | 0.4544 | 0.5342 | N/A | 583.33 MB |
| Round 50 | 6.4461 | 32.03% | 0.2704 | 0.2703 | N/A | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.0000 | 0.0000 | **0.0000** | 250 |
| **Banking malware** | 0.3748 | 0.4792 | **0.4206** | 409 |
| **SMS malware** | 0.6341 | 0.9565 | **0.7626** | 781 |
| **Riskware** | 0.6267 | 0.3617 | **0.4586** | 506 |
| **Benign** | 0.6184 | 0.5251 | **0.5680** | 358 |
| **Macro Average** | **0.4508** | **0.4645** | **0.4420** | 2,304 |
| **Weighted Average** | **0.4508** | **0.4645** | **0.5222** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware          0      134      57         22       37 |   250
Banking         4      196     161         20       28 |   409
SMS             0       11     747         16        7 |   781
Riskware        1      137     141        183       44 |   506
Benign          2       45      72         51      188 |   358
```
