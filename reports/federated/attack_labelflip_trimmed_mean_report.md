# Experiment Report: attack_labelflip_trimmed_mean

**Generated:** 2026-09-14 22:20:23  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: False, SecAgg: False)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **90.23%** |
| **Test Macro F1** | **0.8815** |
| **Test Weighted F1** | **0.9009** |
| **Test Macro Precision** | **0.8934** |
| **Test Macro Recall** | **0.8727** |
| **Differential Privacy Guarantee** | **ε = None, δ = None** |
| **Secure Aggregation** | **Disabled** |
| **Best Validation Round** | **Round 49** (Val F1: 0.8810, Val Acc: 89.84%) |
| **Training Time** | **78.15s** |
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
| Round 01 | 1.2058 | 59.90% | 0.5063 | 0.5585 | N/A | 11.90 MB |
| Round 05 | 0.5872 | 84.11% | 0.8164 | 0.8405 | N/A | 59.52 MB |
| Round 10 | 0.5176 | 86.20% | 0.8409 | 0.8612 | N/A | 119.05 MB |
| Round 15 | 0.4632 | 87.93% | 0.8573 | 0.8786 | N/A | 178.57 MB |
| Round 20 | 0.4535 | 88.37% | 0.8606 | 0.8825 | N/A | 238.09 MB |
| Round 25 | 0.4042 | 88.63% | 0.8661 | 0.8854 | N/A | 297.62 MB |
| Round 30 | 0.4072 | 88.54% | 0.8644 | 0.8845 | N/A | 357.14 MB |
| Round 35 | 0.3986 | 88.63% | 0.8654 | 0.8852 | N/A | 416.66 MB |
| Round 40 | 0.4017 | 89.67% | 0.8785 | 0.8961 | N/A | 476.19 MB |
| Round 45 | 0.3769 | 89.67% | 0.8778 | 0.8955 | N/A | 535.71 MB |
| Round 49 🌟 (Best) | 0.3697 | 89.84% | 0.8810 | 0.8973 | N/A | 583.33 MB |
| Round 50 | 0.3762 | 89.24% | 0.8740 | 0.8910 | N/A | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.8645 | 0.7400 | **0.7974** | 250 |
| **Banking malware** | 0.8565 | 0.8900 | **0.8729** | 409 |
| **SMS malware** | 0.9366 | 0.9834 | **0.9594** | 781 |
| **Riskware** | 0.8921 | 0.9150 | **0.9034** | 506 |
| **Benign** | 0.9172 | 0.8352 | **0.8743** | 358 |
| **Macro Average** | **0.8934** | **0.8727** | **0.8815** | 2,304 |
| **Weighted Average** | **0.8934** | **0.8727** | **0.9009** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware        185       26      13         16       10 |   250
Banking        10      364      18         13        4 |   409
SMS             0        9     768          3        1 |   781
Riskware       11       10      10        463       12 |   506
Benign          8       16      11         24      299 |   358
```
