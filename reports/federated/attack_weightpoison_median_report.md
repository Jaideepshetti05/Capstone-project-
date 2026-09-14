# Experiment Report: attack_weightpoison_median

**Generated:** 2026-09-14 22:30:01  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: False, SecAgg: False)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **89.76%** |
| **Test Macro F1** | **0.8799** |
| **Test Weighted F1** | **0.8966** |
| **Test Macro Precision** | **0.8927** |
| **Test Macro Recall** | **0.8717** |
| **Differential Privacy Guarantee** | **ε = None, δ = None** |
| **Secure Aggregation** | **Disabled** |
| **Best Validation Round** | **Round 42** (Val F1: 0.8733, Val Acc: 89.06%) |
| **Training Time** | **76.39s** |
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
| Round 01 | 1.1791 | 61.98% | 0.5486 | 0.5906 | N/A | 11.90 MB |
| Round 05 | 0.5395 | 82.38% | 0.7975 | 0.8213 | N/A | 59.52 MB |
| Round 10 | 0.4365 | 85.68% | 0.8343 | 0.8558 | N/A | 119.05 MB |
| Round 15 | 0.4117 | 86.98% | 0.8480 | 0.8690 | N/A | 178.57 MB |
| Round 20 | 0.3909 | 87.50% | 0.8529 | 0.8744 | N/A | 238.09 MB |
| Round 25 | 0.3848 | 87.76% | 0.8568 | 0.8767 | N/A | 297.62 MB |
| Round 30 | 0.3680 | 87.93% | 0.8592 | 0.8785 | N/A | 357.14 MB |
| Round 35 | 0.3622 | 88.54% | 0.8664 | 0.8841 | N/A | 416.66 MB |
| Round 40 | 0.3534 | 88.80% | 0.8699 | 0.8874 | N/A | 476.19 MB |
| Round 42 🌟 (Best) | 0.3484 | 89.06% | 0.8733 | 0.8900 | N/A | 500.00 MB |
| Round 45 | 0.3502 | 88.89% | 0.8708 | 0.8881 | N/A | 535.71 MB |
| Round 50 | 0.3471 | 88.72% | 0.8685 | 0.8865 | N/A | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.8716 | 0.7600 | **0.8120** | 250 |
| **Banking malware** | 0.8095 | 0.9144 | **0.8588** | 409 |
| **SMS malware** | 0.9266 | 0.9859 | **0.9553** | 781 |
| **Riskware** | 0.9151 | 0.8518 | **0.8823** | 506 |
| **Benign** | 0.9410 | 0.8464 | **0.8912** | 358 |
| **Macro Average** | **0.8927** | **0.8717** | **0.8799** | 2,304 |
| **Weighted Average** | **0.8927** | **0.8717** | **0.8966** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware        190       27      15         12        6 |   250
Banking        11      374      14          8        2 |   409
SMS             0       10     770          1        0 |   781
Riskware       12       32      20        431       11 |   506
Benign          5       19      12         19      303 |   358
```
