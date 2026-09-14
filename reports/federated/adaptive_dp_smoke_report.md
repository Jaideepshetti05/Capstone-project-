# Experiment Report: adaptive_dp_smoke

**Generated:** 2026-08-27 19:12:23  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: True, SecAgg: True)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **44.53%** |
| **Test Macro F1** | **0.2823** |
| **Test Weighted F1** | **0.3700** |
| **Test Macro Precision** | **0.3884** |
| **Test Macro Recall** | **0.3307** |
| **Differential Privacy Guarantee** | **ε = 7.663, δ = 1e-05** |
| **Secure Aggregation** | **Enabled (Pairwise Additive Masking)** |
| **Best Validation Round** | **Round 2** (Val F1: 0.3084, Val Acc: 45.57%) |
| **Training Time** | **112.00s** |
| **Total Communication** | **59.52 MB** |

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
| **Rounds (T)** | 5 |
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
| Round 02 🌟 (Best) | 1.5899 | 45.57% | 0.3084 | 0.3907 | 5.04 | 23.81 MB |
| Round 05 | 1.3693 | 45.75% | 0.2921 | 0.3702 | 7.66 | 59.52 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.0631 | 0.0280 | **0.0388** | 250 |
| **Banking malware** | 0.2921 | 0.5134 | **0.3723** | 409 |
| **SMS malware** | 0.5674 | 0.9104 | **0.6991** | 781 |
| **Riskware** | 0.4314 | 0.1739 | **0.2479** | 506 |
| **Benign** | 0.5882 | 0.0279 | **0.0533** | 358 |
| **Macro Average** | **0.3884** | **0.3307** | **0.2823** | 2,304 |
| **Weighted Average** | **0.3884** | **0.3307** | **0.3700** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware          7       96     117         29        1 |   250
Banking         6      210     171         18        4 |   409
SMS             2       63     711          5        0 |   781
Riskware       57      219     140         88        2 |   506
Benign         39      131     114         64       10 |   358
```
