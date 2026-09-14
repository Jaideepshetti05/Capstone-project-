# Experiment Report: secagg_only

**Generated:** 2026-08-26 20:00:00  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: False, SecAgg: True)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **89.93%** |
| **Test Macro F1** | **0.8819** |
| **Test Weighted F1** | **0.8983** |
| **Test Macro Precision** | **0.8929** |
| **Test Macro Recall** | **0.8744** |
| **Differential Privacy Guarantee** | **ε = None, δ = None** |
| **Secure Aggregation** | **Enabled (Pairwise Additive Masking)** |
| **Best Validation Round** | **Round 45** (Val F1: 0.8905, Val Acc: 90.71%) |
| **Training Time** | **105.38s** |
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
| Round 01 | 1.1715 | 61.55% | 0.5391 | 0.5842 | N/A | 11.90 MB |
| Round 05 | 0.5380 | 83.68% | 0.8135 | 0.8352 | N/A | 59.52 MB |
| Round 10 | 0.4560 | 87.24% | 0.8511 | 0.8713 | N/A | 119.05 MB |
| Round 15 | 0.4237 | 87.15% | 0.8507 | 0.8706 | N/A | 178.57 MB |
| Round 20 | 0.4059 | 87.93% | 0.8604 | 0.8783 | N/A | 238.09 MB |
| Round 25 | 0.3834 | 88.80% | 0.8711 | 0.8873 | N/A | 297.62 MB |
| Round 30 | 0.3852 | 88.28% | 0.8662 | 0.8819 | N/A | 357.14 MB |
| Round 35 | 0.3764 | 89.84% | 0.8826 | 0.8977 | N/A | 416.66 MB |
| Round 40 | 0.3791 | 89.41% | 0.8766 | 0.8935 | N/A | 476.19 MB |
| Round 45 🌟 (Best) | 0.3594 | 90.71% | 0.8905 | 0.9066 | N/A | 535.71 MB |
| Round 50 | 0.3663 | 90.36% | 0.8883 | 0.9028 | N/A | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.8540 | 0.7720 | **0.8109** | 250 |
| **Banking malware** | 0.8315 | 0.9169 | **0.8721** | 409 |
| **SMS malware** | 0.9188 | 0.9846 | **0.9506** | 781 |
| **Riskware** | 0.9313 | 0.8577 | **0.8930** | 506 |
| **Benign** | 0.9290 | 0.8408 | **0.8827** | 358 |
| **Macro Average** | **0.8929** | **0.8744** | **0.8819** | 2,304 |
| **Weighted Average** | **0.8929** | **0.8744** | **0.8983** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware        193       25      13         10        9 |   250
Banking        11      375      16          6        1 |   409
SMS             0       11     769          1        0 |   781
Riskware       12       22      25        434       13 |   506
Benign         10       18      14         15      301 |   358
```
