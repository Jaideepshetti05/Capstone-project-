# Experiment Report: attack_labelflip_median

**Generated:** 2026-09-14 22:21:37  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: False, SecAgg: False)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **90.15%** |
| **Test Macro F1** | **0.8804** |
| **Test Weighted F1** | **0.9000** |
| **Test Macro Precision** | **0.8931** |
| **Test Macro Recall** | **0.8714** |
| **Differential Privacy Guarantee** | **ε = None, δ = None** |
| **Secure Aggregation** | **Disabled** |
| **Best Validation Round** | **Round 45** (Val F1: 0.8813, Val Acc: 89.93%) |
| **Training Time** | **65.57s** |
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
| Round 01 | 1.2188 | 60.85% | 0.5204 | 0.5722 | N/A | 11.90 MB |
| Round 05 | 0.5997 | 82.73% | 0.7998 | 0.8253 | N/A | 59.52 MB |
| Round 10 | 0.5303 | 85.33% | 0.8280 | 0.8522 | N/A | 119.05 MB |
| Round 15 | 0.4779 | 87.41% | 0.8539 | 0.8734 | N/A | 178.57 MB |
| Round 20 | 0.4567 | 87.67% | 0.8579 | 0.8761 | N/A | 238.09 MB |
| Round 25 | 0.4143 | 88.63% | 0.8684 | 0.8854 | N/A | 297.62 MB |
| Round 30 | 0.4117 | 89.76% | 0.8797 | 0.8969 | N/A | 357.14 MB |
| Round 35 | 0.4003 | 89.06% | 0.8722 | 0.8897 | N/A | 416.66 MB |
| Round 40 | 0.4034 | 89.76% | 0.8794 | 0.8970 | N/A | 476.19 MB |
| Round 45 🌟 (Best) | 0.3770 | 89.93% | 0.8813 | 0.8982 | N/A | 535.71 MB |
| Round 50 | 0.3737 | 89.58% | 0.8780 | 0.8950 | N/A | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.8632 | 0.7320 | **0.7922** | 250 |
| **Banking malware** | 0.8509 | 0.9071 | **0.8781** | 409 |
| **SMS malware** | 0.9320 | 0.9834 | **0.9570** | 781 |
| **Riskware** | 0.8998 | 0.9051 | **0.9025** | 506 |
| **Benign** | 0.9195 | 0.8296 | **0.8722** | 358 |
| **Macro Average** | **0.8931** | **0.8714** | **0.8804** | 2,304 |
| **Weighted Average** | **0.8931** | **0.8714** | **0.9000** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware        183       28      15         15        9 |   250
Banking         9      371      16         10        3 |   409
SMS             0       10     768          2        1 |   781
Riskware       12       11      12        458       13 |   506
Benign          8       16      13         24      297 |   358
```
