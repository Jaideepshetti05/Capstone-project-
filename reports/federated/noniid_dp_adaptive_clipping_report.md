# Experiment Report: noniid_dp_adaptive_clipping

**Generated:** 2026-09-14 22:14:33  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: True, SecAgg: True)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **54.12%** |
| **Test Macro F1** | **0.4149** |
| **Test Weighted F1** | **0.4808** |
| **Test Macro Precision** | **0.4015** |
| **Test Macro Recall** | **0.4456** |
| **Differential Privacy Guarantee** | **ε = 25.5143, δ = 1e-05** |
| **Secure Aggregation** | **Enabled (Pairwise Additive Masking)** |
| **Best Validation Round** | **Round 45** (Val F1: 0.4139, Val Acc: 53.82%) |
| **Training Time** | **1866.06s** |
| **Total Communication** | **595.23 MB** |

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
| **Partition** | dirichlet (alpha = 0.1) |
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
| Client 00 | 61 (0.8%) | 1 | 37 | 0 | 1 | 22 | Banking malware (60.66%) | 1.16 b | 0.634 |
| Client 01 | 1247 (15.5%) | 0 | 76 | 205 | 14 | 952 | Benign (76.34%) | 1.04 b | 0.608 |
| Client 02 | 117 (1.4%) | 0 | 115 | 2 | 0 | 0 | Banking malware (98.29%) | 0.12 b | 0.806 |
| Client 03 | 2122 (26.3%) | 1 | 891 | 0 | 1230 | 0 | Riskware (57.96%) | 0.99 b | 0.603 |
| Client 04 | 3383 (42.0%) | 864 | 0 | 2517 | 2 | 0 | SMS malware (74.4%) | 0.83 b | 0.552 |
| Client 05 | 171 (2.1%) | 0 | 66 | 3 | 0 | 102 | Benign (59.65%) | 1.08 b | 0.650 |
| Client 06 | 314 (3.9%) | 0 | 0 | 0 | 136 | 178 | Benign (56.69%) | 0.99 b | 0.625 |
| Client 07 | 318 (3.9%) | 0 | 0 | 0 | 318 | 0 | Riskware (100.0%) | -0.00 b | 0.780 |
| Client 08 | 52 (0.7%) | 6 | 41 | 4 | 1 | 0 | Banking malware (78.85%) | 1.02 b | 0.618 |
| Client 09 | 277 (3.4%) | 4 | 203 | 0 | 70 | 0 | Banking malware (73.29%) | 0.92 b | 0.589 |
| **Total Global Train** | **8,062** | **876** | **1,429** | **2,731** | **1,772** | **1,254** | **SMS (33.87%)** | **2.20 b** | **0.000** |

---

## 4. Round-by-Round Validation History

| Global Round | Validation Loss | Validation Accuracy | Validation Macro F1 | Validation Weighted F1 | Epsilon (ε) | Cumulative Comm. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Round 01 | 1.6273 | 34.72% | 0.1720 | 0.2409 | 4.16 | 11.90 MB |
| Round 05 | 1.6025 | 38.80% | 0.1973 | 0.2356 | 7.66 | 59.52 MB |
| Round 10 | 1.8174 | 40.45% | 0.2178 | 0.2624 | 10.79 | 119.05 MB |
| Round 15 | 1.9281 | 43.66% | 0.2706 | 0.3245 | 13.30 | 178.57 MB |
| Round 20 | 1.9732 | 50.43% | 0.3751 | 0.4315 | 15.82 | 238.09 MB |
| Round 25 | 1.9677 | 49.74% | 0.3781 | 0.4369 | 18.33 | 297.62 MB |
| Round 30 | 2.0662 | 50.95% | 0.3903 | 0.4536 | 19.91 | 357.14 MB |
| Round 35 | 1.9466 | 50.95% | 0.3842 | 0.4530 | 21.31 | 416.66 MB |
| Round 40 | 1.8695 | 52.00% | 0.3997 | 0.4656 | 22.71 | 476.19 MB |
| Round 45 🌟 (Best) | 1.8345 | 53.82% | 0.4139 | 0.4762 | 24.11 | 535.71 MB |
| Round 50 | 1.7315 | 53.21% | 0.4123 | 0.4771 | 25.51 | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.3770 | 0.3680 | **0.3725** | 250 |
| **Banking malware** | 0.5839 | 0.4597 | **0.5144** | 409 |
| **SMS malware** | 0.6086 | 0.9398 | **0.7388** | 781 |
| **Riskware** | 0.4380 | 0.4605 | **0.4489** | 506 |
| **Benign** | 0.0000 | 0.0000 | **0.0000** | 358 |
| **Macro Average** | **0.4015** | **0.4456** | **0.4149** | 2,304 |
| **Weighted Average** | **0.4015** | **0.4456** | **0.4808** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware         92       50      31         77        0 |   250
Banking        22      188     131         68        0 |   409
SMS             0       33     734         14        0 |   781
Riskware       59       18     196        233        0 |   506
Benign         71       33     114        140        0 |   358
```
