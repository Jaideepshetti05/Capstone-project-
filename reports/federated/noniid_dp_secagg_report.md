# Experiment Report: noniid_dp_secagg

**Generated:** 2026-08-26 20:49:09  
**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Algorithm:** FedAvg (DP: True, SecAgg: True)  
**Model Architecture:** PyTorch MLP (`MalwareMLP`, 156,037 parameters)  

---

## 1. Executive Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | **52.65%** |
| **Test Macro F1** | **0.4041** |
| **Test Weighted F1** | **0.4692** |
| **Test Macro Precision** | **0.3830** |
| **Test Macro Recall** | **0.4366** |
| **Differential Privacy Guarantee** | **ε = 25.5143, δ = 1e-05** |
| **Secure Aggregation** | **Enabled (Pairwise Additive Masking)** |
| **Best Validation Round** | **Round 46** (Val F1: 0.3979, Val Acc: 51.56%) |
| **Training Time** | **1394.48s** |
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
| Round 05 | 1.6053 | 38.72% | 0.1975 | 0.2351 | 7.66 | 59.52 MB |
| Round 10 | 1.8695 | 40.19% | 0.2120 | 0.2574 | 10.79 | 119.05 MB |
| Round 15 | 2.0856 | 42.71% | 0.2494 | 0.3070 | 13.30 | 178.57 MB |
| Round 20 | 2.2274 | 46.88% | 0.3236 | 0.3780 | 15.82 | 238.09 MB |
| Round 25 | 2.3400 | 47.57% | 0.3540 | 0.4057 | 18.33 | 297.62 MB |
| Round 30 | 2.5639 | 47.74% | 0.3573 | 0.4160 | 19.91 | 357.14 MB |
| Round 35 | 2.5515 | 48.18% | 0.3624 | 0.4203 | 21.31 | 416.66 MB |
| Round 40 | 2.5292 | 50.61% | 0.3872 | 0.4494 | 22.71 | 476.19 MB |
| Round 45 | 2.6235 | 50.95% | 0.3936 | 0.4511 | 24.11 | 535.71 MB |
| Round 46 🌟 (Best) | 2.4928 | 51.56% | 0.3979 | 0.4590 | 24.39 | 547.62 MB |
| Round 50 | 2.5053 | 50.43% | 0.3819 | 0.4469 | 25.51 | 595.23 MB |

---

## 5. Locked Test Set Performance (2,304 Samples)

| Class Name | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Adware** | 0.3621 | 0.3520 | **0.3570** | 250 |
| **Banking malware** | 0.5037 | 0.4939 | **0.4988** | 409 |
| **SMS malware** | 0.6040 | 0.8963 | **0.7216** | 781 |
| **Riskware** | 0.4451 | 0.4407 | **0.4429** | 506 |
| **Benign** | 0.0000 | 0.0000 | **0.0000** | 358 |
| **Macro Average** | **0.3830** | **0.4366** | **0.4041** | 2,304 |
| **Weighted Average** | **0.3830** | **0.4366** | **0.4692** | 2,304 |

### Confusion Matrix
```
Pred ->   Adware  Banking     SMS  Riskware   Benign | Total
Adware         88       62      25         75        0 |   250
Banking        23      202     117         67        0 |   409
SMS             0       67     700         14        0 |   781
Riskware       63       24     196        223        0 |   506
Benign         69       46     121        122        0 |   358
```
