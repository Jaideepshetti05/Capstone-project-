# Privacy-Preserving Malware Detection Using Federated Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality](https://img.shields.io/badge/Unit%20Tests-27%20Passed-brightgreen.svg)]()

A research-oriented, privacy-preserving, and Byzantine-robust Federated Learning framework for decentralized Android malware classification on the **CICMalDroid 2020** benchmark dataset.

---

## 1. Project Overview

Mobile malware threats have escalated in complexity and velocity, necessitating collaborative threat intelligence across distributed enterprise fleets, mobile carriers, and endpoint security agents. However, centralizing private application telemetry, system call invocations, and user activity traces introduces severe data privacy liabilities and regulatory non-compliance (GDPR, HIPAA, CCPA).

This capstone project implements a **Privacy-Preserving and Byzantine-Robust Federated Learning Framework** that trains deep neural network detectors across decentralized endpoint clients without transmitting private runtime telemetry to a central server.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                Central Federated Server                │
                  │  - Global Model w_global (MalwareMLP, 156k parameters) │
                  │  - Server Aggregation: FedAvg, FedMedian, Trimmed Mean │
                  │  - Secure Aggregation Decryption / Mask Cancellation   │
                  │  - Centralized Validation Tracking & Test Evaluation   │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                      Downlink: w_t           │         Uplink: y_k (Masked or Clipped)
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
    │  Local Client 0  │            │  Local Client 1  │            │  Local Client K  │
    │ - Private Shard  │            │ - Private Shard  │            │ - Private Shard  │
    │ - DP-SGD Engine  │            │ - DP-SGD Engine  │            │ - DP-SGD Engine  │
    │ - Per-Sample Clip│            │ - Per-Sample Clip│            │ - Per-Sample Clip│
    │ - SecAgg Masking │            │ - SecAgg Masking │            │ - SecAgg Masking │
    └──────────────────┘            └──────────────────┘            └──────────────────┘
```

---

## 2. Research Problem & Gap

### Research Problem
> *"How can federated learning be used for privacy-preserving Android malware detection while maintaining high multi-class detection performance and robustness under heterogeneous non-IID client distributions and potentially malicious or poisoned client updates?"*

### Research Gap
1. **Tradeoff Collapse under Heterogeneity:** Prior literature evaluates Federated Learning on balanced IID distributions. Real-world mobile endpoints exhibit extreme label skew where specific malware variants (e.g., Banking Trojans or SMS Malware) appear intermittently.
2. **Privacy vs Utility Tension:** Naive addition of Differential Privacy noise ($\sigma$) causes catastrophic performance degradation on minority classes.
3. **Byzantine Vulnerability:** Standard Federated Averaging ($\text{FedAvg}$) is vulnerable to even a single malicious client conducting data poisoning (label-flipping) or model weight poisoning.
4. **Lack of End-to-End Integration:** Prior works investigate DP, Secure Aggregation, or robust aggregation in isolation. This project provides a unified architecture integrating all four mechanisms.

---

## 3. Dataset & Preprocessing Pipeline

### CICMalDroid 2020 Dataset
- **Domain:** Android Malware Dynamic Analysis.
- **Original Samples:** 11,598 dynamically analyzed APKs across 5 balanced/unbalanced classes:
  1. **Adware** (Class 0, 1,253 samples)
  2. **Banking Malware** (Class 1, 2,100 samples)
  3. **SMS Malware** (Class 2, 3,904 samples)
  4. **Riskware** (Class 3, 2,546 samples)
  5. **Benign** (Class 4, 1,795 samples)
- **Features:** 470 raw dynamic system call and IPC binder frequency features.

### Preprocessing & Data Cleaning (`src/preprocess_and_compare.py`)
1. **Integrity Filtering:** 35 all-zero feature records dropped. 45 duplicate/conflicting records dropped. 11,518 clean samples retained.
2. **Multicollinearity Reduction:** 27 perfectly collinear features (Pearson $|r| = 1.0$) pruned, yielding **443 dynamic features**.
3. **Rigorous Split Isolation:** Stratified split:
   - **Training Set:** 8,062 samples (70%)
   - **Validation Set:** 1,152 samples (10%)
   - **Locked Test Set:** 2,304 samples (20%) — strictly untouched during training and model selection.
4. **Normalization:** `StandardScaler` fitted strictly on training partition to prevent cross-split data leakage.

---

## 4. System Architecture & Algorithms

### 4.1 Neural Network Model (`MalwareMLP`)
Implemented in `src/models/mlp.py`:
- **Input Layer:** 443 features
- **Hidden Layers:**
  - Layer 1: $\text{Linear}(443 \to 256) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{Dropout}(0.2)$
  - Layer 2: $\text{Linear}(256 \to 128) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{Dropout}(0.2)$
  - Layer 3: $\text{Linear}(128 \to 64) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{Dropout}(0.1)$
- **Output Layer:** $\text{Linear}(64 \to 5)$ class logits
- **Total Trainable Parameters:** 156,037 (~609.52 KB uncompressed float32 payload)

### 4.2 Federated Optimization
- **FedAvg (McMahan et al., 2017):**
  $$\mathbf{w}_{t+1} = \sum_{k=1}^K \frac{n_k}{N} \mathbf{w}_{t+1}^k$$
- **FedProx (Li et al., 2020):** Local objective augmented with proximal penalty:
  $$\min_{\mathbf{w}} \mathcal{L}_k(\mathbf{w}) + \frac{\mu}{2} \|\mathbf{w} - \mathbf{w}_t\|^2$$

### 4.3 Data Heterogeneity Modeling (`src/federated/partition.py`)
- **Stratified IID:** Uniform class distributions across clients.
- **Non-IID Dirichlet Skew:** Proportions sampled from $\text{Dirichlet}(\alpha \cdot \mathbf{1}_K)$ for $\alpha \in \{1.0, 0.5, 0.1\}$. Validated with Shannon Entropy and Total Variation Distance (TVD).

### 4.4 Differential Privacy (DP-SGD & RDP Accountant)
Implemented in `src/federated/dp_client.py` and `src/federated/dp_accountant.py`:
- **Per-Sample Gradient Clipping:** Exact autograd $L_2$ clipping:
  $$\mathbf{g}_i \leftarrow \mathbf{g}_i \cdot \min\left(1, \frac{C}{\|\mathbf{g}_i\|_2}\right)$$
- **Calibrated Noise Injection:**
  $$\tilde{\mathbf{g}} = \frac{1}{B} \left(\sum_{i=1}^B \mathbf{g}_i + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right)\right)$$
- **Rényi Differential Privacy (RDP):** Subsampled Gaussian mechanism composition (Wang et al., 2019) converted to tight $(\varepsilon, \delta)$-DP guarantees at $\delta = 10^{-5}$.

### 4.5 Adaptive Gradient Clipping (`src/federated/adaptive_clipping.py`)
Implements Andrew et al. (NeurIPS 2021) quantile adaptation:
$$C_{t+1} = C_t \cdot \exp\left(\eta \left(\hat{q}_t - (1 - \gamma)\right)\right), \quad C_{t+1} \in [C_{\min}, C_{\max}]$$

### 4.6 Pairwise Secure Aggregation (`src/federated/secure_aggregation.py`)
Pairwise zero-sum additive secret masking (Bonawitz et al., 2017):
$$\mathbf{y}_u = \frac{n_u}{N} \mathbf{w}_u + \sum_{v > u} \mathbf{s}_{u,v} - \sum_{v < u} \mathbf{s}_{v,u}, \quad \sum_{k=1}^K \mathbf{y}_k = \sum_{k=1}^K \frac{n_k}{N} \mathbf{w}_k$$
Cancellation mathematically verified to $< 10^{-6}$ machine precision (empirical error: $7.24 \times 10^{-9}$).

### 4.7 Byzantine Robust Aggregation (`src/federated/robust_aggregation.py`)
- **Coordinate-wise Median (FedMedian):** Coordinate-level median across client parameter tensors.
- **Coordinate-wise Trimmed Mean (FedTrimmedMean):** Trims $\beta$-fraction lowest and highest updates per parameter before averaging.

### 4.8 Threat Models (`src/federated/attacks.py`)
- **Targeted Evasion Data Poisoning:** Malicious clients flip all malware labels ($y \in \{0, 1, 2, 3\}$) to Benign ($y=4$).
- **Model Weight Poisoning:** Malicious clients inject high-variance Byzantine noise or sign-inverted updates into model weights.

---

## 5. Experimental Results

All experiments were executed on 50 communication rounds across 10 clients with AdamW optimizer ($B=64, E=2, \eta=0.001$).

### 5.1 Centralized vs Federated Baselines

| Model / Strategy | Paradigm | Test Accuracy | Test Macro F1 | Test Weighted F1 | Inference Latency |
|:---|:---|:---:|:---:|:---:|:---:|
| **Random Forest (300 trees)** | Centralized | **96.09%** | **0.9535** | **0.9610** | 0.0254 ms |
| **Centralized MLP** | Centralized | **91.28%** | **0.8957** | **0.9124** | 0.0061 ms |
| **Logistic Regression** | Centralized | 88.28% | 0.8616 | 0.8814 | 0.0022 ms |
| **FedAvg (IID, Plaintext)** | Federated | **90.19%** | **0.8853** | **0.9015** | 0.0063 ms |

> *Insight:* Plaintext FedAvg achieves 90.19% accuracy, trailing centralized deep learning by only 1.09%, demonstrating that decentralized collaborative training preserves malware detection efficacy.

### 5.2 Non-IID Data Skew & FedProx Regularization

| Partitioning Scheme | Algorithm | Test Accuracy | Test Macro F1 | Weighted F1 | Validation Acc |
|:---|:---|:---:|:---:|:---:|:---:|
| **Stratified IID** | FedAvg | **90.19%** | **0.8853** | **0.9015** | 89.93% |
| **Dirichlet ($\alpha=1.0$)** | FedAvg | 89.80% | 0.8781 | 0.8974 | 89.24% |
| **Dirichlet ($\alpha=0.5$)** | FedAvg | 89.93% | 0.8845 | 0.8989 | 89.76% |
| **Dirichlet ($\alpha=0.1$)** | FedAvg | 69.92% | 0.5836 | 0.6559 | 71.70% |
| **Dirichlet ($\alpha=0.1$)** | **FedProx ($\mu=0.01$)** | **72.92%** | **0.6138** | **0.6974** | **74.13%** |

> *Insight:* Extreme label skew ($\alpha=0.1$) induces severe client gradient drift, dropping accuracy to 69.92%. Proximal regularization ($\text{FedProx}, \mu=0.01$) curbs local drift and recovers **+3.00% accuracy** and **+3.02% Macro F1**.

### 5.3 Differential Privacy & Secure Aggregation (Phase 6)

| Experiment | DP Noise ($\sigma$) | Clipping ($C$) | SecAgg | Privacy Budget ($\varepsilon$) | Test Accuracy | Test Macro F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`secagg_only`** | 0.0 | None | **ON** | Confidential | **89.93%** | **0.8819** |
| **`dp_low_noise`** | 0.5 | 1.0 | OFF | $\varepsilon = 49.99$ | 79.99% | 0.7718 |
| **`dp_med_noise`** | 1.0 | 1.0 | OFF | $\varepsilon = 25.51$ | 74.57% | 0.7040 |
| **`dp_secagg`** | 1.0 | 1.0 | **ON** | $\varepsilon = 25.51$ | 74.57% | 0.7041 |
| **`dp_high_noise`** | 2.0 | 1.0 | OFF | **$\varepsilon = 8.71$** | 66.02% | 0.5592 |
| **`noniid_dp_secagg`** | 1.0 | 1.0 | **ON** | $\varepsilon = 25.51$ | 52.65% | 0.4041 |

> *Key Finding:* Comparing `dp_med_noise` (SecAgg OFF) vs `dp_secagg` (SecAgg ON) demonstrates **zero accuracy loss** (both 74.57%), confirming that pairwise masking provides mathematical confidentiality without utility cost.

---

## 6. Repository Structure

```
implementation/
├── data/
│   └── processed/variant_a/      # Preprocessed Train, Val, Test CSVs
├── models/
│   ├── centralized/              # Trained baseline weights
│   └── federated/                # Checkpoints per experiment
├── reports/
│   ├── centralized_baseline_report.md
│   ├── dataset_audit_report.md
│   ├── preprocessing_decision_report.md
│   ├── figures/                  # Publication-ready plots
│   └── federated/                # Per-experiment markdown reports
├── results/
│   └── federated/                # Per-experiment JSON metric artifacts
├── src/
│   ├── data_audit.py             # Dataset verification
│   ├── preprocess_and_compare.py # Feature pruning & normalization
│   ├── train_centralized.py      # Centralized baselines
│   ├── run_federated.py          # Unified CLI orchestrator
│   ├── generate_final_benchmarks.py # Benchmark table aggregator
│   ├── generate_final_plots.py   # Publication figure generator
│   ├── models/
│   │   └── mlp.py                # MalwareMLP PyTorch architecture
│   └── federated/
│       ├── client.py             # Local client training
│       ├── dp_client.py          # DP-SGD per-sample clipping
│       ├── dp_accountant.py      # RDP privacy accountant
│       ├── adaptive_clipping.py  # Quantile clipping controller
│       ├── secure_aggregation.py # Pairwise zero-sum SecAgg
│       ├── robust_aggregation.py # FedMedian & FedTrimmedMean
│       ├── attacks.py            # Poisoning threat models
│       ├── fedavg.py             # Sample-weighted aggregation
│       ├── partition.py          # Stratified IID & Dirichlet Non-IID
│       ├── evaluation.py         # Multi-class metrics & ROC-AUC
│       └── utils.py              # Reproducibility seeds & cost models
├── test_adaptive_clipping.py     # Unit tests for adaptive clipping
├── test_attacks.py               # Unit tests for threat models
├── test_evaluation.py            # Unit tests for ROC-AUC
├── test_federated_integrity.py   # 20 pipeline integrity assertions
├── test_privacy_integrity.py     # Unit tests for DP and SecAgg
├── test_robust_aggregation.py    # Unit tests for robust defenses
├── demo.py                       # Fast live demonstration script
└── README.md
```

---

## 7. Installation & Quickstart

### Prerequisites
- Python 3.10+
- PyTorch 2.0+

```bash
# Clone repository
git clone https://github.com/Jaideepshetti05/Capstone-project-.git
cd Capstone-project-/implementation

# Install dependencies
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn tabulate pytest
```

### Running Unit Tests
```bash
python -m pytest
```

### Executing Federated Experiments
```bash
# Plaintext IID FedAvg baseline (50 rounds)
python src/run_federated.py --experiment iid --num_rounds 50

# Extreme Non-IID with FedProx (mu=0.01)
python src/run_federated.py --experiment fedprox_a01_mu001 --num_rounds 50

# Privacy-Preserving DP-SGD with Secure Aggregation
python src/run_federated.py --experiment dp_secagg --num_rounds 50

# Adaptive Quantile Gradient Clipping
python src/run_federated.py --experiment dp_adaptive_clipping --num_rounds 50

# Byzantine Robust Aggregation under 20% Label-Flipping Attack
python src/run_federated.py --experiment attack_labelflip_trimmed_mean --num_rounds 50
```

### Running Live Interactive Demo
```bash
python demo.py
```

---

## 8. Limitations & Scope Clarification

To ensure absolute academic and research integrity, the following implementation boundaries are explicitly stated:
1. **Simulation Model:** The federated clients are executed within a **single-node in-memory Python simulation**. Real-world multi-device network deployment (via gRPC or mobile APK clients) is simulated rather than physically deployed across smartphones.
2. **Feature Space:** Features represent **tabular system call and binder frequencies** from the CICMalDroid 2020 dataset; the pipeline does not decompile or sandbox raw `.apk` binaries at runtime.
3. **Cryptographic Primitives:** Secure Aggregation simulates pairwise zero-sum random masking with exact cancellation; it does not implement Diffie-Hellman key exchanges or Shamir's secret sharing for client dropout recovery.

---

## 9. Conclusion

This project proves that privacy-preserving, Byzantine-robust Federated Learning is highly viable for Android malware detection. Collaborative training retains **90.19% test accuracy** (within 1.09% of centralized deep learning), provides strict $(\varepsilon=25.51, \delta=10^{-5})$ differential privacy without compromising confidentiality via Secure Aggregation, mitigates severe non-IID label skew via FedProx, and defends against poisoning attacks via coordinate-wise trimmed mean and median aggregation.
