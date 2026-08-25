# Privacy-Preserving Malware Detection — Federated Learning Experimental Plan

**Project:** Privacy-Preserving Malware Detection Using Federated Learning  
**Dataset:** CICMalDroid 2020 (Variant A: 443 Dynamic Syscall & Binder Features, 5 Classes)  
**Lead ML & Research Engineer:** Capstone AI Research Team  
**Status:** Phase 5 — Federated Learning Baseline Benchmarking  

---

## 1. System Architecture & Overview

The federated learning (FL) framework coordinates decentralized mobile security clients (e.g., enterprise endpoints, mobile carriers, or security vendors) to collaboratively train a high-accuracy Android malware detection model without sharing sensitive, raw execution traces.

```
                  ┌──────────────────────────────────────────────┐
                  │          Central Federated Server            │
                  │  - Holds Global Model w_global (MLP)         │
                  │  - Broadcasts Global Model to Clients        │
                  │  - Aggregates Client Updates via FedAvg      │
                  │  - Evaluates Centralized Val & Locked Test   │
                  └──────────────┬───────────────────────────────┘
                                 │
                 Downlink: w_t   │   Uplink: (w_{t+1}^k, n_k)
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Local Client 0  │   │  Local Client 1  │   │  Local Client K  │
│ - Private Shard  │   │ - Private Shard  │   │ - Private Shard  │
│ - Local SGD/AdamW│   │ - Local SGD/AdamW│   │ - Local SGD/AdamW│
│ - No Val/Test    │   │ - No Val/Test    │   │ - No Val/Test    │
└──────────────────┘   └──────────────────┘   └──────────────────┘
```

---

## 2. Client & Server Roles

### 2.1 Central Server
- **Model Orchestration:** Maintains the canonical global model parameters $\mathbf{w}_t \in \mathbb{R}^d$.
- **Aggregation:** Executes sample-weighted Federated Averaging ($\text{FedAvg}$) after receiving updates from participating clients.
- **Centralized Evaluation:** Evaluates the aggregated global model $\mathbf{w}_{t+1}$ on the centralized validation set after every communication round to monitor convergence.
- **Locked Test Evaluation:** Evaluates the selected optimal model checkpoint **once** on the locked test set after the training protocol terminates.

### 2.2 Local Clients
- **Data Privacy:** Each client $k \in \{0, \dots, K-1\}$ maintains a strictly private partition $\mathcal{D}_k = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n_k}$ of the training data.
- **Local Optimization:** Initializes local model parameters with $\mathbf{w}_t$, optimizes local loss $\mathcal{L}_k(\mathbf{w})$ using mini-batch AdamW for $E=2$ local epochs.
- **Transmission:** Returns only the updated parameter tensor $\mathbf{w}_{t+1}^k$ and sample count $n_k$.
- **Isolation:** Clients have zero access to validation/test sets and cannot inspect peer data.

---

## 3. Data Partitioning Strategies

### 3.1 Stratified IID Partitioning
To establish an ideal, baseline federated scenario:
- The 8,062 training samples are partitioned across $K=10$ clients.
- For each class $c \in \{0, 1, 2, 3, 4\}$, sample indices are deterministically shuffled and split into $K$ equal subsets.
- Result: Each client receives approximately $806$ samples with identical class proportions matching the global population ($10.9\%$ Adware, $17.7\%$ Banking, $33.9\%$ SMS, $22.0\%$ Riskware, $15.6\%$ Benign).

### 3.2 Non-IID Dirichlet Partitioning
Real-world endpoint environments exhibit severe label distribution skew. To simulate this heterogeneity:
- For each class $c$, sample proportion vector $\mathbf{q}_c \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_K)$.
- Allocate samples of class $c$ to client $k$ proportional to $q_{c,k}$.
- **Concentration Parameter $\alpha$:**
  - $\alpha = 1.0$: Mild label heterogeneity.
  - $\alpha = 0.5$: Moderate non-IID class imbalance.
  - $\alpha = 0.1$: Extreme non-IID distribution (clients specialize in only 1–2 malware categories).

---

## 4. Federated Averaging (FedAvg) Formulation

Let $N = \sum_{k=1}^K n_k$ denote the total number of training samples across all participating clients.
At communication round $t$:

1. **Broadcast:** The server sends $\mathbf{w}_t$ to all active clients $S_t \subseteq \{0, \dots, K-1\}$.
2. **Local Update:** Each client $k \in S_t$ optimizes its local objective:
   $$\mathbf{w}_{t+1}^k \leftarrow \text{LocalTrain}(\mathbf{w}_t, \mathcal{D}_k, E, B, \eta)$$
3. **Server Aggregation:** The server computes the weighted average:
   $$\mathbf{w}_{t+1} = \sum_{k \in S_t} \frac{n_k}{N_t} \mathbf{w}_{t+1}^k \quad \text{where} \quad N_t = \sum_{k \in S_t} n_k$$

*Note: Unweighted averaging is strictly rejected because it biases the global model toward smaller clients.*

---

## 5. Model Architecture (`MalwareMLP`)

The federated model uses the verified PyTorch `MalwareMLP` architecture:

```
Layer                  Input Dim    Output Dim    Activation    Regularization
Linear 1               443          256           ReLU          BatchNorm1d + Dropout(0.2)
Linear 2               256          128           ReLU          BatchNorm1d + Dropout(0.2)
Linear 3               128          64            ReLU          BatchNorm1d + Dropout(0.1)
Classification Head    64           5             Linear        Logits Output
```

- **Total Trainable Parameters:** 156,037 (~609.52 KB uncompressed float32 payload).
- **Loss Function:** Multi-class Cross-Entropy Loss:
  $$\mathcal{L}(\hat{\mathbf{y}}, y) = -\sum_{c=0}^4 y_c \log \frac{\exp(\hat{y}_c)}{\sum_{j=0}^4 \exp(\hat{y}_j)}$$

---

## 6. Experimental Protocol & Hyperparameters

| Hyperparameter | Symbol | Value | Justification |
|:---|:---:|:---:|:---|
| Number of Clients | $K$ | 10 | Standard cross-silo / mobile endpoint cluster |
| Client Participation Rate | $C$ | 1.0 (100%) | Full participation baseline |
| Communication Rounds | $T$ | 50 | Sufficient for convergence verification |
| Local Epochs | $E$ | 2 | Balances local compute vs client drift |
| Local Batch Size | $B$ | 64 | Consistent with centralized MLP training |
| Optimizer | — | AdamW | Adaptive learning rates with $L_2$ weight decay ($10^{-4}$) |
| Learning Rate | $\eta$ | $0.001$ | Standard initial rate matching centralized baseline |
| Random Seed | — | 42 | Deterministic across Python, NumPy, PyTorch |

---

## 7. Evaluation & Leakage Prevention Protocol

1. **Zero Data Leakage:**
   - Preprocessing `StandardScaler` was fitted **strictly on $X_{\text{train}}$**.
   - Client partitioning is applied **only to $X_{\text{train}}$ / $y_{\text{train}}$** (8,062 samples).
   - Validation ($1,152$ samples) and Test ($2,304$ samples) sets remain permanently centralized.
2. **Round-by-Round Validation:**
   - Evaluated on $X_{\text{val}}$ after every round to track convergence without test leakage.
3. **Locked Test Evaluation:**
   - The test set is evaluated **exactly once** using the best global model checkpoint selected by validation Macro F1.

---

## 8. Communication Cost Accounting

Assuming standard uncompressed 32-bit floating-point parameters (4 bytes / parameter):

- **Model Size:** $156,037 \times 4 \text{ bytes} = 624,148 \text{ bytes} \approx 609.52 \text{ KB}$.
- **Per-Round Downlink (Server $\rightarrow$ 10 Clients):** $10 \times 609.52 \text{ KB} = 5.95 \text{ MB}$.
- **Per-Round Uplink (10 Clients $\rightarrow$ Server):** $10 \times 609.52 \text{ KB} = 5.95 \text{ MB}$.
- **Total Per Round:** $11.90 \text{ MB} \approx 0.0119 \text{ GB}$.
- **Total Across 50 Rounds:** $50 \times 11.90 \text{ MB} = \mathbf{595.23 \text{ MB}} \approx \mathbf{0.58 \text{ GB}}$.

---

## 9. Planned Experiments Matrix

| Exp ID | Algorithm | Partition | Alpha ($\alpha$) | Clients | Rounds | Local Epochs | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp 1** | FedAvg | IID (Stratified) | N/A | 10 | 50 | 2 | **Active (Phase 5A)** |
| **Exp 2** | FedAvg | Non-IID Dirichlet | $\alpha=1.0$ | 10 | 50 | 2 | Next |
| **Exp 3** | FedAvg | Non-IID Dirichlet | $\alpha=0.5$ | 10 | 50 | 2 | Next |
| **Exp 4** | FedAvg | Non-IID Dirichlet | $\alpha=0.1$ | 10 | 50 | 2 | Next |
| **Exp 5** | FedAvg + DP | Non-IID / IID | TBD | 10 | 50 | 2 | Phase 6 (Privacy) |
