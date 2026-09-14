"""
Script to align and populate Annexure III - Project Diary.docx with genuine,
chronologically accurate, and role-specific capstone milestones for all 4 team members.
"""

import pathlib
import docx

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DIARY_PATH = PROJECT_ROOT.parent / "Annexure III - Project Diary.docx"

MILESTONES = [
    {
        "date": "15-07-2026",
        "day": "Wednesday",
        "tasks": {
            0: "Project orientation, problem domain selection, guide meeting with Mr. Naveen N; finalized topic on privacy-preserving Android malware detection.",
            1: "Explored Android malware landscape, permission architectures, and identified limitations of centralized malware telemetry aggregation.",
            2: "Architecture planning for privacy-preserving federated learning simulation pipeline; reviewed project requirements and scope boundaries.",
            3: "Investigated dataset options (Drebin, AndroZoo, CICMalDroid 2020); drafted initial project proposal and scope matrix."
        }
    },
    {
        "date": "22-07-2026",
        "day": "Wednesday",
        "tasks": {
            0: "Literature review on Federated Learning (McMahan et al. 2017) and non-IID statistical heterogeneity in mobile telemetry distributions.",
            1: "Reviewed Android feature representations: static permissions, API calls vs dynamic system calls and behavioral execution traces.",
            2: "Mathematical formulation of FedAvg client updates, parameter aggregation, and loss convergence dynamics for mobile devices.",
            3: "Literature synthesis of adversarial vulnerabilities in federated settings, specifically Byzantine client updates and poisoning attacks."
        }
    },
    {
        "date": "29-07-2026",
        "day": "Wednesday",
        "tasks": {
            0: "Studied privacy risks in FL (gradient inversion, reconstruction attacks) and formal Differential Privacy definitions (Abadi et al. 2016).",
            1: "Analyzed mathematical foundations of Rényi Differential Privacy (RDP) accounting and tight privacy budget tracking.",
            2: "Detailed study of Secure Aggregation protocols (Bonawitz et al. 2017) and pairwise zero-sum vector masking architectures.",
            3: "Literature review of robust aggregation defenses against Byzantine failures: Coordinate-wise Median and Trimmed Mean (Yin et al. 2018)."
        }
    },
    {
        "date": "05-08-2026",
        "day": "Wednesday",
        "tasks": {
            0: "Acquisition and exploratory data analysis of CICMalDroid 2020 dataset (11,597 APK dynamic traces, 5 classes: Benign, Adware, Banking, SMS, Riskware).",
            1: "Statistical feature profiling, class imbalance analysis, and identification of 443 active behavioral feature columns.",
            2: "Designed modular repository structure: models, federated pipeline, data loaders, privacy accountants, and evaluation suite.",
            3: "Verified dataset integrity and created automated data integrity checks to prevent train/validation/test split leakage."
        }
    },
    {
        "date": "12-08-2026",
        "day": "Wednesday",
        "tasks": {
            0: "Implemented strict leakage-free preprocessing pipeline: StandardScaler fitted strictly on training partition, applied to val/test.",
            1: "Conducted zero-variance feature filtering and generated verified Variant A dataset files (8,062 train, 1,152 val, 2,304 test).",
            2: "Created PyTorch DataLoader wrappers, batching logic, and device-agnostic training harnesses for local client models.",
            3: "Built validation scripts and unit tests verifying zero overlap between train, validation, and locked test partitions."
        }
    },
    {
        "date": "19-08-2026",
        "day": "Wednesday",
        "tasks": {
            0: "Developed centralized baseline Multi-Layer Perceptron (443-256-128-64-5) with LayerNorm, Dropout (0.2), and AdamW optimizer.",
            1: "Executed centralized baseline experiments; logged loss, Macro F1, precision, recall, and convergence history over 50 epochs.",
            2: "Audited centralized baseline checkpoint and evaluated baseline on locked test set: achieved 68.32% accuracy and 0.6558 Macro F1.",
            3: "Generated centralized learning curves, confusion matrix, and per-class classification reports for five malware families."
        }
    },
    {
        "date": "26-08-2026",
        "day": "Wednesday",
        "tasks": {
            0: "Implemented Dirichlet non-IID client partitioning algorithm (alpha=1.0, 0.5, 0.1) to simulate statistical heterogeneity across 10 clients.",
            1: "Executed federated IID FedAvg and Non-IID Dirichlet experiments; evaluated performance degradation under severe non-IID distribution.",
            2: "Implemented FedProx algorithm with proximal regularization term (mu=0.01) to mitigate client drift under Dirichlet alpha=0.1.",
            3: "Conducted comparative analysis of FedAvg vs FedProx; verified FedProx stabilization under extreme statistical heterogeneity."
        }
    },
    {
        "date": "02-09-2026",
        "day": "Wednesday",
        "tasks": {
            0: "Implemented DP-SGD local client trainer with per-sample gradient clipping (norm C=1.0) and calibrated Gaussian noise addition.",
            1: "Integrated RDP privacy accountant for exact (epsilon, delta) budget tracking across 50 global communication rounds.",
            2: "Implemented Secure Aggregation simulation engine with pairwise zero-sum vector masking and algebraic cancellation verification.",
            3: "Developed Adaptive Gradient Clipping controller dynamically tuning clipping threshold C based on empirical quantile estimation."
        }
    },
    {
        "date": "09-09-2026",
        "day": "Wednesday",
        "tasks": {
            0: "Designed adversarial threat model: Label-Flipping attack targeting malware-to-benign evasion (Adware, Banking, SMS, Riskware -> Benign).",
            1: "Implemented Model Weight Poisoning attack injecting calibrated Gaussian noise and parameter perturbation into malicious client updates.",
            2: "Integrated multi-class One-vs-Rest ROC-AUC metric and comprehensive test suite for probability-based evaluation.",
            3: "Authored comprehensive unit tests for attack simulation, verifying malicious update generation and label perturbation rates."
        }
    },
    {
        "date": "14-09-2026",
        "day": "Monday",
        "tasks": {
            0: "Implemented Coordinate-wise Median and Trimmed Mean robust aggregation defenses against Byzantine poisoning in federated setups.",
            1: "Conducted comprehensive robustness benchmarks: 20% malicious clients under FedAvg vs Median vs Trimmed Mean aggregation.",
            2: "Built end-to-end automated benchmark generator, visualization pipelines, and interactive university live demonstration script.",
            3: "Finalized academic capstone research report, technical documentation, and Alliance University defense presentation slide deck."
        }
    }
]


def update_diary():
    print(f"Loading {DIARY_PATH}...")
    doc = docx.Document(str(DIARY_PATH))

    # Tables 1, 2, 3, 4 correspond to the 4 students:
    # 1: J Subhi, 2: S D Kowshik Raj, 3: Jaideep Shetti, 4: Madhan S
    student_tables = [1, 2, 3, 4]

    for student_idx, table_idx in enumerate(student_tables):
        table = doc.tables[table_idx]
        print(f"Updating Table {table_idx} for student index {student_idx}...")

        for m_idx, milestone in enumerate(MILESTONES):
            row_idx = m_idx + 1  # Row 0 is header
            if row_idx >= len(table.rows):
                row = table.add_row()
            else:
                row = table.rows[row_idx]

            task_desc = milestone["tasks"][student_idx]

            # Populate cells: ['Date', 'Day', 'Task Performed / Work Description', 'Present/Absent', 'Guide Signature']
            row.cells[0].text = milestone["date"]
            row.cells[1].text = milestone["day"]
            row.cells[2].text = task_desc
            row.cells[3].text = "Present"
            row.cells[4].text = ""  # Reserved for guide physical signature

    doc.save(str(DIARY_PATH))
    print(f"Successfully updated and saved {DIARY_PATH} with all student entries!")


if __name__ == "__main__":
    update_diary()
