"""
Presentation Generator for Alliance University Capstone Project Defense.
Uses python-pptx to load the official ASAC template and populate slides with
rigorous, verified experimental results, architecture, publication figures, and defense materials.
"""

import sys
import json
import pathlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT.parent / "PPT Template July2026-Dec2026-V!.pptx"
OUTPUT_PATH = PROJECT_ROOT.parent / "Final_Capstone_Presentation.pptx"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Color Palette
NAVY = RGBColor(14, 34, 61)
GOLD = RGBColor(197, 160, 89)
DARK_GRAY = RGBColor(60, 60, 60)
WHITE = RGBColor(255, 255, 255)


def populate_title_slide(slide):
    """Slide 1: Official Title Slide with Team Metadata."""
    for shape in slide.shapes:
        if shape.has_text_frame:
            text = shape.text_frame.text
            if "Capstone Project" in text:
                continue
            if "Review No" in text:
                tf = shape.text_frame
                tf.clear()
                lines = [
                    "Review No\t: Final Capstone Defense Review",
                    "Batch No\t: ASAC-CP-167",
                    "",
                    "Presented by:",
                    "  1. J SUBHI (Team Leader)        - 2023BIFT07AED012 (IT-DA)",
                    "  2. S D KOWSHIK RAJ              - 2023BIFT07AED009 (IT-DA)",
                    "  3. JAIDEEP SHETTI               - 2023BIFT07AED055 (IT-CC)",
                    "  4. MADHAN S                     - 2023BIFT07AED030 (IT-DA)",
                    "",
                    "Project Guide: Mr. Naveen N (Assistant Professor, ASAC)",
                    "Date of Review: 14 September 2026"
                ]
                for i, l in enumerate(lines):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    p.text = l
                    p.font.size = Pt(13 if i < 9 else 14)
                    p.font.color.rgb = NAVY
                    p.font.bold = (i in [0, 1, 3, 9, 10])
            elif "Project Title:" in text:
                tf = shape.text_frame
                tf.clear()
                p0 = tf.paragraphs[0]
                p0.text = "Project Title:"
                p0.font.size = Pt(14)
                p0.font.bold = True
                p0.font.color.rgb = GOLD
                
                p1 = tf.add_paragraph()
                p1.text = "Privacy-Preserving Malware Detection Using Federated Learning"
                p1.font.size = Pt(22)
                p1.font.bold = True
                p1.font.color.rgb = NAVY


def set_content_slide(slide, title_text, bullets):
    """Generic helper to populate title and bullet points on a slide."""
    title_shape = None
    body_shape = None

    for shape in slide.shapes:
        if shape.has_text_frame:
            t = shape.text_frame.text.strip()
            if any(roman in t for roman in ["I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "Outline", "THANK"]):
                title_shape = shape
            elif len(t) > 0 and body_shape is None:
                body_shape = shape

    if title_shape and title_shape.has_text_frame:
        tf = title_shape.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.bold = True
        p.font.size = Pt(24)
        p.font.color.rgb = NAVY

    if body_shape and body_shape.has_text_frame:
        tf = body_shape.text_frame
        tf.clear()
        for idx, (head, sub) in enumerate(bullets):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.text = f"{head}: " if head else ""
            p.font.bold = True
            p.font.size = Pt(13)
            p.font.color.rgb = NAVY
            
            run = p.add_run()
            run.text = sub
            run.font.bold = False
            run.font.size = Pt(12)
            run.font.color.rgb = DARK_GRAY
            p.space_after = Pt(8)


def add_figure_slide(prs, title_text, img_path, takeaways):
    """Adds a slide with title, embedded publication-quality plot, and key takeaways."""
    blank_layout = prs.slide_layouts[3]  # Title Only
    slide = prs.slides.add_slide(blank_layout)

    # Title
    title_shape = slide.shapes.title
    if title_shape:
        title_shape.text = title_text
        p = title_shape.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(22)
        p.font.color.rgb = NAVY

    # Add Image
    if img_path.exists():
        slide.shapes.add_picture(
            str(img_path),
            Inches(0.6),
            Inches(1.5),
            width=Inches(7.2)
        )

    # Add Takeaways Textbox on the Right
    tx_box = slide.shapes.add_textbox(Inches(8.0), Inches(1.5), Inches(4.8), Inches(5.0))
    tf = tx_box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "Key Empirical Takeaways:"
    p0.font.bold = True
    p0.font.size = Pt(14)
    p0.font.color.rgb = GOLD
    p0.space_after = Pt(10)

    for head, sub in takeaways:
        p = tf.add_paragraph()
        p.text = f"• {head}: " if head else "• "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = NAVY

        run = p.add_run()
        run.text = sub
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = DARK_GRAY
        p.space_after = Pt(8)


def generate():
    print(f"Loading template: {TEMPLATE_PATH}")
    prs = Presentation(str(TEMPLATE_PATH))

    # Slide 1: Title
    populate_title_slide(prs.slides[0])

    # Slide 2: Outline
    set_content_slide(
        prs.slides[1],
        "Agenda & Presentation Outline",
        [
            ("1. Project Overview", "The threat landscape of Android malware and privacy perils of centralized telemetry."),
            ("2. Literature Review", "Critical synthesis of federated learning, differential privacy, and Byzantine defenses."),
            ("3. Research Gap", "Lack of unified frameworks handling privacy, client heterogeneity, and adversarial poisoning."),
            ("4. System Architecture", "Multi-Layer Perceptron, DP-SGD with adaptive clipping, SecAgg, and robust aggregators."),
            ("5. Threat Models & Defenses", "Label-flipping & model weight poisoning attacks defended by Trimmed Mean & Median."),
            ("6. Experimental Benchmarks", "Verified results across 10 clients, 50 rounds, Non-IID Dirichlet partitions, and attack scenarios."),
            ("7. Live Demonstration", "Interactive end-to-end execution of federated round, attack injection, and robust defense.")
        ]
    )

    # Slide 3: Project Overview
    set_content_slide(
        prs.slides[2],
        "I. Project Overview & Problem Context",
        [
            ("Problem Context", "Android commands 70%+ mobile market share, confronting over 350,000 new malicious variants daily."),
            ("Privacy Bottleneck", "Centralized malware analysis requires uploading sensitive user behavioral traces, violating GDPR and CCPA regulations."),
            ("Our Solution", "A decentralized Federated Learning paradigm where edge clients collaboratively train a defense model without transmitting raw execution traces."),
            ("Privacy Mechanics", "Integrates Local Differential Privacy (DP-SGD) and Secure Aggregation (pairwise zero-sum masking) to mathematically eliminate telemetry leakage."),
            ("Byzantine Resilience", "Hardened against adversarial poisoning via Coordinate-wise Median and Trimmed Mean aggregation rules.")
        ]
    )

    # Slide 4: Literature Review
    set_content_slide(
        prs.slides[3],
        "II. Literature Review & Technical Background",
        [
            ("Federated Optimization", "McMahan et al. (AISTATS 2017) formulated FedAvg; Li et al. (MLSys 2020) proposed FedProx to constrain client drift via proximal regularization."),
            ("Differential Privacy in FL", "Abadi et al. (ACM CCS 2016) introduced DP-SGD and moments accountant; Mironov (CSF 2017) formalized Rényi Differential Privacy (RDP)."),
            ("Secure Aggregation", "Bonawitz et al. (ACM CCS 2017) designed cryptographic pairwise masking to compute sum-aggregates without exposing individual client weights."),
            ("Byzantine Robust Defenses", "Blanchard et al. (NeurIPS 2017, Krum) and Yin et al. (ICML 2018, Median & Trimmed Mean) established theoretical breakdown points against poisoning."),
            ("Android Malware Benchmarks", "Mahdavifar et al. (IEEE Trans. Reliability 2020) published CICMalDroid 2020 containing 11,597 behavioral traces across 5 malware families.")
        ]
    )

    # Slide 5: Research Gap
    set_content_slide(
        prs.slides[4],
        "III. Identified Research Gap",
        [
            ("Absence of Unified Triad", "Existing literature evaluates privacy (DP), security (SecAgg), or Byzantine robustness in isolation—neglecting severe conflicts between them."),
            ("Clip-Norm Sensitivity", "Standard DP-SGD relies on fixed manual clipping (C=1.0), which either destroys model signal (over-clipping) or explodes noise addition."),
            ("Extreme Non-IID Mobile Skew", "Real mobile malware distributions are heavily skewed (Dirichlet alpha=0.1); standard FedAvg suffers massive client drift and loss divergence."),
            ("Vulnerability to Poisoning", "Standard FedAvg collapses under 20% label-flipping or weight poisoning; need proven robust aggregation defenses."),
            ("Our Contribution", "First unified FL benchmark for Android malware combining Adaptive DP-SGD, pairwise SecAgg, and Trimmed Mean/Median defenses on CICMalDroid 2020.")
        ]
    )

    # Slide 6: Requirements Analysis
    set_content_slide(
        prs.slides[5],
        "IV. System Requirements & Dataset Profile",
        [
            ("Dataset Specification", "CICMalDroid 2020 Variant A: 11,597 complete behavioral traces (8,062 train, 1,152 validation, 2,304 locked test)."),
            ("5 Malware Categories", "Benign (1,795), Adware (1,253), Banking (2,100), SMS (3,904), Riskware (2,545 traces)."),
            ("Feature Architecture", "443 active dynamic behavioral features (system calls, permissions, intents, network sockets) standardized leakage-free."),
            ("Software Environment", "Python 3.14.0, PyTorch 2.10.0+cpu, scikit-learn 1.6.1, NumPy 2.4.1, python-pptx, pytest."),
            ("Hardware Platform", "Intel64 multi-core processor, 16GB RAM, reproducible seeded deterministic simulation harnesses.")
        ]
    )

    # Slide 7: Proposed Methodology
    set_content_slide(
        prs.slides[6],
        "V. Proposed Methodology & Architecture",
        [
            ("MLP Classification Engine", "443 input features -> 256 -> 128 -> 64 -> 5 logits with LayerNorm, Dropout (0.2), and AdamW optimization (156,037 parameters)."),
            ("Adaptive DP-SGD Engine", "Per-sample gradient clipping with dynamic threshold adaptation tracking empirical 90th percentile gradient norms (Andrew et al.)."),
            ("Rényi Privacy Accountant", "Continuous privacy loss composition over 50 communication rounds achieving tight provable (epsilon, delta) guarantees."),
            ("Secure Aggregation Layer", "Pairwise zero-sum vector masking guaranteeing server learns strictly the global aggregate with zero single-client parameter visibility."),
            ("Adversarial Defense Modules", "Coordinate-wise Median and Trimmed Mean (beta=0.2) replacing arithmetic mean to neutralize 20% Byzantine malicious clients.")
        ]
    )

    # Slide 8: Experimental Work Plan & Timeline
    set_content_slide(
        prs.slides[7],
        "VI. Experimental Design & Milestone Execution",
        [
            ("Phase 1: Baselines", "Centralized MLP (91.28% Acc, 0.8957 F1) vs Federated FedAvg (90.19% Acc, 0.8853 F1) over 50 rounds across 10 clients."),
            ("Phase 2: Heterogeneity", "Non-IID Dirichlet partitions (alpha=1.0, 0.5, 0.1); FedProx (mu=0.01) mitigating client drift under extreme non-IID conditions."),
            ("Phase 3: Privacy & Security", "DP-SGD noise scaling (sigma=0.5, 1.0, 2.0); SecAgg algebraic cancellation verified with 0.000000 reconstruction error."),
            ("Phase 4: Adaptive Clipping", "Dynamic clipping controller adjusting C from 1.0 to empirical quantile; benchmarked across IID and Non-IID partitions."),
            ("Phase 5: Robustness Under Attack", "Evaluation of 20% malicious clients executing Label-Flipping and Weight-Poisoning against FedAvg, Trimmed Mean, and Median.")
        ]
    )

    # --- Append Deep-Dive Experimental Result Slides ---
    print("Appending empirical figure slides...")

    # Slide 9: Centralized vs Federated Figure
    add_figure_slide(
        prs,
        "VII. Empirical Results: Centralized vs Federated Performance",
        FIGURES_DIR / "fig1_centralized_vs_federated.png",
        [
            ("Centralized MLP", "Achieves 91.28% test accuracy and 0.8957 Macro F1."),
            ("Federated FedAvg (IID)", "Reaches 90.19% test accuracy and 0.8853 Macro F1 across 10 clients."),
            ("Minimal Penalty", "The decentralization gap is only 1.09% accuracy, proving federated feasibility without raw data sharing.")
        ]
    )

    # Slide 10: Non-IID Dirichlet Skew Figure
    add_figure_slide(
        prs,
        "VIII. Empirical Results: Non-IID Data Skew & FedProx Defense",
        FIGURES_DIR / "fig2_noniid_convergence.png",
        [
            ("Dirichlet Skew", "Severe non-IID skew (alpha=0.1) drops FedAvg to 69.92% accuracy and 0.5836 Macro F1."),
            ("FedProx Stabilization", "Adding proximal regularization (mu=0.01) lifts accuracy to 72.92% and Macro F1 to 0.6138 (+3.0% boost)."),
            ("Client Drift Mitigation", "Penalizing local parameter deviations prevents local over-fitting on skewed client distributions.")
        ]
    )

    # Slide 11: Privacy-Utility Tradeoff Figure
    add_figure_slide(
        prs,
        "IX. Empirical Results: Privacy–Utility Pareto Curve (DP-SGD)",
        FIGURES_DIR / "fig3_privacy_utility_tradeoff.png",
        [
            ("Noise Scaling", "sigma=0.5 yields 79.99% Acc (eps=49.99); sigma=1.0 yields 74.57% Acc (eps=25.51)."),
            ("High Privacy", "sigma=2.0 provides strong privacy (eps=8.71) while sustaining 66.02% accuracy."),
            ("Zero-Sum SecAgg", "Pairwise vector masking verified with 0.000000 cancellation error, concealing individual client gradients.")
        ]
    )

    # Slide 12: Adaptive Gradient Clipping Trajectory
    add_figure_slide(
        prs,
        "X. Empirical Results: Adaptive Gradient Clipping Dynamics",
        FIGURES_DIR / "fig5_adaptive_clipping_trajectory.png",
        [
            ("Threshold Adaptation", "Clipping threshold C smoothly adapts from 1.0 to empirical 90th percentile (C=10.0)."),
            ("Quantile Tracking", "Observed clipping fraction converges toward the target 10% unclipped margin."),
            ("Accuracy Gain", "Adaptive clipping improves test accuracy from 74.57% to 75.35% and Macro F1 from 0.7040 to 0.7122.")
        ]
    )

    # Slide 13: Byzantine Poisoning Defenses
    add_figure_slide(
        prs,
        "XI. Empirical Results: Byzantine Poisoning Attacks & Robust Defenses",
        FIGURES_DIR / "fig4_byzantine_robustness.png",
        [
            ("Weight Poisoning Collapse", "Standard FedAvg collapses from 89.58% to 57.03% accuracy under 20% weight poisoning."),
            ("Trimmed Mean Defense", "Trimmed Mean (beta=0.2) achieves 90.02% accuracy and 0.8830 Macro F1 (+32.99% recovery over FedAvg)."),
            ("Coordinate-wise Median", "Median achieves 89.76% accuracy and 0.8799 Macro F1, completely neutralizing malicious updates.")
        ]
    )

    # Slide 14: References
    set_content_slide(
        prs.slides[8],
        "XII. Selected References (IEEE Format)",
        [
            ("[1] H. B. McMahan et al.", "'Communication-Efficient Learning of Deep Networks from Decentralized Data', AISTATS, PMLR 54:1273-1282, 2017."),
            ("[2] M. Abadi et al.", "'Deep Learning with Differential Privacy', Proc. ACM SIGSAC Conference on Computer and Communications Security (CCS), pp. 308-318, 2016."),
            ("[3] K. Bonawitz et al.", "'Practical Secure Aggregation for Privacy-Preserving Machine Learning', ACM CCS, pp. 1175-1191, 2017."),
            ("[4] D. Yin, Y. Chen, R. Kannan, and P. Bartlett", "'Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates', ICML, PMLR 80:5650-5659, 2018."),
            ("[5] G. Andrew et al.", "'Differentially Private Learning with Adaptive Clipping', NeurIPS, vol. 34, pp. 17455-17466, 2021."),
            ("[6] S. Mahdavifar et al.", "'Classifying Dynamic Android Malware Using Android Device Behavioral Features', IEEE Trans. Reliability, 2020.")
        ]
    )

    # Slide 15: Conclusion & Thank You
    prs.slides[9].shapes[2].text_frame.text = (
        "THANK YOU!\n\n"
        "Presented by Team ASAC-CP-167:\n"
        "J Subhi | S D Kowshik Raj | Jaideep Shetti | Madhan S\n"
        "Department of Information Technology\n"
        "Alliance School of Advanced Computing, Alliance University"
    )

    prs.save(str(OUTPUT_PATH))
    print(f"Successfully generated 15-slide defense presentation at: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
