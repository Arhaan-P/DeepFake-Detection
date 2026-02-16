"""
Research Paper Figures, Tables & Metrics Generator
====================================================
Generates all figures, tables, and metrics needed for the research paper:
1. Confusion Matrix (pooled from LOOCV)
2. All 4 test cases: TP, FP, TN, FN with examples
3. ROC Curve, Precision-Recall Curve, DET Curve
4. Per-fold LOOCV bar chart
5. Ablation study comparison chart
6. GradCAM joint importance chart
7. Similarity score distribution
8. Per-identity accuracy heatmap
9. LaTeX-ready tables

Usage:
    python scripts/evaluation/generate_paper_figures.py

Author: DeepFake Detection Project
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    det_curve, f1_score, accuracy_score, precision_score, recall_score
)

# Output directory
OUTPUT_DIR = Path('outputs/paper_figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style settings for publication-quality figures
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
})


def load_loocv_results():
    """Load LOOCV results from JSON."""
    results_path = Path('outputs/evaluation/loocv/loocv_results.json')
    if not results_path.exists():
        print(f"ERROR: LOOCV results not found at {results_path}")
        print("Run: python scripts/evaluation/evaluate.py --loocv")
        sys.exit(1)
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Extract pooled labels and scores
    raw_labels = data['pooled_labels']
    raw_scores = data['pooled_scores']
    
    # Flatten nested lists
    labels = np.array([l[0] if isinstance(l, list) else l for l in raw_labels])
    scores = np.array([s[0] if isinstance(s, list) else s for s in raw_scores])
    
    return data, labels, scores


def load_ablation_results():
    """Load ablation study results."""
    path = Path('outputs/ablation/ablation_results.json')
    if not path.exists():
        print(f"WARNING: Ablation results not found at {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)


def load_gradcam_results():
    """Load GradCAM results."""
    path = Path('outputs/gradcam/aggregate/gradcam_results.json')
    if not path.exists():
        print(f"WARNING: GradCAM results not found at {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================
# FIGURE 1: Confusion Matrix
# ============================================================
def plot_confusion_matrix(labels, scores, threshold=0.5):
    """Generate publication-quality confusion matrix."""
    predictions = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    
    # Normalized confusion matrix
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # --- Raw counts ---
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['DEEPFAKE', 'AUTHENTIC'],
                yticklabels=['DEEPFAKE', 'AUTHENTIC'],
                annot_kws={'size': 18, 'weight': 'bold'},
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted Label', fontsize=14)
    axes[0].set_ylabel('True Label', fontsize=14)
    axes[0].set_title('(a) Confusion Matrix (Counts)', fontsize=15, fontweight='bold')
    
    # --- Normalized (%) ---
    annot_labels = np.array([[f'{v:.1f}%' for v in row] for row in cm_norm])
    sns.heatmap(cm_norm, annot=annot_labels, fmt='', cmap='Blues',
                xticklabels=['DEEPFAKE', 'AUTHENTIC'],
                yticklabels=['DEEPFAKE', 'AUTHENTIC'],
                annot_kws={'size': 18, 'weight': 'bold'},
                ax=axes[1], vmin=0, vmax=100,
                cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_xlabel('Predicted Label', fontsize=14)
    axes[1].set_ylabel('True Label', fontsize=14)
    axes[1].set_title('(b) Confusion Matrix (Normalized %)', fontsize=15, fontweight='bold')
    
    plt.tight_layout(pad=2.0)
    path = OUTPUT_DIR / 'fig1_confusion_matrix.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    
    # Extract TP, TN, FP, FN
    tn, fp, fn, tp = cm.ravel()
    return {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'cm': cm}


# ============================================================
# FIGURE 2: All 4 Cases (TP, FP, TN, FN) Explanation
# ============================================================
def plot_four_cases(cm_stats, labels, scores, threshold=0.5):
    """Visualize TP, FP, TN, FN with example similarity scores."""
    predictions = (scores >= threshold).astype(int)
    
    # Find examples of each case
    tp_mask = (labels == 1) & (predictions == 1)
    tn_mask = (labels == 0) & (predictions == 0)
    fp_mask = (labels == 0) & (predictions == 1)
    fn_mask = (labels == 1) & (predictions == 0)
    
    cases = {
        'True Positive (TP)': {
            'count': cm_stats['TP'],
            'color': '#2ecc71',
            'desc': 'Authentic video correctly\nidentified as AUTHENTIC',
            'example_scores': scores[tp_mask][:5] if tp_mask.any() else [],
            'icon': '\u2713'  # checkmark
        },
        'True Negative (TN)': {
            'count': cm_stats['TN'],
            'color': '#3498db',
            'desc': 'Deepfake video correctly\nidentified as DEEPFAKE',
            'example_scores': scores[tn_mask][:5] if tn_mask.any() else [],
            'icon': '\u2713'
        },
        'False Positive (FP)': {
            'count': cm_stats['FP'],
            'color': '#e74c3c',
            'desc': 'Deepfake video wrongly\nidentified as AUTHENTIC',
            'example_scores': scores[fp_mask][:5] if fp_mask.any() else [],
            'icon': '\u2717'  # x mark
        },
        'False Negative (FN)': {
            'count': cm_stats['FN'],
            'color': '#f39c12',
            'desc': 'Authentic video wrongly\nidentified as DEEPFAKE',
            'example_scores': scores[fn_mask][:5] if fn_mask.any() else [],
            'icon': '\u2717'
        }
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (case_name, info) in enumerate(cases.items()):
        ax = axes[idx]
        
        # Big count display
        ax.text(0.5, 0.65, str(info['count']),
                transform=ax.transAxes, fontsize=48, fontweight='bold',
                ha='center', va='center', color=info['color'])
        
        # Case name
        ax.text(0.5, 0.9, case_name,
                transform=ax.transAxes, fontsize=16, fontweight='bold',
                ha='center', va='center', color='black')
        
        # Description
        ax.text(0.5, 0.38, info['desc'],
                transform=ax.transAxes, fontsize=12,
                ha='center', va='center', color='#555555',
                style='italic')
        
        # Example scores
        if len(info['example_scores']) > 0:
            score_text = 'Example scores: ' + ', '.join(
                [f'{s:.3f}' for s in info['example_scores'][:3]]
            )
            ax.text(0.5, 0.15, score_text,
                    transform=ax.transAxes, fontsize=10,
                    ha='center', va='center', color='#777777')
        
        # Styling
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Border color
        for spine in ax.spines.values():
            spine.set_color(info['color'])
            spine.set_linewidth(3)
    
    total = cm_stats['TP'] + cm_stats['TN'] + cm_stats['FP'] + cm_stats['FN']
    accuracy = (cm_stats['TP'] + cm_stats['TN']) / total * 100
    fig.suptitle(f'Classification Outcomes — Pooled LOOCV (N={total}, Accuracy={accuracy:.1f}%)',
                 fontsize=17, fontweight='bold', y=1.02)
    
    plt.tight_layout(pad=1.5)
    path = OUTPUT_DIR / 'fig2_four_cases_tp_fp_tn_fn.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    
    return cases


# ============================================================
# FIGURE 3: ROC Curve
# ============================================================
def plot_roc_curve(labels, scores):
    """Publication-quality ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_thresh = thresholds[optimal_idx]
    
    # Find EER point
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2.5,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    
    # Mark optimal threshold
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
            label=f'Optimal (J={j_scores[optimal_idx]:.3f}, t={optimal_thresh:.3f})')
    
    # Mark EER point
    ax.plot(fpr[eer_idx], tpr[eer_idx], 'g^', markersize=10,
            label=f'EER = {eer:.4f}')
    
    # Shade AUC
    ax.fill_between(fpr, tpr, alpha=0.15, color='blue')
    
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=14)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve\nPooled LOOCV Results', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'fig3_roc_curve.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    
    return roc_auc, eer, optimal_thresh


# ============================================================
# FIGURE 4: Precision-Recall Curve
# ============================================================
def plot_pr_curve(labels, scores):
    """Publication-quality Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    ax.plot(recall, precision, 'b-', linewidth=2.5,
            label=f'PR Curve (AP = {pr_auc:.4f})')
    
    # Baseline (proportion of positives)
    baseline = labels.sum() / len(labels)
    ax.axhline(y=baseline, color='r', linestyle='--', alpha=0.5,
               label=f'Baseline (P(Y=1) = {baseline:.2f})')
    
    ax.fill_between(recall, precision, alpha=0.15, color='blue')
    
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve\nPooled LOOCV Results', fontsize=15, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'fig4_precision_recall_curve.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    
    return pr_auc


# ============================================================
# FIGURE 5: Similarity Score Distribution
# ============================================================
def plot_similarity_distribution(labels, scores, threshold=0.5):
    """Distribution of similarity scores for authentic vs deepfake pairs."""
    authentic_scores = scores[labels == 1]
    deepfake_scores = scores[labels == 0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(authentic_scores, bins=50, alpha=0.65, label='Authentic Pairs',
            color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.hist(deepfake_scores, bins=50, alpha=0.65, label='Deepfake Pairs',
            color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
               label=f'Decision Threshold (t={threshold:.3f})')
    
    # Add overlap region annotation
    ax.set_xlabel('Similarity Score', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Distribution of Gait Similarity Scores', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Stats text box
    stats_text = (
        f'Authentic: $\\mu$={authentic_scores.mean():.3f}, $\\sigma$={authentic_scores.std():.3f}\n'
        f'Deepfake:  $\\mu$={deepfake_scores.mean():.3f}, $\\sigma$={deepfake_scores.std():.3f}'
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'fig5_similarity_distribution.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# FIGURE 6: Per-Fold LOOCV Results Bar Chart
# ============================================================
def plot_loocv_per_fold(loocv_data):
    """Bar chart of per-fold LOOCV metrics."""
    folds = loocv_data['per_fold']
    persons = [f['test_person'] for f in folds]
    accuracies = [f['accuracy'] * 100 for f in folds]
    f1_scores = [f['f1'] * 100 for f in folds]
    aucs = [f['roc_auc'] * 100 for f in folds]
    
    x = np.arange(len(persons))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x, f1_scores, width, label='F1 Score', color='#2ecc71', alpha=0.85)
    bars3 = ax.bar(x + width, aucs, width, label='AUC-ROC', color='#e74c3c', alpha=0.85)
    
    # Mean lines
    agg = loocv_data['aggregate']
    ax.axhline(y=agg['accuracy'] * 100, color='#3498db', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=agg['f1'] * 100, color='#2ecc71', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=agg['roc_auc'] * 100, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Test Subject (Held-Out Fold)', fontsize=14)
    ax.set_ylabel('Score (%)', fontsize=14)
    ax.set_title('Leave-One-Out Cross-Validation: Per-Fold Results', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(persons, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([60, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'fig6_loocv_per_fold.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# FIGURE 7: Ablation Study Comparison
# ============================================================
def plot_ablation_study(ablation_data):
    """Ablation study bar chart comparing model variants."""
    if ablation_data is None:
        print("  Skipping ablation plot — no data")
        return
    
    variants = list(ablation_data.keys())
    nice_names = {
        'CNN-Only': 'CNN-Only',
        'LSTM-Only': 'BiLSTM-Only',
        'Transformer-Only': 'Transformer-Only',
        'Full Hybrid': 'Full Hybrid\n(CNN+BiLSTM+Transformer)'
    }
    display_names = [nice_names.get(v, v) for v in variants]
    
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    x = np.arange(len(variants))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [ablation_data[v][metric] for v in variants]
        ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
    
    ax.set_xlabel('Model Variant', fontsize=14)
    ax.set_ylabel('Score (%)', fontsize=14)
    ax.set_title('Ablation Study: Model Architecture Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([82, 98])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add param counts above bars
    for j, v in enumerate(variants):
        params = ablation_data[v]['params']
        ax.text(j + 1.5 * width, 83, f'{params/1000:.0f}K',
                ha='center', fontsize=9, color='#555555', style='italic')
    ax.text(0.02, 0.02, 'Numbers below bars: parameter count',
            transform=ax.transAxes, fontsize=9, color='#888888')
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'fig7_ablation_study.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# FIGURE 8: GradCAM Joint Importance
# ============================================================
def plot_gradcam_importance(gradcam_data):
    """GradCAM joint importance bar chart."""
    if gradcam_data is None:
        print("  Skipping GradCAM plot — no data")
        return
    
    # Joint importance
    joints = gradcam_data['joint_importance']
    names = list(joints.keys())
    values = list(joints.values())
    
    # Color by body region
    region_colors = {
        'Shoulder': '#3498db', 'Hip': '#2ecc71',
        'Knee': '#e74c3c', 'Ankle': '#f39c12',
        'Heel': '#9b59b6', 'Foot': '#1abc9c'
    }
    colors = []
    for name in names:
        for region, color in region_colors.items():
            if region in name:
                colors.append(color)
                break
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Joint importance ---
    sorted_idx = np.argsort(values)[::-1]
    sorted_names = [names[i] for i in sorted_idx]
    sorted_values = [values[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]
    
    axes[0].barh(range(len(sorted_names)), sorted_values, color=sorted_colors, alpha=0.85)
    axes[0].set_yticks(range(len(sorted_names)))
    axes[0].set_yticklabels(sorted_names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Normalized Importance', fontsize=13)
    axes[0].set_title('(a) Joint Importance (GradCAM)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add legend for body regions
    region_patches = [mpatches.Patch(color=c, label=r) for r, c in region_colors.items()]
    axes[0].legend(handles=region_patches, loc='lower right', fontsize=9)
    
    # --- Feature group importance ---
    groups = gradcam_data['group_importance']
    group_names = ['Coordinates\n(12×3D = 36 dims)', 'Velocities\n(12×3D = 36 dims)', 'Joint Angles\n(6 dims)']
    group_values = [groups['coords'], groups['velocities'], groups['angles']]
    group_colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    axes[1].pie(group_values, labels=group_names, colors=group_colors,
                autopct='%1.1f%%', startangle=140,
                textprops={'fontsize': 11}, pctdistance=0.75,
                wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
    axes[1].set_title('(b) Feature Group Contribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout(pad=2.0)
    path = OUTPUT_DIR / 'fig8_gradcam_importance.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# FIGURE 9: DET Curve
# ============================================================
def plot_det_curve(labels, scores):
    """Detection Error Tradeoff curve (standard in biometric papers)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    ax.plot(fpr * 100, fnr * 100, 'b-', linewidth=2.5, label='Our Model')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.3)
    
    # Mark EER point
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100
    ax.plot(fpr[eer_idx] * 100, fnr[eer_idx] * 100, 'ro', markersize=10,
            label=f'EER = {eer:.2f}%')
    
    ax.set_xlabel('False Positive Rate (%)', fontsize=14)
    ax.set_ylabel('False Negative Rate (%)', fontsize=14)
    ax.set_title('Detection Error Tradeoff (DET) Curve', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'fig9_det_curve.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# FIGURE 10: EER vs. Per-Fold 
# ============================================================
def plot_eer_per_fold(loocv_data):
    """EER per fold scatter plot."""
    folds = loocv_data['per_fold']
    persons = [f['test_person'] for f in folds]
    eers = [f['eer'] * 100 for f in folds]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = ['#e74c3c' if e > 15 else '#f39c12' if e > 10 else '#2ecc71' for e in eers]
    bars = ax.bar(persons, eers, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Mean EER line
    mean_eer = loocv_data['aggregate']['eer'] * 100
    ax.axhline(y=mean_eer, color='blue', linestyle='--', linewidth=2,
               label=f'Mean EER = {mean_eer:.2f}%')
    
    ax.set_xlabel('Test Subject', fontsize=14)
    ax.set_ylabel('Equal Error Rate (%)', fontsize=14)
    ax.set_title('EER per LOOCV Fold', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    path = OUTPUT_DIR / 'fig10_eer_per_fold.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# TABLE GENERATORS
# ============================================================
def generate_tables(loocv_data, cm_stats, ablation_data, gradcam_data, roc_auc, eer, pr_auc, labels, scores, threshold):
    """Generate all tables as formatted text files."""
    
    predictions = (scores >= threshold).astype(int)
    accuracy = accuracy_score(labels, predictions) * 100
    precision = precision_score(labels, predictions, zero_division=0) * 100
    recall = recall_score(labels, predictions, zero_division=0) * 100
    f1 = f1_score(labels, predictions, zero_division=0) * 100
    
    tables_text = []
    
    # ---- TABLE 1: Overall Metrics ----
    tables_text.append("=" * 70)
    tables_text.append("TABLE 1: Overall LOOCV Performance Metrics (Pooled)")
    tables_text.append("=" * 70)
    tables_text.append(f"{'Metric':<30} {'Value':<20} {'Std Dev':<15}")
    tables_text.append("-" * 65)
    
    agg = loocv_data['aggregate']
    rows = [
        ('Accuracy (%)', f"{agg['accuracy']*100:.2f}", f"± {agg['accuracy_std']*100:.2f}"),
        ('Precision (%)', f"{agg['precision']*100:.2f}", f"± {agg['precision_std']*100:.2f}"),
        ('Recall (%)', f"{agg['recall']*100:.2f}", f"± {agg['recall_std']*100:.2f}"),
        ('F1 Score (%)', f"{agg['f1']*100:.2f}", f"± {agg['f1_std']*100:.2f}"),
        ('ROC-AUC (%)', f"{agg['roc_auc']*100:.2f}", f"± {agg['roc_auc_std']*100:.2f}"),
        ('Equal Error Rate (%)', f"{agg['eer']*100:.2f}", f"± {agg['eer_std']*100:.2f}"),
        ('Pooled ROC-AUC (%)', f"{agg['pooled_roc_auc']*100:.2f}", '—'),
        ('Pooled EER (%)', f"{agg['pooled_eer']*100:.2f}", '—'),
        ('Number of Folds', str(agg['n_folds']), '—'),
    ]
    for name, val, std in rows:
        tables_text.append(f"{name:<30} {val:<20} {std:<15}")
    
    # ---- TABLE 2: Confusion Matrix Stats ----
    tables_text.append("\n" + "=" * 70)
    tables_text.append("TABLE 2: Pooled Confusion Matrix Statistics")
    tables_text.append("=" * 70)
    total = cm_stats['TP'] + cm_stats['TN'] + cm_stats['FP'] + cm_stats['FN']
    tables_text.append(f"{'Outcome':<25} {'Count':<10} {'Percentage':<15} {'Description':<40}")
    tables_text.append("-" * 90)
    tables_text.append(f"{'True Positive (TP)':<25} {cm_stats['TP']:<10} {cm_stats['TP']/total*100:.1f}%{'':<10} {'Authentic correctly classified':<40}")
    tables_text.append(f"{'True Negative (TN)':<25} {cm_stats['TN']:<10} {cm_stats['TN']/total*100:.1f}%{'':<10} {'Deepfake correctly classified':<40}")
    tables_text.append(f"{'False Positive (FP)':<25} {cm_stats['FP']:<10} {cm_stats['FP']/total*100:.1f}%{'':<10} {'Deepfake misclassified as authentic':<40}")
    tables_text.append(f"{'False Negative (FN)':<25} {cm_stats['FN']:<10} {cm_stats['FN']/total*100:.1f}%{'':<10} {'Authentic misclassified as deepfake':<40}")
    tables_text.append(f"{'TOTAL':<25} {total:<10} {'100.0%':<15} {'':<40}")
    
    # ---- TABLE 3: Per-Fold LOOCV ----
    tables_text.append("\n" + "=" * 70)
    tables_text.append("TABLE 3: Per-Fold LOOCV Results")
    tables_text.append("=" * 70)
    tables_text.append(f"{'Subject':<12} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'AUC':>10} {'EER':>10} {'N':>6}")
    tables_text.append("-" * 80)
    for fold in loocv_data['per_fold']:
        tables_text.append(
            f"{fold['test_person']:<12} "
            f"{fold['accuracy']*100:>9.2f}% "
            f"{fold['f1']*100:>9.2f}% "
            f"{fold['precision']*100:>9.2f}% "
            f"{fold['recall']*100:>9.2f}% "
            f"{fold['roc_auc']*100:>9.2f}% "
            f"{fold['eer']*100:>9.2f}% "
            f"{fold['n_samples']:>5d}"
        )
    tables_text.append("-" * 80)
    tables_text.append(
        f"{'MEAN':<12} "
        f"{agg['accuracy']*100:>9.2f}% "
        f"{agg['f1']*100:>9.2f}% "
        f"{agg['precision']*100:>9.2f}% "
        f"{agg['recall']*100:>9.2f}% "
        f"{agg['roc_auc']*100:>9.2f}% "
        f"{agg['eer']*100:>9.2f}% "
        f"{'':>5s}"
    )
    tables_text.append(
        f"{'STD':<12} "
        f"{agg['accuracy_std']*100:>9.2f}% "
        f"{agg['f1_std']*100:>9.2f}% "
        f"{agg['precision_std']*100:>9.2f}% "
        f"{agg['recall_std']*100:>9.2f}% "
        f"{agg['roc_auc_std']*100:>9.2f}% "
        f"{agg['eer_std']*100:>9.2f}% "
        f"{'':>5s}"
    )
    
    # ---- TABLE 4: Ablation Study ----
    if ablation_data:
        tables_text.append("\n" + "=" * 70)
        tables_text.append("TABLE 4: Ablation Study Results")
        tables_text.append("=" * 70)
        tables_text.append(f"{'Variant':<25} {'Params':>8} {'Acc%':>8} {'F1%':>8} {'Prec%':>8} {'Rec%':>8} {'AUC%':>8}")
        tables_text.append("-" * 75)
        for name, data in ablation_data.items():
            tables_text.append(
                f"{name:<25} "
                f"{data['params']:>7d} "
                f"{data['accuracy']:>7.2f} "
                f"{data['f1']:>7.2f} "
                f"{data['precision']:>7.2f} "
                f"{data['recall']:>7.2f} "
                f"{data['auc']:>7.2f}"
            )
    
    # ---- TABLE 5: GradCAM Results ----
    if gradcam_data:
        tables_text.append("\n" + "=" * 70)
        tables_text.append("TABLE 5: GradCAM Feature Importance Analysis")
        tables_text.append("=" * 70)
        tables_text.append(f"{'Joint/Feature':<20} {'Importance':>12}")
        tables_text.append("-" * 35)
        joints_sorted = sorted(gradcam_data['joint_importance'].items(), key=lambda x: x[1], reverse=True)
        for name, val in joints_sorted:
            tables_text.append(f"{name:<20} {val:>11.4f}")
        
        tables_text.append(f"\n{'Feature Group':<20} {'Contribution':>12}")
        tables_text.append("-" * 35)
        for name, val in gradcam_data['group_importance'].items():
            tables_text.append(f"{name:<20} {val*100:>10.1f}%")
    
    # Write all tables
    tables_path = OUTPUT_DIR / 'all_tables.txt'
    with open(tables_path, 'w') as f:
        f.write('\n'.join(tables_text))
    print(f"  Saved: {tables_path}")
    
    return '\n'.join(tables_text)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("RESEARCH PAPER FIGURES & TABLES GENERATOR")
    print("=" * 60)
    
    # Load all data
    print("\n[1/4] Loading data...")
    loocv_data, labels, scores = load_loocv_results()
    ablation_data = load_ablation_results()
    gradcam_data = load_gradcam_results()
    
    print(f"  LOOCV: {len(labels)} pooled samples, {loocv_data['aggregate']['n_folds']} folds")
    if ablation_data:
        print(f"  Ablation: {len(ablation_data)} variants")
    if gradcam_data:
        print(f"  GradCAM: {gradcam_data['n_samples']} samples analyzed")
    
    # Use default threshold (0.5) for confusion matrix
    threshold = 0.5
    
    # Generate all figures
    print("\n[2/4] Generating figures...")
    
    print("\n  Figure 1: Confusion Matrix")
    cm_stats = plot_confusion_matrix(labels, scores, threshold)
    print(f"    TP={cm_stats['TP']}, TN={cm_stats['TN']}, FP={cm_stats['FP']}, FN={cm_stats['FN']}")
    
    print("\n  Figure 2: Four Cases (TP/FP/TN/FN)")
    cases = plot_four_cases(cm_stats, labels, scores, threshold)
    
    print("\n  Figure 3: ROC Curve")
    roc_auc, eer, optimal_thresh = plot_roc_curve(labels, scores)
    print(f"    AUC={roc_auc:.4f}, EER={eer:.4f}, Optimal Threshold={optimal_thresh:.4f}")
    
    print("\n  Figure 4: Precision-Recall Curve")
    pr_auc = plot_pr_curve(labels, scores)
    print(f"    PR-AUC={pr_auc:.4f}")
    
    print("\n  Figure 5: Similarity Score Distribution")
    plot_similarity_distribution(labels, scores, threshold)
    
    print("\n  Figure 6: LOOCV Per-Fold Results")
    plot_loocv_per_fold(loocv_data)
    
    print("\n  Figure 7: Ablation Study Comparison")
    plot_ablation_study(ablation_data)
    
    print("\n  Figure 8: GradCAM Feature Importance")
    plot_gradcam_importance(gradcam_data)
    
    print("\n  Figure 9: DET Curve")
    plot_det_curve(labels, scores)
    
    print("\n  Figure 10: EER Per Fold")
    plot_eer_per_fold(loocv_data)
    
    # Generate tables
    print("\n[3/4] Generating tables...")
    tables = generate_tables(loocv_data, cm_stats, ablation_data, gradcam_data,
                             roc_auc, eer, pr_auc, labels, scores, threshold)
    
    # Print summary
    print("\n[4/4] Summary")
    print("=" * 60)
    print(f"Total figures generated: 10")
    print(f"Tables generated: 5")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nAll files:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        size = f.stat().st_size / 1024
        print(f"  {f.name:<45} {size:>7.1f} KB")
    
    print("\n" + "=" * 60)
    print("DONE! Figures ready for research paper.")
    print("=" * 60)


if __name__ == "__main__":
    main()
