"""
Visualization Utilities for Gait-Based Deepfake Detection
==========================================================
Includes:
1. Training curves plotting
2. Confusion matrix
3. ROC/AUC curves
4. Embedding space and gait-attention plots

Author: DeepFake Detection Project
"""

from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


def plot_training_curves(history: List[Dict], save_path: Optional[str] = None):
    """
    Plot training and validation curves.

    Args:
        history: List of epoch metrics dictionaries
        save_path: Optional path to save figure
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, train_acc, "b-", label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, val_acc, "r-", label="Val Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Training and Validation Accuracy", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str] = ["Deepfake", "Authentic"],
    save_path: Optional[str] = None,
):
    """
    Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: Class names
        save_path: Optional path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={"size": 16},
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    plt.show()

    # Print metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    precision = tp / (tp + fp + 1e-8) * 100
    recall = tp / (tp + fn + 1e-8) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\nMetrics:")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall:    {recall:.2f}%")
    print(f"  F1-Score:  {f1:.2f}%")


def plot_roc_curve(
    y_true: np.ndarray, y_scores: np.ndarray, save_path: Optional[str] = None
):
    """
    Plot ROC curve with AUC.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Predicted probabilities for positive class
        save_path: Optional path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC curve to {save_path}")

    plt.show()

    print(f"\nAUC: {roc_auc:.4f}")

    # Find optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  TPR at threshold: {tpr[optimal_idx]:.4f}")
    print(f"  FPR at threshold: {fpr[optimal_idx]:.4f}")


def plot_precision_recall_curve(
    y_true: np.ndarray, y_scores: np.ndarray, save_path: Optional[str] = None
):
    """
    Plot Precision-Recall curve.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        save_path: Optional path to save figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        recall, precision, "b-", linewidth=2, label=f"PR Curve (AUC = {pr_auc:.4f})"
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved PR curve to {save_path}")

    plt.show()


def visualize_gait_attention(
    video_frames: np.ndarray,
    attention_weights: np.ndarray,
    pose_landmarks: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Visualize attention weights overlaid on video frames with pose.

    Args:
        video_frames: (T, H, W, 3) - video frames
        attention_weights: (T,) - importance per frame
        pose_landmarks: (T, 33, 3) - pose keypoints
        save_path: Optional path to save figure
    """
    num_frames = len(video_frames)

    # Select key frames to display
    num_display = min(8, num_frames)
    indices = np.linspace(0, num_frames - 1, num_display).astype(int)

    fig, axes = plt.subplots(2, num_display // 2, figsize=(16, 8))
    axes = axes.flatten()

    # Connections for drawing skeleton
    connections = [
        (11, 13),
        (13, 15),  # Left arm
        (12, 14),
        (14, 16),  # Right arm
        (11, 12),  # Shoulders
        (11, 23),
        (12, 24),  # Torso
        (23, 24),  # Hips
        (23, 25),
        (25, 27),
        (27, 31),  # Left leg
        (24, 26),
        (26, 28),
        (28, 32),  # Right leg
    ]

    for ax_idx, frame_idx in enumerate(indices):
        frame = video_frames[frame_idx].copy()
        attention = attention_weights[frame_idx]
        landmarks = pose_landmarks[frame_idx]

        h, w = frame.shape[:2]

        # Draw skeleton
        for start, end in connections:
            pt1 = (int(landmarks[start, 0] * w), int(landmarks[start, 1] * h))
            pt2 = (int(landmarks[end, 0] * w), int(landmarks[end, 1] * h))

            # Color based on attention (red = high, blue = low)
            color = (int(255 * attention), 0, int(255 * (1 - attention)))
            cv2.line(frame, pt1, pt2, color, 2)

        # Draw keypoints
        for i, (x, y, z) in enumerate(landmarks[:, :3]):
            pt = (int(x * w), int(y * h))
            cv2.circle(frame, pt, 3, (0, 255, 0), -1)

        axes[ax_idx].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[ax_idx].set_title(
            f"Frame {frame_idx}\nAttention: {attention:.3f}", fontsize=10
        )
        axes[ax_idx].axis("off")

    plt.suptitle("Temporal Attention Visualization", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention visualization to {save_path}")

    plt.show()


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    person_names: List[str],
    save_path: Optional[str] = None,
):
    """
    Visualize embedding space using t-SNE.

    Args:
        embeddings: (N, embedding_dim) - gait embeddings
        labels: (N,) - person indices
        person_names: List of person names
        save_path: Optional path to save figure
    """
    from sklearn.manifold import TSNE

    # Apply t-SNE
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1)
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each person with different color
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=person_names[label],
            s=50,
            alpha=0.7,
        )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title("Gait Embedding Space Visualization", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved embedding visualization to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test visualization functions with dummy data
    print("Testing visualization utilities...")

    # Test training curves
    history = [
        {
            "epoch": i + 1,
            "train_loss": 1.0 - i * 0.08,
            "val_loss": 1.1 - i * 0.07,
            "train_accuracy": 50 + i * 4,
            "val_accuracy": 48 + i * 3.5,
        }
        for i in range(10)
    ]

    print("\nPlotting training curves...")
    plot_training_curves(history)

    # Test confusion matrix
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])

    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred)

    # Test ROC curve
    y_scores = np.array([0.1, 0.4, 0.8, 0.9, 0.2, 0.3, 0.7, 0.15, 0.85, 0.95])

    print("\nPlotting ROC curve...")
    plot_roc_curve(y_true, y_scores)

    print("\n✓ Visualization tests complete!")
