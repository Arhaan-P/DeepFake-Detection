"""
Evaluation Script for Gait-Based Deepfake Detection
====================================================
Comprehensive evaluation with multiple metrics and visualizations.
Supports Leave-One-Out Cross-Validation (LOOCV) with subject-level splits.

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/checkpoint_epoch_best.pth
    python evaluate.py --checkpoint outputs/checkpoints/checkpoint_epoch_best.pth --save_plots
    python evaluate.py --loocv --features_file data/gait_features/gait_features.pkl

Author: DeepFake Detection Project
"""

import os
import argparse
import json
import pickle
import time
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    det_curve
)

# Setup logging
from utils.logger import setup_logging, close_logging

from models.full_pipeline import create_model
from utils.data_loader import create_data_loaders


def check_device():
    """Check and return available device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("DEVICE INFORMATION")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"{'='*60}\n")
    return device


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model config if available
    config_path = Path(checkpoint_path).parent.parent / 'model_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        model = create_model(config)
    else:
        model = create_model()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']+1}")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    
    return model, checkpoint


def evaluate_verification(model, dataloader, device):
    """Evaluate model on verification task."""
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_similarities = []
    all_confidences = []
    
    results_per_identity = defaultdict(lambda: {
        'correct': 0, 'total': 0, 'similarities': []
    })
    
    print("\nRunning evaluation on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            video_features = batch['video_features'].to(device)
            claimed_features = batch['claimed_features'].to(device)
            labels = batch['label'].to(device)
            true_identities = batch['true_identity']
            claimed_identities = batch['claimed_identity']
            
            outputs = model(video_features, claimed_features, mode='verification')
            
            predictions = outputs['is_authentic'].cpu().numpy()
            similarities = outputs['similarity'].cpu().numpy()
            confidences = outputs['confidence'].cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_labels.extend(labels_np)
            all_predictions.extend(predictions)
            all_similarities.extend(similarities)
            all_confidences.extend(confidences)
            
            # Track per-identity results
            for i, (true_id, claimed_id) in enumerate(zip(true_identities, claimed_identities)):
                is_correct = (predictions[i] > 0.5) == labels_np[i]
                results_per_identity[true_id]['correct'] += int(is_correct)
                results_per_identity[true_id]['total'] += 1
                results_per_identity[true_id]['similarities'].append(similarities[i])
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_similarities = np.array(all_similarities)
    all_confidences = np.array(all_confidences)
    
    # Binary predictions
    binary_predictions = (all_predictions > 0.5).astype(int)
    
    return {
        'labels': all_labels,
        'predictions': all_predictions,
        'binary_predictions': binary_predictions,
        'similarities': all_similarities,
        'confidences': all_confidences,
        'per_identity': dict(results_per_identity)
    }


def compute_metrics(results):
    """Compute comprehensive metrics."""
    labels = results['labels']
    predictions = results['predictions']
    binary_predictions = results['binary_predictions']
    similarities = results['similarities']
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, binary_predictions)
    metrics['precision'] = precision_score(labels, binary_predictions, zero_division=0)
    metrics['recall'] = recall_score(labels, binary_predictions, zero_division=0)
    metrics['f1'] = f1_score(labels, binary_predictions, zero_division=0)
    
    # AUC metrics
    if len(np.unique(labels)) > 1:
        metrics['roc_auc'] = roc_auc_score(labels, similarities)
        metrics['pr_auc'] = average_precision_score(labels, similarities)
    else:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(labels, binary_predictions)
    
    # Classification report
    metrics['classification_report'] = classification_report(
        labels, binary_predictions,
        target_names=['DEEPFAKE', 'AUTHENTIC'],
        output_dict=True
    )
    
    # ROC curve data
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    
    # Precision-Recall curve data
    precision, recall, pr_thresholds = precision_recall_curve(labels, similarities)
    metrics['pr_curve'] = {
        'precision': precision, 
        'recall': recall, 
        'thresholds': pr_thresholds
    }
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    metrics['optimal_threshold'] = thresholds[optimal_idx]
    
    # Metrics at optimal threshold
    optimal_predictions = (similarities > metrics['optimal_threshold']).astype(int)
    metrics['optimal_accuracy'] = accuracy_score(labels, optimal_predictions)
    metrics['optimal_f1'] = f1_score(labels, optimal_predictions, zero_division=0)
    
    # Per-identity accuracy
    per_identity_acc = {}
    for identity, data in results['per_identity'].items():
        if data['total'] > 0:
            per_identity_acc[identity] = data['correct'] / data['total']
    metrics['per_identity_accuracy'] = per_identity_acc
    
    # Equal Error Rate (EER)
    metrics['eer'], metrics['eer_threshold'] = compute_eer(labels, similarities)
    
    # DET curve data
    try:
        fpr_det, fnr_det, det_thresholds = det_curve(labels, similarities)
        metrics['det_curve'] = {
            'fpr': fpr_det, 'fnr': fnr_det, 'thresholds': det_thresholds
        }
    except Exception:
        metrics['det_curve'] = None
    
    return metrics


def compute_eer(labels: np.ndarray, scores: np.ndarray):
    """
    Compute Equal Error Rate (EER).
    
    EER is the point where FAR (False Accept Rate) == FRR (False Reject Rate).
    This is a standard biometric verification metric.
    
    Args:
        labels: Ground truth (0=deepfake, 1=authentic)
        scores: Similarity scores
        
    Returns:
        eer: Equal Error Rate (0-1)
        eer_threshold: Threshold at EER point
    """
    if len(np.unique(labels)) < 2:
        return 0.0, 0.5
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find intersection of FPR and FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx]) if eer_idx < len(thresholds) else 0.5
    
    return eer, eer_threshold


def run_loocv(
    features_file: str,
    enrolled_file: str,
    model_config: dict = None,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: torch.device = None,
    output_dir: str = 'outputs/loocv'
) -> dict:
    """
    Leave-One-Out Cross-Validation (LOOCV) with subject-level folds.
    
    For N subjects, trains N models — each holding out 1 subject for testing.
    Reports mean ± std of all metrics across folds.
    
    Args:
        features_file: Path to gait_features.pkl
        enrolled_file: Path to enrolled_identities.pkl
        model_config: Model config dict (None = defaults)
        epochs: Training epochs per fold
        batch_size: Batch size
        lr: Learning rate
        device: torch device
        output_dir: Directory to save LOOCV results
        
    Returns:
        Dictionary with per-fold metrics and aggregated results
    """
    from models.full_pipeline import create_model
    from utils.data_loader import GaitDataset
    from torch.utils.data import DataLoader
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load features to get subject list
    with open(features_file, 'rb') as f:
        all_features = pickle.load(f)
    
    persons = sorted(set(f['identity'] for f in all_features.values()))
    n_folds = len(persons)
    
    print(f"\n{'='*60}")
    print(f"LEAVE-ONE-OUT CROSS-VALIDATION ({n_folds} folds)")
    print(f"{'='*60}")
    print(f"Subjects: {persons}")
    print(f"Epochs per fold: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    fold_metrics = []
    all_fold_labels = []
    all_fold_scores = []
    
    total_start = time.time()
    
    for fold_idx, test_person in enumerate(persons):
        fold_start = time.time()
        train_persons = [p for p in persons if p != test_person]
        
        print(f"\n{'─'*50}")
        print(f"Fold {fold_idx + 1}/{n_folds} — Test: {test_person}, Train: {train_persons}")
        print(f"{'─'*50}")
        
        # Create datasets
        train_dataset = GaitDataset(
            features_file=features_file,
            enrolled_identities_file=enrolled_file,
            person_list=train_persons,
            mode='verification'
        )
        test_dataset = GaitDataset(
            features_file=features_file,
            enrolled_identities_file=enrolled_file,
            person_list=[test_person],
            mode='verification'
        )
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"  Skipping fold — insufficient data")
            continue
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                   shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0, pin_memory=True)
        
        # Determine input_dim from data
        sample = next(iter(train_loader))
        input_dim = sample['video_features'].shape[-1]
        
        # Create fresh model for this fold
        fold_config = model_config or {}
        fold_config['input_dim'] = input_dim
        model = create_model(fold_config).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
        )
        
        # Train this fold
        model.train()
        best_loss = float('inf')
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                video_features = batch['video_features'].to(device)
                claimed_features = batch['claimed_features'].to(device)
                labels = batch['label'].to(device).squeeze(-1).long()
                
                optimizer.zero_grad()
                outputs = model(video_features, claimed_features, mode='verification')
                logits = outputs['verification']['logits']
                loss = criterion(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = deepcopy(model.state_dict())
        
        # Load best model and evaluate
        model.load_state_dict(best_state)
        results = evaluate_verification(model, test_loader, device)
        
        # Store raw predictions for aggregate analysis
        all_fold_labels.extend(results['labels'].tolist())
        all_fold_scores.extend(results['similarities'].tolist())
        
        # Compute fold metrics
        fm = {}
        fm['test_person'] = test_person
        fm['accuracy'] = accuracy_score(results['labels'], results['binary_predictions'])
        fm['f1'] = f1_score(results['labels'], results['binary_predictions'], zero_division=0)
        fm['precision'] = precision_score(results['labels'], results['binary_predictions'], zero_division=0)
        fm['recall'] = recall_score(results['labels'], results['binary_predictions'], zero_division=0)
        
        if len(np.unique(results['labels'])) > 1:
            fm['roc_auc'] = roc_auc_score(results['labels'], results['similarities'])
            fm['eer'], _ = compute_eer(results['labels'], results['similarities'])
        else:
            fm['roc_auc'] = 0.0
            fm['eer'] = 0.0
        
        fm['n_samples'] = len(results['labels'])
        fold_metrics.append(fm)
        
        fold_time = time.time() - fold_start
        print(f"  Acc={fm['accuracy']:.4f}  F1={fm['f1']:.4f}  "
              f"AUC={fm['roc_auc']:.4f}  EER={fm['eer']:.4f}  "
              f"({fm['n_samples']} samples, {fold_time:.1f}s)")
        
        # Clean up GPU memory
        del model, optimizer
        torch.cuda.empty_cache()
    
    total_time = time.time() - total_start
    
    # Aggregate results
    all_fold_labels = np.array(all_fold_labels)
    all_fold_scores = np.array(all_fold_scores)
    
    # Overall metrics from pooled predictions
    aggregate = {}
    aggregate['accuracy'] = float(np.mean([f['accuracy'] for f in fold_metrics]))
    aggregate['accuracy_std'] = float(np.std([f['accuracy'] for f in fold_metrics]))
    aggregate['f1'] = float(np.mean([f['f1'] for f in fold_metrics]))
    aggregate['f1_std'] = float(np.std([f['f1'] for f in fold_metrics]))
    aggregate['precision'] = float(np.mean([f['precision'] for f in fold_metrics]))
    aggregate['precision_std'] = float(np.std([f['precision'] for f in fold_metrics]))
    aggregate['recall'] = float(np.mean([f['recall'] for f in fold_metrics]))
    aggregate['recall_std'] = float(np.std([f['recall'] for f in fold_metrics]))
    aggregate['roc_auc'] = float(np.mean([f['roc_auc'] for f in fold_metrics]))
    aggregate['roc_auc_std'] = float(np.std([f['roc_auc'] for f in fold_metrics]))
    aggregate['eer'] = float(np.mean([f['eer'] for f in fold_metrics]))
    aggregate['eer_std'] = float(np.std([f['eer'] for f in fold_metrics]))
    
    # Pooled EER and AUC from all fold predictions combined
    if len(np.unique(all_fold_labels)) > 1:
        aggregate['pooled_roc_auc'] = float(roc_auc_score(all_fold_labels, all_fold_scores))
        aggregate['pooled_eer'], _ = compute_eer(all_fold_labels, all_fold_scores)
    else:
        aggregate['pooled_roc_auc'] = 0.0
        aggregate['pooled_eer'] = 0.0
    
    aggregate['n_folds'] = len(fold_metrics)
    aggregate['total_time'] = total_time
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"LOOCV RESULTS ({len(fold_metrics)} folds, {total_time:.1f}s total)")
    print(f"{'='*60}")
    print(f"Accuracy:   {aggregate['accuracy']:.4f} ± {aggregate['accuracy_std']:.4f}")
    print(f"F1 Score:   {aggregate['f1']:.4f} ± {aggregate['f1_std']:.4f}")
    print(f"Precision:  {aggregate['precision']:.4f} ± {aggregate['precision_std']:.4f}")
    print(f"Recall:     {aggregate['recall']:.4f} ± {aggregate['recall_std']:.4f}")
    print(f"ROC-AUC:    {aggregate['roc_auc']:.4f} ± {aggregate['roc_auc_std']:.4f}")
    print(f"EER:        {aggregate['eer']:.4f} ± {aggregate['eer_std']:.4f}")
    print(f"\nPooled AUC: {aggregate['pooled_roc_auc']:.4f}")
    print(f"Pooled EER: {aggregate['pooled_eer']:.4f}")
    print(f"{'='*60}")
    
    # Per-fold table
    print("\nPer-Fold Breakdown:")
    print(f"{'Subject':<12} {'Acc':>7} {'F1':>7} {'AUC':>7} {'EER':>7} {'N':>5}")
    print("─" * 50)
    for fm in fold_metrics:
        print(f"{fm['test_person']:<12} {fm['accuracy']:>7.4f} {fm['f1']:>7.4f} "
              f"{fm['roc_auc']:>7.4f} {fm['eer']:>7.4f} {fm['n_samples']:>5d}")
    
    # Save results
    loocv_results = {
        'aggregate': aggregate,
        'per_fold': fold_metrics,
        'pooled_labels': all_fold_labels.tolist(),
        'pooled_scores': all_fold_scores.tolist()
    }
    
    results_path = os.path.join(output_dir, 'loocv_results.json')
    with open(results_path, 'w') as f:
        json.dump(loocv_results, f, indent=2)
    print(f"\nSaved LOOCV results to: {results_path}")
    
    return loocv_results


def print_results(metrics):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n--- Overall Metrics ---")
    print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1 Score:     {metrics['f1']:.4f}")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"EER:          {metrics.get('eer', 0):.4f}")
    
    print("\n--- Optimal Threshold ---")
    print(f"Threshold:    {metrics['optimal_threshold']:.4f}")
    print(f"Accuracy:     {metrics['optimal_accuracy']:.4f}")
    print(f"F1 Score:     {metrics['optimal_f1']:.4f}")
    
    print("\n--- Confusion Matrix ---")
    cm = metrics['confusion_matrix']
    print(f"                  Predicted")
    print(f"                  DEEPFAKE  AUTHENTIC")
    print(f"Actual DEEPFAKE      {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"       AUTHENTIC     {cm[1,0]:4d}     {cm[1,1]:4d}")
    
    print("\n--- Per-Class Metrics ---")
    report = metrics['classification_report']
    for class_name in ['DEEPFAKE', 'AUTHENTIC']:
        print(f"\n{class_name}:")
        print(f"  Precision: {report[class_name]['precision']:.4f}")
        print(f"  Recall:    {report[class_name]['recall']:.4f}")
        print(f"  F1-Score:  {report[class_name]['f1-score']:.4f}")
        print(f"  Support:   {int(report[class_name]['support'])}")
    
    print("\n--- Per-Identity Accuracy ---")
    for identity, acc in sorted(metrics['per_identity_accuracy'].items()):
        print(f"  {identity}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n" + "=" * 60)


def save_plots(metrics, results, output_dir):
    """Save evaluation plots."""
    from utils.visualization import (
        plot_confusion_matrix, plot_roc_curve, 
        plot_precision_recall_curve, plot_embedding_space
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        results['labels'],
        results['binary_predictions'],
        classes=['DEEPFAKE', 'AUTHENTIC'],
        save_path=cm_path
    )
    print(f"Saved: {cm_path}")
    
    # ROC Curve
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(
        results['labels'],
        results['similarities'],
        save_path=roc_path
    )
    print(f"Saved: {roc_path}")
    
    # Precision-Recall Curve
    pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plot_precision_recall_curve(
        results['labels'],
        results['similarities'],
        save_path=pr_path
    )
    print(f"Saved: {pr_path}")
    
    # Similarity Distribution
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    authentic_sims = results['similarities'][results['labels'] == 1]
    deepfake_sims = results['similarities'][results['labels'] == 0]
    
    ax.hist(authentic_sims, bins=50, alpha=0.7, label='Authentic', color='green')
    ax.hist(deepfake_sims, bins=50, alpha=0.7, label='Deepfake', color='red')
    
    ax.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold')
    ax.axvline(x=metrics['optimal_threshold'], color='blue', linestyle='--', 
               label=f"Optimal Threshold ({metrics['optimal_threshold']:.3f})")
    
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Similarity Scores')
    ax.legend()
    
    dist_path = os.path.join(output_dir, 'similarity_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dist_path}")
    
    # Per-Identity Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    identities = list(metrics['per_identity_accuracy'].keys())
    accuracies = [metrics['per_identity_accuracy'][id] for id in identities]
    
    bars = ax.bar(range(len(identities)), accuracies, color='steelblue')
    
    # Color bars by accuracy
    for bar, acc in zip(bars, accuracies):
        if acc >= 0.9:
            bar.set_color('green')
        elif acc >= 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.set_xticks(range(len(identities)))
    ax.set_xticklabels(identities, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Identity Classification Accuracy')
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    ax.legend()
    
    id_path = os.path.join(output_dir, 'per_identity_accuracy.png')
    plt.savefig(id_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {id_path}")
    
    print(f"\nAll plots saved to: {output_dir}")


def save_results(metrics, results, output_dir):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics summary
    summary = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'roc_auc': float(metrics['roc_auc']),
        'pr_auc': float(metrics['pr_auc']),
        'eer': float(metrics.get('eer', 0)),
        'eer_threshold': float(metrics.get('eer_threshold', 0.5)),
        'optimal_threshold': float(metrics['optimal_threshold']),
        'optimal_accuracy': float(metrics['optimal_accuracy']),
        'optimal_f1': float(metrics['optimal_f1']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'per_identity_accuracy': {k: float(v) for k, v in metrics['per_identity_accuracy'].items()}
    }
    
    summary_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to: {summary_path}")
    
    # Save detailed predictions
    predictions_path = os.path.join(output_dir, 'predictions.pkl')
    with open(predictions_path, 'wb') as f:
        pickle.dump({
            'labels': results['labels'],
            'predictions': results['predictions'],
            'similarities': results['similarities'],
            'confidences': results['confidences']
        }, f)
    print(f"Saved predictions to: {predictions_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Gait-Based Deepfake Detection Model'
    )
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (not needed for --loocv)')
    parser.add_argument('--features_file', type=str,
                        default='data/gait_features/gait_features.pkl',
                        help='Path to features file')
    parser.add_argument('--enrolled_file', type=str,
                        default='data/gait_features/enrolled_identities.pkl',
                        help='Path to enrolled identities file')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--test_only', action='store_true',
                        help='Evaluate only on test set (default)')
    parser.add_argument('--eval_all', action='store_true',
                        help='Evaluate on all splits')
    parser.add_argument('--loocv', action='store_true',
                        help='Run Leave-One-Out Cross-Validation (train N models)')
    parser.add_argument('--loocv_epochs', type=int, default=30,
                        help='Epochs per LOOCV fold (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for LOOCV training')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                        help='Directory to save results')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save evaluation plots')
    parser.add_argument('--save_results', action='store_true',
                        help='Save evaluation results to files')
    
    args = parser.parse_args()
    
    # LOOCV mode — separate workflow
    if args.loocv:
        if not os.path.exists(args.features_file):
            raise FileNotFoundError(f"Features file not found: {args.features_file}")
        
        device = check_device()
        
        # Load model config if available
        config_path = Path('outputs/model_config.json')
        model_config = None
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        
        loocv_dir = os.path.join(args.output_dir, 'loocv')
        run_loocv(
            features_file=args.features_file,
            enrolled_file=args.enrolled_file,
            model_config=model_config,
            epochs=args.loocv_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            output_dir=loocv_dir
        )
        return
    
    # Standard evaluation — requires checkpoint
    if args.checkpoint is None:
        parser.error("--checkpoint is required (unless using --loocv)")
    
    # Check inputs
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.features_file):
        raise FileNotFoundError(f"Features file not found: {args.features_file}")
    
    # Setup
    device = check_device()
    
    # Load model
    model, checkpoint = load_model(args.checkpoint, device)
    
    # Create data loaders (split by person — no data leakage)
    print("\nLoading data...")
    train_loader, val_loader, train_persons, val_persons = create_data_loaders(
        args.features_file,
        args.enrolled_file,
        mode='verification',
        batch_size=args.batch_size,
        train_ratio=0.8
    )
    
    # Use validation set as test set
    test_loader = val_loader
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Test persons: {val_persons}")
    
    # Evaluate
    results = evaluate_verification(model, test_loader, device)
    metrics = compute_metrics(results)
    
    # Print results
    print_results(metrics)
    
    # Evaluate on other splits if requested
    if args.eval_all:
        print("\n--- Validation Set ---")
        val_results = evaluate_verification(model, val_loader, device)
        val_metrics = compute_metrics(val_results)
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation ROC-AUC:  {val_metrics['roc_auc']:.4f}")
        
        print("\n--- Training Set ---")
        train_results = evaluate_verification(model, train_loader, device)
        train_metrics = compute_metrics(train_results)
        print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Training ROC-AUC:  {train_metrics['roc_auc']:.4f}")
    
    # Save plots
    if args.save_plots:
        print("\nGenerating plots...")
        save_plots(metrics, results, args.output_dir)
    
    # Save results
    if args.save_results:
        print("\nSaving results...")
        save_results(metrics, results, args.output_dir)
    
    return metrics


if __name__ == "__main__":
    logger = setup_logging('evaluate')
    try:
        main()
    finally:
        close_logging(logger)
