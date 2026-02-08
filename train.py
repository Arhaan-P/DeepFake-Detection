"""
Training Script for Gait-Based Deepfake Detection
===================================================
Complete training pipeline with:
1. GPU/CUDA support (required)
2. Detailed progress monitoring
3. Multi-loss training (verification + triplet)
4. Checkpoint saving
5. TensorBoard logging
6. Early stopping

Author: DeepFake Detection Project
"""

import os
import time
import argparse
from datetime import datetime
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Setup logging
from utils.logger import setup_logging, close_logging

from models.full_pipeline import GaitDeepfakeDetector, create_model
from utils.data_loader import GaitDataset, create_data_loaders


def check_device():
    """Check and display device information. ALWAYS use GPU if available."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: Running on CPU! Training will be slow.")
        print("Please ensure CUDA is properly installed for GPU acceleration.")
    
    print("=" * 60)
    return device


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute classification metrics."""
    preds = predictions.cpu().numpy()
    labs = labels.cpu().numpy()
    
    # Accuracy
    accuracy = (preds == labs).mean() * 100
    
    # Precision, Recall, F1 for each class
    metrics = {'accuracy': accuracy}
    
    for cls in [0, 1]:
        tp = ((preds == cls) & (labs == cls)).sum()
        fp = ((preds == cls) & (labs != cls)).sum()
        fn = ((preds != cls) & (labs == cls)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        cls_name = 'authentic' if cls == 1 else 'deepfake'
        metrics[f'{cls_name}_precision'] = precision * 100
        metrics[f'{cls_name}_recall'] = recall * 100
        metrics[f'{cls_name}_f1'] = f1 * 100
    
    return metrics


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, 
                total_epochs, use_triplet=False):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    all_predictions = []
    all_labels = []
    
    # Print header once
    print(f"Training: ", end='', flush=True)
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        video_features = batch['video_features'].to(device)
        claimed_features = batch['claimed_features'].to(device)
        labels = batch['label'].to(device).squeeze()
        
        # Forward pass
        optimizer.zero_grad()
        
        output = model(video_features, claimed_features, mode='verification')
        logits = output['verification']['logits']
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute accuracy
        predictions = output['is_authentic']
        accuracy = (predictions == labels).float().mean() * 100
        
        # Update meters
        loss_meter.update(loss.item(), video_features.size(0))
        acc_meter.update(accuracy.item(), video_features.size(0))
        
        # Store for epoch metrics
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            progress = (batch_idx + 1) / len(train_loader) * 100
            print(f"\r{progress:5.1f}% | Loss={loss_meter.avg:.4f} Acc={acc_meter.avg:.2f}%", 
                  end='', flush=True)
    
    print()  # New line after completion
    
    # Compute epoch metrics
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    epoch_metrics = compute_metrics(all_predictions, all_labels)
    epoch_metrics['loss'] = loss_meter.avg
    
    # Diagnostic: embedding stats and gradient norms (last batch)
    with torch.no_grad():
        vid_emb = output['video_embedding']
        sim = output['similarity']
        print(f"  [Diag] VidEmb mean={vid_emb.mean():.4f} std={vid_emb.std():.4f}")
        print(f"  [Diag] P(auth) mean={sim.mean():.4f} std={sim.std():.4f} | "
              f"Predictions: auth={all_predictions.sum().int()}/{len(all_predictions)} "
              f"fake={(1-all_predictions).sum().int()}/{len(all_predictions)}")
    
    # Gradient norms
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"  [Diag] GradNorm={total_norm:.4f}")
    
    return epoch_metrics


def validate(model, val_loader, criterion, device, epoch, total_epochs):
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter()
    all_predictions = []
    all_labels = []
    all_similarities = []
    
    print(f"Validation: ", end='', flush=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            video_features = batch['video_features'].to(device)
            claimed_features = batch['claimed_features'].to(device)
            labels = batch['label'].to(device).squeeze()
            
            output = model(video_features, claimed_features, mode='verification')
            logits = output['verification']['logits']
            
            loss = criterion(logits, labels)
            loss_meter.update(loss.item(), video_features.size(0))
            
            predictions = output['is_authentic']
            similarities = output['similarity']
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_similarities.append(similarities.cpu())
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                progress = (batch_idx + 1) / len(val_loader) * 100
                acc = (torch.cat(all_predictions) == torch.cat(all_labels)).float().mean() * 100
                print(f"\r{progress:5.1f}% | Loss={loss_meter.avg:.4f} Acc={acc:.2f}%", 
                      end='', flush=True)
    
    print()  # New line after completion
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_similarities = torch.cat(all_similarities)
    
    metrics = compute_metrics(all_predictions, all_labels)
    metrics['loss'] = loss_meter.avg
    metrics['avg_similarity'] = all_similarities.mean().item()
    
    # Diagnostic: per-class similarity and prediction distribution
    auth_mask = all_labels == 1
    fake_mask = all_labels == 0
    if auth_mask.any() and fake_mask.any():
        print(f"  [Diag] Sim(auth)={all_similarities[auth_mask].mean():.4f} "
              f"Sim(fake)={all_similarities[fake_mask].mean():.4f} | "
              f"Pred: auth={all_predictions.sum().int()}/{len(all_predictions)} "
              f"fake={(1-all_predictions).sum().int()}/{len(all_predictions)}")
    
    return metrics



def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path, is_best=False, feature_stats=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    # Save feature normalization stats (critical for inference)
    if feature_stats is not None:
        checkpoint['feature_stats'] = {
            'mean': feature_stats['mean'].cpu(),
            'std': feature_stats['std'].cpu()
        }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"  ★ New best model saved! Val Acc: {metrics['val_accuracy']:.2f}%")


def train(args):
    """Main training function."""

    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore', message='.*pin_memory.*')
    warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')
    
    # ========================================
    # 1. Setup
    # ========================================
    print("\n" + "=" * 60)
    print("GAIT-BASED DEEPFAKE DETECTION TRAINING")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check device
    device = check_device()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    
    # ========================================
    # 2. Data Loading
    # ========================================
    print("\n" + "-" * 60)
    print("Loading Data...")
    print("-" * 60)
    
    train_loader, val_loader, train_persons, val_persons = create_data_loaders(
        features_file=args.features_file,
        enrolled_identities_file=args.enrolled_file,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        mode='verification',
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Get feature normalization stats from training dataset (saved in checkpoint for inference)
    feature_stats = train_loader.dataset.get_feature_stats()
    
    print(f"\nDataset Statistics:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Train persons: {train_persons}")
    print(f"  Val persons: {val_persons}")
    
    # Determine input dimension from data
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['video_features'].shape[-1]
    print(f"  Input dimension: {input_dim}")
    
    # ========================================
    # 3. Model Creation
    # ========================================
    print("\n" + "-" * 60)
    print("Creating Model...")
    print("-" * 60)
    
    model_config = {
        'input_dim': input_dim,
        'encoder_hidden_dims': (64, 128),
        'encoder_output_dim': 128,
        'lstm_hidden': 64,
        'lstm_layers': 1,
        'transformer_d_model': 128,
        'transformer_heads': 4,
        'transformer_layers': 2,
        'embedding_dim': 128,
        'verification_hidden': 64,
        'dropout': args.dropout,
        'use_multi_scale_encoder': args.multi_scale
    }
    
    model = create_model(model_config)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1e6:.2f} MB (float32)")
    
    # Save model config
    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # ========================================
    # 4. Loss, Optimizer, Scheduler
    # ========================================
    print("\n" + "-" * 60)
    print("Setting up Training...")
    print("-" * 60)
    
    # Class-weighted loss to handle any residual imbalance
    # Class 0 = deepfake, Class 1 = authentic
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=7,
        min_lr=args.lr * 0.01,
        verbose=True
    )
    
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=7)")
    print(f"  Early stopping patience: {args.patience}")
    
    # ========================================
    # 5. Training Loop
    # ========================================
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    best_val_acc = 0.0
    training_history = []
    
    total_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"EPOCH [{epoch+1}/{args.epochs}]")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, 
            device, epoch, args.epochs
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion,
            device, epoch, args.epochs
        )
        
        # Update scheduler (ReduceLROnPlateau uses val metric)
        scheduler.step(val_metrics['accuracy'])
        
        # Epoch time
        epoch_time = time.time() - epoch_start_time
        eta = epoch_time * (args.epochs - epoch - 1) / 60
        
        # Print epoch summary
        print(f"\n{'-'*60}")
        print(f"EPOCH [{epoch+1}/{args.epochs}] SUMMARY")
        print(f"{'-'*60}")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
        print(f"  Val Authentic F1: {val_metrics['authentic_f1']:.2f}% | "
              f"Val Deepfake F1: {val_metrics['deepfake_f1']:.2f}%")
        print(f"  Epoch Time: {epoch_time:.2f}s | ETA: {eta:.1f} min")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('F1/authentic', val_metrics['authentic_f1'], epoch)
        writer.add_scalar('F1/deepfake', val_metrics['deepfake_f1'], epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save history
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_authentic_f1': val_metrics['authentic_f1'],
            'val_deepfake_f1': val_metrics['deepfake_f1'],
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        training_history.append(history_entry)
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
        
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'train_accuracy': train_metrics['accuracy'], 'val_accuracy': val_metrics['accuracy']},
            str(output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth'),
            is_best=is_best,
            feature_stats=feature_stats
        )
        
        # Early stopping
        if early_stopping(val_metrics['accuracy']):
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING TRIGGERED at epoch {epoch+1}")
            print(f"{'='*60}")
            break
    
    # ========================================
    # 6. Training Complete
    # ========================================
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_dir / 'checkpoints'}")
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    writer.close()
    
    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Train Gait-Based Deepfake Detector')
    
    # Data
    parser.add_argument('--features_file', type=str, 
                        default='data/gait_features/gait_features.pkl',
                        help='Path to extracted gait features')
    parser.add_argument('--enrolled_file', type=str,
                        default='data/gait_features/enrolled_identities.pkl',
                        help='Path to enrolled identities')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Model
    parser.add_argument('--multi_scale', action='store_true', 
                        help='Use multi-scale encoder')
    
    # Data loading
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--num_workers', type=int, default=0, help='Data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Train
    train(args)


if __name__ == "__main__":
    logger = setup_logging('train')
    try:
        main()
    finally:
        close_logging(logger)
