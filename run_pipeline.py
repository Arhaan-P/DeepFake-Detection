"""
Main Pipeline Script - Gait-Based Deepfake Detection
=====================================================
Orchestrates the entire pipeline from raw videos to trained model.

Usage:
    python run_pipeline.py --mode full          # Run entire pipeline
    python run_pipeline.py --mode augment       # Only data augmentation
    python run_pipeline.py --mode preprocess    # Only feature extraction
    python run_pipeline.py --mode train         # Only training
    python run_pipeline.py --mode evaluate      # Only evaluation
    python run_pipeline.py --mode demo          # Run demo inference

Author: DeepFake Detection Project
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time

import torch


def check_device():
    """Check and display device information."""
    print("\n" + "=" * 60)
    print("SYSTEM CHECK")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print("✓ GPU is available - models will run on CUDA")
    else:
        print("✗ GPU not available - models will run on CPU (slower)")
    
    print("=" * 60)
    return device


def run_command(command: list, description: str):
    """Run a command and display output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}\n")
    
    start_time = time.time()
    result = subprocess.run(command, capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ Completed in {elapsed/60:.1f} minutes")
    else:
        print(f"\n✗ Failed with return code {result.returncode}")
    
    return result.returncode == 0


def stage_augment(videos_dir: str, output_dir: str):
    """Stage 1: Data Augmentation."""
    return run_command(
        [sys.executable, 'augment_videos.py',
         '--input_dir', videos_dir,
         '--output_dir', output_dir],
        "Stage 1: Data Augmentation"
    )


def stage_preprocess(videos_dir: str, augmented_dir: str, output_file: str):
    """Stage 2: Feature Extraction."""
    cmd = [sys.executable, 'preprocess_videos.py',
           '--videos_dir', videos_dir,
           '--output', output_file]
    
    if augmented_dir and os.path.exists(augmented_dir):
        cmd.extend(['--augmented_dir', augmented_dir])
    
    return run_command(cmd, "Stage 2: Feature Extraction")


def stage_train(features_file: str, enrolled_file: str, 
                output_dir: str, epochs: int, batch_size: int):
    """Stage 3: Model Training."""
    return run_command(
        [sys.executable, 'train.py',
         '--features_file', features_file,
         '--enrolled_file', enrolled_file,
         '--output_dir', output_dir,
         '--epochs', str(epochs),
         '--batch_size', str(batch_size)],
        "Stage 3: Model Training"
    )


def stage_evaluate(checkpoint: str, features_file: str, 
                   enrolled_file: str, output_dir: str):
    """Stage 4: Model Evaluation."""
    return run_command(
        [sys.executable, 'evaluate.py',
         '--checkpoint', checkpoint,
         '--features_file', features_file,
         '--enrolled_file', enrolled_file,
         '--output_dir', output_dir,
         '--save_plots',
         '--save_results'],
        "Stage 4: Model Evaluation"
    )


def stage_demo(checkpoint: str, enrolled_file: str, 
               video_path: str, claimed_identity: str):
    """Stage 5: Demo Inference."""
    return run_command(
        [sys.executable, 'inference.py',
         '--video', video_path,
         '--claimed_identity', claimed_identity,
         '--checkpoint', checkpoint,
         '--enrolled_file', enrolled_file,
         '--visualize'],
        "Stage 5: Demo Inference"
    )


def full_pipeline(args):
    """Run the complete pipeline."""
    print("\n" + "=" * 60)
    print("GAIT-BASED DEEPFAKE DETECTION PIPELINE")
    print("=" * 60)
    
    stages_status = {}
    
    # Stage 1: Data Augmentation
    if not args.skip_augment:
        success = stage_augment(args.videos_dir, args.augmented_dir)
        stages_status['augmentation'] = success
        if not success and not args.continue_on_error:
            return stages_status
    else:
        print("\nSkipping augmentation (--skip_augment)")
    
    # Stage 2: Feature Extraction
    if not args.skip_preprocess:
        success = stage_preprocess(
            args.videos_dir, 
            args.augmented_dir,
            args.features_file
        )
        stages_status['preprocessing'] = success
        if not success and not args.continue_on_error:
            return stages_status
    else:
        print("\nSkipping preprocessing (--skip_preprocess)")
    
    # Stage 3: Training
    if not args.skip_train:
        success = stage_train(
            args.features_file,
            args.enrolled_file,
            args.output_dir,
            args.epochs,
            args.batch_size
        )
        stages_status['training'] = success
        if not success and not args.continue_on_error:
            return stages_status
    else:
        print("\nSkipping training (--skip_train)")
    
    # Stage 4: Evaluation
    if not args.skip_evaluate:
        checkpoint = os.path.join(args.output_dir, 'checkpoints', 'checkpoint_epoch_best.pth')
        success = stage_evaluate(
            checkpoint,
            args.features_file,
            args.enrolled_file,
            os.path.join(args.output_dir, 'evaluation')
        )
        stages_status['evaluation'] = success
    else:
        print("\nSkipping evaluation (--skip_evaluate)")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    for stage, success in stages_status.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {stage}: {status}")
    
    return stages_status


def main():
    parser = argparse.ArgumentParser(
        description='Gait-Based Deepfake Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python run_pipeline.py --mode full
    
    # Run only augmentation
    python run_pipeline.py --mode augment
    
    # Run training with custom parameters
    python run_pipeline.py --mode train --epochs 100 --batch_size 16
    
    # Run demo inference
    python run_pipeline.py --mode demo --video path/to/video.mp4 --claimed_identity "Arhaan"
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'augment', 'preprocess', 'train', 
                                 'evaluate', 'demo'],
                        help='Pipeline mode to run')
    
    # Directory arguments
    parser.add_argument('--videos_dir', type=str, default='data/videos',
                        help='Directory containing original videos')
    parser.add_argument('--augmented_dir', type=str, default='data/augmented_videos',
                        help='Directory for augmented videos')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for model and results')
    
    # Feature arguments
    parser.add_argument('--features_file', type=str, 
                        default='data/gait_features/gait_features.pkl',
                        help='Path to features file')
    parser.add_argument('--enrolled_file', type=str,
                        default='data/gait_features/enrolled_identities.pkl',
                        help='Path to enrolled identities file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    # Demo arguments
    parser.add_argument('--video', type=str, default=None,
                        help='Video path for demo mode')
    parser.add_argument('--claimed_identity', type=str, default=None,
                        help='Claimed identity for demo mode')
    
    # Skip stages
    parser.add_argument('--skip_augment', action='store_true',
                        help='Skip augmentation stage')
    parser.add_argument('--skip_preprocess', action='store_true',
                        help='Skip preprocessing stage')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training stage')
    parser.add_argument('--skip_evaluate', action='store_true',
                        help='Skip evaluation stage')
    parser.add_argument('--continue_on_error', action='store_true',
                        help='Continue pipeline even if a stage fails')
    
    args = parser.parse_args()
    
    # Check device
    check_device()
    
    # Run appropriate mode
    if args.mode == 'full':
        full_pipeline(args)
    
    elif args.mode == 'augment':
        stage_augment(args.videos_dir, args.augmented_dir)
    
    elif args.mode == 'preprocess':
        stage_preprocess(args.videos_dir, args.augmented_dir, args.features_file)
    
    elif args.mode == 'train':
        if not os.path.exists(args.features_file):
            print(f"Error: Features file not found: {args.features_file}")
            print("Please run preprocessing first: python run_pipeline.py --mode preprocess")
            return
        stage_train(
            args.features_file,
            args.enrolled_file,
            args.output_dir,
            args.epochs,
            args.batch_size
        )
    
    elif args.mode == 'evaluate':
        checkpoint = os.path.join(args.output_dir, 'checkpoints', 'checkpoint_epoch_best.pth')
        if not os.path.exists(checkpoint):
            print(f"Error: Checkpoint not found: {checkpoint}")
            print("Please run training first: python run_pipeline.py --mode train")
            return
        stage_evaluate(
            checkpoint,
            args.features_file,
            args.enrolled_file,
            os.path.join(args.output_dir, 'evaluation')
        )
    
    elif args.mode == 'demo':
        if not args.video or not args.claimed_identity:
            print("Error: Demo mode requires --video and --claimed_identity")
            return
        checkpoint = os.path.join(args.output_dir, 'checkpoints', 'checkpoint_epoch_best.pth')
        stage_demo(checkpoint, args.enrolled_file, args.video, args.claimed_identity)


if __name__ == "__main__":
    main()
