"""
Inference Script for Gait-Based Deepfake Detection
===================================================
Predict whether a video is AUTHENTIC or DEEPFAKE
based on the claimed identity.

Usage:
    python inference.py --video path/to/video.mp4 --claimed_identity "PersonName"
    python inference.py --video path/to/video.mp4 --claimed_identity "Arhaan" --visualize

Author: DeepFake Detection Project
"""

import os
import argparse
from pathlib import Path
import json
import pickle

import torch
import numpy as np
import cv2
from tqdm import tqdm

from models.full_pipeline import GaitDeepfakeDetector, create_model
from utils.pose_extraction import GaitFeatureExtractor


def find_latest_best_checkpoint(checkpoint_dir: str = 'outputs/checkpoints') -> str:
    """Find the latest best checkpoint by epoch number (numeric sort)."""
    import re
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    best_checkpoints = list(checkpoint_path.glob('checkpoint_epoch_*_best.pth'))
    if not best_checkpoints:
        raise FileNotFoundError(f"No best checkpoints found in {checkpoint_dir}")
    
    # Sort numerically by epoch number, not lexicographically
    def epoch_num(p):
        m = re.search(r'epoch_(\d+)_best', p.name)
        return int(m.group(1)) if m else 0
    
    best_checkpoints.sort(key=epoch_num)
    latest_best = str(best_checkpoints[-1])
    return latest_best


def check_device():
    """Check and return available device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model and feature normalization stats from checkpoint.
    
    Returns:
        model: GaitDeepfakeDetector in eval mode
        feature_stats: dict with 'mean' and 'std' tensors, or None
    """
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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
    
    # Load feature normalization stats (critical for correct inference)
    # Keep stats on CPU since normalization is applied before .to(device)
    feature_stats = checkpoint.get('feature_stats', None)
    if feature_stats is not None:
        feature_stats = {k: v.cpu() for k, v in feature_stats.items()}
        print(f"Loaded feature normalization stats (dim={feature_stats['mean'].shape[0]})")
    else:
        print("WARNING: No feature_stats in checkpoint! Retrain the model.")
    
    print(f"Model loaded successfully (epoch {checkpoint['epoch']+1})")
    return model, feature_stats


def load_enrolled_identity(enrolled_file: str, 
                           person_name: str,
                           device: torch.device,
                           feature_stats: dict = None) -> torch.Tensor:
    """Load enrolled identity features (pre-computed from MediaPipe)."""
    with open(enrolled_file, 'rb') as f:
        enrolled_identities = pickle.load(f)
    
    if person_name not in enrolled_identities:
        available = list(enrolled_identities.keys())
        raise ValueError(f"Person '{person_name}' not enrolled. "
                        f"Available: {available}")
    
    enrolled = enrolled_identities[person_name]
    
    # Use pre-computed gait features (same pipeline as training)
    # MediaPipe: 12 gait landmarks × 3D coords + 6 angles + 36 velocities = 78 dims
    feature_list = []
    
    # Normalized coordinates (T, 12, 3) -> (T, 36)
    coords = enrolled['avg_normalized_coords']
    coords_flat = coords.reshape(coords.shape[0], -1)
    feature_list.append(coords_flat)
    
    # Joint angles (T, 6)
    angles = enrolled['avg_joint_angles']
    feature_list.append(angles)
    
    # Velocities (T, 12, 3) -> (T, 36)
    velocities = enrolled['avg_velocities']
    velocities_flat = velocities.reshape(velocities.shape[0], -1)
    feature_list.append(velocities_flat)
    
    # Combine: 36 + 6 + 36 = 78 dims
    features = np.concatenate(feature_list, axis=1)
    features_tensor = torch.FloatTensor(features)
    
    # Apply same z-score normalization as training
    if feature_stats is not None:
        features_tensor = (features_tensor - feature_stats['mean']) / feature_stats['std']
    
    features_tensor = features_tensor.unsqueeze(0).to(device)
    
    return features_tensor


def extract_video_features(video_path: str, 
                            sequence_length: int = 60,
                            device: torch.device = None,
                            feature_stats: dict = None) -> torch.Tensor:
    """Extract gait features from video."""
    print(f"Extracting gait features from {video_path}...")
    
    extractor = GaitFeatureExtractor(sequence_length=sequence_length)
    result = extractor.process_video(video_path)
    extractor.close()
    
    if result is None:
        raise ValueError(f"Could not extract features from {video_path}")
    
    # Prepare features
    gait_features = result['gait_features']
    
    # Normalized coordinates
    coords = gait_features['normalized_coords']
    coords_flat = coords.reshape(coords.shape[0], -1)
    
    # Joint angles
    angles = gait_features['joint_angles']
    
    # Velocities
    velocities = gait_features['velocities']
    velocities_flat = velocities.reshape(velocities.shape[0], -1)
    
    # Combine
    features = np.concatenate([coords_flat, angles, velocities_flat], axis=1)
    features_tensor = torch.FloatTensor(features)
    
    # Apply same z-score normalization as training
    if feature_stats is not None:
        features_tensor = (features_tensor - feature_stats['mean']) / feature_stats['std']
    
    features_tensor = features_tensor.unsqueeze(0)
    
    if device:
        features_tensor = features_tensor.to(device)
    
    print(f"Extracted features shape: {features_tensor.shape}")
    return features_tensor, result


def predict(model: GaitDeepfakeDetector,
            video_features: torch.Tensor,
            claimed_features: torch.Tensor,
            threshold: float = 0.7737) -> dict:
    """Make prediction for a single identity comparison (threshold from LOOCV)."""
    model.eval()
    
    with torch.no_grad():
        output = model(video_features, claimed_features, mode='verification')
        
        is_authentic = output['is_authentic'].item()
        similarity = output['similarity'].item()
        confidence = output['confidence'].item()
        
        # Adjust prediction based on threshold
        final_prediction = similarity > threshold
        
    return {
        'prediction': 'AUTHENTIC' if final_prediction else 'DEEPFAKE',
        'is_authentic': final_prediction,
        'similarity': similarity,
        'confidence': confidence,
        'model_prediction': is_authentic
    }


def check_all_identities(model: GaitDeepfakeDetector,
                         video_features: torch.Tensor,
                         enrolled_file: str,
                         device: torch.device,
                         feature_stats: dict = None,
                         threshold: float = 0.7737) -> dict:
    """Check video against ALL enrolled identities to distinguish
    identity mismatch from suspected deepfake.
    
    Returns:
        dict with 'scores' (name->similarity), 'best_match' (name or None),
        'best_similarity' (float)
    """
    with open(enrolled_file, 'rb') as f:
        enrolled_identities = pickle.load(f)
    
    scores = {}
    for name in enrolled_identities:
        identity_features = load_enrolled_identity(
            enrolled_file, name, device, feature_stats=feature_stats
        )
        result = predict(model, video_features, identity_features, threshold)
        scores[name] = result['similarity']
    
    # Find best match above threshold
    best_name = max(scores, key=scores.get)
    best_sim = scores[best_name]
    
    return {
        'scores': scores,
        'best_match': best_name if best_sim > threshold else None,
        'best_similarity': best_sim
    }


def visualize_result(video_path: str,
                     result: dict,
                     extraction_result: dict,
                     output_path: str = None):
    """Visualize prediction result with attention."""
    from utils.visualization import visualize_gait_attention
    
    # Load video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    
    # Get pose landmarks
    pose_sequence = extraction_result['pose_sequence']
    
    # Create dummy attention (uniform for now - can be replaced with Grad-CAM)
    attention = np.ones(len(pose_sequence)) * 0.5
    
    # Highlight key frames (simple heuristic based on movement)
    if len(pose_sequence) > 1:
        movement = np.diff(pose_sequence[:, :, :2], axis=0)
        movement_magnitude = np.abs(movement).sum(axis=(1, 2))
        movement_magnitude = np.concatenate([[movement_magnitude[0]], movement_magnitude])
        attention = movement_magnitude / (movement_magnitude.max() + 1e-8)
    
    # Interpolate attention to match frame count
    if len(attention) != len(frames):
        old_indices = np.linspace(0, len(attention) - 1, len(attention))
        new_indices = np.linspace(0, len(attention) - 1, len(frames))
        attention = np.interp(new_indices, old_indices, attention)
    
    # Interpolate pose landmarks to match frame count
    if len(pose_sequence) != len(frames):
        old_indices = np.linspace(0, len(pose_sequence) - 1, len(pose_sequence))
        new_indices = np.linspace(0, len(pose_sequence) - 1, len(frames))
        pose_interp = np.zeros((len(frames), pose_sequence.shape[1], pose_sequence.shape[2]))
        for i in range(pose_sequence.shape[1]):
            for j in range(pose_sequence.shape[2]):
                pose_interp[:, i, j] = np.interp(new_indices, old_indices, pose_sequence[:, i, j])
        pose_sequence = pose_interp
    
    visualize_gait_attention(frames, attention, pose_sequence, output_path)


def print_result(result: dict, claimed_identity: str, video_path: str,
                 identity_check: dict = None):
    """Print prediction result with 3-way verdict:
    
    Case 1 - AUTHENTIC: Gait matches claimed identity
    Case 2 - IDENTITY MISMATCH: Gait matches a DIFFERENT enrolled person  
    Case 3 - SUSPECTED DEEPFAKE: Gait matches NO enrolled identity
    """
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    
    print("\n" + "=" * 60)
    print("DEEPFAKE DETECTION RESULT")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Claimed Identity: {claimed_identity}")
    print("-" * 60)
    
    if result['is_authentic']:
        # Case 1: AUTHENTIC
        verdict = 'AUTHENTIC'
        print(f"Verdict:    {GREEN}AUTHENTIC{RESET}")
        print(f"Status:     Verified as {claimed_identity}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("=" * 60)
        print(f"The gait pattern matches the enrolled identity of {claimed_identity}.")
    
    elif identity_check and identity_check['best_match'] is not None:
        # Case 2: IDENTITY MISMATCH — real person, wrong claim
        actual = identity_check['best_match']
        verdict = 'IDENTITY_MISMATCH'
        print(f"Verdict:    {YELLOW}IDENTITY MISMATCH{RESET}")
        print(f"Status:     This is NOT {claimed_identity}")
        print(f"Actual:     Gait matches {GREEN}{actual}{RESET} "
              f"(similarity: {identity_check['best_similarity']:.4f})")
        print(f"Claimed:    {claimed_identity} "
              f"(similarity: {result['similarity']:.4f})")
        print("=" * 60)
        print(f"The video is REAL but the person is {actual}, not {claimed_identity}.")
        print("No deepfake detected — this is an identity claim mismatch.")
        
        # Show top matches
        sorted_scores = sorted(identity_check['scores'].items(), 
                               key=lambda x: x[1], reverse=True)
        print("\nAll identity scores:")
        for name, score in sorted_scores[:5]:
            marker = " <-- best match" if name == actual else ""
            print(f"  {name:15s}: {score:.4f}{marker}")
        if len(sorted_scores) > 5:
            print(f"  ... and {len(sorted_scores) - 5} more")
    
    else:
        # Case 3: SUSPECTED DEEPFAKE — matches nobody
        verdict = 'SUSPECTED_DEEPFAKE'
        print(f"Verdict:    {RED}SUSPECTED DEEPFAKE{RESET}")
        print(f"Status:     Gait matches NO enrolled identity")
        print(f"Claimed:    {claimed_identity} "
              f"(similarity: {result['similarity']:.4f})")
        print("=" * 60)
        print(f"WARNING: The gait pattern does not match {claimed_identity}")
        print("or any other enrolled identity!")
        print("This video is likely a deepfake — the walking pattern is synthetic.")
        
        if identity_check:
            sorted_scores = sorted(identity_check['scores'].items(),
                                   key=lambda x: x[1], reverse=True)
            print("\nClosest identity scores (all below threshold):")
            for name, score in sorted_scores[:3]:
                print(f"  {name:15s}: {score:.4f}")
    
    print("=" * 60 + "\n")
    result['verdict'] = verdict
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Gait-Based Deepfake Detection Inference'
    )
    
    # Required arguments
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file')
    parser.add_argument('--claimed_identity', type=str, required=True,
                        help='Name of the claimed identity')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, 
                        default=None,
                        help='Path to model checkpoint (auto-finds latest best if not specified)')
    parser.add_argument('--enrolled_file', type=str,
                        default='data/gait_features/enrolled_identities.pkl',
                        help='Path to enrolled identities file')
    
    # Options
    parser.add_argument('--threshold', type=float, default=0.7737,
                        help='Decision threshold (default 0.7737 from LOOCV Youden)')
    parser.add_argument('--sequence_length', type=int, default=60,
                        help='Sequence length for feature extraction')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize attention on video')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Auto-find latest best checkpoint if not specified
    if args.checkpoint is None:
        args.checkpoint = find_latest_best_checkpoint()
        print(f"Auto-selected checkpoint: {args.checkpoint}")
    
    # Check inputs
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.enrolled_file):
        raise FileNotFoundError(f"Enrolled file not found: {args.enrolled_file}")
    
    # Setup
    device = check_device()
    
    # Load model and feature normalization stats
    model, feature_stats = load_model(args.checkpoint, device)
    
    # Load claimed identity (with same normalization as training)
    print(f"\nLoading enrolled identity: {args.claimed_identity}")
    claimed_features = load_enrolled_identity(
        args.enrolled_file, 
        args.claimed_identity, 
        device,
        feature_stats=feature_stats
    )
    
    # Extract video features (with same normalization as training)
    video_features, extraction_result = extract_video_features(
        args.video, 
        args.sequence_length,
        device,
        feature_stats=feature_stats
    )
    
    # Make prediction against claimed identity
    result = predict(model, video_features, claimed_features, args.threshold)
    
    # If mismatch, check all enrolled identities to distinguish
    # identity mismatch (real person, wrong claim) from deepfake
    identity_check = None
    if not result['is_authentic']:
        print("\nClaimed identity mismatch — checking all enrolled identities...")
        identity_check = check_all_identities(
            model, video_features, args.enrolled_file,
            device, feature_stats, args.threshold
        )
    
    # Print result with 3-way verdict
    print_result(result, args.claimed_identity, args.video, identity_check)
    
    # Visualize if requested
    if args.visualize:
        output_path = args.output or f"visualization_{Path(args.video).stem}.png"
        visualize_result(args.video, result, extraction_result, output_path)
    
    # Return result for programmatic use
    return result


if __name__ == "__main__":
    main()
