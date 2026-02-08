"""
GradCAM Visualization Runner
=============================
Generates explainability visualizations for the trained model.
Produces temporal heatmaps, joint importance, and feature group charts.

Usage:
    python scripts/evaluation/run_gradcam.py
    python scripts/evaluation/run_gradcam.py --video data/videos/Arhaan_S1.mp4 --claimed_identity Aarav
    python scripts/evaluation/run_gradcam.py --all_identities

Author: DeepFake Detection Project
"""

import os
import json
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from models.full_pipeline import create_model
from utils.gradcam import (
    GaitGradCAM, JointImportanceAnalyzer,
    plot_temporal_heatmap, plot_joint_importance, plot_feature_group_importance,
    generate_explainability_report, GAIT_KEYPOINT_NAMES, JOINT_ANGLE_NAMES
)
from utils.pose_extraction import GaitFeatureExtractor


def load_model_and_stats(checkpoint_path, device):
    """Load model and feature stats from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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
    
    feature_stats = checkpoint.get('feature_stats', None)
    if feature_stats is not None:
        feature_stats = {k: v.cpu() for k, v in feature_stats.items()}
    
    print(f"Model loaded (epoch {checkpoint['epoch']+1})")
    return model, feature_stats


def load_features_from_video(video_path, feature_stats=None, seq_len=60):
    """Extract and normalize features from a video file."""
    print(f"Extracting gait features from {video_path}...")
    extractor = GaitFeatureExtractor()
    features = extractor.extract_from_video(video_path, sequence_length=seq_len)
    
    if features is None or len(features) == 0:
        raise ValueError(f"Could not extract features from {video_path}")
    
    features = torch.tensor(features, dtype=torch.float32)
    if features.dim() == 2:
        features = features.unsqueeze(0)
    
    # Normalize
    if feature_stats is not None:
        features = (features - feature_stats['mean']) / (feature_stats['std'] + 1e-8)
    
    return features


def load_enrolled_features(enrolled_file, person_name, feature_stats=None, seq_len=60):
    """Load enrolled identity features and compose 78-dim vector."""
    with open(enrolled_file, 'rb') as f:
        enrolled = pickle.load(f)
    
    if person_name not in enrolled:
        raise ValueError(f"Identity '{person_name}' not found. Available: {list(enrolled.keys())}")
    
    data = enrolled[person_name]
    
    if isinstance(data, dict):
        # Try direct 78-dim features first
        features = data.get('features', data.get('mean_features'))
        
        if features is None and 'avg_normalized_coords' in data:
            # Compose 78-dim from avg gait components:
            # coords(T,12,3)→(T,36) + angles(T,6) + velocities(T,12,3)→(T,36) = (T,78)
            coords = data['avg_normalized_coords'].reshape(data['avg_normalized_coords'].shape[0], -1)
            angles = data['avg_joint_angles']
            velocities = data['avg_velocities'].reshape(data['avg_velocities'].shape[0], -1)
            features = np.concatenate([coords, angles, velocities], axis=1).astype(np.float64)
    else:
        features = data
    
    if features is None:
        raise ValueError(f"Could not extract features for '{person_name}' from enrolled data")
    
    features = np.asarray(features, dtype=np.float64)
    
    # Build a sequence from enrolled features
    if features.ndim == 1:
        features = np.tile(features, (seq_len, 1))
    elif features.ndim == 2 and features.shape[0] > seq_len:
        features = features[:seq_len]
    elif features.ndim == 2 and features.shape[0] < seq_len:
        repeats = seq_len // features.shape[0] + 1
        features = np.tile(features, (repeats, 1))[:seq_len]
    
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    if feature_stats is not None:
        features = (features - feature_stats['mean']) / (feature_stats['std'] + 1e-8)
    
    return features


def run_single_analysis(model, video_features, claimed_features, 
                        video_label, claimed_label, save_dir, device):
    """Run GradCAM analysis on a single video-identity pair."""
    video_features = video_features.to(device)
    claimed_features = claimed_features.to(device)
    
    # Get prediction first
    with torch.no_grad():
        output = model(video_features, claimed_features, mode='verification')
        pred = output['is_authentic'].item()
        sim = output['similarity'].item()
    
    verdict = 'AUTHENTIC' if pred == 1 else 'DEEPFAKE'
    label = f"{verdict} (sim={sim:.4f}) | Video={video_label}, Claimed={claimed_label}"
    
    print(f"\n  Prediction: {verdict} (similarity={sim:.4f})")
    print(f"  Generating explainability report...")
    
    # Generate report
    report = generate_explainability_report(
        model=model,
        video_features=video_features,
        claimed_features=claimed_features,
        prediction_label=label,
        save_dir=save_dir,
        mode='verification'
    )
    
    return report, verdict, sim


def run_aggregate_analysis(model, enrolled_file, features_file, feature_stats,
                           device, save_dir, n_samples=20):
    """Run GradCAM on multiple samples and aggregate importance scores."""
    print(f"\n{'='*60}")
    print("  AGGREGATE ANALYSIS ({} samples)".format(n_samples))
    print(f"{'='*60}")
    
    # Load pre-extracted features (keyed by video path)
    with open(features_file, 'rb') as f:
        all_features = pickle.load(f)
    
    with open(enrolled_file, 'rb') as f:
        enrolled = pickle.load(f)
    
    # Group video keys by person name
    from collections import defaultdict
    person_videos = defaultdict(list)
    for video_key, video_data in all_features.items():
        # Extract person name from video path (e.g., 'data/videos/Arhaan_F1.mp4' → 'Arhaan')
        basename = Path(video_key).stem  # 'Arhaan_F1'
        parts = basename.rsplit('_', 1)
        person_name = parts[0] if len(parts) > 1 else basename
        person_videos[person_name].append((video_key, video_data))
    
    persons = [p for p in person_videos if p in enrolled]
    print(f"  Found {len(persons)} enrolled identities with video data")
    
    # Initialize extractor for 78-dim feature conversion
    extractor = GaitFeatureExtractor()
    
    # Build 78-dim feature converter
    def pose_to_78dim(pose_seq):
        """Convert (T, 33, 3) raw poses to (T, 78) feature vector."""
        # Need (T, 33, 4) for compute_gait_features but we have (T, 33, 3)
        # Pad with visibility=1.0
        if pose_seq.shape[-1] == 3:
            vis = np.ones((*pose_seq.shape[:-1], 1))
            pose_seq_4d = np.concatenate([pose_seq, vis], axis=-1)  # (T, 33, 4)
        else:
            pose_seq_4d = pose_seq
        
        gait_features = extractor.compute_gait_features(pose_seq_4d)
        
        # Flatten: coords(T,12,3)→(T,36) + angles(T,6) + velocities(T,12,3)→(T,36) = (T,78)
        coords = gait_features['normalized_coords'].reshape(pose_seq.shape[0], -1)  # (T, 36)
        angles = gait_features['joint_angles']  # (T, 6)
        velocities = gait_features['velocities'].reshape(pose_seq.shape[0], -1)  # (T, 36)
        
        return np.concatenate([coords, angles, velocities], axis=1)  # (T, 78)
    
    # Collect importance scores
    all_joint_imp = []
    all_angle_imp = []
    all_group_imp = []
    all_temporal_imp = []
    
    analyzer = JointImportanceAnalyzer(model)
    
    sample_count = 0
    n_per_person = max(1, n_samples // len(persons))
    
    for person in persons:
        videos = person_videos[person]
        
        for vid_key, vid_data in videos[:n_per_person]:
            try:
                # Convert raw pose (60, 33, 3) to 78-dim features
                pose_seq = vid_data['pose_sequence']  # (60, 33, 3)
                features_78 = pose_to_78dim(pose_seq)  # (60, 78)
                
                if features_78.shape[0] < 60:
                    reps = 60 // features_78.shape[0] + 1
                    features_78 = np.tile(features_78, (reps, 1))[:60]
                
                video_feat = torch.tensor(features_78, dtype=torch.float32).unsqueeze(0)
                if feature_stats:
                    video_feat = (video_feat - feature_stats['mean']) / (feature_stats['std'] + 1e-8)
                video_feat = video_feat.to(device)
                
                # Claimed features (same person = authentic pair)
                claimed_feat = load_enrolled_features(enrolled_file, person, feature_stats)
                claimed_feat = claimed_feat.to(device)
                
                importance = analyzer.compute_joint_importance(
                    video_feat, claimed_feat, mode='verification'
                )
                
                all_joint_imp.append(importance['joint_importance'])
                all_angle_imp.append(importance['angle_importance'])
                all_group_imp.append(importance['group_importance'])
                all_temporal_imp.append(importance['temporal_importance'])
                
                sample_count += 1
                if sample_count % 5 == 0:
                    print(f"  Processed {sample_count}/{n_samples} samples...")
                    
                if sample_count >= n_samples:
                    break
            except Exception as e:
                print(f"    Warning: Skipped {vid_key}: {e}")
                continue
        
        if sample_count >= n_samples:
            break
    
    print(f"  Analyzed {sample_count} samples across {len(persons)} identities")
    
    if sample_count == 0:
        print("  ERROR: No samples could be analyzed")
        return
    
    # Aggregate
    avg_joint = np.mean(all_joint_imp, axis=0)
    avg_angle = np.mean(all_angle_imp, axis=0)
    avg_group = {k: np.mean([g[k] for g in all_group_imp]) for k in all_group_imp[0].keys()}
    avg_temporal = np.mean(all_temporal_imp, axis=0)
    
    # Normalize
    if avg_joint.max() > 0: avg_joint /= avg_joint.max()
    if avg_angle.max() > 0: avg_angle /= avg_angle.max()
    if avg_temporal.max() > 0: avg_temporal /= avg_temporal.max()
    
    # Save plots
    os.makedirs(save_dir, exist_ok=True)
    
    plot_temporal_heatmap(
        avg_temporal,
        title=f'Average Temporal Importance (n={sample_count})',
        save_path=os.path.join(save_dir, 'aggregate_temporal_heatmap.png')
    )
    
    plot_joint_importance(
        avg_joint, avg_angle,
        title=f'Average Joint Importance (n={sample_count})',
        save_path=os.path.join(save_dir, 'aggregate_joint_importance.png')
    )
    
    plot_feature_group_importance(
        avg_group,
        title=f'Feature Group Contribution (n={sample_count})',
        save_path=os.path.join(save_dir, 'aggregate_feature_groups.png')
    )
    
    # Print summary
    print(f"\n  Top 5 Most Important Joints:")
    sorted_idx = np.argsort(avg_joint)[::-1]
    for rank, idx in enumerate(sorted_idx[:5], 1):
        print(f"    {rank}. {GAIT_KEYPOINT_NAMES[idx]:>15s}: {avg_joint[idx]:.4f}")
    
    print(f"\n  Joint Angle Importance:")
    sorted_idx_a = np.argsort(avg_angle)[::-1]
    for rank, idx in enumerate(sorted_idx_a, 1):
        print(f"    {rank}. {JOINT_ANGLE_NAMES[idx]:>20s}: {avg_angle[idx]:.4f}")
    
    print(f"\n  Feature Group Contribution:")
    for group, val in avg_group.items():
        print(f"    {group:>12s}: {val:.1%}")
    
    # Save numeric results
    results = {
        'joint_importance': {GAIT_KEYPOINT_NAMES[i]: float(avg_joint[i]) for i in range(12)},
        'angle_importance': {JOINT_ANGLE_NAMES[i]: float(avg_angle[i]) for i in range(6)},
        'group_importance': {k: float(v) for k, v in avg_group.items()},
        'n_samples': sample_count
    }
    with open(os.path.join(save_dir, 'gradcam_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description='GradCAM Visualization for Gait Deepfake Detection')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--video', type=str, default=None,
                        help='Video file for single analysis')
    parser.add_argument('--claimed_identity', type=str, default=None,
                        help='Claimed identity for single analysis')
    parser.add_argument('--enrolled_file', type=str, 
                        default='data/gait_features/enrolled_identities.pkl')
    parser.add_argument('--features_file', type=str,
                        default='data/gait_features/gait_features.pkl')
    parser.add_argument('--aggregate', action='store_true',
                        help='Run aggregate analysis across many samples')
    parser.add_argument('--n_samples', type=int, default=30,
                        help='Number of samples for aggregate analysis')
    parser.add_argument('--output_dir', type=str, default='outputs/gradcam')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Find checkpoint
    if args.checkpoint is None:
        import re
        ckpt_dir = Path('outputs/checkpoints')
        best_ckpts = list(ckpt_dir.glob('*_best.pth'))
        if best_ckpts:
            def epoch_num(p):
                m = re.search(r'epoch_(\d+)_best', p.name)
                return int(m.group(1)) if m else 0
            best_ckpts.sort(key=epoch_num)
            args.checkpoint = str(best_ckpts[-1])
            print(f"Auto-selected: {args.checkpoint}")
        else:
            raise FileNotFoundError("No checkpoint found in outputs/checkpoints/")
    
    # Load model
    model, feature_stats = load_model_and_stats(args.checkpoint, device)
    
    # Single video analysis
    if args.video and args.claimed_identity:
        save_dir = os.path.join(args.output_dir, 'single')
        os.makedirs(save_dir, exist_ok=True)
        
        video_features = load_features_from_video(args.video, feature_stats)
        claimed_features = load_enrolled_features(
            args.enrolled_file, args.claimed_identity, feature_stats
        )
        
        report, verdict, sim = run_single_analysis(
            model, video_features, claimed_features,
            args.video, args.claimed_identity, save_dir, device
        )
        print(f"\n  Plots saved to {save_dir}/")
    
    # Aggregate analysis (default if no video specified)
    if args.aggregate or not args.video:
        save_dir = os.path.join(args.output_dir, 'aggregate')
        run_aggregate_analysis(
            model, args.enrolled_file, args.features_file,
            feature_stats, device, save_dir, args.n_samples
        )
    
    print(f"\n[DONE] GradCAM analysis complete!")


if __name__ == "__main__":
    main()
