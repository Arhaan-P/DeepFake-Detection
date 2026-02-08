"""
Preprocessing Script - Extract Gait Features from All Videos
=============================================================
Extract pose landmarks and gait features from all videos
and save them for training.

Usage:
    python scripts/preprocessing/preprocess_videos.py --videos_dir data/videos
    python scripts/preprocessing/preprocess_videos.py --videos_dir data/videos --augmented_dir data/augmented
    python scripts/preprocessing/preprocess_videos.py --videos_dir data/videos --use_gpu  # Use TensorFlow MoveNet on GPU

Author: DeepFake Detection Project
"""

import os
import argparse
import pickle
from pathlib import Path
from collections import defaultdict
import time

import numpy as np
from tqdm import tqdm

from utils.pose_extraction import GaitFeatureExtractor


def extract_identity_from_filename(filename: str) -> str:
    """Extract identity from video filename."""
    name = Path(filename).stem
    
    # Handle augmented video names like 'Arhaan_F_hflip.mp4'
    parts = name.split('_')
    if len(parts) >= 2:
        # Check if last part is view indicator
        if parts[-1] in ['F', 'S', 'front', 'side']:
            return '_'.join(parts[:-1])
        # Check if second-to-last is view indicator (augmented case)
        if len(parts) >= 3 and parts[-2] in ['F', 'S', 'front', 'side']:
            return '_'.join(parts[:-2])
    
    # Default: take first part
    return parts[0]


def collect_videos(directories: list) -> dict:
    """Collect all videos from directories grouped by identity."""
    videos_by_identity = defaultdict(list)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            continue
        
        for root, _, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in video_extensions:
                    video_path = os.path.join(root, file)
                    identity = extract_identity_from_filename(file)
                    videos_by_identity[identity].append(video_path)
    
    return dict(videos_by_identity)


def preprocess_videos(videos_dir: str,
                      augmented_dir: str = None,
                      output_file: str = 'data/gait_features/gait_features.pkl',
                      sequence_length: int = 60,
                      batch_size: int = 10,
                      use_gpu: bool = False,
                      resume: bool = True) -> dict:
    """Extract gait features from all videos."""
    
    print("\n" + "=" * 60)
    print("GAIT FEATURE EXTRACTION")
    print("=" * 60)
    
    # Collect videos
    directories = [videos_dir]
    if augmented_dir and os.path.exists(augmented_dir):
        directories.append(augmented_dir)
    
    videos_by_identity = collect_videos(directories)
    
    total_videos = sum(len(v) for v in videos_by_identity.values())
    print(f"\nFound {total_videos} videos from {len(videos_by_identity)} identities:")
    for identity, videos in sorted(videos_by_identity.items()):
        print(f"  {identity}: {len(videos)} videos")
    
    # Check for existing progress (resume capability)
    all_features = {}
    already_processed = set()
    
    if resume and os.path.exists(output_file):
        try:
            with open(output_file, 'rb') as f:
                all_features = pickle.load(f)
            already_processed = set(all_features.keys())
            print(f"\n✓ Resuming: Found {len(already_processed)} already processed videos")
        except Exception as e:
            print(f"Warning: Could not load existing file ({e}). Starting fresh.")
            all_features = {}
    
    # Initialize feature extractor
    print("\nInitializing pose extractor...")
    
    # Auto-detect GPU if not explicitly disabled
    try:
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        use_gpu = use_gpu or gpu_available  # Use GPU if available or explicitly requested
    except:
        gpu_available = False
    
    if use_gpu:
        try:
            from utils.pose_extraction_gpu import MoveNetExtractor
            extractor = MoveNetExtractor(model_name='thunder', sequence_length=sequence_length)
            print("✓ Using GPU-accelerated MoveNet (TensorFlow)")
        except Exception as e:
            print(f"Warning: GPU initialization failed ({e}). Falling back to CPU...")
            from utils.pose_extraction import GaitFeatureExtractor
            extractor = GaitFeatureExtractor(sequence_length=sequence_length)
            print("Using MediaPipe (CPU)")
    else:
        from utils.pose_extraction import GaitFeatureExtractor
        extractor = GaitFeatureExtractor(sequence_length=sequence_length)
        print("Using MediaPipe (CPU)")
    
    # Process videos
    failed_videos = []
    
    start_time = time.time()
    processed = len(already_processed)
    new_processed = 0
    save_interval = 50  # Save every 50 videos
    
    # Create overall progress bar
    identity_list = sorted(videos_by_identity.items())
    remaining = total_videos - len(already_processed)
    print(f"\nProcessing {remaining} remaining videos ({len(already_processed)} already done)...\n")
    
    try:
        for identity, videos in tqdm(identity_list, desc="Overall Progress", unit="person"):
            # Filter out already processed videos
            videos_to_process = [v for v in videos if v not in already_processed]
            
            if not videos_to_process:
                continue  # Skip this identity entirely
            
            for video_path in videos_to_process:
                try:
                    result = extractor.process_video(video_path)
                    
                    if result is not None:
                        result['identity'] = identity
                        all_features[video_path] = result
                        processed += 1
                        new_processed += 1
                    else:
                        failed_videos.append((video_path, "No features extracted"))
                except Exception as e:
                    failed_videos.append((video_path, str(e)))
                
                # Save checkpoint periodically
                if new_processed > 0 and new_processed % save_interval == 0:
                    os.makedirs(Path(output_file).parent, exist_ok=True)
                    with open(output_file, 'wb') as f:
                        pickle.dump(all_features, f)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted! Saving progress...")
        os.makedirs(Path(output_file).parent, exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(all_features, f)
        print(f"✓ Saved {len(all_features)} videos to {output_file}")
        print("Run the same command again to resume.")
        return all_features
    
    # Close extractor (only MediaPipe has close method)
    if hasattr(extractor, 'close'):
        extractor.close()
    
    # Save features
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(all_features, f)
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Processed: {processed}/{total_videos} videos")
    print(f"Failed: {len(failed_videos)} videos")
    print(f"Time: {elapsed/60:.1f} minutes ({elapsed/processed:.2f} sec/video)")
    print(f"Output: {output_file}")
    
    if failed_videos:
        print("\nFailed videos:")
        for video, error in failed_videos[:10]:
            print(f"  - {Path(video).name}: {error}")
        if len(failed_videos) > 10:
            print(f"  ... and {len(failed_videos) - 10} more")
    
    # Feature statistics
    print("\n--- Feature Statistics ---")
    
    # Count per identity
    identity_counts = defaultdict(int)
    for video_path, features in all_features.items():
        identity_counts[features['identity']] += 1
    
    for identity, count in sorted(identity_counts.items()):
        print(f"  {identity}: {count} samples")
    
    # Feature dimensions
    if all_features:
        sample = next(iter(all_features.values()))
        gait = sample['gait_features']
        print(f"\nFeature dimensions:")
        print(f"  Pose sequence: {sample['pose_sequence'].shape}")
        print(f"  Normalized coords: {gait['normalized_coords'].shape}")
        print(f"  Joint angles: {gait['joint_angles'].shape}")
        print(f"  Velocities: {gait['velocities'].shape}")
        print(f"  Accelerations: {gait['accelerations'].shape}")
    
    return all_features


def verify_features(features_file: str):
    """Verify extracted features file."""
    print(f"\nVerifying features file: {features_file}")
    
    with open(features_file, 'rb') as f:
        all_features = pickle.load(f)
    
    print(f"Total samples: {len(all_features)}")
    
    # Check feature shapes
    shapes = defaultdict(list)
    identities = defaultdict(int)
    
    for video_path, features in all_features.items():
        identities[features['identity']] += 1
        shapes['pose_sequence'].append(features['pose_sequence'].shape)
        gait = features['gait_features']
        shapes['normalized_coords'].append(gait['normalized_coords'].shape)
        shapes['joint_angles'].append(gait['joint_angles'].shape)
    
    print("\nIdentity distribution:")
    for identity, count in sorted(identities.items()):
        print(f"  {identity}: {count}")
    
    print("\nFeature shape verification:")
    for name, shape_list in shapes.items():
        unique_shapes = set(shape_list)
        if len(unique_shapes) == 1:
            print(f"  {name}: ✓ Consistent {unique_shapes.pop()}")
        else:
            print(f"  {name}: ✗ Inconsistent shapes: {unique_shapes}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract Gait Features from Videos'
    )
    
    # Input arguments
    parser.add_argument('--videos_dir', type=str, default='data/videos',
                        help='Directory containing original videos')
    parser.add_argument('--augmented_dir', type=str, default=None,
                        help='Directory containing augmented videos')
    
    # Output arguments
    parser.add_argument('--output', type=str, 
                        default='data/gait_features/gait_features.pkl',
                        help='Output file for extracted features')
    
    # Options
    parser.add_argument('--sequence_length', type=int, default=60,
                        help='Sequence length for feature extraction')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for progress updates')
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing features file')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU-accelerated MoveNet instead of MediaPipe')
    
    args = parser.parse_args()
    
    if args.verify:
        if os.path.exists(args.output):
            verify_features(args.output)
        else:
            print(f"Features file not found: {args.output}")
        return
    
    if not os.path.exists(args.videos_dir):
        raise FileNotFoundError(f"Videos directory not found: {args.videos_dir}")
    
    preprocess_videos(
        args.videos_dir,
        args.augmented_dir,
        args.output,
        args.sequence_length,
        args.batch_size,
        args.use_gpu
    )
    
    # Auto-enroll after extraction
    print("\n" + "=" * 60)
    print("ENROLLING IDENTITIES")
    print("=" * 60)
    
    from enroll_identities import enroll_from_features
    
    enrolled_file = args.output.replace('gait_features.pkl', 'enrolled_identities.pkl')
    enroll_from_features(args.output, enrolled_file)


if __name__ == "__main__":
    from utils.logger import setup_logging, close_logging
    logger = setup_logging('preprocess')
    try:
        main()
    finally:
        close_logging(logger)
