"""
Identity Enrollment Script
===========================
Extract and store reference gait features for each identity.
These enrolled features will be used for verification.

Usage:
    python scripts/enrollment/enroll_identities.py --videos_dir data/videos
    python scripts/enrollment/enroll_identities.py --features_dir data/gait_features --output enrolled_identities.pkl

Author: DeepFake Detection Project
"""

import os
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from utils.pose_extraction import GaitFeatureExtractor


def extract_identity_from_filename(filename: str) -> str:
    """Extract identity from video filename (e.g., 'Arhaan_F.mp4' -> 'Arhaan')."""
    name = Path(filename).stem
    if '_' in name:
        return name.rsplit('_', 1)[0]
    return name


def get_videos_by_identity(videos_dir: str) -> dict:
    """Group videos by identity."""
    videos_by_identity = defaultdict(list)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    for file in os.listdir(videos_dir):
        if Path(file).suffix.lower() in video_extensions:
            identity = extract_identity_from_filename(file)
            videos_by_identity[identity].append(os.path.join(videos_dir, file))
    
    return dict(videos_by_identity)


def enroll_from_videos(videos_dir: str, 
                       output_file: str,
                       sequence_length: int = 60,
                       min_videos: int = 1) -> dict:
    """Enroll identities from video files."""
    print(f"\nScanning videos directory: {videos_dir}")
    
    videos_by_identity = get_videos_by_identity(videos_dir)
    
    print(f"Found {len(videos_by_identity)} identities:")
    for identity, videos in sorted(videos_by_identity.items()):
        print(f"  {identity}: {len(videos)} videos")
    
    # Initialize feature extractor
    extractor = GaitFeatureExtractor(sequence_length=sequence_length)
    
    enrolled_identities = {}
    
    for identity, videos in tqdm(sorted(videos_by_identity.items()), 
                                  desc="Enrolling identities"):
        print(f"\n{'='*40}")
        print(f"Enrolling: {identity}")
        print(f"{'='*40}")
        
        all_pose_sequences = []
        all_gait_features = []
        valid_videos = 0
        
        for video_path in videos:
            print(f"  Processing: {Path(video_path).name}")
            
            try:
                result = extractor.process_video(video_path)
                
                if result is not None:
                    all_pose_sequences.append(result['pose_sequence'])
                    all_gait_features.append(result['gait_features'])
                    valid_videos += 1
                    print(f"    ✓ Extracted {result['frame_count']} frames")
                else:
                    print(f"    ✗ Failed to extract features")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        if valid_videos >= min_videos:
            # Compute average pose sequence
            avg_pose = np.mean(all_pose_sequences, axis=0)
            
            # Compute average gait features
            avg_normalized_coords = np.mean(
                [gf['normalized_coords'] for gf in all_gait_features], axis=0
            )
            avg_joint_angles = np.mean(
                [gf['joint_angles'] for gf in all_gait_features], axis=0
            )
            avg_velocities = np.mean(
                [gf['velocities'] for gf in all_gait_features], axis=0
            )
            avg_accelerations = np.mean(
                [gf['accelerations'] for gf in all_gait_features], axis=0
            )
            
            enrolled_identities[identity] = {
                'avg_pose_sequence': avg_pose,
                'avg_normalized_coords': avg_normalized_coords,
                'avg_joint_angles': avg_joint_angles,
                'avg_velocities': avg_velocities,
                'avg_accelerations': avg_accelerations,
                'num_videos': valid_videos,
                'video_paths': videos,
                'all_pose_sequences': all_pose_sequences,
                'all_gait_features': all_gait_features
            }
            
            print(f"  ✓ Enrolled with {valid_videos} videos")
        else:
            print(f"  ✗ Not enough valid videos (need {min_videos}, got {valid_videos})")
    
    extractor.close()
    
    # Save enrolled identities
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(enrolled_identities, f)
    
    print(f"\n{'='*60}")
    print(f"ENROLLMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total identities enrolled: {len(enrolled_identities)}")
    print(f"Saved to: {output_file}")
    
    return enrolled_identities


def enroll_from_features(features_file: str, 
                         output_file: str) -> dict:
    """Enroll identities from pre-extracted features file."""
    print(f"\nLoading features from: {features_file}")
    
    with open(features_file, 'rb') as f:
        all_features = pickle.load(f)
    
    # Group features by identity
    features_by_identity = defaultdict(list)
    
    for video_path, features in all_features.items():
        identity = features.get('identity') or extract_identity_from_filename(video_path)
        features_by_identity[identity].append(features)
    
    print(f"Found {len(features_by_identity)} identities")
    
    enrolled_identities = {}
    
    for identity, feature_list in sorted(features_by_identity.items()):
        print(f"\nEnrolling: {identity} ({len(feature_list)} samples)")
        
        # Extract pose sequences and gait features
        pose_sequences = [f['pose_sequence'] for f in feature_list]
        gait_features = [f['gait_features'] for f in feature_list]
        
        # Compute averages
        avg_pose = np.mean(pose_sequences, axis=0)
        avg_normalized_coords = np.mean(
            [gf['normalized_coords'] for gf in gait_features], axis=0
        )
        avg_joint_angles = np.mean(
            [gf['joint_angles'] for gf in gait_features], axis=0
        )
        avg_velocities = np.mean(
            [gf['velocities'] for gf in gait_features], axis=0
        )
        avg_accelerations = np.mean(
            [gf['accelerations'] for gf in gait_features], axis=0
        )
        
        enrolled_identities[identity] = {
            'avg_pose_sequence': avg_pose,
            'avg_normalized_coords': avg_normalized_coords,
            'avg_joint_angles': avg_joint_angles,
            'avg_velocities': avg_velocities,
            'avg_accelerations': avg_accelerations,
            'num_videos': len(feature_list),
            'all_pose_sequences': pose_sequences,
            'all_gait_features': gait_features
        }
    
    # Save enrolled identities
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(enrolled_identities, f)
    
    print(f"\n{'='*60}")
    print(f"ENROLLMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total identities enrolled: {len(enrolled_identities)}")
    print(f"Saved to: {output_file}")
    
    return enrolled_identities


def show_enrolled_summary(enrolled_file: str):
    """Display summary of enrolled identities."""
    with open(enrolled_file, 'rb') as f:
        enrolled_identities = pickle.load(f)
    
    print(f"\n{'='*60}")
    print("ENROLLED IDENTITIES SUMMARY")
    print(f"{'='*60}")
    
    total_videos = 0
    for identity, data in sorted(enrolled_identities.items()):
        num_videos = data['num_videos']
        total_videos += num_videos
        avg_pose_shape = data['avg_pose_sequence'].shape
        print(f"  {identity}:")
        print(f"    Videos: {num_videos}")
        print(f"    Pose shape: {avg_pose_shape}")
    
    print(f"\nTotal: {len(enrolled_identities)} identities, {total_videos} videos")


def main():
    parser = argparse.ArgumentParser(
        description='Enroll identities for Gait-Based Deepfake Detection'
    )
    
    # Mode selection
    parser.add_argument('--from_videos', action='store_true',
                        help='Enroll directly from video files')
    parser.add_argument('--from_features', action='store_true',
                        help='Enroll from pre-extracted features file')
    parser.add_argument('--show_summary', action='store_true',
                        help='Show summary of enrolled identities')
    
    # Input arguments
    parser.add_argument('--videos_dir', type=str, default='data/videos',
                        help='Directory containing video files')
    parser.add_argument('--features_file', type=str, 
                        default='data/gait_features/gait_features.pkl',
                        help='Path to pre-extracted features file')
    
    # Output arguments
    parser.add_argument('--output', type=str, 
                        default='data/gait_features/enrolled_identities.pkl',
                        help='Output file for enrolled identities')
    
    # Options
    parser.add_argument('--sequence_length', type=int, default=60,
                        help='Sequence length for feature extraction')
    parser.add_argument('--min_videos', type=int, default=1,
                        help='Minimum videos required for enrollment')
    
    args = parser.parse_args()
    
    # Default to from_videos if no mode specified
    if not args.from_videos and not args.from_features and not args.show_summary:
        args.from_videos = True
    
    if args.show_summary:
        if os.path.exists(args.output):
            show_enrolled_summary(args.output)
        else:
            print(f"Enrolled file not found: {args.output}")
        return
    
    if args.from_videos:
        if not os.path.exists(args.videos_dir):
            raise FileNotFoundError(f"Videos directory not found: {args.videos_dir}")
        enroll_from_videos(
            args.videos_dir,
            args.output,
            args.sequence_length,
            args.min_videos
        )
    
    elif args.from_features:
        if not os.path.exists(args.features_file):
            raise FileNotFoundError(f"Features file not found: {args.features_file}")
        enroll_from_features(args.features_file, args.output)


if __name__ == "__main__":
    main()
