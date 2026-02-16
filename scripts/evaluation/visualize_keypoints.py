"""
Gait Keypoint Matching Visualizations
======================================
Generates publication-quality figures for research paper:
1. Skeleton overlay sequences (gait cycle visualization)
2. Authentic vs Deepfake gait comparison with similarity scores
3. Joint angle time-series comparison across subjects
4. t-SNE embedding space visualization
5. Joint importance heatmap on canonical skeleton

Usage:
    python scripts/evaluation/visualize_keypoints.py
    python scripts/evaluation/visualize_keypoints.py --subjects Arhaan,Ananya,Som,Teja,Vedant

Author: DeepFake Detection Project
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from models.full_pipeline import create_model
from utils.pose_extraction import GaitFeatureExtractor
from utils.gradcam import JointImportanceAnalyzer, GAIT_KEYPOINT_NAMES, JOINT_ANGLE_NAMES


# ── Constants ────────────────────────────────────────────────────────────────

# Skeleton bone connections (MediaPipe 33-landmark indices)
SKELETON_CONNECTIONS = [
    (11, 13), (13, 15),           # Left arm
    (12, 14), (14, 16),           # Right arm
    (11, 12),                      # Shoulders
    (11, 23), (12, 24),           # Torso
    (23, 24),                      # Hips
    (23, 25), (25, 27), (27, 31), # Left leg
    (24, 26), (26, 28), (28, 32), # Right leg
]

# Gait-only skeleton connections (using indices into GAIT_LANDMARKS array 0-11)
GAIT_BONE_CONNECTIONS = [
    (0, 1),                       # Shoulders
    (0, 2), (1, 3),               # Torso (shoulder→hip)
    (2, 3),                       # Hips
    (2, 4), (4, 6), (6, 10),     # Left: hip→knee→ankle→foot
    (3, 5), (5, 7), (7, 11),     # Right: hip→knee→ankle→foot
    (6, 8),                       # Left: ankle→heel
    (7, 9),                       # Right: ankle→heel
]

# GAIT_LANDMARKS indices into MediaPipe 33
GAIT_LANDMARK_INDICES = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

GAIT_LANDMARK_NAMES = [
    'L_Shoulder', 'R_Shoulder', 'L_Hip', 'R_Hip',
    'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle',
    'L_Heel', 'R_Heel', 'L_Foot', 'R_Foot'
]

# Canonical skeleton layout for the importance diagram (normalized 0-1 coords)
# (x, y) positions arranged as a frontal stick figure
CANONICAL_SKELETON = {
    'L_Shoulder': (0.60, 0.25), 'R_Shoulder': (0.40, 0.25),
    'L_Hip':      (0.57, 0.50), 'R_Hip':      (0.43, 0.50),
    'L_Knee':     (0.60, 0.68), 'R_Knee':     (0.40, 0.68),
    'L_Ankle':    (0.62, 0.85), 'R_Ankle':    (0.38, 0.85),
    'L_Heel':     (0.64, 0.90), 'R_Heel':     (0.36, 0.90),
    'L_Foot':     (0.67, 0.93), 'R_Foot':     (0.33, 0.93),
}

CANONICAL_BONES = [
    ('L_Shoulder', 'R_Shoulder'),
    ('L_Shoulder', 'L_Hip'), ('R_Shoulder', 'R_Hip'),
    ('L_Hip', 'R_Hip'),
    ('L_Hip', 'L_Knee'), ('L_Knee', 'L_Ankle'), ('L_Ankle', 'L_Foot'),
    ('R_Hip', 'R_Knee'), ('R_Knee', 'R_Ankle'), ('R_Ankle', 'R_Foot'),
    ('L_Ankle', 'L_Heel'), ('R_Ankle', 'R_Heel'),
]


# ── Model loading (adapted from run_gradcam.py) ─────────────────────────────

def load_model_and_stats(checkpoint_path, device):
    """Load trained model and feature normalization stats."""
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

    print(f"  Model loaded (epoch {checkpoint['epoch']+1})")
    return model, feature_stats


def find_best_checkpoint(checkpoints_dir):
    """Find the best checkpoint file (highest epoch with _best suffix)."""
    ckpt_dir = Path(checkpoints_dir)
    best_files = sorted(ckpt_dir.glob('*_best.pth'))
    if not best_files:
        raise FileNotFoundError(f"No *_best.pth found in {ckpt_dir}")
    # Sort by epoch number
    def epoch_num(p):
        name = p.stem  # e.g. checkpoint_epoch_30_best
        parts = name.split('_')
        for i, part in enumerate(parts):
            if part == 'epoch' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return 0
    best_files.sort(key=epoch_num)
    return str(best_files[-1])


def load_enrolled_features(enrolled_file, person_name, feature_stats=None, seq_len=60):
    """Load enrolled identity features and compose 78-dim vector."""
    with open(enrolled_file, 'rb') as f:
        enrolled = pickle.load(f)

    if person_name not in enrolled:
        raise ValueError(f"'{person_name}' not found. Available: {list(enrolled.keys())}")

    data = enrolled[person_name]

    if isinstance(data, dict):
        features = data.get('features', data.get('mean_features'))
        if features is None and 'avg_normalized_coords' in data:
            coords = data['avg_normalized_coords'].reshape(data['avg_normalized_coords'].shape[0], -1)
            angles = data['avg_joint_angles']
            velocities = data['avg_velocities'].reshape(data['avg_velocities'].shape[0], -1)
            features = np.concatenate([coords, angles, velocities], axis=1).astype(np.float64)
    else:
        features = data

    if features is None:
        raise ValueError(f"Could not extract features for '{person_name}'")

    features = np.asarray(features, dtype=np.float64)
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


def load_video_features(features_file, video_key, feature_stats=None, seq_len=60):
    """Load pre-extracted 78-dim features for a video from the features pickle."""
    with open(features_file, 'rb') as f:
        all_features = pickle.load(f)

    if video_key not in all_features:
        raise ValueError(f"Video key '{video_key}' not found in features file")

    data = all_features[video_key]
    extractor = GaitFeatureExtractor.__new__(GaitFeatureExtractor)
    extractor.GAIT_LANDMARKS = GaitFeatureExtractor.GAIT_LANDMARKS

    pose_seq = data.get('pose_sequence', None)
    if pose_seq is not None:
        if pose_seq.shape[-1] == 3:
            vis = np.ones((*pose_seq.shape[:-1], 1))
            pose_seq = np.concatenate([pose_seq, vis], axis=-1)
        gait_features = extractor.compute_gait_features(pose_seq)
        coords = gait_features['normalized_coords'].reshape(pose_seq.shape[0], -1)
        angles = gait_features['joint_angles']
        velocities = gait_features['velocities'].reshape(pose_seq.shape[0], -1)
        features = np.concatenate([coords, angles, velocities], axis=1).astype(np.float64)
    else:
        features = data.get('features', data.get('gait_features'))
        if features is None:
            raise ValueError(f"No usable features for '{video_key}'")
        features = np.asarray(features, dtype=np.float64)

    # Normalize sequence length
    if features.shape[0] > seq_len:
        features = features[:seq_len]
    elif features.shape[0] < seq_len:
        repeats = seq_len // features.shape[0] + 1
        features = np.tile(features, (repeats, 1))[:seq_len]

    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    if feature_stats is not None:
        features = (features - feature_stats['mean']) / (feature_stats['std'] + 1e-8)
    return features


# ── Face blurring utility ────────────────────────────────────────────────────

def blur_face(frame, landmarks):
    """
    Blur the face region using MediaPipe pose landmarks.

    Args:
        frame: BGR frame (H, W, 3)
        landmarks: (33, 3) normalized landmarks [x, y, z]

    Returns:
        Frame with face blurred
    """
    h, w = frame.shape[:2]
    
    # Use nose (0) and shoulders (11, 12) to estimate face region
    nose = landmarks[0, :2]
    l_shoulder = landmarks[11, :2]
    r_shoulder = landmarks[12, :2]

    # Shoulder midpoint
    shoulder_mid = (l_shoulder + r_shoulder) / 2
    shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)

    # Face box: centered on nose, width ~ 0.8x shoulder width, extends upward
    face_radius = max(int(shoulder_width * w * 0.45), 30)

    cx = int(nose[0] * w)
    cy = int(nose[1] * h)

    x1 = max(0, cx - face_radius)
    y1 = max(0, cy - int(face_radius * 1.3))  # Extend up more for forehead
    x2 = min(w, cx + face_radius)
    y2 = min(h, cy + int(face_radius * 0.5))   # Slight below nose

    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (99, 99), 30)
        frame[y1:y2, x1:x2] = blurred

    return frame


def draw_skeleton(frame, landmarks, connections, color=(0, 255, 0),
                  keypoint_color=(0, 255, 0), thickness=2, radius=4,
                  gait_only=False):
    """
    Draw skeleton overlay on a frame.

    Args:
        frame: BGR frame
        landmarks: (33, 3) or (12, 3) normalized landmarks
        connections: List of (start, end) index pairs
        color: BGR bone color
        keypoint_color: BGR keypoint color  
        thickness: Line thickness
        radius: Keypoint circle radius
        gait_only: If True, landmarks are already gait-only (12 joints)
    """
    h, w = frame.shape[:2]

    # Draw bones
    for start, end in connections:
        try:
            pt1 = (int(landmarks[start, 0] * w), int(landmarks[start, 1] * h))
            pt2 = (int(landmarks[end, 0] * w), int(landmarks[end, 1] * h))
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
        except (IndexError, ValueError):
            continue

    # Draw keypoints
    n_points = landmarks.shape[0]
    indices = range(n_points) if gait_only else GAIT_LANDMARK_INDICES
    for i in indices:
        try:
            idx = i if gait_only else i
            if not gait_only and i not in GAIT_LANDMARK_INDICES:
                continue
            pt = (int(landmarks[idx, 0] * w), int(landmarks[idx, 1] * h))
            cv2.circle(frame, pt, radius, keypoint_color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
        except (IndexError, ValueError):
            continue

    return frame


# ── Figure 1: Skeleton Overlay Sequence ──────────────────────────────────────

def generate_skeleton_overlay(subjects, videos_dir, output_dir):
    """
    Generate skeleton overlay grid showing gait cycle for each subject.
    Picks side-view video, extracts 8 frames, overlays gait skeleton.
    """
    print("\n" + "=" * 60)
    print("  FIGURE 1: Skeleton Overlay Sequences")
    print("=" * 60)

    extractor = GaitFeatureExtractor()
    os.makedirs(output_dir, exist_ok=True)

    for subject in subjects:
        # Find a side-view video
        video_path = None
        for view in ['S1', 'S2', 'S3', 'F1', 'F2']:
            candidate = Path(videos_dir) / f"{subject}_{view}.mp4"
            if candidate.exists():
                video_path = str(candidate)
                break
        if video_path is None:
            print(f"  [SKIP] No video found for {subject}")
            continue

        print(f"  Processing {subject} ({Path(video_path).name})...")

        # Read video and extract frames + landmarks
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 8:
            print(f"  [SKIP] Too few frames ({total_frames})")
            cap.release()
            continue

        # Select 8 evenly-spaced frame indices
        frame_indices = np.linspace(0, total_frames - 1, 8).astype(int)
        frames = []
        all_landmarks = []

        for fidx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue

            landmarks = extractor.extract_pose_from_frame(frame)
            if landmarks is None:
                continue

            # Blur face before storing
            frame = blur_face(frame, landmarks)
            frames.append((fidx, frame, landmarks))

        cap.release()

        if len(frames) < 4:
            print(f"  [SKIP] Could not extract enough frames with poses for {subject}")
            continue

        # Create figure: 2 rows x 4 cols
        n_display = min(8, len(frames))
        n_cols = 4
        n_rows = (n_display + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for ax_idx in range(len(axes)):
            if ax_idx < n_display:
                fidx, frame, landmarks = frames[ax_idx]
                frame_draw = frame.copy()

                # Draw skeleton
                draw_skeleton(frame_draw, landmarks, SKELETON_CONNECTIONS,
                              color=(255, 255, 255), keypoint_color=(0, 255, 0),
                              thickness=2, radius=5)

                # Convert BGR→RGB for matplotlib
                axes[ax_idx].imshow(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB))
                time_s = fidx / fps if fps > 0 else 0
                axes[ax_idx].set_title(f'Frame {fidx} ({time_s:.2f}s)',
                                        fontsize=10, fontweight='bold')
            axes[ax_idx].axis('off')

        fig.suptitle(f'Gait Skeleton Overlay — {subject} (Side View)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'skeleton_overlay_{subject}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"    Saved: {save_path}")


# ── Figure 2: Authentic vs Deepfake Comparison ──────────────────────────────

def generate_auth_vs_fake(subject_pairs, videos_dir, enrolled_file,
                          features_file, model, feature_stats, device,
                          output_dir):
    """
    Generate side-by-side authentic vs deepfake (mismatched gait) comparison.
    Shows skeleton overlays with similarity scores from the model.
    """
    print("\n" + "=" * 60)
    print("  FIGURE 2: Authentic vs Deepfake Comparison")
    print("=" * 60)

    extractor = GaitFeatureExtractor()
    os.makedirs(output_dir, exist_ok=True)

    for subject_a, subject_b in subject_pairs:
        # Find a video for subject_a
        video_path = None
        for view in ['S1', 'S2', 'F1']:
            candidate = Path(videos_dir) / f"{subject_a}_{view}.mp4"
            if candidate.exists():
                video_path = str(candidate)
                break
        if video_path is None:
            print(f"  [SKIP] No video for {subject_a}")
            continue

        print(f"  Comparing {subject_a} vs {subject_b} ({Path(video_path).name})...")

        # Get similarity scores from model
        video_key = None
        with open(features_file, 'rb') as f:
            all_features = pickle.load(f)
        for key in all_features:
            stem = Path(key).stem
            if stem.startswith(subject_a) and ('S1' in stem or 'F1' in stem):
                video_key = key
                break
        if video_key is None:
            # Try any video for this subject
            for key in all_features:
                if Path(key).stem.startswith(subject_a + '_'):
                    video_key = key
                    break

        if video_key is None:
            print(f"  [SKIP] No features found for {subject_a}")
            continue

        try:
            video_feat = load_video_features(features_file, video_key, feature_stats)
            enrolled_a = load_enrolled_features(enrolled_file, subject_a, feature_stats)
            enrolled_b = load_enrolled_features(enrolled_file, subject_b, feature_stats)
        except Exception as e:
            print(f"  [SKIP] Error loading features: {e}")
            continue

        video_feat_dev = video_feat.to(device)
        enrolled_a_dev = enrolled_a.to(device)
        enrolled_b_dev = enrolled_b.to(device)

        is_auth_a, sim_a, conf_a = model.verify_identity(video_feat_dev, enrolled_a_dev)
        is_auth_b, sim_b, conf_b = model.verify_identity(video_feat_dev, enrolled_b_dev)

        # Extract frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_indices = np.linspace(0, total_frames - 1, 4).astype(int)

        frames_data = []
        for fidx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            landmarks = extractor.extract_pose_from_frame(frame)
            if landmarks is None:
                continue
            frame = blur_face(frame, landmarks)
            frames_data.append((fidx, frame, landmarks))
        cap.release()

        if len(frames_data) < 2:
            print(f"  [SKIP] Not enough frames with poses")
            continue

        n_frames = min(4, len(frames_data))

        # Create figure: 2 rows (authentic on top, fake on bottom) x n_frames columns
        fig, axes = plt.subplots(2, n_frames, figsize=(4.5 * n_frames, 9))
        if n_frames == 1:
            axes = axes.reshape(2, 1)

        for col in range(n_frames):
            fidx, frame, landmarks = frames_data[col]

            # Top row: Authentic (green skeleton)
            frame_auth = frame.copy()
            draw_skeleton(frame_auth, landmarks, SKELETON_CONNECTIONS,
                          color=(0, 200, 0), keypoint_color=(0, 255, 0),
                          thickness=3, radius=6)
            axes[0, col].imshow(cv2.cvtColor(frame_auth, cv2.COLOR_BGR2RGB))
            axes[0, col].axis('off')

            # Bottom row: Mismatched identity (red skeleton)
            frame_fake = frame.copy()
            draw_skeleton(frame_fake, landmarks, SKELETON_CONNECTIONS,
                          color=(0, 0, 200), keypoint_color=(0, 0, 255),
                          thickness=3, radius=6)
            axes[1, col].imshow(cv2.cvtColor(frame_fake, cv2.COLOR_BGR2RGB))
            axes[1, col].axis('off')

        # Row labels
        verdict_a = "AUTHENTIC" if is_auth_a else "DEEPFAKE"
        verdict_b = "AUTHENTIC" if is_auth_b else "DEEPFAKE"

        auth_color = '#2ecc71' if is_auth_a else '#e74c3c'
        fake_color = '#2ecc71' if is_auth_b else '#e74c3c'

        axes[0, 0].set_ylabel(
            f'{verdict_a}\nClaimed: {subject_a}\nSimilarity: {sim_a:.3f}',
            fontsize=11, fontweight='bold', color=auth_color,
            rotation=0, labelpad=120, va='center')
        axes[1, 0].set_ylabel(
            f'{verdict_b}\nClaimed: {subject_b}\nSimilarity: {sim_b:.3f}',
            fontsize=11, fontweight='bold', color=fake_color,
            rotation=0, labelpad=120, va='center')

        fig.suptitle(
            f'Gait Verification: {subject_a}\'s Video — '
            f'Matching Own Gait (top) vs {subject_b}\'s Gait (bottom)',
            fontsize=13, fontweight='bold', y=1.02)

        # Legend
        auth_patch = mpatches.Patch(color='#2ecc71', label=f'Claimed: {subject_a} (own identity)')
        fake_patch = mpatches.Patch(color='#e74c3c', label=f'Claimed: {subject_b} (wrong identity)')
        fig.legend(handles=[auth_patch, fake_patch], loc='lower center',
                   ncol=2, fontsize=11, frameon=True,
                   bbox_to_anchor=(0.5, -0.04))

        plt.tight_layout()
        save_path = os.path.join(output_dir,
                                 f'auth_vs_fake_{subject_a}_vs_{subject_b}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"    Saved: {save_path}")
        print(f"    {subject_a} self-match: sim={sim_a:.4f} ({verdict_a})")
        print(f"    {subject_a} vs {subject_b}: sim={sim_b:.4f} ({verdict_b})")


# ── Figure 3: Joint Angle Time-Series ────────────────────────────────────────

def generate_angle_timeseries(subjects, features_file, output_dir):
    """
    Compare joint angle trajectories across subjects over a gait cycle.
    Shows 6 angle time-series (2x3 grid) with one line per subject.
    """
    print("\n" + "=" * 60)
    print("  FIGURE 3: Joint Angle Time-Series Comparison")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    with open(features_file, 'rb') as f:
        all_features = pickle.load(f)

    extractor = GaitFeatureExtractor.__new__(GaitFeatureExtractor)
    extractor.GAIT_LANDMARKS = GaitFeatureExtractor.GAIT_LANDMARKS

    angle_names = ['Left Knee', 'Right Knee', 'Left Hip', 'Right Hip',
                   'Left Ankle', 'Right Ankle']
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(subjects), 3)))

    # Collect angle data per subject
    subject_angles = {}
    for subject in subjects:
        # Find a non-augmented video for this subject
        for key in all_features:
            stem = Path(key).stem
            if stem.startswith(subject + '_') and '_aug' not in stem.lower():
                data = all_features[key]
                pose_seq = data.get('pose_sequence', None)
                if pose_seq is not None:
                    if pose_seq.shape[-1] == 3:
                        vis = np.ones((*pose_seq.shape[:-1], 1))
                        pose_seq = np.concatenate([pose_seq, vis], axis=-1)
                    gait_feat = extractor.compute_gait_features(pose_seq)
                    subject_angles[subject] = gait_feat['joint_angles']  # (T, 6)
                    print(f"  Loaded angles for {subject} ({stem})")
                    break

    if len(subject_angles) < 2:
        print("  [SKIP] Need at least 2 subjects with angle data")
        return

    # Plot 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for angle_idx in range(6):
        ax = axes[angle_idx]
        for subj_idx, (subject, angles) in enumerate(subject_angles.items()):
            seq_len = angles.shape[0]
            # Normalize x-axis to percentage of gait cycle
            x_pct = np.linspace(0, 100, seq_len)
            ax.plot(x_pct, angles[:, angle_idx],
                    color=colors[subj_idx], linewidth=1.5,
                    label=subject, alpha=0.85)

        ax.set_title(angle_names[angle_idx], fontsize=12, fontweight='bold')
        ax.set_xlabel('Gait Cycle (%)', fontsize=10)
        ax.set_ylabel('Angle (degrees)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(subjects), 5),
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Joint Angle Trajectories Across Subjects',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'joint_angle_comparison.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {save_path}")


# ── Figure 4: t-SNE Embedding Space ─────────────────────────────────────────

def generate_embedding_tsne(features_file, model, feature_stats, device,
                            output_dir):
    """
    Visualize gait embedding space using t-SNE.
    Each subject's videos form a cluster; inter-subject separation is key.
    """
    print("\n" + "=" * 60)
    print("  FIGURE 4: t-SNE Embedding Space")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    with open(features_file, 'rb') as f:
        all_features = pickle.load(f)

    extractor = GaitFeatureExtractor.__new__(GaitFeatureExtractor)
    extractor.GAIT_LANDMARKS = GaitFeatureExtractor.GAIT_LANDMARKS

    embeddings = []
    labels = []
    person_names = []
    person_map = {}

    for key in sorted(all_features.keys()):
        stem = Path(key).stem
        # Skip augmented videos to reduce clutter
        # Augmented videos have names like Arhaan_S1_speed_0.8 or contain 'aug'
        if '_aug' in stem.lower():
            continue
        # Original videos are {Name}_{View} e.g. Arhaan_S1, Ananya_F2
        parts = stem.split('_')
        # Valid original: exactly 2 parts where part[1] is F1-F3 or S1-S3
        if len(parts) != 2 or parts[1] not in ('F1','F2','F3','S1','S2','S3'):
            continue
        person_name = parts[0]

        # Assign label index
        if person_name not in person_map:
            person_map[person_name] = len(person_names)
            person_names.append(person_name)

        data = all_features[key]
        pose_seq = data.get('pose_sequence', None)
        if pose_seq is None:
            continue

        try:
            if pose_seq.shape[-1] == 3:
                vis = np.ones((*pose_seq.shape[:-1], 1))
                pose_seq = np.concatenate([pose_seq, vis], axis=-1)
            gait_feat = extractor.compute_gait_features(pose_seq)
            coords = gait_feat['normalized_coords'].reshape(pose_seq.shape[0], -1)
            angles = gait_feat['joint_angles']
            velocities = gait_feat['velocities'].reshape(pose_seq.shape[0], -1)
            feat_78 = np.concatenate([coords, angles, velocities], axis=1).astype(np.float32)

            # Normalize length to 60
            if feat_78.shape[0] > 60:
                feat_78 = feat_78[:60]
            elif feat_78.shape[0] < 60:
                repeats = 60 // feat_78.shape[0] + 1
                feat_78 = np.tile(feat_78, (repeats, 1))[:60]

            feat_tensor = torch.tensor(feat_78, dtype=torch.float32).unsqueeze(0)
            if feature_stats is not None:
                feat_tensor = (feat_tensor - feature_stats['mean']) / (feature_stats['std'] + 1e-8)

            feat_tensor = feat_tensor.to(device)
            emb = model.get_embedding(feat_tensor)  # (1, 128)
            embeddings.append(emb.cpu().numpy().squeeze(0))
            labels.append(person_map[person_name])
        except Exception as e:
            continue

    if len(embeddings) < 5:
        print("  [SKIP] Not enough embeddings for t-SNE")
        return

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    print(f"  Computing t-SNE for {len(embeddings)} samples, "
          f"{len(person_names)} subjects...")

    from sklearn.manifold import TSNE
    perplexity = min(15, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                max_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    cmap = plt.cm.tab20 if len(unique_labels) > 10 else plt.cm.tab10
    colors = cmap(np.linspace(0, 1, len(unique_labels)))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'p', '<', '>', '8']

    for i, label in enumerate(unique_labels):
        mask = labels == label
        marker = markers[i % len(markers)]
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[colors[i]], marker=marker,
                   label=person_names[label], s=80, alpha=0.8,
                   edgecolors='white', linewidth=0.5)

        # Annotate centroid
        centroid = emb_2d[mask].mean(axis=0)
        ax.annotate(person_names[label], centroid,
                    fontsize=8, fontweight='bold', alpha=0.7,
                    ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points')

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Gait Embedding Space — Inter-Subject Separability',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
              frameon=True, ncol=1)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'embedding_tsne.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {save_path}")


# ── Figure 5: Joint Importance on Canonical Skeleton ─────────────────────────

def generate_importance_skeleton(features_file, enrolled_file, model,
                                 feature_stats, device, output_dir,
                                 n_samples=20):
    """
    Draw a canonical stick figure with joints colored by GradCAM importance.
    Aggregates importance across multiple samples.
    """
    print("\n" + "=" * 60)
    print("  FIGURE 5: Joint Importance Skeleton")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    with open(features_file, 'rb') as f:
        all_features = pickle.load(f)
    with open(enrolled_file, 'rb') as f:
        enrolled = pickle.load(f)

    extractor = GaitFeatureExtractor.__new__(GaitFeatureExtractor)
    extractor.GAIT_LANDMARKS = GaitFeatureExtractor.GAIT_LANDMARKS

    analyzer = JointImportanceAnalyzer(model)

    # Group videos by person
    person_videos = defaultdict(list)
    for key in all_features:
        stem = Path(key).stem
        if '_aug' in stem.lower():
            continue
        parts = stem.rsplit('_', 1)
        person_name = parts[0] if len(parts) > 1 else stem
        if person_name in enrolled:
            person_videos[person_name].append(key)

    # Collect joint importance across samples
    all_joint_imp = []
    all_angle_imp = []
    count = 0

    for person_name, video_keys in person_videos.items():
        if count >= n_samples:
            break
        for vk in video_keys[:2]:  # Max 2 videos per person
            if count >= n_samples:
                break
            try:
                video_feat = load_video_features(features_file, vk, feature_stats)
                claimed_feat = load_enrolled_features(enrolled_file, person_name,
                                                       feature_stats)
                video_feat_dev = video_feat.to(device)
                claimed_feat_dev = claimed_feat.to(device)

                result = analyzer.compute_joint_importance(
                    video_feat_dev, claimed_feat_dev, mode='verification')

                all_joint_imp.append(result['joint_importance'])
                all_angle_imp.append(result['angle_importance'])
                count += 1
            except Exception as e:
                continue

    if count == 0:
        print("  [SKIP] Could not compute any joint importance")
        return

    print(f"  Aggregated importance over {count} samples")

    # Average and normalize
    avg_joint = np.mean(all_joint_imp, axis=0)   # (12,)
    avg_angle = np.mean(all_angle_imp, axis=0)   # (6,)

    # Normalize joint importance to 0-1
    if avg_joint.max() > avg_joint.min():
        norm_joint = (avg_joint - avg_joint.min()) / (avg_joint.max() - avg_joint.min())
    else:
        norm_joint = np.ones_like(avg_joint) * 0.5

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(1.0, 0.15)  # Inverted: head at top
    ax.set_aspect('equal')
    ax.axis('off')

    # Color map
    cmap = plt.cm.RdYlGn_r  # Red=important, Green=less important
    norm = Normalize(vmin=0, vmax=1)

    # Draw head (small circle above shoulders)
    head_center = (0.50, 0.18)
    head_circle = plt.Circle(head_center, 0.04, fill=True,
                              facecolor='#cccccc', edgecolor='#666666',
                              linewidth=1.5, zorder=5)
    ax.add_patch(head_circle)
    # Neck line
    ax.plot([0.50, 0.50], [0.22, 0.25], color='#888888', linewidth=2, zorder=3)

    # Draw bones
    for name_a, name_b in CANONICAL_BONES:
        pos_a = CANONICAL_SKELETON[name_a]
        pos_b = CANONICAL_SKELETON[name_b]
        idx_a = GAIT_LANDMARK_NAMES.index(name_a)
        idx_b = GAIT_LANDMARK_NAMES.index(name_b)
        avg_imp = (norm_joint[idx_a] + norm_joint[idx_b]) / 2
        bone_color = cmap(norm(avg_imp))
        ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]],
                color=bone_color, linewidth=3.5, zorder=3, solid_capstyle='round')

    # Draw joints
    for i, name in enumerate(GAIT_LANDMARK_NAMES):
        pos = CANONICAL_SKELETON[name]
        imp = norm_joint[i]
        color = cmap(norm(imp))
        size = 150 + 350 * imp  # Radius proportional to importance

        ax.scatter(pos[0], pos[1], c=[color], s=size, zorder=6,
                   edgecolors='white', linewidths=1.5)

        # Label with name and value
        offset_x = 0.08 if 'L_' in name else -0.08
        ha = 'left' if 'L_' in name else 'right'
        ax.annotate(f'{name}\n({imp:.2f})',
                    xy=pos, fontsize=8, fontweight='bold',
                    ha=ha, va='center',
                    xytext=(offset_x, 0), textcoords='offset fontsize',
                    color='#333333')

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, aspect=30)
    cbar.set_label('Normalized Importance', fontsize=11)

    ax.set_title('Joint Importance for Deepfake Detection\n'
                 '(Aggregated Gradient Attribution)',
                 fontsize=14, fontweight='bold', pad=20)

    save_path = os.path.join(output_dir, 'joint_importance_skeleton.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate gait keypoint matching visualizations for research paper')
    parser.add_argument('--videos_dir', type=str, default='data/videos',
                        help='Directory with video files')
    parser.add_argument('--features_file', type=str,
                        default='data/gait_features/gait_features.pkl',
                        help='Pre-extracted features file')
    parser.add_argument('--enrolled_file', type=str,
                        default='data/gait_features/enrolled_identities.pkl',
                        help='Enrolled identities file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path (auto-detect if omitted)')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/keypoint_visualizations',
                        help='Output directory for figures')
    parser.add_argument('--subjects', type=str,
                        default='Arhaan,Ananya,Som,Teja,Vedant',
                        help='Comma-separated list of subjects')
    parser.add_argument('--figures', type=str, default='1,2,3,4,5',
                        help='Comma-separated figure numbers to generate (1-5)')
    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(',')]
    figures = [int(f.strip()) for f in args.figures.split(',')]

    print("=" * 60)
    print("  GAIT KEYPOINT MATCHING VISUALIZATIONS")
    print("=" * 60)
    print(f"  Subjects: {subjects}")
    print(f"  Figures:  {figures}")
    print(f"  Output:   {args.output_dir}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device:   {device}")
    if device.type == 'cuda':
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")

    # Load model (needed for Figures 2, 4, 5)
    model = None
    feature_stats = None
    needs_model = bool(set(figures) & {2, 4, 5})

    if needs_model:
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            ckpt_path = find_best_checkpoint('outputs/checkpoints')
        model, feature_stats = load_model_and_stats(ckpt_path, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate figures
    if 1 in figures:
        generate_skeleton_overlay(subjects, args.videos_dir, args.output_dir)

    if 2 in figures:
        # Create subject pairs for comparison
        pairs = []
        if len(subjects) >= 2:
            pairs.append((subjects[0], subjects[1]))
        if len(subjects) >= 4:
            pairs.append((subjects[2], subjects[3]))
        if len(subjects) >= 5:
            pairs.append((subjects[0], subjects[4]))

        generate_auth_vs_fake(pairs, args.videos_dir, args.enrolled_file,
                              args.features_file, model, feature_stats,
                              device, args.output_dir)

    if 3 in figures:
        generate_angle_timeseries(subjects, args.features_file, args.output_dir)

    if 4 in figures:
        generate_embedding_tsne(args.features_file, model, feature_stats,
                                device, args.output_dir)

    if 5 in figures:
        generate_importance_skeleton(args.features_file, args.enrolled_file,
                                     model, feature_stats, device,
                                     args.output_dir, n_samples=20)

    print("\n" + "=" * 60)
    print("  ALL FIGURES COMPLETE")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
