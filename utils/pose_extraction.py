"""
Gait Feature Extraction using MediaPipe Pose
=============================================
Extracts skeletal pose keypoints from videos for gait analysis.

This module:
1. Processes videos frame by frame
2. Extracts 33 body landmarks using MediaPipe
3. Computes gait-specific features (angles, velocities, etc.)
4. Saves normalized pose sequences for training

Author: DeepFake Detection Project
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import pickle
from typing import Dict, List, Optional

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class GaitFeatureExtractor:
    """
    Extracts gait features from video using MediaPipe Pose.
    """
    
    # Key body landmarks for gait analysis
    # MediaPipe Pose has 33 landmarks
    POSE_LANDMARKS = {
        'nose': 0,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
        'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32
    }
    
    # Landmarks most important for gait (lower body focus)
    GAIT_LANDMARKS = [
        11, 12,  # Shoulders (for upper body sway)
        23, 24,  # Hips
        25, 26,  # Knees
        27, 28,  # Ankles
        29, 30,  # Heels
        31, 32   # Foot tips
    ]
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 sequence_length: int = 60):
        """
        Initialize the gait feature extractor.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            sequence_length: Fixed sequence length for output (frames)
        """
        # Download model if needed
        model_path = self._get_pose_model(min_detection_confidence, min_tracking_confidence)
        
        # Create PoseLandmarker with options
        # Use IMAGE mode for frame-by-frame processing via detect()
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.sequence_length = sequence_length
    
    def _get_pose_model(self, min_det_conf: float, min_track_conf: float) -> str:
        """Get or download the MediaPipe Pose model."""
        # Use a standard model path — MediaPipe handles downloading
        import os
        
        model_dir = os.path.expanduser("~/.mediapipe/models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "pose_landmarker_lite.task")
        
        # If model doesn't exist, download it
        if not os.path.exists(model_path):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            print(f"Downloading MediaPipe Pose model to {model_path}...")
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"✓ Model downloaded to {model_path}")
            except Exception as e:
                print(f"Failed to download model: {e}")
                print("Using default model location...")
                return model_path
        
        return model_path
    
    def extract_pose_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose landmarks from a single frame.
        
        Args:
            frame: Input frame (BGR, from OpenCV)
            
        Returns:
            Landmarks array of shape (33, 3) with [x, y, z] coordinates
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create Image object for MediaPipe
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect pose
        detection_result = self.landmarker.detect(image)
        
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            return None
        
        # Extract landmarks (take first detected person)
        pose_landmarks = detection_result.pose_landmarks[0]
        
        # Convert to numpy array
        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in pose_landmarks
        ])
        
        return landmarks
    
    def extract_gait_landmarks(self, full_landmarks: np.ndarray) -> np.ndarray:
        """
        Extract only gait-relevant landmarks.
        
        Args:
            full_landmarks: Full 33 landmarks array
            
        Returns:
            Gait landmarks array (12, 4)
        """
        return full_landmarks[self.GAIT_LANDMARKS]
    
    def compute_joint_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Compute key joint angles for gait analysis.
        
        Args:
            landmarks: Full pose landmarks (33, 4)
            
        Returns:
            Dictionary of joint angles in degrees
        """
        def angle_between_points(p1, p2, p3):
            """Compute angle at p2 formed by p1-p2-p3"""
            v1 = p1[:2] - p2[:2]
            v2 = p3[:2] - p2[:2]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        
        angles = {}
        
        # Left knee angle (hip-knee-ankle)
        angles['left_knee'] = angle_between_points(
            landmarks[23], landmarks[25], landmarks[27]
        )
        
        # Right knee angle
        angles['right_knee'] = angle_between_points(
            landmarks[24], landmarks[26], landmarks[28]
        )
        
        # Left hip angle (shoulder-hip-knee)
        angles['left_hip'] = angle_between_points(
            landmarks[11], landmarks[23], landmarks[25]
        )
        
        # Right hip angle
        angles['right_hip'] = angle_between_points(
            landmarks[12], landmarks[24], landmarks[26]
        )
        
        # Left ankle angle (knee-ankle-foot)
        angles['left_ankle'] = angle_between_points(
            landmarks[25], landmarks[27], landmarks[31]
        )
        
        # Right ankle angle
        angles['right_ankle'] = angle_between_points(
            landmarks[26], landmarks[28], landmarks[32]
        )
        
        return angles
    
    def compute_gait_features(self, pose_sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute comprehensive gait features from pose sequence.
        
        Args:
            pose_sequence: Array of shape (T, 33, 4) - temporal pose sequence
            
        Returns:
            Dictionary containing various gait features
        """
        T = len(pose_sequence)
        features = {}
        
        # 1. Raw normalized coordinates (gait landmarks only)
        gait_coords = np.array([
            self.extract_gait_landmarks(pose)[:, :3]  # x, y, z only
            for pose in pose_sequence
        ])  # Shape: (T, 12, 3)
        
        # Normalize to hip center
        hip_center = (pose_sequence[:, 23, :3] + pose_sequence[:, 24, :3]) / 2
        gait_coords_normalized = gait_coords - hip_center[:, np.newaxis, :]
        features['normalized_coords'] = gait_coords_normalized
        
        # 2. Joint angles over time
        angles_sequence = []
        for pose in pose_sequence:
            angles = self.compute_joint_angles(pose)
            angles_sequence.append(list(angles.values()))
        features['joint_angles'] = np.array(angles_sequence)  # (T, 6)
        
        # 3. Velocities (first derivative)
        velocities = np.diff(gait_coords_normalized, axis=0)
        # Pad to maintain length
        velocities = np.concatenate([velocities, velocities[-1:]], axis=0)
        features['velocities'] = velocities  # (T, 12, 3)
        
        # 4. Accelerations (second derivative)
        accelerations = np.diff(velocities, axis=0)
        accelerations = np.concatenate([accelerations, accelerations[-1:]], axis=0)
        features['accelerations'] = accelerations  # (T, 12, 3)
        
        # 5. Step-related features
        left_ankle = pose_sequence[:, 27, :2]
        right_ankle = pose_sequence[:, 28, :2]
        step_width = np.linalg.norm(left_ankle - right_ankle, axis=1)
        features['step_width'] = step_width  # (T,)
        
        # 6. Body symmetry (left-right comparison)
        left_side = gait_coords_normalized[:, [0, 2, 4, 6, 8, 10], :]  # Left landmarks
        right_side = gait_coords_normalized[:, [1, 3, 5, 7, 9, 11], :]  # Right landmarks
        symmetry = np.abs(left_side[:, :, 0] + right_side[:, :, 0])  # X-coord symmetry
        features['symmetry'] = symmetry.mean(axis=1)  # (T,)
        
        return features
    
    def process_video(self, video_path: str, 
                      use_all_landmarks: bool = True) -> Optional[Dict]:
        """
        Process a video and extract gait features.
        
        Args:
            video_path: Path to video file
            use_all_landmarks: If True, use all 33 landmarks; else use gait-specific
            
        Returns:
            Dictionary with pose sequence and computed features,
            or None if processing fails
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pose_sequence = []
        frame_indices = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.extract_pose_from_frame(frame)
            
            if landmarks is not None:
                pose_sequence.append(landmarks)
                frame_indices.append(frame_idx)
            
            frame_idx += 1
        
        cap.release()
        
        if len(pose_sequence) < 10:
            print(f"Warning: Too few valid poses ({len(pose_sequence)}) in {video_path}")
            return None
        
        pose_sequence = np.array(pose_sequence)
        
        # Normalize sequence length
        pose_sequence = self._normalize_sequence_length(pose_sequence)
        
        # Compute gait features
        gait_features = self.compute_gait_features(pose_sequence)
        
        return {
            'video_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'valid_frames': len(frame_indices),
            'pose_sequence': pose_sequence,  # (T, 33, 4)
            'gait_features': gait_features,
            'frame_indices': frame_indices
        }
    
    def _normalize_sequence_length(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normalize sequence to fixed length via interpolation or sampling.
        
        Args:
            sequence: Input sequence of shape (T, ...)
            
        Returns:
            Normalized sequence of shape (sequence_length, ...)
        """
        T = len(sequence)
        
        if T == self.sequence_length:
            return sequence
        
        # Create interpolation indices
        old_indices = np.linspace(0, T - 1, T)
        new_indices = np.linspace(0, T - 1, self.sequence_length)
        
        # Interpolate for each dimension
        original_shape = sequence.shape[1:]
        sequence_flat = sequence.reshape(T, -1)
        
        interpolated = np.zeros((self.sequence_length, sequence_flat.shape[1]))
        for i in range(sequence_flat.shape[1]):
            interpolated[:, i] = np.interp(new_indices, old_indices, sequence_flat[:, i])
        
        return interpolated.reshape(self.sequence_length, *original_shape)
    
    def close(self):
        """Release MediaPipe resources."""
        # PoseLandmarker handles cleanup automatically with new API
        pass


def extract_features_from_dataset(input_dir: str, 
                                   output_dir: str,
                                   sequence_length: int = 60):
    """
    Extract gait features from all videos in a directory.
    
    Args:
        input_dir: Directory containing videos
        output_dir: Directory to save extracted features
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all videos
    video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi"))
    
    if not video_files:
        print(f"No videos found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} videos")
    print(f"Sequence length: {sequence_length} frames")
    print("-" * 50)
    
    # Initialize extractor
    extractor = GaitFeatureExtractor(sequence_length=sequence_length)
    
    # Process videos
    successful = 0
    failed = 0
    all_features = {}
    
    for video_file in tqdm(video_files, desc="Extracting gait features"):
        try:
            result = extractor.process_video(str(video_file))
            
            if result is not None:
                # Parse person name and view from filename
                # Format: PersonName_View_Aug.mp4 or PersonName_View.mp4
                name_parts = video_file.stem.split('_')
                person_name = name_parts[0]
                view = name_parts[1] if len(name_parts) > 1 else 'unknown'
                
                result['person_name'] = person_name
                result['view'] = view
                result['is_augmented'] = len(name_parts) > 2
                
                # Store with video filename as key
                all_features[video_file.stem] = result
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"\nError processing {video_file}: {e}")
            failed += 1
    
    extractor.close()
    
    # Save all features
    output_file = output_path / "gait_features.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_features, f)
    
    # Also save a summary JSON
    summary = {
        'total_videos': len(video_files),
        'successful': successful,
        'failed': failed,
        'sequence_length': sequence_length,
        'persons': list(set(f['person_name'] for f in all_features.values())),
        'feature_keys': list(all_features[list(all_features.keys())[0]]['gait_features'].keys()) if all_features else []
    }
    
    with open(output_path / "extraction_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 50)
    print("FEATURE EXTRACTION COMPLETE!")
    print("=" * 50)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Persons found: {summary['persons']}")
    print(f"Features saved to: {output_file}")


def create_identity_database(features_file: str, output_file: str):
    """
    Create enrolled identity database from extracted features.
    Groups features by person and creates gait signatures.
    
    Args:
        features_file: Path to gait_features.pkl
        output_file: Path to save identity database
    """
    with open(features_file, 'rb') as f:
        all_features = pickle.load(f)
    
    # Group by person
    person_features = {}
    for video_name, features in all_features.items():
        person = features['person_name']
        if person not in person_features:
            person_features[person] = []
        
        # Only use non-augmented videos for enrollment (original data)
        if not features['is_augmented']:
            person_features[person].append(features)
    
    # Create enrollment signatures (average of pose sequences)
    enrolled_identities = {}
    for person, features_list in person_features.items():
        if not features_list:
            continue
            
        # Average the pose sequences
        pose_sequences = [f['pose_sequence'] for f in features_list]
        avg_pose = np.mean(pose_sequences, axis=0)
        
        # Average joint angles
        angle_sequences = [f['gait_features']['joint_angles'] for f in features_list]
        avg_angles = np.mean(angle_sequences, axis=0)
        
        enrolled_identities[person] = {
            'avg_pose_sequence': avg_pose,
            'avg_joint_angles': avg_angles,
            'num_samples': len(features_list),
            'views': [f['view'] for f in features_list]
        }
    
    with open(output_file, 'wb') as f:
        pickle.dump(enrolled_identities, f)
    
    print(f"Enrolled {len(enrolled_identities)} identities")
    for person, data in enrolled_identities.items():
        print(f"  - {person}: {data['num_samples']} samples, views: {data['views']}")


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "data/augmented_videos"  # Use augmented videos
    OUTPUT_DIR = "data/gait_features"
    SEQUENCE_LENGTH = 60  # Normalize all videos to 60 frames
    
    print("=" * 50)
    print("GAIT FEATURE EXTRACTION")
    print("=" * 50)
    
    # Extract features
    extract_features_from_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        sequence_length=SEQUENCE_LENGTH
    )
    
    # Create identity database
    print("\nCreating identity database...")
    create_identity_database(
        features_file=f"{OUTPUT_DIR}/gait_features.pkl",
        output_file=f"{OUTPUT_DIR}/enrolled_identities.pkl"
    )
    
    print("\n✓ Gait feature extraction complete!")
