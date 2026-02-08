"""
GPU-Accelerated Pose Extraction using TensorFlow MoveNet
=========================================================
Alternative to MediaPipe that runs on GPU via TensorFlow.
NOTE: MediaPipe (pose_extraction.py) is preferred for gait analysis
      due to better angular accuracy. This file is kept for reference only.

Author: DeepFake Detection Project
"""

import os
import numpy as np
import cv2
from pathlib import Path

# Lazy-load TensorFlow to avoid initializing GPU when this module is imported
# but not used. TF is only needed if MoveNetExtractor is instantiated.
_tf = None
_tf_initialized = False


def _init_tensorflow():
    """Initialize TensorFlow lazily on first use."""
    global _tf, _tf_initialized
    if _tf_initialized:
        return _tf
    import tensorflow as tf
    _tf = tf
    _tf_initialized = True
    print("=" * 60)
    print("DEVICE CHECK (TensorFlow MoveNet)")
    print("=" * 60)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU Available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found - running on CPU")
    print("=" * 60)
    return tf


class MoveNetExtractor:
    """
    GPU-accelerated pose extraction using TensorFlow MoveNet.
    MoveNet detects 17 keypoints (COCO format).
    """
    
    # MoveNet keypoint indices
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Gait-relevant keypoints (lower body + torso)
    GAIT_INDICES = [5, 6, 11, 12, 13, 14, 15, 16]  # shoulders, hips, knees, ankles
    
    def __init__(self, model_name: str = 'thunder', sequence_length: int = 60):
        """
        Initialize MoveNet model.
        
        Args:
            model_name: 'lightning' (faster) or 'thunder' (more accurate)
            sequence_length: Number of frames to extract per video
        """
        # Lazy-init TensorFlow only when MoveNetExtractor is actually used
        tf = _init_tensorflow()
        
        self.sequence_length = sequence_length
        self.model_name = model_name
        
        # Load model from TensorFlow Hub
        print(f"Loading MoveNet {model_name} model...")
        
        if model_name == 'lightning':
            model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
            self.input_size = 192
        else:  # thunder
            model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
            self.input_size = 256
        
        # Load model
        self.model = tf.saved_model.load(
            tf.keras.utils.get_file(
                f'movenet_{model_name}',
                f'{model_url}?tf-hub-format=compressed',
                cache_subdir='models',
                untar=True
            )
        )
        self.movenet = self.model.signatures['serving_default']
        
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✓ MoveNet {model_name} loaded (input size: {self.input_size}x{self.input_size})")
        print(f"  Running on: {'GPU' if gpus else 'CPU'}")
    
    def preprocess_frame(self, frame: np.ndarray):
        """Preprocess frame for MoveNet."""
        tf = _tf
        # Resize and pad to square
        img = tf.image.resize_with_pad(
            tf.expand_dims(frame, axis=0),
            self.input_size,
            self.input_size
        )
        img = tf.cast(img, dtype=tf.int32)
        return img
    
    def detect_pose(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect pose in a single frame.
        
        Returns:
            keypoints: Array of shape (17, 3) with [y, x, confidence]
        """
        input_tensor = self.preprocess_frame(frame)
        outputs = self.movenet(input_tensor)
        keypoints = outputs['output_0'].numpy()[0, 0]  # Shape: (17, 3)
        return keypoints
    
    def extract_gait_features(self, keypoints_sequence: np.ndarray) -> dict:
        """
        Extract gait-specific features from keypoint sequence.
        
        Args:
            keypoints_sequence: Array of shape (T, 17, 3)
            
        Returns:
            Dictionary with gait features
        """
        T = len(keypoints_sequence)
        
        # Extract gait keypoints (8 points: shoulders, hips, knees, ankles)
        gait_kps = keypoints_sequence[:, self.GAIT_INDICES, :2]  # (T, 8, 2)
        
        # Normalize to hip center
        left_hip = keypoints_sequence[:, 11, :2]
        right_hip = keypoints_sequence[:, 12, :2]
        hip_center = (left_hip + right_hip) / 2  # (T, 2)
        
        normalized_coords = gait_kps - hip_center[:, np.newaxis, :]
        
        # Compute joint angles
        angles = self._compute_joint_angles(keypoints_sequence)
        
        # Compute velocities
        velocities = np.diff(normalized_coords, axis=0)
        velocities = np.concatenate([velocities, velocities[-1:]], axis=0)
        
        # Compute accelerations
        accelerations = np.diff(velocities, axis=0)
        accelerations = np.concatenate([accelerations, accelerations[-1:]], axis=0)
        
        return {
            'normalized_coords': normalized_coords.astype(np.float32),
            'joint_angles': angles.astype(np.float32),
            'velocities': velocities.astype(np.float32),
            'accelerations': accelerations.astype(np.float32)
        }
    
    def _compute_joint_angles(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute joint angles for gait analysis."""
        T = len(keypoints)
        
        def angle_between(p1, p2, p3):
            """Compute angle at p2 formed by p1-p2-p3."""
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.sum(v1 * v2, axis=-1) / (
                np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
            )
            return np.arccos(np.clip(cos_angle, -1, 1))
        
        angles = []
        
        # Left knee angle (hip-knee-ankle)
        left_knee = angle_between(
            keypoints[:, 11, :2],  # left hip
            keypoints[:, 13, :2],  # left knee
            keypoints[:, 15, :2]   # left ankle
        )
        angles.append(left_knee)
        
        # Right knee angle
        right_knee = angle_between(
            keypoints[:, 12, :2],  # right hip
            keypoints[:, 14, :2],  # right knee
            keypoints[:, 16, :2]   # right ankle
        )
        angles.append(right_knee)
        
        # Left hip angle (shoulder-hip-knee)
        left_hip = angle_between(
            keypoints[:, 5, :2],   # left shoulder
            keypoints[:, 11, :2],  # left hip
            keypoints[:, 13, :2]   # left knee
        )
        angles.append(left_hip)
        
        # Right hip angle
        right_hip = angle_between(
            keypoints[:, 6, :2],   # right shoulder
            keypoints[:, 12, :2],  # right hip
            keypoints[:, 14, :2]   # right knee
        )
        angles.append(right_hip)
        
        return np.stack(angles, axis=1)  # (T, 4)
    
    def process_video(self, video_path: str) -> dict:
        """
        Process video and extract gait features.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with pose sequence and gait features
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames uniformly
        if total_frames <= self.sequence_length:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, 
                                        self.sequence_length, dtype=int)
        
        keypoints_list = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect pose
            keypoints = self.detect_pose(rgb_frame)
            keypoints_list.append(keypoints)
        
        cap.release()
        
        if len(keypoints_list) < 10:
            print(f"Warning: Only {len(keypoints_list)} frames extracted")
            return None
        
        # Convert to numpy array
        pose_sequence = np.array(keypoints_list)
        
        # Pad or truncate to sequence_length
        if len(pose_sequence) < self.sequence_length:
            pad_length = self.sequence_length - len(pose_sequence)
            pose_sequence = np.pad(
                pose_sequence,
                ((0, pad_length), (0, 0), (0, 0)),
                mode='edge'
            )
        else:
            pose_sequence = pose_sequence[:self.sequence_length]
        
        # Extract gait features
        gait_features = self.extract_gait_features(pose_sequence)
        
        return {
            'pose_sequence': pose_sequence,
            'gait_features': gait_features,
            'frame_count': len(keypoints_list),
            'video_fps': fps
        }


def test_gpu():
    """Test if TensorFlow is using GPU."""
    tf = _init_tensorflow()
    print("\nTesting TensorFlow GPU...")
    
    gpus = tf.config.list_physical_devices('GPU')
    # Simple matrix multiplication to test GPU
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
    
    print(f"Matrix multiplication completed on: {'GPU' if gpus else 'CPU'}")
    return len(gpus) > 0


if __name__ == "__main__":
    # Test GPU
    has_gpu = test_gpu()
    
    # Test pose extraction
    print("\nTesting MoveNet pose extraction...")
    extractor = MoveNetExtractor(model_name='thunder', sequence_length=60)
    
    # Test on a sample video if available
    test_video = "data/videos/Arhaan_F.mp4"
    if os.path.exists(test_video):
        print(f"\nProcessing: {test_video}")
        result = extractor.process_video(test_video)
        if result:
            print(f"✓ Pose sequence shape: {result['pose_sequence'].shape}")
            print(f"✓ Gait coords shape: {result['gait_features']['normalized_coords'].shape}")
            print(f"✓ Joint angles shape: {result['gait_features']['joint_angles'].shape}")
    else:
        print(f"Test video not found: {test_video}")
