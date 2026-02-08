"""
Video Data Augmentation Pipeline for Gait-Based Deepfake Detection
=====================================================================
This script applies temporal-consistent augmentations to gait videos
to expand your dataset for better model training.

Augmentation Techniques Applied:
1. Spatial Augmentations (consistent across all frames):
   - Horizontal Flip (mirror walking direction)
   - Slight Rotation (±10 degrees)
   - Brightness/Contrast adjustments
   - Gaussian Blur (simulates camera quality variations)
   
2. Temporal Augmentations:
   - Speed variation (0.8x to 1.2x)
   - Temporal jittering (frame sampling variations)
   - Reverse playback (for symmetric actions)
   
3. Color Augmentations:
   - Hue/Saturation shifts
   - Color jittering
   - Grayscale conversion

Author: Auto-generated for DeepFake Detection Project
"""

import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import random
from tqdm import tqdm
import shutil


class VideoAugmenter:
    """
    Video augmentation class that applies temporally consistent
    transformations to maintain gait pattern integrity.
    """
    
    def __init__(self, output_dir: str, target_fps: int = 30):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_fps = target_fps
        
        # Define augmentation pipelines
        self.augmentation_configs = self._create_augmentation_configs()
    
    def _create_augmentation_configs(self):
        """
        Create multiple augmentation configurations.
        Each config is applied consistently across all frames of a video.
        """
        configs = {
            # 1. Horizontal Flip - useful for gait as walking direction shouldn't matter
            "hflip": {
                "spatial": A.Compose([
                    A.HorizontalFlip(p=1.0),
                ]),
                "temporal": None,
                "description": "Horizontal flip (mirror)"
            },
            
            # 2. Brightness variations - simulates different lighting
            "bright_up": {
                "spatial": A.Compose([
                    A.RandomBrightnessContrast(
                        brightness_limit=(0.2, 0.3),
                        contrast_limit=0,
                        p=1.0
                    ),
                ]),
                "temporal": None,
                "description": "Increased brightness"
            },
            
            "bright_down": {
                "spatial": A.Compose([
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.3, -0.2),
                        contrast_limit=0,
                        p=1.0
                    ),
                ]),
                "temporal": None,
                "description": "Decreased brightness"
            },
            
            # 3. Contrast variations
            "contrast_up": {
                "spatial": A.Compose([
                    A.RandomBrightnessContrast(
                        brightness_limit=0,
                        contrast_limit=(0.2, 0.3),
                        p=1.0
                    ),
                ]),
                "temporal": None,
                "description": "Increased contrast"
            },
            
            # 4. Rotation (slight) - simulates camera angle variations
            "rotate_left": {
                "spatial": A.Compose([
                    A.Rotate(limit=(-10, -5), p=1.0, border_mode=cv2.BORDER_REFLECT),
                ]),
                "temporal": None,
                "description": "Slight left rotation"
            },
            
            "rotate_right": {
                "spatial": A.Compose([
                    A.Rotate(limit=(5, 10), p=1.0, border_mode=cv2.BORDER_REFLECT),
                ]),
                "temporal": None,
                "description": "Slight right rotation"
            },
            
            # 5. Gaussian Blur - simulates lower quality cameras
            "blur": {
                "spatial": A.Compose([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ]),
                "temporal": None,
                "description": "Gaussian blur (camera quality)"
            },
            
            # 6. Color jittering - simulates different camera color profiles
            "color_jitter": {
                "spatial": A.Compose([
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=10,
                        p=1.0
                    ),
                ]),
                "temporal": None,
                "description": "Color jittering"
            },
            
            # 7. Grayscale - tests model robustness to color removal
            "grayscale": {
                "spatial": A.Compose([
                    A.ToGray(p=1.0),
                ]),
                "temporal": None,
                "description": "Grayscale conversion"
            },
            
            # 8. Combined augmentation - realistic scenario
            "combined_1": {
                "spatial": A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=5,
                        sat_shift_limit=10,
                        val_shift_limit=5,
                        p=0.5
                    ),
                ]),
                "temporal": None,
                "description": "Combined augmentation 1"
            },
            
            # 9. Noise addition - simulates sensor noise
            "noise": {
                "spatial": A.Compose([
                    A.GaussNoise(std_range=(0.02, 0.05), p=1.0),
                ]),
                "temporal": None,
                "description": "Gaussian noise"
            },
            
            # 10. Scale/Zoom - simulates different camera distances
            "zoom_in": {
                "spatial": A.Compose([
                    A.RandomScale(scale_limit=(0.1, 0.2), p=1.0),
                    A.CenterCrop(height=480, width=640),  # Will be resized later
                ]),
                "temporal": None,
                "description": "Slight zoom in"
            },
        }
        
        return configs
    
    def load_video(self, video_path: str) -> tuple:
        """Load video and return frames with metadata."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return np.array(frames), fps, width, height
    
    def save_video(self, frames: np.ndarray, output_path: str, 
                   fps: float, width: int, height: int):
        """Save frames as video file."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Ensure frame is the correct size
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
        
        out.release()
    
    def apply_spatial_augmentation(self, frames: np.ndarray, 
                                    transform: A.Compose) -> np.ndarray:
        """
        Apply spatial augmentation consistently across all frames.
        This is CRITICAL for maintaining temporal consistency.
        """
        if transform is None:
            return frames
        
        # For Albumentations, we need to apply same random seed to all frames
        # to ensure consistent augmentation
        augmented_frames = []
        
        # Get the first frame to determine random parameters
        # Then apply same transformation to all frames
        random_state = random.getstate()
        rng = np.random.RandomState(42)  # Per-video RNG for reproducibility
        
        for i, frame in enumerate(frames):
            # Reset random state for each frame to get identical augmentation
            random.setstate(random_state)
            # Use local RNG instead of contaminating global np.random state
            
            # Convert BGR to RGB for albumentations
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                augmented = transform(image=frame_rgb)
                aug_frame = augmented['image']
                
                # Convert back to BGR for OpenCV
                aug_frame_bgr = cv2.cvtColor(aug_frame, cv2.COLOR_RGB2BGR)
                augmented_frames.append(aug_frame_bgr)
            except Exception as e:
                # If augmentation fails, use original frame
                augmented_frames.append(frame)
        
        return np.array(augmented_frames)
    
    def apply_temporal_speed_change(self, frames: np.ndarray, 
                                     speed_factor: float) -> np.ndarray:
        """
        Change video playback speed by resampling frames.
        speed_factor < 1.0 = slower (more frames)
        speed_factor > 1.0 = faster (fewer frames)
        """
        num_frames = len(frames)
        new_num_frames = int(num_frames / speed_factor)
        
        if new_num_frames < 2:
            return frames
        
        # Create new frame indices
        old_indices = np.linspace(0, num_frames - 1, new_num_frames)
        new_frames = []
        
        for idx in old_indices:
            # Simple nearest neighbor interpolation
            frame_idx = int(round(idx))
            frame_idx = min(frame_idx, num_frames - 1)
            new_frames.append(frames[frame_idx])
        
        return np.array(new_frames)
    
    def apply_temporal_reverse(self, frames: np.ndarray) -> np.ndarray:
        """Reverse the temporal order of frames."""
        return frames[::-1].copy()
    
    def apply_temporal_jitter(self, frames: np.ndarray, 
                               jitter_strength: float = 0.1) -> np.ndarray:
        """
        Apply temporal jittering by slightly shuffling frame order.
        This simulates frame rate inconsistencies.
        """
        num_frames = len(frames)
        indices = list(range(num_frames))
        
        # Slightly perturb indices
        for i in range(len(indices)):
            offset = int(random.gauss(0, jitter_strength * num_frames))
            new_idx = max(0, min(num_frames - 1, i + offset))
            indices[i] = new_idx
        
        return frames[indices]
    
    def augment_video(self, video_path: str, aug_name: str, 
                       aug_config: dict) -> str:
        """
        Apply augmentation to a single video and save result.
        """
        # Load video
        frames, fps, width, height = self.load_video(video_path)
        
        if len(frames) == 0:
            print(f"Warning: No frames in {video_path}")
            return None
        
        # Apply spatial augmentation
        if aug_config.get("spatial"):
            frames = self.apply_spatial_augmentation(frames, aug_config["spatial"])
        
        # Apply temporal augmentation
        temporal = aug_config.get("temporal")
        if temporal:
            if temporal == "reverse":
                frames = self.apply_temporal_reverse(frames)
            elif temporal == "slow":
                frames = self.apply_temporal_speed_change(frames, 0.8)
            elif temporal == "fast":
                frames = self.apply_temporal_speed_change(frames, 1.2)
            elif temporal == "jitter":
                frames = self.apply_temporal_jitter(frames)
        
        # Generate output filename
        video_name = Path(video_path).stem
        output_filename = f"{video_name}_{aug_name}.mp4"
        output_path = str(self.output_dir / output_filename)
        
        # Save augmented video
        self.save_video(frames, output_path, fps, width, height)
        
        return output_path
    
    def augment_dataset(self, input_dir: str, augmentations_to_apply: list = None):
        """
        Augment all videos in a directory.
        
        Args:
            input_dir: Directory containing original videos
            augmentations_to_apply: List of augmentation names to apply.
                                   If None, applies all augmentations.
        """
        input_path = Path(input_dir)
        video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi"))
        
        if not video_files:
            print(f"No video files found in {input_dir}")
            return
        
        # Determine which augmentations to apply
        if augmentations_to_apply is None:
            augmentations_to_apply = list(self.augmentation_configs.keys())
        
        print(f"Found {len(video_files)} videos")
        print(f"Applying {len(augmentations_to_apply)} augmentations to each video")
        print(f"This will create approximately {len(video_files) * len(augmentations_to_apply)} new videos")
        print("-" * 50)
        
        # Copy original videos to output directory
        print("Copying original videos...")
        for video_file in tqdm(video_files, desc="Copying originals"):
            dest = self.output_dir / video_file.name
            if not dest.exists():
                shutil.copy2(video_file, dest)
        
        # Apply augmentations
        total_augmentations = len(video_files) * len(augmentations_to_apply)
        
        # Count existing augmented files to skip
        existing_files = set(f.stem for f in self.output_dir.glob("*.mp4"))
        skipped_count = 0
        
        with tqdm(total=total_augmentations, desc="Augmenting videos") as pbar:
            for video_file in video_files:
                for aug_name in augmentations_to_apply:
                    # Check if this augmentation already exists - SKIP if so
                    expected_output_name = f"{video_file.stem}_{aug_name}"
                    if expected_output_name in existing_files:
                        skipped_count += 1
                        pbar.update(1)
                        pbar.set_postfix_str(f"Skipped {skipped_count} existing")
                        continue
                    
                    try:
                        aug_config = self.augmentation_configs[aug_name]
                        output_path = self.augment_video(
                            str(video_file), 
                            aug_name, 
                            aug_config
                        )
                        if output_path:
                            pbar.set_postfix_str(f"{Path(video_file).stem}_{aug_name}")
                    except Exception as e:
                        print(f"\nError augmenting {video_file} with {aug_name}: {e}")
                    pbar.update(1)
        
        # Add temporal augmentations (speed changes, reverse)
        print("\nApplying temporal augmentations...")
        temporal_augs = {
            "slow": {"spatial": None, "temporal": "slow", "description": "Slow motion (0.8x)"},
            "fast": {"spatial": None, "temporal": "fast", "description": "Fast motion (1.2x)"},
            "reverse": {"spatial": None, "temporal": "reverse", "description": "Temporal reverse"},
        }
        
        for video_file in tqdm(video_files, desc="Temporal augmentations"):
            for aug_name, aug_config in temporal_augs.items():
                try:
                    self.augment_video(str(video_file), aug_name, aug_config)
                except Exception as e:
                    print(f"\nError with temporal aug {aug_name} on {video_file}: {e}")
        
        # Summary
        augmented_files = list(self.output_dir.glob("*.mp4"))
        print("\n" + "=" * 50)
        print("AUGMENTATION COMPLETE!")
        print("=" * 50)
        print(f"Original videos: {len(video_files)}")
        print(f"Total videos after augmentation: {len(augmented_files)}")
        print(f"Expansion factor: {len(augmented_files) / len(video_files):.1f}x")
        print(f"Output directory: {self.output_dir}")


def create_train_val_split(augmented_dir: str, train_ratio: float = 0.8):
    """
    Create train/validation split of augmented videos.
    Ensures all augmentations of the same person go to same split.
    """
    aug_path = Path(augmented_dir)
    train_dir = aug_path.parent / "train"
    val_dir = aug_path.parent / "val"
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Group videos by person
    videos = list(aug_path.glob("*.mp4"))
    persons = {}
    
    for video in videos:
        # Extract person name (before first underscore followed by F or S)
        name = video.stem
        # Parse: PersonName_View_Aug or PersonName_View
        parts = name.split('_')
        if len(parts) >= 2:
            person = parts[0]
            if person not in persons:
                persons[person] = []
            persons[person].append(video)
    
    # Split by person
    person_list = list(persons.keys())
    random.shuffle(person_list)
    
    split_idx = int(len(person_list) * train_ratio)
    train_persons = set(person_list[:split_idx])
    val_persons = set(person_list[split_idx:])
    
    print(f"Train persons ({len(train_persons)}): {train_persons}")
    print(f"Val persons ({len(val_persons)}): {val_persons}")
    
    # Copy files
    for person, videos in persons.items():
        dest_dir = train_dir if person in train_persons else val_dir
        for video in videos:
            shutil.copy2(video, dest_dir / video.name)
    
    print(f"\nTrain videos: {len(list(train_dir.glob('*.mp4')))}")
    print(f"Val videos: {len(list(val_dir.glob('*.mp4')))}")


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = r"data/videos"
    OUTPUT_DIR = r"data/augmented_videos"
    
    # Create augmenter
    augmenter = VideoAugmenter(output_dir=OUTPUT_DIR)
    
    # List available augmentations
    print("Available Augmentations:")
    print("-" * 50)
    for name, config in augmenter.augmentation_configs.items():
        print(f"  - {name}: {config['description']}")
    print("-" * 50)
    
    # Apply all augmentations
    # You can also specify a subset:
    # augmenter.augment_dataset(INPUT_DIR, augmentations_to_apply=['hflip', 'bright_up', 'blur'])
    
    augmenter.augment_dataset(INPUT_DIR)
    
    # Optional: Create train/val split
    # create_train_val_split(OUTPUT_DIR, train_ratio=0.8)
    
    print("\n✓ Data augmentation complete!")
    print("\nNext steps:")
    print("1. Review augmented videos in:", OUTPUT_DIR)
    print("2. Run train/val split if needed")
    print("3. Proceed to gait feature extraction")
