"""
Data Loader for Gait-Based Deepfake Detection
==============================================
PyTorch Dataset and DataLoader for training the model.

Handles:
1. Loading extracted gait features
2. Creating train/val splits BY PERSON (critical!)
3. Creating positive/negative pairs for identity verification
4. Batching and data augmentation

Author: DeepFake Detection Project
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


class GaitDataset(Dataset):
    """
    Dataset for gait-based deepfake detection.
    
    Modes:
    1. 'classification': Binary classification (real vs fake)
    2. 'verification': Identity verification with claimed identity
    3. 'triplet': Triplet loss training (anchor, positive, negative)
    """
    
    def __init__(self, 
                 features_file: str,
                 enrolled_identities_file: str,
                 person_list: List[str],
                 mode: str = 'verification',
                 feature_type: str = 'normalized_coords',
                 include_angles: bool = True,
                 include_velocities: bool = True,
                 feature_stats: Optional[Dict] = None):
        """
        Args:
            features_file: Path to gait_features.pkl
            enrolled_identities_file: Path to enrolled_identities.pkl
            person_list: List of person names to include (for train/val split)
            mode: 'classification', 'verification', or 'triplet'
            feature_type: Which feature to use as primary input
            include_angles: Include joint angles in features
            include_velocities: Include velocities in features
            feature_stats: Optional dict with 'mean' and 'std' tensors for normalization.
                           If None, stats are computed from this dataset's data.
        """
        self.mode = mode
        self.feature_type = feature_type
        self.include_angles = include_angles
        self.include_velocities = include_velocities
        
        # Load features
        with open(features_file, 'rb') as f:
            all_features = pickle.load(f)
        
        # Load enrolled identities
        with open(enrolled_identities_file, 'rb') as f:
            self.enrolled_identities = pickle.load(f)
        
        # Filter by person list and prepare raw features
        self.samples = []
        self.person_to_samples = {p: [] for p in person_list}
        
        for video_name, features in all_features.items():
            person = features['identity']
            if person in person_list:
                sample = {
                    'video_name': video_name,
                    'person_name': person,
                    'features': self._prepare_features(features),
                }
                self.samples.append(sample)
                self.person_to_samples[person].append(len(self.samples) - 1)
        
        self.person_list = person_list
        
        # Compute or use provided feature statistics for z-score normalization
        if feature_stats is not None:
            self.feature_mean = feature_stats['mean']
            self.feature_std = feature_stats['std']
        else:
            self._compute_feature_stats()
        
        # Apply normalization to all samples
        for sample in self.samples:
            sample['features'] = (sample['features'] - self.feature_mean) / self.feature_std
        
        print(f"Loaded {len(self.samples)} samples for {len(person_list)} persons")
    
    def _compute_feature_stats(self):
        """Compute per-feature mean and std across all samples for z-score normalization."""
        all_feats = torch.stack([s['features'] for s in self.samples])  # (N, T, D)
        # Compute stats across samples and time
        self.feature_mean = all_feats.mean(dim=(0, 1))  # (D,)
        self.feature_std = all_feats.std(dim=(0, 1))  # (D,)
        # Avoid division by zero for constant features
        self.feature_std[self.feature_std < 1e-6] = 1.0
    
    def get_feature_stats(self) -> Dict:
        """Return feature statistics for use by validation dataset."""
        return {'mean': self.feature_mean, 'std': self.feature_std}
    
    def _prepare_features(self, features: Dict) -> torch.Tensor:
        """Prepare and concatenate features into a single tensor."""
        gait_features = features['gait_features']
        
        # Primary feature: normalized coordinates (T, 12, 3)
        coords = gait_features['normalized_coords']
        coords_flat = coords.reshape(coords.shape[0], -1)  # (T, 36)
        
        feature_list = [coords_flat]
        
        # Add joint angles (T, 6)
        if self.include_angles:
            angles = gait_features['joint_angles']
            feature_list.append(angles)
        
        # Add velocities (T, 12, 3) -> (T, 36)
        if self.include_velocities:
            velocities = gait_features['velocities']
            velocities_flat = velocities.reshape(velocities.shape[0], -1)
            feature_list.append(velocities_flat)
        
        # Concatenate all features
        combined = np.concatenate(feature_list, axis=1)  # (T, feature_dim)
        
        return torch.FloatTensor(combined)
    
    def get_enrolled_embedding(self, person_name: str) -> torch.Tensor:
        """Get the enrolled gait signature for a person, with same normalization."""
        if person_name not in self.enrolled_identities:
            raise ValueError(f"Person {person_name} not enrolled")
        
        enrolled = self.enrolled_identities[person_name]
        
        # Use pre-extracted gait features (MediaPipe: 12 landmarks × 3D)
        feature_list = []
        
        # Normalized coordinates (T, 12, 3) -> (T, 36)
        coords = enrolled['avg_normalized_coords']
        coords_flat = coords.reshape(coords.shape[0], -1)
        feature_list.append(coords_flat)
        
        # Joint angles (T, 6)
        if self.include_angles:
            feature_list.append(enrolled['avg_joint_angles'])
        
        # Velocities (T, 12, 3) -> (T, 36)
        if self.include_velocities:
            velocities = enrolled['avg_velocities']
            velocities_flat = velocities.reshape(velocities.shape[0], -1)
            feature_list.append(velocities_flat)
        
        combined = np.concatenate(feature_list, axis=1)
        tensor = torch.FloatTensor(combined)
        
        # Apply same z-score normalization as training data
        tensor = (tensor - self.feature_mean) / self.feature_std
        
        return tensor
    
    def __len__(self):
        if self.mode == 'triplet':
            return len(self.samples)
        elif self.mode == 'verification':
            # Balanced: 2 pairs per sample (1 positive + 1 negative)
            return len(self.samples) * 2
        else:
            return len(self.samples)
    
    def __getitem__(self, idx):
        if self.mode == 'verification':
            return self._get_verification_sample(idx)
        elif self.mode == 'triplet':
            return self._get_triplet_sample(idx)
        else:
            return self._get_classification_sample(idx)
    
    def _get_classification_sample(self, idx):
        """Simple binary classification (for baseline)."""
        sample = self.samples[idx]
        features = sample['features']
        label = 1  # All training videos are real
        return features, torch.LongTensor([label])
    
    def _get_verification_sample(self, idx):
        """
        Get a sample for identity verification with BALANCED sampling.
        
        50% of indices produce positive (same-identity) pairs,
        50% produce negative (cross-identity) pairs.
        This prevents the model from collapsing to always predicting one class.
        """
        sample_idx = idx // 2
        is_positive = (idx % 2) == 0  # Even = positive pair, Odd = negative pair
        
        sample = self.samples[sample_idx]
        true_identity = sample['person_name']
        video_features = sample['features']
        
        if is_positive:
            # Positive pair: claim the true identity
            claimed_identity = true_identity
            is_match = 1
        else:
            # Negative pair: claim a different (random) identity
            # Use all enrolled identities (not just person_list) so LOOCV test sets
            # with a single person can still generate negative pairs
            all_enrolled = list(self.enrolled_identities.keys())
            other_persons = [p for p in all_enrolled if p != true_identity]
            claimed_identity = random.choice(other_persons)
            is_match = 0
        
        # Get claimed identity's enrolled embedding
        claimed_embedding = self.get_enrolled_embedding(claimed_identity)
        
        return {
            'video_features': video_features,
            'claimed_features': claimed_embedding,
            'label': torch.LongTensor([is_match]),
            'true_identity': true_identity,
            'claimed_identity': claimed_identity
        }
    
    def _get_triplet_sample(self, idx):
        """
        Get triplet for contrastive learning.
        Returns: (anchor, positive, negative)
        """
        sample = self.samples[idx]
        anchor_person = sample['person_name']
        anchor_features = sample['features']
        
        # Get positive (same person, different video)
        positive_indices = [i for i in self.person_to_samples[anchor_person] if i != idx]
        if positive_indices:
            pos_idx = random.choice(positive_indices)
            positive_features = self.samples[pos_idx]['features']
        else:
            positive_features = anchor_features  # Fallback to same sample
        
        # Get negative (different person)
        other_persons = [p for p in self.person_list if p != anchor_person]
        neg_person = random.choice(other_persons)
        neg_idx = random.choice(self.person_to_samples[neg_person])
        negative_features = self.samples[neg_idx]['features']
        
        return {
            'anchor': anchor_features,
            'positive': positive_features,
            'negative': negative_features,
            'anchor_person': anchor_person
        }


def create_data_loaders(features_file: str,
                        enrolled_identities_file: str,
                        batch_size: int = 16,
                        train_ratio: float = 0.8,
                        mode: str = 'verification',
                        num_workers: int = 0,
                        seed: int = 42) -> Tuple[DataLoader, DataLoader, List[str], List[str]]:
    """
    Create train and validation data loaders.
    
    IMPORTANT: Split is done BY PERSON to prevent data leakage.
    
    Args:
        features_file: Path to gait_features.pkl
        enrolled_identities_file: Path to enrolled_identities.pkl
        batch_size: Batch size for training
        train_ratio: Ratio of persons for training
        mode: 'classification', 'verification', or 'triplet'
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, train_persons, val_persons
    """
    # Load features to get person list
    with open(features_file, 'rb') as f:
        all_features = pickle.load(f)
    
    # Get unique persons
    persons = list(set(f['identity'] for f in all_features.values()))
    random.seed(seed)
    random.shuffle(persons)
    
    # Split by person
    split_idx = int(len(persons) * train_ratio)
    train_persons = persons[:split_idx]
    val_persons = persons[split_idx:]
    
    print(f"Train persons ({len(train_persons)}): {train_persons}")
    print(f"Val persons ({len(val_persons)}): {val_persons}")
    
    # Create datasets — train computes stats, val uses train stats
    train_dataset = GaitDataset(
        features_file=features_file,
        enrolled_identities_file=enrolled_identities_file,
        person_list=train_persons,
        mode=mode,
        feature_stats=None  # Compute stats from training data
    )
    
    # Pass training stats to val dataset for consistent normalization
    val_dataset = GaitDataset(
        features_file=features_file,
        enrolled_identities_file=enrolled_identities_file,
        person_list=val_persons,
        mode=mode,
        feature_stats=train_dataset.get_feature_stats()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_persons, val_persons


def collate_verification(batch):
    """Custom collate function for verification mode."""
    video_features = torch.stack([b['video_features'] for b in batch])
    claimed_features = torch.stack([b['claimed_features'] for b in batch])
    labels = torch.cat([b['label'] for b in batch])
    
    return {
        'video_features': video_features,
        'claimed_features': claimed_features,
        'labels': labels,
        'true_identities': [b['true_identity'] for b in batch],
        'claimed_identities': [b['claimed_identity'] for b in batch]
    }


def collate_triplet(batch):
    """Custom collate function for triplet mode."""
    anchors = torch.stack([b['anchor'] for b in batch])
    positives = torch.stack([b['positive'] for b in batch])
    negatives = torch.stack([b['negative'] for b in batch])
    
    return {
        'anchor': anchors,
        'positive': positives,
        'negative': negatives
    }


if __name__ == "__main__":
    # Test the data loader
    FEATURES_FILE = "data/gait_features/gait_features.pkl"
    ENROLLED_FILE = "data/gait_features/enrolled_identities.pkl"
    
    print("Testing data loader...")
    
    train_loader, val_loader, train_persons, val_persons = create_data_loaders(
        features_file=FEATURES_FILE,
        enrolled_identities_file=ENROLLED_FILE,
        batch_size=8,
        mode='verification'
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    for batch in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Video features: {batch['video_features'].shape}")
        print(f"  Claimed features: {batch['claimed_features'].shape}")
        print(f"  Labels: {batch['label'].shape}")
        break
