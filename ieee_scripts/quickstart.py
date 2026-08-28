"""
GaitDeepfake-13 — Quick Start Script
Run this after setup to verify the dataset loads correctly.

Usage:
    python quickstart.py
    python quickstart.py --features path/to/gait_features.pkl
"""

import argparse
import os
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="GaitDeepfake-13 quickstart loader")
    parser.add_argument(
        "--features",
        default=os.path.join("data", "gait_features", "gait_features.pkl"),
        help="Path to gait_features.pkl (default: data/gait_features/gait_features.pkl)",
    )
    parser.add_argument(
        "--enrolled",
        default=os.path.join("data", "gait_features", "enrolled_identities.pkl"),
        help="Path to enrolled_identities.pkl",
    )
    args = parser.parse_args()

    # --- Load gait features ---
    print(f"\nLoading: {args.features}")
    try:
        with open(args.features, "rb") as f:
            dataset = pickle.load(f)

        # dataset is a dict: { video_path -> {gait_features: {normalized_coords, joint_angles, velocities, ...}, identity, ...} }
        video_paths = list(dataset.keys())
        subject_ids = [dataset[v]["identity"] for v in video_paths]

        # Build the 78-dim feature array: coords (36) + angles (6) + velocities (36)
        features_list = []
        for v in video_paths:
            gf = dataset[v]["gait_features"]
            coords = gf["normalized_coords"].reshape(60, -1)  # (60, 36)
            angles = gf["joint_angles"]  # (60, 6)
            velocities = gf["velocities"].reshape(60, -1)  # (60, 36)
            combined = np.concatenate([coords, angles, velocities], axis=1)  # (60, 78)
            features_list.append(combined)

        features = np.array(features_list)  # (N, 60, 78)

        print(f"  Total samples : {len(features)}")
        print(f"  Feature shape : {features.shape[1:]}")
        print(f"  Subjects      : {sorted(set(subject_ids))}")

        # Feature layout sanity check
        assert (
            features.shape[2] == 78
        ), f"Expected 78-dim features, got {features.shape[2]}"
        assert (
            features.shape[1] == 60
        ), f"Expected 60-frame sequences, got {features.shape[1]}"
        print("  Feature layout check passed (60 frames x 78 dims)")

    except FileNotFoundError:
        print(f"  ERROR: File not found — {args.features}")
        print("  Make sure the dataset is in the same directory as this script.")
        return

    # --- Load enrolled identities ---
    print(f"\nLoading: {args.enrolled}")
    try:
        with open(args.enrolled, "rb") as f:
            enrolled = pickle.load(f)

        print(f"  Enrolled subjects : {list(enrolled.keys())}")
        sample = enrolled[list(enrolled.keys())[0]]
        coords = sample["avg_normalized_coords"].reshape(60, -1)
        angles = sample["avg_joint_angles"]
        vels = sample["avg_velocities"].reshape(60, -1)
        sig = np.concatenate([coords, angles, vels], axis=1)
        print(f"  Signature shape   : {sig.shape}")

    except FileNotFoundError:
        print(f"  ERROR: File not found — {args.enrolled}")

    print("\nAll checks passed. Dataset is ready to use.")
    print("See DATASET_INSTRUCTIONS.md for next steps.")


if __name__ == "__main__":
    main()
