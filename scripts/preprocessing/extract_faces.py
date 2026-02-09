"""
Extract Face Frames from Walking Videos
========================================
Extracts a clear frontal face frame from each person's front-view walking video.
These face images are used as "source faces" for FaceFusion face-swapping.

Usage:
    python scripts/preprocessing/extract_faces.py
"""

import cv2
import numpy as np
from pathlib import Path
import sys


def extract_best_face_frame(video_path: str, output_path: str, sample_count: int = 20) -> bool:
    """
    Extract the best face frame from a video.
    Samples frames evenly across the video and picks the one with the largest detected face.

    Args:
        video_path: Path to input video
        output_path: Path to save the extracted face image
        sample_count: Number of frames to sample

    Returns:
        True if a face was found and saved, False otherwise
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 10:
        print(f"  Too few frames in: {video_path}")
        cap.release()
        return False

    # Use OpenCV's built-in face detector (Haar cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    best_frame = None
    best_face_size = 0

    # Sample frames evenly across the video
    frame_indices = np.linspace(total_frames * 0.1, total_frames * 0.9, sample_count, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_size = w * h
            if face_size > best_face_size:
                best_face_size = face_size
                # Save the full frame (FaceFusion handles face detection itself)
                best_frame = frame.copy()

    cap.release()

    if best_frame is not None:
        cv2.imwrite(output_path, best_frame)
        print(f"  Saved face frame ({best_face_size} px area)")
        return True
    else:
        # Fallback: save a frame from the middle of the video even without face detection
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(output_path, frame)
            print(f"  No face detected by Haar; saved middle frame as fallback")
            return True
        return False


def main():
    project_root = Path(__file__).parent.parent.parent
    videos_dir = project_root / "data" / "videos"
    faces_dir = project_root / "data" / "deepfake" / "faces"

    # Create faces directory
    faces_dir.mkdir(parents=True, exist_ok=True)

    # Get unique identity names from video filenames
    video_files = sorted(videos_dir.glob("*.mp4"))
    identities = set()
    for vf in video_files:
        name = vf.stem.rsplit("_", 1)[0]  # "Arhaan_F1" -> "Arhaan"
        identities.add(name)

    identities = sorted(identities)
    print(f"Found {len(identities)} identities: {', '.join(identities)}")
    print(f"Output directory: {faces_dir}\n")

    success_count = 0
    for name in identities:
        print(f"Processing {name}...")

        # Prefer front-view video (_F1) for clearest face
        front_video = videos_dir / f"{name}_F1.mp4"
        if not front_video.exists():
            # Try F2, F3 as alternatives
            for alt in ["F2", "F3"]:
                alt_video = videos_dir / f"{name}_{alt}.mp4"
                if alt_video.exists():
                    front_video = alt_video
                    break

        if not front_video.exists():
            # Last resort: use any available video
            candidates = list(videos_dir.glob(f"{name}_*.mp4"))
            if candidates:
                front_video = candidates[0]
            else:
                print(f"  No videos found for {name}, skipping")
                continue

        output_path = str(faces_dir / f"{name}.jpg")
        print(f"  Source video: {front_video.name}")

        if extract_best_face_frame(str(front_video), output_path):
            success_count += 1

    print(f"\nDone! Extracted {success_count}/{len(identities)} face frames")
    print(f"Face images saved to: {faces_dir}")

    # Print recommended swap pairs
    print("\n" + "=" * 60)
    print("RECOMMENDED FACE-SWAP PAIRS FOR FACEFUSION")
    print("=" * 60)
    print(f"{'#':<4} {'Body (gait owner)':<18} {'Face (swap on)':<18} {'Target Video':<20} {'Output Filename'}")
    print("-" * 95)

    # Generate diverse swap pairs
    pairs = [
        ("Arhaan", "Devika", "Arhaan_F1.mp4"),
        ("Arhaan", "Aarav", "Arhaan_S1.mp4"),
        ("Aarav", "Ananya", "Aarav_F1.mp4"),
        ("Devika", "Arhaan", "Devika_F1.mp4"),
        ("Ananya", "Prakhar", "Ananya_F1.mp4"),
        ("Prakhar", "Bharti", "Prakhar_F1.mp4"),
        ("Bharti", "Som", "Bharti_F1.mp4"),
        ("Som", "Teja", "Som_F1.mp4"),
        ("Teja", "Vibhav", "Teja_F1.mp4"),
        ("Vedant", "Prayag", "Vedant_F1.mp4"),
        ("Prayag", "Vedant2", "Prayag_F1.mp4"),
        ("Vibhav", "A2", "Vibhav_F1.mp4"),
    ]

    for i, (body, face, video) in enumerate(pairs, 1):
        output = f"{body}_body_{face}_face.mp4"
        print(f"{i:<4} {body:<18} {face:<18} {video:<20} {output}")

    print(f"\nUse these pairs in FaceFusion GUI:")
    print(f"  Source (face donor) = data/deepfake/faces/<Face>.jpg")
    print(f"  Target (body video) = data/videos/<Target Video>")
    print(f"  Output             = data/deepfake/<Output Filename>")


if __name__ == "__main__":
    main()
