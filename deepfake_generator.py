"""
Deepfake Generation Script
==========================
Creates face-swapped videos for testing gait detection.
Uses OpenCV DNN + simple face blending for effective deepfakes.

Usage:
    python deepfake_generator.py --target <video> --source <video> --output <output>
    
The BODY gait comes from --target, FACE comes from --source.
We expect the model to detect the gait from --target (body source).
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# OpenCV's pre-trained face detector (DNN)
FACE_DETECTOR_PROTO = 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector.caffemodel'
FACE_DETECTOR_MODEL = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector_fp32.pb'


def get_face_detector():
    """Load OpenCV's pre-trained face detector (DNN)."""
    try:
        net = cv2.dnn.readNetFromCaffe(
            'opencv_face_detector.prototxt',
            'opencv_face_detector_fp32.caffemodel'
        )
        return net
    except:
        # Fallback: use cascade classifier (faster)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        return cascade


def extract_frames(video_path: str, max_frames: int = 300):
    """Extract frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames


def detect_faces_cascade(frame):
    """Detect faces using OpenCV Cascade Classifier."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    return [list(face) for face in faces] if len(faces) > 0 else []


def extract_face_region(frame, bbox, margin: int = 20):
    """Extract face region from frame with optional margin."""
    x, y, w, h = bbox
    frame_h, frame_w = frame.shape[:2]
    
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(frame_w, x + w + margin)
    y_end = min(frame_h, y + h + margin)
    
    return frame[y_start:y_end, x_start:x_end], (x_start, y_start, x_end, y_end)


def blend_face(src_face, dst_frame, src_bbox, dst_bbox, alpha: float = 0.8):
    """Blend source face over destination frame."""
    try:
        # Extract regions
        sx1, sy1, sx2, sy2 = src_bbox
        dx1, dy1, dx2, dy2 = dst_bbox
        
        src_roi = src_face[sy1:sy2, sx1:sx2]
        
        # Resize source face to destination size
        dst_h = dy2 - dy1
        dst_w = dx2 - dx1
        
        if dst_h > 0 and dst_w > 0 and src_roi.size > 0:
            src_roi_resized = cv2.resize(src_roi, (dst_w, dst_h))
            
            # Blend with alpha blending
            if dy2 <= dst_frame.shape[0] and dx2 <= dst_frame.shape[1]:
                dst_frame[dy1:dy2, dx1:dx2] = cv2.addWeighted(
                    src_roi_resized, alpha,
                    dst_frame[dy1:dy2, dx1:dx2],
                    1 - alpha, 0
                )
        
        return dst_frame
    except Exception as e:
        print(f"    Warning: Face blend failed - {str(e)}")
        return dst_frame


def create_deepfake(target_video: str, source_video: str, output_video: str):
    """Create deepfake by swapping faces from source into target.
    
    Args:
        target_video: Video providing the BODY gait
        source_video: Video providing the FACE
        output_video: Output deepfake video path
    """
    print(f"\n{'='*60}")
    print(f"Creating Deepfake")
    print(f"{'='*60}")
    print(f"Target (body):  {target_video}")
    print(f"Source (face):  {source_video}")
    print(f"Output:         {output_video}")
    print(f"{'='*60}\n")
    
    # Extract frames
    print("Loading target video frames (body source)...")
    target_frames = extract_frames(target_video)
    print(f"  ✓ Loaded {len(target_frames)} frames from target")
    
    print("Loading source video frames (face source)...")
    source_frames = extract_frames(source_video)
    print(f"  ✓ Loaded {len(source_frames)} frames from source")
    
    # Get video properties from target
    cap = cv2.VideoCapture(target_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Prepare output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"\nFace detection & swap progress ({len(target_frames)} frames)...")
    print(f"Output resolution: {width}x{height} @ {fps:.1f} FPS\n")
    
    faces_processed = 0
    faces_skipped = 0
    
    # Process each frame
    for i, target_frame in enumerate(tqdm(target_frames, desc="Processing")):
        src_frame_idx = i % len(source_frames) if source_frames else 0
        src_frame = source_frames[src_frame_idx] if source_frames else target_frame.copy()
        
        # Detect faces
        src_faces = detect_faces_cascade(src_frame)
        dst_faces = detect_faces_cascade(target_frame)
        
        # Swap if faces detected
        if src_faces and dst_faces:
            # Use first face detected in each
            src_bbox = src_faces[0]
            dst_bbox = dst_faces[0]
            
            # Extract and blend
            src_roi, src_bbox_full = extract_face_region(src_frame, src_bbox)
            dst_bbox_full = list(extract_face_region(target_frame, dst_bbox)[1])
            
            output_frame = blend_face(src_frame, target_frame.copy(), 
                                     src_bbox_full, dst_bbox_full, alpha=0.85)
            faces_processed += 1
        else:
            output_frame = target_frame
            if i < 10:  # Only print first few skips
                faces_skipped += 1
        
        out.write(output_frame)
    
    out.release()
    
    print(f"\n{'='*60}")
    print(f"✅ Deepfake creation complete!")
    print(f"{'='*60}")
    print(f"Output file: {output_video}")
    print(f"Duration: {len(target_frames) / fps:.1f}s")
    print(f"Frames with face swap: {faces_processed}/{len(target_frames)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create deepfake by swapping faces (source) into target video"
    )
    parser.add_argument('--target', type=str, required=True,
                        help='Target video (provides BODY gait)')
    parser.add_argument('--source', type=str, required=True,
                        help='Source video (provides FACE)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output deepfake video path')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.target).exists():
        raise FileNotFoundError(f"Target video not found: {args.target}")
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Source video not found: {args.source}")
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    create_deepfake(args.target, args.source, args.output)


if __name__ == "__main__":
    main()
