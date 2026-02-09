"""
Verify Gait Preservation After Face-Swap
=========================================
Compares MediaPipe skeletal features between original and face-swapped videos
to confirm that the body/gait was NOT altered during face swapping.

Usage:
    python scripts/evaluation/verify_gait_preservation.py

Expects:
    - Original videos in data/videos/
    - Face-swapped videos in data/deepfake/ with naming: {Body}_body_{Face}_face.mp4
"""

import os
import sys
import warnings
import logging
from pathlib import Path

# Suppress all warnings and noisy logs BEFORE importing anything else
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # Suppress TensorFlow C++ logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'           # Suppress oneDNN messages
os.environ['GLOG_minloglevel'] = '3'                 # Suppress MediaPipe glog
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Redirect stderr temporarily during imports to catch stray C++ warnings
import io
_stderr = sys.stderr
sys.stderr = io.StringIO()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from utils.pose_extraction import GaitFeatureExtractor

# Restore stderr
sys.stderr = _stderr


# ── Pretty print helpers ──────────────────────────────────────────────

BOLD  = "\033[1m"
DIM   = "\033[2m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
CHECK = f"{GREEN}\u2714{RESET}"
CROSS = f"{RED}\u2718{RESET}"
BAR   = f"{DIM}\u2502{RESET}"


def header(text: str):
    width = 66
    print(f"\n{BOLD}{CYAN}{'=' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * width}{RESET}\n")


def subheader(text: str):
    print(f"  {BOLD}{text}{RESET}")


def status_icon(preserved: bool) -> str:
    return f"{CHECK} PRESERVED" if preserved else f"{CROSS} ALTERED"


def progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / total)
    bar = f"{'█' * filled}{'░' * (width - filled)}"
    return f"[{bar}] {current}/{total}"


# ── Thresholds (literature-grounded) ──────────────────────────────────
#
# Multi-metric consensus approach based on:
#   - Pearson r > 0.85 for temporal gait correlations
#     (standard in clinical gait analysis, PMC10886083, PMC11097739)
#   - PCK-style: >=90% of frames with per-landmark error < 5% of torso height
#     (PCK@0.05 on torso-normalized coords, V7Labs HPE guide, MPII standard)
#   - Coordinate cosine similarity > 0.95 for pose shape agreement
#     (skeleton-based gait recognition, PMC9371146)
#
# Verdict = at least 2 of 3 criteria pass → PRESERVED

THRESHOLD_CORR      = 0.85   # Pearson r for step-width & symmetry
THRESHOLD_PCK       = 0.90   # fraction of frames within tolerance
PCK_NORM_FRACTION   = 0.05   # 5% of torso height per landmark
THRESHOLD_COSINE    = 0.95   # mean cosine similarity of pose vectors


# ── Core comparison ───────────────────────────────────────────────────

def compare_gait(original_video: str, swapped_video: str, extractor: GaitFeatureExtractor) -> dict:
    """Compare gait features between original and face-swapped video."""
    # Suppress ALL output during extraction (catches MediaPipe C++ warnings)
    _stderr = sys.stderr
    _stdout = sys.stdout
    _devnull = open(os.devnull, 'w')
    sys.stderr = _devnull
    sys.stdout = _devnull
    _old_fd_err = os.dup(2)
    _old_fd_out = os.dup(1)
    _null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_null_fd, 2)
    os.dup2(_null_fd, 1)

    orig_result = extractor.process_video(original_video)
    swap_result = extractor.process_video(swapped_video)

    os.dup2(_old_fd_err, 2)
    os.dup2(_old_fd_out, 1)
    os.close(_null_fd)
    os.close(_old_fd_err)
    os.close(_old_fd_out)
    _devnull.close()
    sys.stderr = _stderr
    sys.stdout = _stdout

    if orig_result is None or swap_result is None:
        return {"error": "Feature extraction failed for one or both videos"}

    orig_feats = orig_result['gait_features']
    swap_feats = swap_result['gait_features']

    # ── 1. Coordinate distances ──
    # normalized_coords shape: (T, 12, 3) — hip-centered
    orig_coords_3d = orig_feats['normalized_coords']  # (T, 12, 3)
    swap_coords_3d = swap_feats['normalized_coords']
    min_len = min(len(orig_coords_3d), len(swap_coords_3d))
    orig_c = orig_coords_3d[:min_len]
    swap_c = swap_coords_3d[:min_len]

    # Per-landmark L2 distance per frame → (T, 12)
    per_landmark_dist = np.linalg.norm(orig_c - swap_c, axis=2)

    # Torso height estimate: distance from mid-shoulder to mid-hip in original
    # Landmarks 0,1 = L/R hip; 6,7 = L/R shoulder (in gait landmark ordering)
    mid_hip = (orig_c[:, 0, :] + orig_c[:, 1, :]) / 2
    mid_shoulder = (orig_c[:, 6, :] + orig_c[:, 7, :]) / 2
    torso_height = np.linalg.norm(mid_shoulder - mid_hip, axis=1)  # (T,)
    torso_height = np.clip(torso_height, 1e-6, None)  # avoid division by zero

    # PCK: fraction of (frame, landmark) pairs within 5% of torso height
    pck_threshold_per_frame = torso_height[:, np.newaxis] * PCK_NORM_FRACTION  # (T, 1)
    within_threshold = per_landmark_dist < pck_threshold_per_frame  # (T, 12)
    pck_score = float(within_threshold.mean())  # overall PCK

    # Flattened coord L2 (for display)
    orig_flat = orig_c.reshape(min_len, -1)
    swap_flat = swap_c.reshape(min_len, -1)
    coord_distances = np.linalg.norm(orig_flat - swap_flat, axis=1)
    mean_coord_l2 = float(coord_distances.mean())

    # Mean per-landmark distance (in normalized units)
    mean_per_landmark = float(per_landmark_dist.mean())

    # ── 2. Cosine similarity of pose vectors ──
    dot = np.sum(orig_flat * swap_flat, axis=1)
    norm_o = np.linalg.norm(orig_flat, axis=1)
    norm_s = np.linalg.norm(swap_flat, axis=1)
    cosine_sim = dot / (norm_o * norm_s + 1e-8)
    mean_cosine = float(cosine_sim.mean())

    # ── 3. Joint angle distances ──
    orig_angles = orig_feats['joint_angles']
    swap_angles = swap_feats['joint_angles']
    min_a = min(len(orig_angles), len(swap_angles))
    angle_distances = np.linalg.norm(orig_angles[:min_a] - swap_angles[:min_a], axis=1)

    # ── 4. Temporal correlations ──
    orig_step = orig_feats['step_width']
    swap_step = swap_feats['step_width']
    min_s = min(len(orig_step), len(swap_step))
    step_corr = float(np.corrcoef(orig_step[:min_s], swap_step[:min_s])[0, 1])

    orig_sym = orig_feats['symmetry']
    swap_sym = swap_feats['symmetry']
    min_y = min(len(orig_sym), len(swap_sym))
    sym_corr = float(np.corrcoef(orig_sym[:min_y], swap_sym[:min_y])[0, 1])

    # ── Consensus verdict (2 of 3 must pass) ──
    criterion_corr   = (step_corr >= THRESHOLD_CORR) and (sym_corr >= THRESHOLD_CORR)
    criterion_pck    = pck_score >= THRESHOLD_PCK
    criterion_cosine = mean_cosine >= THRESHOLD_COSINE
    pass_count = sum([criterion_corr, criterion_pck, criterion_cosine])
    preserved = pass_count >= 2

    return {
        "num_frames_original": orig_result['valid_frames'],
        "num_frames_swapped": swap_result['valid_frames'],
        "num_frames_compared": min_len,
        # Coordinate metrics
        "mean_l2_coords": mean_coord_l2,
        "mean_per_landmark_l2": mean_per_landmark,
        # PCK (literature: PCK@0.05 on torso height)
        "pck_score": pck_score,
        "pck_pass": criterion_pck,
        # Cosine similarity
        "mean_cosine": mean_cosine,
        "cosine_pass": criterion_cosine,
        # Temporal correlations (literature: Pearson r > 0.85)
        "step_correlation": step_corr,
        "symmetry_correlation": sym_corr,
        "corr_pass": criterion_corr,
        # Angles (informational)
        "mean_l2_angles": float(angle_distances.mean()),
        # Verdict
        "pass_count": pass_count,
        "preserved": preserved,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    videos_dir = project_root / "data" / "videos"
    deepfake_dir = project_root / "data" / "deepfake"

    header("GAIT PRESERVATION VERIFICATION")
    print(f"  {DIM}Compares skeletal pose features between original walking")
    print(f"  videos and their face-swapped counterparts to verify that")
    print(f"  the face swap did NOT alter the body/gait motion.{RESET}\n")

    swapped_videos = sorted(deepfake_dir.glob("*_body_*_face.mp4"))

    if not swapped_videos:
        print(f"  {RED}No face-swapped videos found in data/deepfake/{RESET}")
        print(f"  {DIM}Expected naming: {{Body}}_body_{{Face}}_face.mp4{RESET}")
        print(f"\n  Generate deepfakes first using FaceFusion, then re-run.")
        return

    total = len(swapped_videos)
    print(f"  Found {BOLD}{total}{RESET} face-swapped video(s) to verify\n")
    print(f"  {DIM}Loading MediaPipe pose estimator...{RESET}", end=" ", flush=True)

    # Suppress ALL output during extractor init (catches C++ TF/MediaPipe warnings)
    _stderr_backup = sys.stderr
    _stdout_backup = sys.stdout
    _devnull = open(os.devnull, 'w')
    sys.stderr = _devnull
    sys.stdout = _devnull
    # Also redirect at file-descriptor level to catch C++ stderr writes
    _old_fd_err = os.dup(2)
    _old_fd_out = os.dup(1)
    _null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_null_fd, 2)
    os.dup2(_null_fd, 1)

    extractor = GaitFeatureExtractor()
    # Do a dummy frame to trigger all lazy C++ init warnings now
    import cv2
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    extractor.extract_pose_from_frame(dummy)

    # Also warm up with a real-ish frame to trigger landmark_projection warnings
    dummy_real = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    extractor.extract_pose_from_frame(dummy_real)

    # Restore everything
    os.dup2(_old_fd_err, 2)
    os.dup2(_old_fd_out, 1)
    os.close(_null_fd)
    os.close(_old_fd_err)
    os.close(_old_fd_out)
    _devnull.close()
    sys.stderr = _stderr_backup
    sys.stdout = _stdout_backup

    print(f"{GREEN}done{RESET}\n")

    results = []

    for idx, swap_path in enumerate(swapped_videos, 1):
        body_name = swap_path.stem.split("_body_")[0]
        face_name = swap_path.stem.split("_body_")[1].replace("_face", "")

        # Find original video
        original = videos_dir / f"{body_name}_F1.mp4"
        if not original.exists():
            for suffix in ["F2", "S1", "F3"]:
                alt = videos_dir / f"{body_name}_{suffix}.mp4"
                if alt.exists():
                    original = alt
                    break

        if not original.exists():
            print(f"  {CROSS} Original video not found for {body_name}, skipping\n")
            continue

        print(f"  {DIM}{progress_bar(idx, total)}{RESET}")
        print(f"  {BOLD}{body_name}'s body + {face_name}'s face{RESET}")
        print(f"  {DIM}Original: {original.name}  |  Swapped: {swap_path.name}{RESET}")
        print(f"  {DIM}Extracting & comparing gait features...{RESET}", end=" ", flush=True)

        result = compare_gait(str(original), str(swap_path), extractor)
        result["body_identity"] = body_name
        result["face_identity"] = face_name
        result["swap_file"] = swap_path.name
        results.append(result)

        if "error" in result:
            print(f"\n  {CROSS} {RED}{result['error']}{RESET}\n")
        else:
            print(f"{GREEN}done{RESET}")
            print()
            preserved = result["preserved"]
            pc = result["pass_count"]

            # Status line
            print(f"  {BAR}  Verdict:  {status_icon(preserved)}  {DIM}({pc}/3 criteria passed){RESET}")
            print(f"  {BAR}")

            # Criterion 1: PCK
            pck_icon = f"{GREEN}PASS{RESET}" if result['pck_pass'] else f"{RED}FAIL{RESET}"
            print(f"  {BAR}  PCK@0.05 (torso-norm):  {result['pck_score']:.1%}   [{pck_icon}]  {DIM}(need >= {THRESHOLD_PCK:.0%}){RESET}")

            # Criterion 2: Cosine sim
            cos_icon = f"{GREEN}PASS{RESET}" if result['cosine_pass'] else f"{RED}FAIL{RESET}"
            print(f"  {BAR}  Cosine similarity:      {result['mean_cosine']:.4f}  [{cos_icon}]  {DIM}(need >= {THRESHOLD_COSINE}){RESET}")

            # Criterion 3: Correlations
            corr_icon = f"{GREEN}PASS{RESET}" if result['corr_pass'] else f"{RED}FAIL{RESET}"
            print(f"  {BAR}  Step-width corr:        {result['step_correlation']:.4f}  [{corr_icon}]  {DIM}(need >= {THRESHOLD_CORR}){RESET}")
            print(f"  {BAR}  Symmetry corr:          {result['symmetry_correlation']:.4f}          {DIM}(need >= {THRESHOLD_CORR}){RESET}")

            print(f"  {BAR}")
            print(f"  {BAR}  {DIM}Info: Mean per-landmark L2 = {result['mean_per_landmark_l2']:.4f}  |  Angles L2 = {result['mean_l2_angles']:.2f}  |  Frames = {result['num_frames_compared']}{RESET}")
            print()

    # ── Summary ──
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        print(f"  {RED}No valid comparisons could be made.{RESET}")
        return

    header("RESULTS SUMMARY")

    preserved_count = sum(1 for r in valid_results if r["preserved"])
    total_valid = len(valid_results)

    # Table header
    print(f"  {BOLD}{'Video':<38} {'PCK':>7} {'Cosine':>8} {'StepR':>7} {'Pass':>6} {'Status':>14}{RESET}")
    print(f"  {'─' * 84}")

    for r in valid_results:
        icon = status_icon(r["preserved"])
        print(f"  {r['swap_file']:<38} {r['pck_score']:>6.1%} {r['mean_cosine']:>8.4f} {r['step_correlation']:>7.4f} {r['pass_count']:>4}/3   {icon}")

    print(f"  {'─' * 84}")
    print(f"  {BOLD}{preserved_count}/{total_valid}{RESET} videos have preserved gait  {DIM}(need >= 2/3 criteria){RESET}")
    print()

    # ── Interpretation ──
    header("INTERPRETATION")

    if preserved_count == total_valid:
        print(f"  {CHECK} {BOLD}{GREEN}All gait sequences are preserved.{RESET}")
        print()
        print(f"  {DIM}The face-swap operation modified only the facial region and did")
        print(f"  NOT alter body landmarks, joint angles, or walking dynamics.")
        print(f"  This confirms that the deepfake videos retain the original")
        print(f"  person's gait signature — exactly what a gait-based deepfake")
        print(f"  detector should exploit to flag identity mismatches.{RESET}")
    else:
        altered = total_valid - preserved_count
        print(f"  {CROSS} {BOLD}{RED}{altered} video(s) show gait alteration.{RESET}")
        print()
        print(f"  {DIM}Possible causes:")
        print(f"    - Face mask extended into body/shoulder region")
        print(f"    - Video resolution or compression artifacts")
        print(f"    - Frame count mismatch between original and swap")
        print(f"  Consider re-generating affected videos with tighter face masks.{RESET}")

    # ── Criteria explanation ──
    header("EVALUATION CRITERIA")

    print(f"  Verdict uses {BOLD}multi-metric consensus{RESET} (2 of 3 must pass):")
    print()
    print(f"  {BOLD}1. PCK@0.05{RESET} (Percentage of Correct Keypoints)")
    print(f"     {DIM}% of (frame, landmark) pairs where per-landmark L2 < 5% of torso height.")
    print(f"     Based on PCK metric from MPII benchmark (V7Labs HPE Guide).")
    print(f"     Pass: >= {THRESHOLD_PCK:.0%}{RESET}")
    print()
    print(f"  {BOLD}2. Cosine Similarity{RESET}")
    print(f"     {DIM}Mean cosine similarity of hip-normalized pose vectors per frame.")
    print(f"     Measures shape agreement independent of scale (PMC9371146).")
    print(f"     Pass: >= {THRESHOLD_COSINE}{RESET}")
    print()
    print(f"  {BOLD}3. Temporal Correlations{RESET} (Pearson r)")
    print(f"     {DIM}Step-width correlation + body symmetry correlation over time.")
    print(f"     Standard in clinical gait analysis (PMC10886083, PMC11097739).")
    print(f"     Pass: both >= {THRESHOLD_CORR}{RESET}")
    print()


if __name__ == "__main__":
    main()
