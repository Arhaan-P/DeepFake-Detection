"""
Pluggable Pose Backends (future work, plan Section 3 / experiments E1-E2)
=========================================================================
Every backend runs on a raw video and returns keypoints in ONE canonical
layout: MediaPipe Pose's 33-landmark index space. Joints a backend does not
predict are NaN. This lets the unchanged baseline feature math (hip index 23/24,
the 12 GAIT_LANDMARKS, the six joint angles) run on any backend, which is the
controlled comparison the roadmap asks for: identical videos, identical joints,
identical feature engineering -- only the pose estimator changes.

Backends
--------
mediapipe          MediaPipe Pose Landmarker, model lite|full|heavy, IMAGE or
                   VIDEO (temporal tracking) running mode. `lite` + IMAGE is the
                   frozen baseline (utils/pose_extraction.py). Also records the
                   metric `pose_world_landmarks` (RGB-only 3D estimate, E2).
rtmpose            RTMPose Halpe-26 (body + heels + toes) via rtmlib/onnxruntime.
vitpose            ViTPose coco_25 (body + feet) via rtmlib/onnxruntime.
rtmw3d             RTMW3D whole-body 3D pose from RGB via rtmlib (E2).
openpose           OpenPose BODY_25 through cv2.dnn; weights supplied by user.

2D-only backends (rtmpose, vitpose, openpose) have z = 0. A matched control
for them is the MediaPipe baseline with z dropped (see gait_descriptors
`coords_2d` family), so a z-less detector is not penalised for missing depth.

Author: DeepFake Detection Project
"""

import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

NUM_CANONICAL = 33

# Canonical (MediaPipe) indices of the 12 gait landmarks used by the baseline.
GAIT_LANDMARKS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
GAIT_JOINT_NAMES = [
    "L_Shoulder",
    "R_Shoulder",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
    "L_Heel",
    "R_Heel",
    "L_Foot",
    "R_Foot",
]

# backend-native index -> canonical MediaPipe index
HALPE26_TO_MP = {
    0: 0,
    5: 11,
    6: 12,
    7: 13,
    8: 14,
    9: 15,
    10: 16,
    11: 23,
    12: 24,
    13: 25,
    14: 26,
    15: 27,
    16: 28,
    24: 29,
    25: 30,
    20: 31,
    21: 32,
}
# easy_ViTPose "coco_25": COCO-17 reordered with neck(5) and mid-hip(14),
# followed by the six foot points.
COCO25_TO_MP = {
    0: 0,
    6: 11,
    7: 12,
    8: 13,
    9: 14,
    10: 15,
    11: 16,
    12: 23,
    13: 24,
    15: 25,
    16: 26,
    17: 27,
    18: 28,
    21: 29,
    24: 30,
    19: 31,
    22: 32,
}
# COCO-WholeBody (RTMW / RTMW3D): body 0-16, then feet 17-22.
WHOLEBODY_TO_MP = {
    0: 0,
    5: 11,
    6: 12,
    7: 13,
    8: 14,
    9: 15,
    10: 16,
    11: 23,
    12: 24,
    13: 25,
    14: 26,
    15: 27,
    16: 28,
    19: 29,
    22: 30,
    17: 31,
    20: 32,
}
# OpenPose BODY_25
BODY25_TO_MP = {
    0: 0,
    5: 11,
    2: 12,
    6: 13,
    3: 14,
    7: 15,
    4: 16,
    12: 23,
    9: 24,
    13: 25,
    10: 26,
    14: 27,
    11: 28,
    21: 29,
    24: 30,
    19: 31,
    22: 32,
}

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/1/pose_landmarker_{variant}.task"
)
RTMLIB_VITPOSE_URL = (
    "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco_25/"
    "vitpose-{size}-coco_25.onnx"
)


@dataclass
class PoseTrack:
    """Per-frame pose output for one video, canonical 33-slot layout.

    Frames without a detection are kept as all-NaN rows so missing-detection
    rate is measurable (utils/pose_quality.py).
    """

    keypoints: np.ndarray  # (N, 33, 3) x,y image-normalised [0,1], z relative
    confidence: np.ndarray  # (N, 33) visibility / score, NaN where missing
    fps: float
    width: int
    height: int
    backend: str
    world: Optional[np.ndarray] = None  # (N, 33, 3) 3D estimate, if any
    latency_ms: Optional[np.ndarray] = None  # (N,) inference time per frame
    meta: Dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.keypoints.shape[0])

    def detected(self) -> np.ndarray:
        """Boolean (N,) mask of frames where both hips were found."""
        return np.all(np.isfinite(self.keypoints[:, [23, 24], :2]), axis=(1, 2))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arrays = {
            "keypoints": self.keypoints.astype(np.float32),
            "confidence": self.confidence.astype(np.float32),
            "fps": np.float32(self.fps),
            "width": np.int32(self.width),
            "height": np.int32(self.height),
            "backend": np.array(self.backend),
        }
        if self.world is not None:
            arrays["world"] = self.world.astype(np.float32)
        if self.latency_ms is not None:
            arrays["latency_ms"] = self.latency_ms.astype(np.float32)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "PoseTrack":
        z = np.load(path, allow_pickle=False)
        return cls(
            keypoints=z["keypoints"].astype(np.float64),
            confidence=z["confidence"].astype(np.float64),
            fps=float(z["fps"]),
            width=int(z["width"]),
            height=int(z["height"]),
            backend=str(z["backend"]),
            world=z["world"].astype(np.float64) if "world" in z else None,
            latency_ms=z["latency_ms"] if "latency_ms" in z else None,
        )


def _empty(n: int):
    return (
        np.full((n, NUM_CANONICAL, 3), np.nan),
        np.full((n, NUM_CANONICAL), np.nan),
    )


def _cache_download(url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


def _pick_person(keypoints: np.ndarray, scores: np.ndarray) -> int:
    """Index of the most prominent person: largest confident bounding box."""
    best, best_area = 0, -1.0
    for i in range(len(keypoints)):
        ok = scores[i] > 0.3
        if ok.sum() < 3:
            continue
        pts = keypoints[i][ok]
        area = float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))
        if area > best_area:
            best, best_area = i, area
    return best


class PoseBackend:
    """Base class. Subclasses implement `_infer(frame_bgr, timestamp_ms)`."""

    name = "base"

    def _infer(self, frame: np.ndarray, timestamp_ms: int):
        """Return (kp (33,3), conf (33,), world (33,3) or None) or None."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset temporal state between videos (tracking backends)."""

    def process_video(self, video_path: str, max_frames: int = 0) -> PoseTrack:
        import cv2  # lazy: PoseTrack / pure-numpy users need no OpenCV

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise OSError(f"Cannot open video {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reset()

        kps, confs, worlds, lat = [], [], [], []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and idx >= max_frames):
                break
            t0 = time.perf_counter()
            out = self._infer(frame, int(round(idx * 1000.0 / fps)))
            lat.append((time.perf_counter() - t0) * 1000.0)
            if out is None:
                kp, cf = _empty(1)
                kps.append(kp[0])
                confs.append(cf[0])
                worlds.append(np.full((NUM_CANONICAL, 3), np.nan))
            else:
                kp, cf, w = out
                kps.append(kp)
                confs.append(cf)
                worlds.append(w if w is not None else np.full((33, 3), np.nan))
            idx += 1
        cap.release()

        world = np.array(worlds)
        return PoseTrack(
            keypoints=np.array(kps).reshape(-1, NUM_CANONICAL, 3),
            confidence=np.array(confs).reshape(-1, NUM_CANONICAL),
            fps=float(fps),
            width=width,
            height=height,
            backend=self.name,
            world=world if np.isfinite(world).any() else None,
            latency_ms=np.array(lat),
        )


class MediaPipeBackend(PoseBackend):
    """MediaPipe Pose Landmarker. variant=lite + mode=image is the baseline."""

    def __init__(
        self,
        variant: str = "lite",
        mode: str = "image",
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        model_dir: str = "~/.mediapipe/models",
    ):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._vision = vision
        self.variant, self.mode = variant, mode
        self.name = f"mediapipe_{variant}" + ("_video" if mode == "video" else "")
        model_path = _cache_download(
            MEDIAPIPE_MODEL_URL.format(variant=variant), os.path.expanduser(model_dir)
        )
        self._options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=(
                vision.RunningMode.VIDEO
                if mode == "video"
                else vision.RunningMode.IMAGE
            ),
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
        )
        self._landmarker = None
        self.reset()

    def reset(self) -> None:
        # VIDEO mode requires monotonically increasing timestamps per video,
        # so a fresh landmarker is created for every video.
        if self._landmarker is not None:
            self._landmarker.close()
        self._landmarker = self._vision.PoseLandmarker.create_from_options(
            self._options
        )

    def _infer(self, frame, timestamp_ms):
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self.mode == "video":
            res = self._landmarker.detect_for_video(image, timestamp_ms)
        else:
            res = self._landmarker.detect(image)
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks[0]
        kp = np.array([[lm.x, lm.y, lm.z] for lm in lms])
        conf = np.array([lm.visibility for lm in lms])
        world = None
        if res.pose_world_landmarks:
            world = np.array([[w.x, w.y, w.z] for w in res.pose_world_landmarks[0]])
        return kp, conf, world


class _RtmlibBackend(PoseBackend):
    """Shared plumbing for rtmlib top-down models (detector + pose model)."""

    mapping: Dict[int, int] = {}

    def _to_canonical(self, kpts, scores, width, height, z=None):
        i = _pick_person(kpts, scores)
        kp, conf = _empty(1)
        kp, conf = kp[0], conf[0]
        for src, dst in self.mapping.items():
            kp[dst, 0] = kpts[i, src, 0] / width
            kp[dst, 1] = kpts[i, src, 1] / height
            kp[dst, 2] = 0.0 if z is None else z[i, src]
            conf[dst] = scores[i, src]
        return kp, conf, i


class RTMPoseBackend(_RtmlibBackend):
    """RTMPose Halpe-26 (includes heels and big toes) via rtmlib."""

    mapping = HALPE26_TO_MP

    def __init__(self, mode: str = "balanced", device: str = "cpu"):
        from rtmlib import BodyWithFeet

        self.name = f"rtmpose_{mode}"
        self._model = BodyWithFeet(mode=mode, backend="onnxruntime", device=device)

    def _infer(self, frame, timestamp_ms):
        kpts, scores = self._model(frame)
        if len(kpts) == 0:
            return None
        kp, conf, _ = self._to_canonical(kpts, scores, frame.shape[1], frame.shape[0])
        return kp, conf, None


class ViTPoseBackend(_RtmlibBackend):
    """ViTPose coco_25 (body + feet) via rtmlib, YOLOX person detector."""

    mapping = COCO25_TO_MP

    def __init__(self, size: str = "b", device: str = "cpu"):
        from rtmlib import YOLOX, ViTPose

        self.name = f"vitpose_{size}"
        self._det = YOLOX(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            model_input_size=(640, 640),
            backend="onnxruntime",
            device=device,
        )
        self._pose = ViTPose(
            RTMLIB_VITPOSE_URL.format(size=size),
            model_input_size=(192, 256),
            backend="onnxruntime",
            device=device,
        )

    def _infer(self, frame, timestamp_ms):
        boxes = self._det(frame)
        if len(boxes) == 0:
            return None
        kpts, scores = self._pose(frame, bboxes=boxes)
        kpts = np.asarray(kpts).reshape(len(boxes), -1, 2)
        scores = np.asarray(scores).reshape(len(boxes), -1)
        kp, conf, _ = self._to_canonical(kpts, scores, frame.shape[1], frame.shape[0])
        return kp, conf, None


class RTMW3DBackend(_RtmlibBackend):
    """RTMW3D whole-body 3D pose from a single RGB frame (E2 3D estimator).

    `keypoints` holds image-normalised 2D plus the model's relative depth;
    `world` holds the model's 3D output (crop-space x,y rescaled to the crop
    height, model z), i.e. a scale-normalised 3D skeleton.
    """

    mapping = WHOLEBODY_TO_MP

    def __init__(self, device: str = "cpu"):
        from rtmlib import Wholebody3d

        self.name = "rtmw3d"
        self._model = Wholebody3d(backend="onnxruntime", device=device)

    def _infer(self, frame, timestamp_ms):
        kpts3d, scores, _, kpts2d = self._model(frame)
        if len(kpts2d) == 0:
            return None
        kpts3d = np.asarray(kpts3d)
        kp, conf, i = self._to_canonical(
            np.asarray(kpts2d), np.asarray(scores), frame.shape[1], frame.shape[0]
        )
        world = np.full((NUM_CANONICAL, 3), np.nan)
        h_in = float(self._model.pose_model.model_input_size[1])
        for src, dst in self.mapping.items():
            world[dst, 0] = kpts3d[i, src, 0] / h_in
            world[dst, 1] = kpts3d[i, src, 1] / h_in
            world[dst, 2] = kpts3d[i, src, 2]
            kp[dst, 2] = kpts3d[i, src, 2]
        return kp, conf, world


class OpenPoseBackend(PoseBackend):
    """OpenPose BODY_25 through OpenCV DNN (single-person heatmap argmax).

    Requires the official `pose_deploy.prototxt` and `pose_iter_584000.caffemodel`
    from the OpenPose model zoo; they are not downloaded automatically. Taking
    the per-heatmap argmax assumes one walker per frame, which holds for the
    GaitDeepfake-13 recordings but not for crowded scenes.
    """

    name = "openpose_body25"

    def __init__(self, prototxt: str, caffemodel: str, input_height: int = 368):
        import cv2

        self._net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self._h = input_height

    def _infer(self, frame, timestamp_ms):
        import cv2

        h, w = frame.shape[:2]
        in_w = int(round(self._h * w / h / 8) * 8)
        blob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255, (in_w, self._h), (0, 0, 0), swapRB=False, crop=False
        )
        self._net.setInput(blob)
        out = self._net.forward()  # (1, 78, H', W'): 25 heatmaps + bg + PAFs
        kp, conf = _empty(1)
        kp, conf = kp[0], conf[0]
        for src, dst in BODY25_TO_MP.items():
            hm = out[0, src]
            _, score, _, loc = cv2.minMaxLoc(hm)
            if score < 0.05:
                continue
            kp[dst] = [loc[0] / hm.shape[1], loc[1] / hm.shape[0], 0.0]
            conf[dst] = score
        if not np.isfinite(kp[[23, 24], 0]).all():
            return None
        return kp, conf, None


def available_backends() -> List[str]:
    return [
        "mediapipe_lite",
        "mediapipe_full",
        "mediapipe_heavy",
        "mediapipe_lite_video",
        "mediapipe_full_video",
        "mediapipe_heavy_video",
        "rtmpose_lightweight",
        "rtmpose_balanced",
        "rtmpose_performance",
        "vitpose_s",
        "vitpose_b",
        "vitpose_l",
        "rtmw3d",
        "openpose_body25",
    ]


def create_backend(name: str, **kwargs) -> PoseBackend:
    """Build a backend from a name in `available_backends()`."""
    if name.startswith("mediapipe_"):
        parts = name.split("_")
        mode = "video" if parts[-1] == "video" else "image"
        return MediaPipeBackend(variant=parts[1], mode=mode, **kwargs)
    if name.startswith("rtmpose_"):
        return RTMPoseBackend(mode=name.split("_", 1)[1], **kwargs)
    if name.startswith("vitpose_"):
        return ViTPoseBackend(size=name.split("_", 1)[1], **kwargs)
    if name == "rtmw3d":
        return RTMW3DBackend(**kwargs)
    if name == "openpose_body25":
        return OpenPoseBackend(**kwargs)
    raise ValueError(f"Unknown backend '{name}'. Options: {available_backends()}")
