"""Native pose cache and detector injection; no dependence on legacy cache semantics."""

from dataclasses import asdict
import json
import numpy as np
from .contracts import PoseSequence
from .media import probe_video, decode_frames, sha256_file
from .skeleton_contracts import InsufficientGait


class RTMPoseDetector:
    """RTMLib Body COCO17, explicit local ONNX paths (no implicit downloads).

    Native scores are preserved. Multiple people abort extraction instead of
    selecting the largest person. Detector/pose weights are frozen ONNX sessions.
    """

    layout = "coco17"

    def __init__(self, detection_model, pose_model):
        from rtmlib import Body
        import onnxruntime as ort

        def input_size(path):
            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            shape = session.get_inputs()[0].shape
            if (
                len(shape) != 4
                or shape[1] != 3
                or not all(isinstance(x, int) and x > 0 for x in shape[2:])
            ):
                raise ValueError("Requires static NCHW RGB detector/pose ONNX inputs")
            return (shape[3], shape[2])

        self.backend_id = (
            "rtmpose-coco17:"
            + sha256_file(detection_model)
            + ":"
            + sha256_file(pose_model)
        )
        self.runtime = Body(
            det=detection_model,
            pose=pose_model,
            det_input_size=input_size(detection_model),
            pose_input_size=input_size(pose_model),
            backend="onnxruntime",
            device="cpu",
            to_openpose=False,
        )

    def __call__(self, frame):
        return self.runtime(frame)


class NativePoseExtractor:
    def __init__(
        self, detector, original_take_id=None, variant_id="raw", max_center_jump=0.25
    ):
        self.detector = detector
        self.original_take_id = original_take_id
        self.variant_id = variant_id
        self.max_center_jump = max_center_jump  # provisional fraction of frame diagonal

    def extract(self, video_path):
        meta = probe_video(video_path)
        width, height = (
            (meta.height, meta.width)
            if meta.rotation % 180
            else (meta.width, meta.height)
        )
        count = {"coco17": 17, "halpe26": 26, "mediapipe33": 33}[self.detector.layout]
        xy, conf, obs, boxes, tracks = [], [], [], [], []
        previous_center = None
        for _, _, frame in decode_frames(meta):
            points, scores = self.detector(frame)
            points, scores = np.asarray(points), np.asarray(scores)
            if len(points) > 1:
                raise InsufficientGait("Multiple people; identity continuity ambiguous")
            if len(points) == 0:
                xy.append(np.zeros((count, 2)))
                conf.append(np.zeros(count))
                obs.append(np.zeros(count, bool))
                boxes.append(np.zeros(4))
                tracks.append(-1)
                continue
            p, s = points[0], scores[0]
            if p.shape != (count, 2) or s.shape != (count,):
                raise ValueError(
                    "Detector layout does not match declared native layout"
                )
            observed = np.isfinite(p).all(axis=1) & np.isfinite(s) & (s > 0)
            if observed.sum() < 4:
                raise InsufficientGait("No stable body detection")
            bbox = np.concatenate([p[observed].min(0), p[observed].max(0)])
            center = (bbox[:2] + bbox[2:]) / 2
            if previous_center is not None and np.linalg.norm(
                center - previous_center
            ) > self.max_center_jump * np.hypot(meta.width, meta.height):
                raise InsufficientGait("Track discontinuity")
            previous_center = center
            xy.append(np.where(observed[:, None], p, 0))
            conf.append(np.where(observed, s, 0))
            obs.append(observed)
            boxes.append(bbox)
            tracks.append(0)
        seq = PoseSequence(
            np.array(xy),
            np.array(conf),
            np.array(obs),
            np.array(meta.timestamps),
            self.detector.layout,
            width,
            height,
            self.original_take_id or meta.sha256,
            meta.sha256,
            self.detector.backend_id + f":display-rotation={meta.rotation}",
            np.arange(len(xy)),
            np.array(tracks),
            np.array(boxes),
            self.variant_id,
        )
        seq.validate()
        return seq


def save_pose(sequence, path):
    sequence.validate()
    fields = asdict(sequence)
    arrays = {k: v for k, v in fields.items() if isinstance(v, np.ndarray)}
    metadata = {k: v for k, v in fields.items() if k not in arrays}
    # Exclusive creation: caches are immutable artifacts too.
    with open(path, "xb") as f:
        np.savez_compressed(f, **arrays, metadata=json.dumps(metadata, sort_keys=True))


def load_pose(path):
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"]))
        sequence = PoseSequence(
            **metadata, **{k: z[k] for k in z.files if k != "metadata"}
        )
    sequence.validate()
    return sequence


class MediaPipeDetector:
    """Native 33-point compatibility baseline in IMAGE mode, local frozen weights."""

    layout = "mediapipe33"

    def __init__(self, model_path):
        import mediapipe as mp

        self.mp = mp
        self.backend_id = f"mediapipe-{mp.__version__}-image:" + sha256_file(model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=2,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def __call__(self, frame):
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame[..., ::-1]),
        )
        result = self.landmarker.detect(image)
        height, width = frame.shape[:2]
        xy = np.array(
            [
                [[p.x * width, p.y * height] for p in person]
                for person in result.pose_landmarks
            ]
        )
        scores = np.array(
            [
                [min(p.visibility, p.presence) for p in person]
                for person in result.pose_landmarks
            ]
        )
        return xy, scores

    def close(self):
        self.landmarker.close()
