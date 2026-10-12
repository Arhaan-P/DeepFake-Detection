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

    def __init__(
        self,
        detection_model,
        pose_model,
        *,
        expected_detector_sha256=None,
        expected_pose_sha256=None,
    ):
        for path, expected in (
            (detection_model, expected_detector_sha256),
            (pose_model, expected_pose_sha256),
        ):
            if expected is not None and sha256_file(path) != expected:
                raise ValueError("RTMPose artifact checksum mismatch")
        from rtmlib import Body
        import onnxruntime as ort

        ort.preload_dlls()

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
            device="cuda",
            to_openpose=False,
        )
        for name, model in (
            ("detector", self.runtime.det_model),
            ("pose", self.runtime.pose_model),
        ):
            if "CUDAExecutionProvider" not in model.session.get_providers():
                raise RuntimeError(f"{name} did not initialize with CUDA")
            model.session.disable_fallback()

    def __call__(self, frame):
        self.last_bboxes = self.runtime.det_model(frame)
        if len(self.last_bboxes) == 0:
            # RTMLib's pose-only API defaults to a full-image crop for an empty
            # bbox list. That is inappropriate after a failed person detector.
            return np.empty((0, 17, 2)), np.empty((0, 17))
        return self.runtime.pose_model(frame, bboxes=self.last_bboxes)


class NativePoseExtractor:
    def __init__(
        self,
        detector,
        original_take_id=None,
        variant_id="raw",
        max_center_jump=0.25,
        collect_diagnostics=False,
    ):
        self.detector = detector
        self.original_take_id = original_take_id
        self.variant_id = variant_id
        self.max_center_jump = max_center_jump  # provisional fraction of frame diagonal
        self.collect_diagnostics = collect_diagnostics

    def extract(self, video_path):
        meta = probe_video(video_path)
        width, height = (
            (meta.height, meta.width)
            if meta.rotation % 180
            else (meta.width, meta.height)
        )
        count = {"coco17": 17, "halpe26": 26, "mediapipe33": 33}[self.detector.layout]
        xy, conf, obs, boxes, tracks = [], [], [], [], []
        counts, events, blur, dark = [], [], [], []
        candidates, candidate_scores, candidate_masks = [], [], []
        previous_center = None
        track_id = 0
        interrupted = False
        for _, _, frame in decode_frames(meta):
            if self.collect_diagnostics:
                import cv2

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                edge = np.concatenate(
                    [
                        gray[:8].ravel(),
                        gray[-8:].ravel(),
                        gray[:, :8].ravel(),
                        gray[:, -8:].ravel(),
                    ]
                )
                dark.append(float((edge < 8).mean()))
            points, scores = self.detector(frame)
            points, scores = np.asarray(points), np.asarray(scores)
            counts.append(len(points))
            if len(points) and (
                points.shape != (len(points), count, 2)
                or scores.shape != (len(points), count)
            ):
                raise ValueError(
                    "Detector layout does not match declared native layout"
                )
            if self.collect_diagnostics:
                points = points.reshape(-1, count, 2)
                scores = scores.reshape(-1, count)
                mask = (
                    np.isfinite(points).all(axis=-1)
                    & np.isfinite(scores)
                    & (scores > 0)
                )
                candidates.append(np.where(np.isfinite(points), points, 0))
                candidate_scores.append(np.where(np.isfinite(scores), scores, 0))
                candidate_masks.append(mask)
            event = 0
            if len(points) > 1:
                if not self.collect_diagnostics:
                    raise InsufficientGait(
                        "Multiple people; identity continuity ambiguous"
                    )
                event = 2
            if len(points) == 0 or event == 2:
                xy.append(np.zeros((count, 2)))
                conf.append(np.zeros(count))
                obs.append(np.zeros(count, bool))
                boxes.append(np.zeros(4))
                tracks.append(-1)
                events.append(event or 1)
                # A brief missing detection does not itself prove a new person.
                # Ambiguous multi-person frames require a fresh continuity ID.
                interrupted = interrupted or event == 2
                continue
            p, s = points[0], scores[0]
            if p.shape != (count, 2) or s.shape != (count,):
                raise ValueError(
                    "Detector layout does not match declared native layout"
                )
            observed = np.isfinite(p).all(axis=1) & np.isfinite(s) & (s > 0)
            if observed.sum() < 4:
                if not self.collect_diagnostics:
                    raise InsufficientGait("No stable body detection")
                event = 4
            bbox = (
                np.concatenate([p[observed].min(0), p[observed].max(0)])
                if observed.any()
                else np.zeros(4)
            )
            detected_boxes = getattr(self.detector, "last_bboxes", None)
            if detected_boxes is not None and len(detected_boxes) == 1:
                bbox = np.asarray(detected_boxes[0][:4])
            center = (bbox[:2] + bbox[2:]) / 2
            if previous_center is not None and np.linalg.norm(
                center - previous_center
            ) > self.max_center_jump * np.hypot(meta.width, meta.height):
                if not self.collect_diagnostics:
                    raise InsufficientGait("Track discontinuity")
                event = 3
            if interrupted or event in (3, 4):
                track_id += 1
            interrupted = event == 4
            previous_center = center
            # Preserve all finite estimator values, including low/zero scores.
            xy.append(np.where(np.isfinite(p), p, 0))
            conf.append(np.where(np.isfinite(s), s, 0))
            obs.append(observed)
            boxes.append(bbox)
            tracks.append(track_id if event != 4 else -1)
            events.append(event)
        extra = {}
        if self.collect_diagnostics:
            max_people = max(counts, default=0)
            cxy = np.zeros((len(xy), max_people, count, 2))
            cconf = np.zeros((len(xy), max_people, count))
            cmask = np.zeros((len(xy), max_people, count), bool)
            for i, people in enumerate(counts):
                cxy[i, :people] = candidates[i]
                cconf[i, :people] = candidate_scores[i]
                cmask[i, :people] = candidate_masks[i]
            extra = dict(
                candidate_xy=cxy, candidate_confidence=cconf, candidate_observed=cmask
            )
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
            detection_counts=np.array(counts) if self.collect_diagnostics else None,
            tracking_events=np.array(events) if self.collect_diagnostics else None,
            blur_scores=np.array(blur) if self.collect_diagnostics else None,
            border_dark_fraction=np.array(dark) if self.collect_diagnostics else None,
            **extra,
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
