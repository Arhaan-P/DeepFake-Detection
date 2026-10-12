"""E0-native feature artifacts using Phase 2 seals, writes, quarantine and ledger.

Features are un-z-scored. Model statistics/checkpoint identity are separately bound
by the scorer. Legacy pickle caches are never elevated to managed study artifacts.
"""

import json
from importlib.metadata import version
from pathlib import Path
import tempfile
import time

import numpy as np

from .capture_registry import write_new_json
from .dataset_processing import (
    Ledger,
    final_test_denied,
    protect_sources,
    quarantine,
    source_issue,
)
from .e0 import JOINT_NAMES, validate_features, worker
from .media import probe_video, sha256_file
from .offline_eval import digest, seal, sealed_ok
from .pipeline import CacheIntegrityError
from .splits import validate_split

SCHEMA = "e0-native-feature-v1"
POSE_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"


def feature_recipe(pose_sha256=POSE_SHA256):
    root = Path(__file__).resolve().parents[1]
    return {
        "recipe_version": "e0-native60x78-v1",
        "pose_extractor": "MediaPipe-Pose-Lite-IMAGE; normalized-image-xyz",
        "pose_checkpoint_sha256": pose_sha256,
        "min_detection_confidence": 0.5,
        "min_pose_presence_confidence": 0.5,
        "decoder": "historical-cv2-VideoCapture; native orientation handling",
        "feature_schema": {
            "joints": list(JOINT_NAMES),
            "blocks": [
                "hip-relative-xyz36",
                "image-plane-angle-degrees6",
                "adjacent-index-difference36-repeat-last",
            ],
            "angles": [
                "left_knee",
                "right_knee",
                "left_hip",
                "right_hip",
                "left_ankle",
                "right_ankle",
            ],
        },
        "sequence_length": 60,
        "feature_dimension": 78,
        "resampling": "linear-detected-pose-index; minimum10; no physical velocity",
        "feature_normalization": "hip-center; no body-scale; z-score applied by checkpoint scorer",
        "quality_gate": "historical-min10-native33xyz-finite-v1; provisional",
        "runtime_versions": {
            name: version(name) for name in ("numpy", "mediapipe", "opencv-python")
        },
        "implementation_sha256": {
            p: sha256_file(root / p)
            for p in (
                "utils/gait_features.py",
                "utils/pose_extraction.py",
                "scripts/gait_verification/e0_worker.py",
            )
        },
    }


class E0FeatureStore:
    """Private exclusive sealed JSON, matching Phase 2's integrity trust boundary."""

    def __init__(self, root, recipe):
        self.root, self.recipe = Path(root), recipe
        self.recipe_id = digest(recipe, 64)

    def expected(self, asset):
        return {
            "schema_version": SCHEMA,
            "asset_id": asset.asset_id,
            "original_take_id": asset.original_take_id,
            "variant_id": asset.variant_id,
            "parent_asset_id": asset.parent_asset_id,
            "source_sha256": asset.sha256,
            "recipe": self.recipe,
        }

    def path(self, asset):
        return self.root / (digest(self.expected(asset), 64) + ".json")

    def get(self, asset):
        path = self.path(asset)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
            if not sealed_ok(record) or any(
                record.get(k) != v for k, v in self.expected(asset).items()
            ):
                raise ValueError("seal or recipe/lineage mismatch")
            if record.get("status") not in (
                "FEATURED",
                "INSUFFICIENT_GAIT",
            ) or not isinstance(record.get("reason"), str):
                raise ValueError("invalid E0 feature status")
            if record["status"] == "FEATURED":
                validate_features(record["values"])
            elif record.get("values") is not None:
                raise ValueError("features on an abstention")
            if not isinstance(
                record.get("runtime_sec"), (float, int)
            ) or not np.isfinite(record["runtime_sec"]):
                raise ValueError("invalid feature runtime")
            return record
        except (OSError, ValueError, KeyError, TypeError):
            raise CacheIntegrityError("E0_FEATURE_INTEGRITY") from None

    def put(self, asset, values, runtime_sec):
        values = validate_features(values).tolist() if values is not None else None
        record = seal(
            self.expected(asset)
            | {
                "status": "FEATURED" if values is not None else "INSUFFICIENT_GAIT",
                "reason": "OK" if values is not None else "TOO_FEW_NATIVE_POSES",
                "values": values,
                "runtime_sec": runtime_sec,
            }
        )
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_new_json(self.path(asset), record)


class MediaPipeE0Extractor:
    """Exact legacy extraction, separate from RTMPose; local weights only."""

    def __init__(self, model_path, recipe):
        self.model_path, self.recipe = Path(model_path).expanduser(), recipe
        if sha256_file(self.model_path) != recipe["pose_checkpoint_sha256"]:
            raise ValueError("MediaPipe checkpoint checksum mismatch")

    def __call__(self, video):
        probe_video(video)  # invalid media is an error, not too-few-poses quality
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "features.npy"
            result = worker(
                "extract",
                "--video",
                video,
                "--checkpoint",
                self.model_path,
                "--sha256",
                self.recipe["pose_checkpoint_sha256"],
                "--output",
                out,
            )
            if result["status"] == "INSUFFICIENT_GAIT":
                return None
            return validate_features(np.load(out, allow_pickle=False))


def process_e0(
    registry, split, data_root, store, asset_ids, extractor, should_stop=lambda: False
):
    """Explicit originals only. Retry missing work; preserve/quarantine invalid work.

    This has no enrollment, threshold or split mutation. All role checks precede
    source access. Failures stay visible; a valid cached rerun extracts nothing.
    """
    if split is not None:
        validate_split(split, registry.subjects())
    ids = sorted(set(asset_ids))
    if not ids:
        raise ValueError("Explicit E0 asset selection required")
    results = []
    ledger = Ledger(store.root / "ledger.jsonl")
    scope_checked = False
    for asset_id in ids:
        if should_stop():
            break
        row = {"asset_id": asset_id}
        results.append(row)
        try:
            asset = registry.asset(asset_id)
            if final_test_denied(asset, split):
                row.update(status="REFUSED", reason="LOCKED_FINAL_DENIED")
                continue
            if not asset.is_original:
                row.update(status="REFUSED", reason="E0_ORIGINALS_ONLY")
                continue
            if not scope_checked:
                protect_sources(registry, data_root, store.root)
                scope_checked = True
            if split is not None:
                registry.recording(asset_id, split["assignments"][asset.subject_id])
            issue = source_issue(registry, asset, data_root)
            if issue:
                row.update(status="REFUSED", reason=issue)
                continue
            try:
                record = store.get(asset)
            except CacheIntegrityError:
                quarantine(store.path(asset))
                record = None
            if record is not None:
                row.update(status="SKIPPED_VALID", reason=record["reason"])
            else:
                started = time.perf_counter()
                values = extractor(registry.path_of(asset, data_root))
                issue = source_issue(registry, asset, data_root)
                if issue:
                    raise ValueError(issue)
                store.put(asset, values, time.perf_counter() - started)
                row.update(
                    status="DONE",
                    reason="FEATURED" if values is not None else "INSUFFICIENT_GAIT",
                )
            ledger.append(
                task_id=digest(store.expected(asset)), **row, state=row["status"]
            )
        except Exception as exc:
            row.update(status="FAILED_RETRYABLE", reason=type(exc).__name__)
    return {
        "schema_version": "e0-processing-run-v1",
        "status": "INTERRUPTED"
        if len(results) != len(ids)
        else (
            "COMPLETE"
            if all(r["status"] in ("DONE", "SKIPPED_VALID") for r in results)
            else "PARTIAL"
        ),
        "recipe_id": store.recipe_id,
        "requested_assets": len(ids),
        "results": results,
        "evidence": "Engineering processing only; no scientific suitability established",
    }
