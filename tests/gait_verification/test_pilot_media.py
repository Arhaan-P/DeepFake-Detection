"""Actual decoder/stabilizer integration on generated media, not human inference."""

from dataclasses import replace
import shutil
import subprocess
import numpy as np
import pytest
from gait_verification.contracts import Recording
from gait_verification.media import probe_video, decode_frames, sha256_file
from gait_verification.stabilization import stabilize, StabilizationConfig

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)


def source_video(tmp_path):
    path = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x192:rate=30:duration=1.4",
            "-c:v",
            "libx264",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.mark.parametrize("smoothing", [None, 10, 15])
def test_raw_and_both_stabilizers_source_lineage_and_time(
    tmp_path, monkeypatch, smoothing
):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    filters = subprocess.check_output(
        ["ffmpeg", "-hide_banner", "-filters"], stderr=subprocess.DEVNULL, text=True
    )
    if "vidstabdetect" not in filters:
        pytest.skip("libvidstab unavailable")
    path = source_video(tmp_path)
    meta = probe_video(path)
    record = Recording(
        "raw",
        "take",
        "S001",
        "session1",
        "toward",
        "2026-10-12",
        meta.sha256,
        "development",
    )
    result = stabilize(
        path,
        tmp_path / "result",
        record,
        StabilizationConfig(enabled=smoothing is not None, smoothing=smoothing or 10),
    )
    assert (
        result["recording"]["original_take_id"] == "take"
        and result["input"]["sha256"] == meta.sha256
    )
    assert result["status"] == "OK" and sha256_file(path) == meta.sha256
    assert len(result["output"]["timestamps"]) == len(meta.timestamps)
    a = np.array(meta.timestamps)
    b = np.array(result["output"]["timestamps"])
    np.testing.assert_allclose(a - a[0], b - b[0], atol=0.0001, rtol=0)
    if smoothing is None:
        assert (
            result["output"]["path"] == str(path.resolve()) and result["commands"] == []
        )
    else:
        assert (tmp_path / "result" / "transforms.trf").is_file()
    with pytest.raises(ValueError):
        stabilize(
            path,
            tmp_path / "bad",
            replace(record, source_sha256="bad"),
            StabilizationConfig(enabled=True),
        )


def test_actual_portrait_rotation_keeps_source_pts(tmp_path):
    source = source_video(tmp_path)
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(source),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )
    raw_meta, meta = probe_video(source), probe_video(rotated)
    assert meta.rotation == 90
    raw = list(decode_frames(raw_meta))
    portrait = list(decode_frames(meta))
    assert portrait[0][2].shape[:2] == (128, 192)
    np.testing.assert_array_equal(portrait[0][2], np.rot90(raw[0][2]))
    np.testing.assert_array_equal(meta.timestamps, raw_meta.timestamps)
