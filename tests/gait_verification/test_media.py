"""Actual FFmpeg synthetic VFR media regression; no real biometric data needed."""

import shutil
import subprocess
import numpy as np
import pytest
from gait_verification.media import probe_video, decode_frames, sha256_file
from gait_verification.stabilization import stabilize, StabilizationConfig
from gait_verification.contracts import Recording


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)
def test_stabilization_preserves_irregular_pts_and_original(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    filters = subprocess.check_output(
        ["ffmpeg", "-hide_banner", "-filters"], stderr=subprocess.DEVNULL, text=True
    )
    if "vidstabdetect" not in filters:
        pytest.skip("libvidstab unavailable")
    source = tmp_path / "vfr.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x192:rate=30:duration=1.2",
            "-vf",
            "select='not(eq(mod(n,7),0))'",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
    )
    m = probe_video(source)
    assert len(set(np.round(np.diff(m.timestamps), 4))) > 1
    r = Recording(
        "raw",
        "take",
        "synthetic",
        "session",
        "toward",
        "2026-10-12",
        m.sha256,
        "development",
    )
    result = stabilize(
        source, tmp_path / "derivative", r, StabilizationConfig(enabled=True)
    )
    assert result["status"] == "OK"
    np.testing.assert_allclose(
        np.array(result["input"]["timestamps"])
        + result["timing_transform"]["pts_offset_sec"],
        result["output"]["timestamps"],
        atol=1e-6,
    )
    assert sha256_file(source) == m.sha256
    assert len(list(decode_frames(m))) == len(m.timestamps)
    with pytest.raises(FileExistsError):
        stabilize(source, tmp_path / "derivative", r, StabilizationConfig(enabled=True))
