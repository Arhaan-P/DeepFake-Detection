"""Source hashes and presentation timestamps from ffprobe, never nominal FPS time."""

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
from .contracts import VideoMetadata


class InvalidMedia(ValueError):
    """Unreadable, truncated or structurally unsupported video (maps to INVALID_INPUT)."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probe_video(path):
    path = Path(path).resolve(strict=True)
    try:
        result = _run_ffprobe(path)
    except subprocess.CalledProcessError as exc:
        raise InvalidMedia("ffprobe could not read the video") from exc
    if result.stderr.strip():
        raise InvalidMedia("ffprobe reported decode errors")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise InvalidMedia("ffprobe returned no stream data") from exc
    if len(data.get("streams", [])) != 1:
        raise InvalidMedia("Exactly one video stream required")
    stream = data["streams"][0]
    pts = tuple(float(f["pts_time"]) for f in data.get("frames", []))
    if not pts or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise InvalidMedia("Missing/nonmonotonic presentation timestamps")
    return _metadata(path, stream, pts)


def _run_ffprobe(path):
    return subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "stream=index,width,height,codec_name,r_frame_rate,avg_frame_rate,time_base,duration:stream_side_data=rotation:frame=pts_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def _metadata(path, stream, pts):
    return VideoMetadata(
        str(path),
        sha256_file(path),
        stream["width"],
        stream["height"],
        stream["r_frame_rate"],
        stream["avg_frame_rate"],
        stream["time_base"],
        pts,
        stream["codec_name"],
        int(
            next(
                (
                    d["rotation"]
                    for d in stream.get("side_data_list", [])
                    if "rotation" in d
                ),
                0,
            )
        ),
        float(stream["duration"])
        if stream.get("duration") not in (None, "N/A")
        else None,
    )


def decode_frames(metadata):
    """Decode native orientation in presentation order; require one frame per PTS."""
    if metadata.rotation % 90:
        raise InvalidMedia("Only right-angle display rotations are supported")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-noautorotate",
        "-i",
        metadata.path,
        "-map",
        "0:v:0",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    import tempfile

    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        size = metadata.width * metadata.height * 3
        try:
            for i, pts in enumerate(metadata.timestamps):
                raw = proc.stdout.read(size)
                if len(raw) != size:
                    raise InvalidMedia("Decoded frames do not match source PTS")
                frame = np.frombuffer(raw, np.uint8).reshape(
                    metadata.height, metadata.width, 3
                )
                # Explicit lossless display-orientation transform, PTS unchanged.
                yield (
                    i,
                    pts,
                    np.ascontiguousarray(np.rot90(frame, metadata.rotation // 90)),
                )
            if proc.stdout.read(1):
                raise InvalidMedia("Additional decoded frame without PTS")
            if proc.wait() != 0:
                raise InvalidMedia("ffmpeg decode failed")
            errors.seek(0)
            if errors.read():
                raise InvalidMedia("ffmpeg reported damaged media")
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.terminate()
                proc.wait()


def metadata_dict(metadata):
    return asdict(metadata)
