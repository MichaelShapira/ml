"""
Small media helpers for the demo notebook.

Everything here runs client side (notebook), not in the endpoint. The point is
to be able to build every H3 input out of H3's own output: generate one clip
with t2va, then pull its first frame, its last frame and its stereo soundtrack
back out and feed them to i2va / l2va / fl2va / ref2va. That keeps the demos
self-contained - no external asset URLs to rot - and it exercises the fact that
H3 emits real 32 kHz stereo audio, not a silent MP4.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ffmpeg_exe",
    "ffprobe_json",
    "probe",
    "MediaInfo",
    "first_frame",
    "last_frame",
    "frame_at",
    "extract_audio",
    "trim",
    "data_uri",
    "upload",
]


def ffmpeg_exe() -> str:
    """Resolve an ffmpeg binary, preferring the one imageio-ffmpeg bundles."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ffmpeg not found. Install it with `pip install imageio-ffmpeg` "
            "or `conda install -c conda-forge ffmpeg`."
        ) from exc


def _ffprobe_exe() -> str | None:
    return shutil.which("ffprobe")


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError("ffmpeg failed:\n  " + "\n  ".join(tail))
    return proc.stdout


def ffprobe_json(path: str | Path) -> dict:
    exe = _ffprobe_exe()
    if not exe:
        raise RuntimeError("ffprobe not on PATH; install ffmpeg to use probe()")
    out = _run([exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    return json.loads(out)


@dataclass
class MediaInfo:
    duration_s: float
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool
    audio_sample_rate: int | None
    audio_channels: int | None

    def __str__(self) -> str:
        geom = f"{self.width}x{self.height}" if self.width else "no video"
        fps = f" @ {self.fps:.3g} fps" if self.fps else ""
        audio = (
            f", audio {self.audio_channels}ch {self.audio_sample_rate} Hz"
            if self.has_audio
            else ", no audio"
        )
        return f"{self.duration_s:.3f}s, {geom}{fps}{audio}"


def probe(path: str | Path) -> MediaInfo:
    meta = ffprobe_json(path)
    video = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in meta.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps = None
    if video and video.get("avg_frame_rate", "0/0") not in ("0/0", ""):
        num, _, den = video["avg_frame_rate"].partition("/")
        try:
            fps = float(num) / float(den or 1)
        except (TypeError, ValueError, ZeroDivisionError):
            fps = None
    return MediaInfo(
        duration_s=float(meta.get("format", {}).get("duration", 0.0) or 0.0),
        width=int(video["width"]) if video and video.get("width") else None,
        height=int(video["height"]) if video and video.get("height") else None,
        fps=fps,
        has_audio=audio is not None,
        audio_sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        audio_channels=int(audio["channels"]) if audio and audio.get("channels") else None,
    )


def frame_at(video: str | Path, out_png: str | Path, seconds: float) -> Path:
    """Extract a single frame as PNG.

    ``-ss`` before ``-i`` seeks on keyframes (fast, approximate); putting it
    after ``-i`` decodes up to the timestamp, which is what you want for an
    exact frame. These clips are short, so accuracy wins.
    """
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([ffmpeg_exe(), "-y", "-v", "error", "-i", str(video), "-ss", f"{max(seconds, 0):.3f}",
          "-frames:v", "1", str(out)])
    return out


def first_frame(video: str | Path, out_png: str | Path) -> Path:
    return frame_at(video, out_png, 0.0)


def last_frame(video: str | Path, out_png: str | Path, back_off_s: float = 0.05) -> Path:
    """Extract the final frame.

    Backing off a hair from the reported duration avoids landing past the last
    decodable frame, which yields an empty file instead of an error.
    """
    duration = probe(video).duration_s
    return frame_at(video, out_png, max(duration - back_off_s, 0.0))


def extract_audio(video: str | Path, out_wav: str | Path, *, sample_rate: int = 32000) -> Path:
    """Pull the soundtrack out as stereo WAV at H3's native 32 kHz."""
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([ffmpeg_exe(), "-y", "-v", "error", "-i", str(video), "-vn",
          "-ac", "2", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(out)])
    return out


def trim(src: str | Path, out: str | Path, *, start_s: float = 0.0, duration_s: float | None = None) -> Path:
    """Cut a clip. Re-encodes, because reference clips must start on a real frame."""
    dst = Path(out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-i", str(src), "-ss", f"{start_s:.3f}"]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", str(dst)]
    _run(cmd)
    return dst


def data_uri(path: str | Path) -> str:
    """Inline a file as a ``data:`` URI.

    Fine for keyframe PNGs. Do not do this for video references: the async
    invocation payload goes through S3 and base64 inflates by a third, so upload
    those with :func:`upload` instead.
    """
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def upload(path: str | Path, s3_uri_prefix: str, *, s3_client=None) -> str:
    """Upload a local file under an ``s3://bucket/prefix`` and return its URI."""
    import boto3

    s3 = s3_client or boto3.client("s3")
    p = Path(path)
    if not s3_uri_prefix.startswith("s3://"):
        raise ValueError(f"expected an s3:// prefix, got {s3_uri_prefix!r}")
    bucket, _, key_prefix = s3_uri_prefix[5:].partition("/")
    key = f"{key_prefix.rstrip('/')}/{p.name}" if key_prefix else p.name
    content_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    s3.upload_file(str(p), bucket, key, ExtraArgs={"ContentType": content_type})
    return f"s3://{bucket}/{key}"
