"""Stage 1, ingest: pick a small set of sharp, evenly spaced keyframes from a phone video.

The selected frames are also re-encoded as a short H.264 clip, so frame i of the
clip SAM tracks is exactly frame i that VGGT reconstructs. That one-to-one
correspondence is what makes lifting masks into 3D a lookup instead of a search.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ..core.errors import InputError


@dataclass
class FrameSet:
    count: int
    size: tuple[int, int]  # (width, height)
    source_seconds: list[float]
    sharpness: list[float]
    duration_s: float


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError as error:
        raise InputError("That file doesn't look like a video we can read.") from error


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: high for crisp edges, low for motion blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_keyframes(
    video: Path,
    out_dir: Path,
    target: int = 48,
    candidate_fps: float = 12.0,
    max_side: int = 1024,
    min_frames: int = 12,
) -> FrameSet:
    """Write `out_dir/frames/NNN.jpg` and `out_dir/clip.mp4`; return the selection.

    Candidates are decoded at `candidate_fps`, split into `target` equal time bins,
    and the sharpest candidate in each bin wins. Bins whose best frame is far
    blurrier than the video's typical frame are dropped rather than polluting the
    reconstruction.
    """
    duration = probe_duration(video)
    if duration < 3:
        raise InputError("The video is too short. Film for 20 to 60 seconds while circling the table.")

    frames_dir = out_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        scale = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video),
                "-vf",
                f"fps={candidate_fps},{scale}",
                "-q:v",
                "2",
                f"{tmp}/%05d.jpg",
            ],
            check=True,
        )
        candidates = sorted(Path(tmp).glob("*.jpg"))
        if len(candidates) < min_frames:
            raise InputError("The video has too few frames to reconstruct. Film for at least 20 seconds.")
        grays = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in candidates]
        brightness = float(np.mean([g.mean() for g in grays[:: max(1, len(grays) // 20)]]))
        if brightness < 35:
            raise InputError("The video is too dark. Turn on more light and film again.")
        scores = np.array([sharpness(g) for g in grays])

        bins = np.array_split(np.arange(len(candidates)), min(target, len(candidates)))
        picks = [int(b[np.argmax(scores[b])]) for b in bins if len(b)]
        floor = 0.3 * float(np.median(scores))
        picks = [i for i in picks if scores[i] >= floor]
        if len(picks) < min_frames:
            raise InputError("Most of the video is blurry. Move the phone more slowly and keep it steady.")

        for n, index in enumerate(picks):
            shutil.copy(candidates[index], frames_dir / f"{n:03d}.jpg")

    height, width = cv2.imread(str(frames_dir / "000.jpg")).shape[:2]
    encode_clip(frames_dir, out_dir / "clip.mp4")
    selection = FrameSet(
        count=len(picks),
        size=(width, height),
        source_seconds=[round(i / candidate_fps, 3) for i in picks],
        sharpness=[round(float(scores[i]), 1) for i in picks],
        duration_s=round(duration, 2),
    )
    (out_dir / "frames.json").write_text(json.dumps(asdict(selection), indent=2) + "\n")
    return selection


def encode_clip(frames_dir: Path, clip: Path, fps: int = 10) -> None:
    """H.264 MP4 of the keyframes (SAM 3.1 rejects VP9/WebM), moov atom up front."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "%03d.jpg"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            "-movflags",
            "+faststart",
            str(clip),
        ],
        check=True,
    )


def load_frameset(job_dir: Path) -> FrameSet:
    data = json.loads((job_dir / "frames.json").read_text())
    data["size"] = tuple(data["size"])
    return FrameSet(**data)
