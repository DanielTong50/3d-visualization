"""Stage 2, reconstruct: VGGT predicts cameras and depth for all keyframes in one pass.

Runs on a GPU inside the Modal image defined in modal_app.py. Only this module
imports torch; everything downstream works on the saved `recon.npz`.

Output arrays (N frames at VGGT resolution h x w):
    points      (N, h, w, 3) float32  world points from depth + cameras
    confidence  (N, h, w)    float32  VGGT depth confidence
    colors      (N, h, w, 3) uint8
    extrinsic   (N, 3, 4)    float32  camera-from-world, OpenCV convention
    intrinsic   (N, 3, 3)    float32  at VGGT resolution
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

PATCH = 14
LONG_SIDE = 518


def vggt_size(width: int, height: int) -> tuple[int, int]:
    """Resize target keeping aspect: long side 518, both sides multiples of the 14 px patch."""
    scale = LONG_SIDE / max(width, height)
    return (
        max(PATCH, round(width * scale / PATCH) * PATCH),
        max(PATCH, round(height * scale / PATCH) * PATCH),
    )


def load_images(paths: list[Path]) -> tuple[np.ndarray, tuple[int, int]]:
    """Frames as float32 (N, 3, h, w) in [0, 1]. Unlike VGGT's own loader this never
    crops, so portrait phone video keeps its full field of view."""
    with Image.open(paths[0]) as first:
        size = vggt_size(*first.size)
    stack = []
    for path in paths:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BICUBIC))
        stack.append(pixels.transpose(2, 0, 1))
    return np.stack(stack).astype(np.float32) / 255.0, size


DEFAULT_WEIGHTS = "facebook/VGGT-1B-Commercial"


def load_model(weights: str = DEFAULT_WEIGHTS):
    import torch
    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained(weights).to("cuda").eval()
    model.weights_name = weights
    torch.cuda.reset_peak_memory_stats()
    return model


def run_vggt(model, frames_dir: Path, out_path: Path) -> dict:
    import torch
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    paths = sorted(frames_dir.glob("*.jpg"))
    images_np, (width, height) = load_images(paths)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.cuda.reset_peak_memory_stats()

    started = time.monotonic()
    images = torch.from_numpy(images_np).to("cuda")[None]
    with torch.no_grad():
        with torch.autocast("cuda", dtype=dtype):
            tokens, patch_start = model.aggregator(images)
        pose_enc = model.camera_head(tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth, depth_conf = model.depth_head(tokens, images, patch_start)
    torch.cuda.synchronize()
    inferred = time.monotonic()

    extrinsic = extrinsic.squeeze(0).float().cpu().numpy()
    intrinsic = intrinsic.squeeze(0).float().cpu().numpy()
    depth = depth.squeeze(0).float().cpu().numpy()
    points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic).astype(np.float32)
    colors = (images_np.transpose(0, 2, 3, 1) * 255).round().astype(np.uint8)

    np.savez_compressed(
        out_path,
        points=points,
        confidence=depth_conf.squeeze(0).float().cpu().numpy().astype(np.float32),
        colors=colors,
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
    )
    return {
        "frames": len(paths),
        "resolution": [width, height],
        "weights": model.weights_name,
        "gpu": torch.cuda.get_device_name(),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "inference_seconds": round(inferred - started, 2),
    }


def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    """World-space camera centers from camera-from-world [R | t]: C = -R^T t."""
    rotation, translation = extrinsic[:, :, :3], extrinsic[:, :, 3]
    return -np.einsum("nji,nj->ni", rotation, translation)
