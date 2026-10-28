"""Stage 6, splat: photo-real appearance with 3D Gaussian splatting, separated per object.

Gaussians are initialized from the lifted, gravity-aligned points, and every
Gaussian inherits the label of the point it came from (0 = background, k =
object k). The scene is optimized jointly against the keyframes with gsplat
(Ye et al., 2024), so objects and background are lit and composited
consistently. Labels are carried through densification as a non-trainable
parameter, so after training each object's splats can be lifted out and
attached to its rigid body, exactly like its points and collider.

Cameras are VGGT's, mapped into the gravity frame. Every eighth keyframe is held
out and only used to report test PSNR, so the quality number in the diagnostics
is measured on views the optimizer never saw.

Densification uses MCMC (Kheradmand et al., 2024): a fixed budget of Gaussians
that are relocated rather than cloned and pruned, with no opacity resets. A
budget matters twice here: training time, and the browser, which pays for every
splat on every frame. Rasterization is anti-aliased (gsplat's Mip-Splatting-style
2D filter), so splats stay correct at the resolutions the viewer renders at.

Runs on a GPU in its own image (gsplat ships CUDA wheels for torch 2.4 and
Python 3.10). Writes `splats.npz` in gravity-aligned VGGT units and the refined
cameras in `cameras.npz`; the package stage converts to meters and object-local
frames.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .jobs import load_arrays

SH_C0 = 0.28209479177387814
ITERATIONS = 6000  # 15k measured worse on held-out views (28.5 vs 29.0 dB) and 9x slower
SH_DEGREE = 1
HOLDOUT_EVERY = 8
POSE_REFINE_START = 300
TEST_ALIGN_STEPS = 120
MAX_INIT_POINTS = 300_000
OBJECT_INIT_SHARE = 0.5  # of the initial points, at most this share goes to objects
STRATEGY = "mcmc"  # or "default": gsplat's clone/split/prune with opacity resets
ANTIALIASED = True
CAP_MAX = 450_000  # MCMC budget of Gaussians
OPACITY_REG = 0.01  # MCMC's regularizers: unused Gaussians fade and shrink, so they get relocated
SCALE_REG = 0.01


def initial_points(lift: dict[str, np.ndarray], object_indices: list[int]) -> tuple[np.ndarray, ...]:
    """Lifted points, colors and labels to start the Gaussians from, at most MAX_INIT_POINTS.

    Objects get up to OBJECT_INIT_SHARE of the budget (in proportion to their point
    counts) and the background the rest. A close-up clip can lift a million points
    on one object; keeping them all once left no budget for the background, so the
    table had no Gaussians of its own and the object's had to draw it.
    """
    rng = np.random.default_rng(0)
    objects = [(index, len(lift[f"obj{index}_points"])) for index in object_indices]
    object_total = sum(count for _, count in objects)
    object_budget = min(object_total, int(OBJECT_INIT_SHARE * MAX_INIT_POINTS))
    points, colors, labels = [], [], []
    for index, count in objects:
        keep = min(count, max(1, round(object_budget * count / max(object_total, 1))))
        pick = np.sort(rng.choice(count, keep, replace=False))
        points.append(lift[f"obj{index}_points"][pick])
        colors.append(lift[f"obj{index}_colors"][pick])
        labels.append(np.full(keep, index, dtype=np.int64))
    background = lift["background_points"]
    keep = min(len(background), MAX_INIT_POINTS - sum(len(p) for p in points))
    pick = np.sort(rng.choice(len(background), keep, replace=False))
    points.insert(0, background[pick])
    colors.insert(0, lift["background_colors"][pick])
    labels.insert(0, np.zeros(keep, dtype=np.int64))
    return np.concatenate(points).astype(np.float32), np.concatenate(colors), np.concatenate(labels)


def aligned_cameras(recon: dict[str, np.ndarray], transform: np.ndarray, image_size: tuple[int, int]):
    """World-to-camera 4x4 matrices in the gravity frame, and intrinsics at image resolution.

    VGGT's extrinsics map VGGT-world to camera; the gravity frame is
    world' = T world, so camera-from-world' = E T^-1. VGGT's intrinsics are at its
    518-px working resolution; frames were resized per axis, so K scales per axis.
    """
    n = len(recon["extrinsic"])
    viewmats = np.tile(np.eye(4), (n, 1, 1))
    viewmats[:, :3, :] = recon["extrinsic"]
    viewmats = viewmats @ np.linalg.inv(transform)
    _, height_v, width_v = recon["confidence"].shape
    width, height = image_size
    Ks = recon["intrinsic"].astype(np.float64).copy()
    Ks[:, 0, :] *= width / width_v
    Ks[:, 1, :] *= height / height_v
    return viewmats.astype(np.float32), Ks.astype(np.float32)


def se3(delta):
    """(C, 6) [rotation vector, translation] -> (C, 4, 4) rigid transforms (Rodrigues)."""
    import torch

    omega, t = delta[:, :3], delta[:, 3:]
    theta = omega.norm(dim=1, keepdim=True).clamp_min(1e-9)
    k = omega / theta
    zero = torch.zeros_like(k[:, 0])
    K = torch.stack([zero, -k[:, 2], k[:, 1], k[:, 2], zero, -k[:, 0], -k[:, 1], k[:, 0], zero], 1).view(-1, 3, 3)
    eye = torch.eye(3, device=delta.device).expand_as(K)
    R = eye + torch.sin(theta)[..., None] * K + (1 - torch.cos(theta))[..., None] * (K @ K)
    M = torch.eye(4, device=delta.device).repeat(len(delta), 1, 1)
    M[:, :3, :3] = R
    M[:, :3, 3] = t
    return M


def psnr(a, b) -> float:
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(1 / max(mse, 1e-10))


def ssim_loss(x, y):
    """1 - SSIM with an 11-px Gaussian window; x, y are (1, 3, H, W) in [0, 1]."""
    import torch
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    window = g.expand(3, 1, 11, 11)

    def blur(t):
        return F.conv2d(t, window, padding=5, groups=3)

    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return 1 - ssim.mean()


def train(
    job_dir: Path,
    strategy_name: str = STRATEGY,
    antialiased: bool = ANTIALIASED,
    output: str = "splats",
) -> dict:
    """Optimize the scene's splats; writes `<output>.npz`, `<output>.json` and, for the default
    output, `cameras.npz` and `splat_test_view.jpg`. Other outputs are for ablations."""
    import torch
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy
    from scipy.spatial import cKDTree

    recon = load_arrays(job_dir / "recon.npz", ["extrinsic", "intrinsic", "confidence"])
    lift = load_arrays(job_dir / "lift.npz")
    lift_info = json.loads((job_dir / "lift.json").read_text())
    indices = [o["index"] for o in lift_info["objects"]]
    means, colors, labels = initial_points(lift, indices)

    frames = sorted((job_dir / "frames").glob("*.jpg"))
    images = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in frames]).astype(np.float32) / 255
    height, width = images.shape[1:3]
    viewmats, Ks = aligned_cameras(recon, lift["transform"], (width, height))
    test_ids = list(range(HOLDOUT_EVERY // 2, len(frames), HOLDOUT_EVERY))
    train_ids = [i for i in range(len(frames)) if i not in test_ids]
    mcmc = strategy_name == "mcmc"
    mode = "antialiased" if antialiased else "classic"

    device = "cuda"
    dist, _ = cKDTree(means).query(means, k=4)
    init_scale = np.log(np.clip(dist[:, 1:].mean(axis=1), 1e-6, None))
    n = len(means)
    init_opacity = 0.5 if mcmc else 0.1
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.tensor(means, device=device)),
            "scales": torch.nn.Parameter(
                torch.tensor(np.repeat(init_scale[:, None], 3, 1), dtype=torch.float32, device=device)
            ),
            "quats": torch.nn.Parameter(
                torch.tensor(np.tile([1.0, 0, 0, 0], (n, 1)), dtype=torch.float32, device=device)
            ),
            "opacities": torch.nn.Parameter(
                torch.full((n,), math.log(init_opacity / (1 - init_opacity)), device=device)
            ),
            "sh0": torch.nn.Parameter(
                torch.tensor((colors / 255.0 - 0.5) / SH_C0, dtype=torch.float32, device=device)[:, None, :]
            ),
            "shN": torch.nn.Parameter(torch.zeros((n, (SH_DEGREE + 1) ** 2 - 1, 3), device=device)),
            # Not optimized: rides along through densification so every splat keeps its object.
            "labels": torch.nn.Parameter(torch.tensor(labels, dtype=torch.float32, device=device), requires_grad=False),
        }
    )
    scene_scale = float(np.linalg.norm(np.percentile(means, 95, axis=0) - np.percentile(means, 5, axis=0))) / 2
    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr}], eps=1e-15) for k, lr in lrs.items()}
    means_decay = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1 / ITERATIONS))
    if mcmc:
        strategy = MCMCStrategy(
            cap_max=CAP_MAX, refine_start_iter=500, refine_stop_iter=int(ITERATIONS * 0.8), refine_every=100
        )
        state = strategy.initialize_state()
    else:
        strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(ITERATIONS * 0.6), reset_every=3000)
        state = strategy.initialize_state(scene_scale=scene_scale)

    # VGGT's cameras come from a 518-px pass, so they are refined jointly with the
    # splats: a small rigid correction per camera and one shared focal-length scale.
    camera_delta = torch.nn.Parameter(torch.zeros((len(frames), 6), device=device))
    log_focal = torch.nn.Parameter(torch.zeros(1, device=device))
    camera_optimizer = torch.optim.Adam([{"params": [camera_delta], "lr": 2e-4}, {"params": [log_focal], "lr": 2e-4}])

    gt = torch.tensor(images, device=device)
    view_t = torch.tensor(viewmats, device=device)
    K_t = torch.tensor(Ks, device=device)

    import torch.nn.functional as F

    def zoom(image, i: int):
        """Scaling the focal length about the principal point is the same as zooming the
        image about it. gsplat gives no gradient for intrinsics, but this zoom is
        differentiable, so it is how the shared focal correction is learned."""
        s_ = torch.exp(log_focal)[0]
        cx = K_t[i, 0, 2] / width * 2 - 1
        cy = K_t[i, 1, 2] / height * 2 - 1
        theta = torch.zeros((1, 2, 3), device=device)
        theta[0, 0, 0] = theta[0, 1, 1] = 1 / s_
        theta[0, 0, 2] = cx * (1 - 1 / s_)
        theta[0, 1, 2] = cy * (1 - 1 / s_)
        grid = F.affine_grid(theta, (1, 3, height, width), align_corners=False)
        return F.grid_sample(image.permute(0, 3, 1, 2), grid, padding_mode="border", align_corners=False).permute(
            0, 2, 3, 1
        )

    def render(i: int, degree: int):
        image, alpha, info = raster(i, degree)
        return zoom(image, i), alpha, info

    def raster(i: int, degree: int):
        return rasterization(
            params["means"],
            params["quats"],
            torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]),
            torch.cat([params["sh0"], params["shN"]], 1),
            se3(camera_delta[i : i + 1]) @ view_t[i : i + 1],
            K_t[i : i + 1],
            width,
            height,
            sh_degree=degree,
            packed=False,
            rasterize_mode=mode,
        )

    started = time.monotonic()
    rng = np.random.default_rng(0)
    for step in range(ITERATIONS):
        i = int(rng.choice(train_ids))
        degree = min(SH_DEGREE, step // 1000)
        image, _, info = render(i, degree)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        pred = image[0].permute(2, 0, 1)[None]
        target = gt[i].permute(2, 0, 1)[None]
        loss = 0.8 * (pred - target).abs().mean() + 0.2 * ssim_loss(pred, target)
        if mcmc:
            loss = loss + OPACITY_REG * torch.sigmoid(params["opacities"]).mean()
            loss = loss + SCALE_REG * torch.exp(params["scales"]).mean()
        loss.backward()
        if mcmc:
            # MCMC relocates and perturbs Gaussians after the optimizer has stepped.
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            strategy.step_post_backward(params, optimizers, state, step, info, lr=means_decay.get_last_lr()[0])
        else:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if step >= POSE_REFINE_START:
            camera_optimizer.step()
        camera_optimizer.zero_grad(set_to_none=True)
        means_decay.step()
    torch.cuda.synchronize()
    seconds = time.monotonic() - started

    # Test-time pose alignment: a held-out view's camera never trained, so with the splats
    # frozen, only that view's correction is fit before scoring (as in pose-optimizing NeRFs).
    for p_ in params.values():
        p_.requires_grad_(False)
    for i in test_ids:
        align = torch.optim.Adam([camera_delta], lr=1e-3)
        for _ in range(TEST_ALIGN_STEPS):
            image, _, _ = render(i, SH_DEGREE)
            mask = torch.zeros(len(frames), 1, device=device)
            mask[i] = 1
            loss = (image[0] - gt[i]).abs().mean()
            loss.backward()
            camera_delta.grad *= mask
            align.step()
            align.zero_grad(set_to_none=True)

    default_output = output == "splats"
    test_view = job_dir / ("splat_test_view.jpg" if default_output else f"{output}_test_view.jpg")
    with torch.no_grad():

        def mean_psnr(ids):
            return float(np.mean([psnr(render(i, SH_DEGREE)[0][0].clamp(0, 1), gt[i]) for i in ids])) if ids else None

        metrics = {"test_psnr": mean_psnr(test_ids), "train_psnr": mean_psnr(train_ids[:: max(1, len(train_ids) // 8)])}
        if test_ids:  # held-out view: ground truth | render, for the docs and for eyeballing
            i = test_ids[len(test_ids) // 2]
            rendered = (render(i, SH_DEGREE)[0][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            truth = (images[i] * 255).astype(np.uint8)
            Image.fromarray(np.concatenate([truth, rendered], axis=1)).save(test_view, quality=90)
        out = {k: v.detach().cpu().numpy() for k, v in params.items()}
        refined = (se3(camera_delta) @ view_t).cpu().numpy()
        focal_scale = float(torch.exp(log_focal)[0])

    np.savez_compressed(
        job_dir / f"{output}.npz",
        means=out["means"],
        quats=out["quats"],
        log_scales=out["scales"],
        opacity_logits=out["opacities"],
        sh0=out["sh0"][:, 0, :],
        shN=out["shN"],
        labels=np.rint(out["labels"]).astype(np.int16),
    )
    if default_output:
        # Refined world-to-camera matrices (gravity frame, VGGT units) and intrinsics at
        # keyframe resolution with the learned focal correction: the capture path.
        refined_Ks = Ks.copy()
        refined_Ks[:, :2, :2] *= focal_scale
        np.savez_compressed(job_dir / "cameras.npz", viewmats=refined.astype(np.float32), Ks=refined_Ks)
    info = {
        "gaussians": int(len(out["means"])),
        "iterations": ITERATIONS,
        "strategy": strategy_name,
        "antialiased": antialiased,
        "sh_degree": SH_DEGREE,
        "train_views": len(train_ids),
        "test_views": len(test_ids),
        "train_seconds": round(seconds, 1),
        "resolution": [width, height],
        "gpu": torch.cuda.get_device_name(),
        "focal_correction": round(focal_scale, 4),
        "mean_camera_rotation_correction_deg": round(
            float(camera_delta[train_ids, :3].norm(dim=1).mean()) * 180 / math.pi, 3
        ),
        **{k: (round(v, 2) if v is not None else None) for k, v in metrics.items()},
    }
    (job_dir / f"{output}.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


# ---------- Post-processing (CPU) ----------

MIN_OPACITY = 0.005
NEEDLE_RATIO = 10
# Metric clean-up thresholds, converted to scene units with the scene scale.
CUT_MARGIN_M = 0.015  # background cut-out and table fill grow this far around each object
# Object splats larger than this (or this share of the object's longest side) smear when the
# object moves. MCMC draws with fewer, larger Gaussians than clone-and-split densification: at the
# old 5 mm, 16% of a scanned mug was dropped and it rendered see-through.
MAX_OBJECT_SIGMA_M = 0.01
MAX_OBJECT_SIGMA_SHARE = 0.08
TABLE_LAYER_MARGIN_M = 0.03  # the table layer reaches this far past the table's outline
TABLE_LAYER_MIN_HEIGHT_M = 0.12  # and this high above it (or 1.5x the tallest object)


def cleanup_options(voxel: float, scale: float) -> dict:
    """Keyword arguments for `assign` in gravity-aligned VGGT units."""
    return {
        "margin": 3 * voxel,
        "cut_margin": max(3 * voxel, CUT_MARGIN_M / scale),
        "max_object_sigma": MAX_OBJECT_SIGMA_M / scale,
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def subset(splats: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in splats.items()}


def assign(
    splats: dict[str, np.ndarray],
    boxes: dict[int, tuple[np.ndarray, np.ndarray]],
    margin: float,
    cut_margin: float | None = None,
    max_object_sigma: float | None = None,
) -> np.ndarray:
    """Final owner of every splat: its training label, corrected by geometry (-1 = dropped).

    The optimizer draws each object with whatever Gaussians are near it, so a label
    from initialization is only a strong hint. Measured on a scanned mug, the old rules
    (drop flat splats, faint splats and every background splat touching the object)
    removed over half of it and left a see-through shell. So:

    - A background splat whose center lies inside an object's box, above the table,
      joins that object: it was drawing the object.
    - An object splat that drifted outside its object's box (grown by `margin`) is
      dropped: left behind it would be a ghost on the table. So are object needles (one
      axis over NEEDLE_RATIO x the middle one; flat discs are how surfaces are drawn, so
      they stay) and object splats larger than `max_object_sigma` (or a share of the
      object's size), which smear when the object moves.
    - A background splat above the table is dropped if its center lies within
      `cut_margin` of an object's box (it drew the object's edges from some view), or if
      it is large (over half the cut margin) and its extent reaches into that zone: it
      would smudge the empty spot once the object moves.
    - A background splat below the table top within an object's footprint is dropped:
      the optimizer parks splats there to help draw the object from above.
    - Splats fainter than MIN_OPACITY are dropped.
    """
    labels = splats["labels"].astype(np.int64).copy()
    means = splats["means"]
    sigma = np.sort(np.exp(splats["log_scales"]), axis=1)  # ascending
    cut = margin if cut_margin is None else cut_margin
    above = means[:, 1] > margin
    extent = 2.5 * sigma[:, 2:]
    large = sigma[:, 2] > 0.5 * cut
    for index, (lo, hi) in boxes.items():
        inside = np.all((means >= lo) & (means <= hi), axis=1)
        labels[(labels == 0) & inside & above] = index
    is_object = labels > 0
    labels[is_object & (sigma[:, 2] > NEEDLE_RATIO * sigma[:, 1])] = -1
    for index, (lo, hi) in boxes.items():
        own = labels == index
        if max_object_sigma is not None:
            limit = max(max_object_sigma, MAX_OBJECT_SIGMA_SHARE * float(np.max(hi - lo)))
            labels[own & (sigma[:, 2] > limit)] = -1
        inside = np.all((means >= lo - margin) & (means <= hi + margin), axis=1)
        labels[own & ~inside] = -1
        near = np.all((means >= lo - cut) & (means <= hi + cut), axis=1)
        reaches = np.all((means + extent >= lo - cut) & (means - extent <= hi + cut), axis=1)
        labels[(labels == 0) & (near | (reaches & large)) & above] = -1
        over_xz = np.all((means[:, [0, 2]] >= lo[[0, 2]] - cut) & (means[:, [0, 2]] <= hi[[0, 2]] + cut), axis=1)
        labels[(labels == 0) & over_xz & (means[:, 1] < -margin)] = -1
    labels[sigmoid(splats["opacity_logits"]) < MIN_OPACITY] = -1
    return labels


def table_layer(background: dict[str, np.ndarray], polygon_m: np.ndarray, scale: float, top_m: float) -> np.ndarray:
    """Which background splats belong to the table top (True) rather than the backdrop.

    The table layer is everything over the table's outline (grown by
    TABLE_LAYER_MARGIN_M), from just below its surface up to `top_m`: the surface and
    anything left on it. The rest (walls, floor, the room) is backdrop, trained only
    from the filmed side, which viewers fade out away from the capture path.
    """
    from scipy.spatial import Delaunay

    meters = background["means"] * scale
    center = polygon_m.mean(axis=0)
    radius = max(float(np.linalg.norm(polygon_m - center, axis=1).max()), 1e-6)
    grown = center + (polygon_m - center) * (1 + TABLE_LAYER_MARGIN_M / radius)
    over = Delaunay(grown).find_simplex(meters[:, [0, 2]]) >= 0
    return over & (meters[:, 1] > -0.02) & (meters[:, 1] < top_m)


def fill_under(
    background: dict[str, np.ndarray],
    footprints: list[np.ndarray],
    spacing: float,
    table_band: float,
    grow: float = 0.0,
) -> dict[str, np.ndarray]:
    """Replace each object's footprint on the table with flat, opaque splats.

    The table under an object was never seen, so the optimizer painted it with
    whatever explained the object from above: lifting the object would reveal a
    mug-colored smudge. Every background splat near the table inside the (grown)
    footprint is removed, and the area is re-covered. Each fill splat takes the color
    of the nearest visible table splats (inverse-distance weighted), which continues
    the table's texture inward.
    """
    from scipy.spatial import Delaunay, cKDTree

    if not footprints:
        return background
    grown_outlines = []
    for outline in footprints:
        center = outline.mean(axis=0)
        radius = max(np.ptp(outline, axis=0).max() / 2, 1e-9)
        grown_outlines.append(center + (outline - center) * (1 + (spacing + grow) / radius))
    xz = background["means"][:, [0, 2]]
    covered = np.zeros(len(xz), dtype=bool)
    for grown in grown_outlines:
        covered |= Delaunay(grown).find_simplex(xz) >= 0
    background = subset(background, ~(covered & (background["means"][:, 1] < 2 * table_band)))
    near_table = np.abs(background["means"][:, 1]) < table_band
    if near_table.sum() < 8:
        return background
    table = subset(background, near_table)
    tree = cKDTree(table["means"][:, [0, 2]])
    fills = []
    for outline, grown in zip(footprints, grown_outlines, strict=True):
        lo, hi = outline.min(axis=0) - spacing - grow, outline.max(axis=0) + spacing + grow
        gx, gz = np.meshgrid(np.arange(lo[0], hi[0], spacing), np.arange(lo[1], hi[1], spacing))
        grid = np.stack([gx.ravel(), gz.ravel()], axis=1)
        # Footprints are convex, so a Delaunay triangulation is an exact inside test.
        grid = grid[Delaunay(grown).find_simplex(grid) >= 0]
        if len(grid):
            fills.append(grid)
    if not fills:
        return background
    grid = np.concatenate(fills)
    dist, idx = tree.query(grid, k=8)
    weights = 1 / np.maximum(dist, 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    n = len(grid)
    fill = {
        "means": np.stack([grid[:, 0], np.zeros(n), grid[:, 1]], axis=1).astype(np.float32),
        "quats": np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        "log_scales": np.tile(np.log([spacing * 0.75, spacing * 0.05, spacing * 0.75]).astype(np.float32), (n, 1)),
        "opacity_logits": np.full(n, logit(0.97), np.float32),
        "sh0": np.einsum("nk,nkc->nc", weights, table["sh0"][idx]).astype(np.float32),
        "shN": np.zeros((n,) + background["shN"].shape[1:], np.float32),
        "labels": np.zeros(n, np.int16),
    }
    return {k: np.concatenate([background[k], fill[k]]) for k in background}


def write_spz(
    path: Path, splats: dict[str, np.ndarray], offset: np.ndarray, scale: float, antialiased: bool = False
) -> int:
    """Write splats to Niantic SPZ (v3), converting to meters and shifting by -offset.

    gsplat stores quaternions as (w, x, y, z); SPZ wants (x, y, z, w). Scene
    coordinates are right-handed with +y up and +z toward the viewer, SPZ's RUB.
    """
    import spz

    cloud = spz.GaussianCloud()
    cloud.antialiased = antialiased
    cloud.sh_degree = int(round(math.sqrt(splats["shN"].shape[1] + 1))) - 1
    cloud.positions = ((splats["means"] * scale) - offset).astype(np.float32).ravel()
    cloud.scales = (splats["log_scales"] + math.log(scale)).astype(np.float32).ravel()
    quats = splats["quats"] / np.linalg.norm(splats["quats"], axis=1, keepdims=True)
    cloud.rotations = quats[:, [1, 2, 3, 0]].astype(np.float32).ravel()
    cloud.alphas = splats["opacity_logits"].astype(np.float32)
    cloud.colors = splats["sh0"].astype(np.float32).ravel()
    # SPZ orders SH per splat coefficient-major with the color channel inner: gsplat's
    # shN is (N, coefficients, 3), so a plain reshape matches.
    cloud.sh = splats["shN"].astype(np.float32).ravel()
    options = spz.PackOptions()
    options.version = 3
    options.from_coord = spz.RUB
    if not spz.save_spz(cloud, options, str(path)):
        raise RuntimeError(f"could not write {path}")
    return len(quats)


def render_background_preview(job_dir: Path, frame: int | None = None, overrides: dict | None = None) -> Path:
    """Render one keyframe with every object removed: what the table looks like once the
    objects are picked up. Uses the same assignment and hole filling as packaging."""
    import torch
    from gsplat import rasterization

    trained = load_arrays(job_dir / "splats.npz")
    lift = load_arrays(job_dir / "lift.npz")
    lift_info = json.loads((job_dir / "lift.json").read_text())
    recon = load_arrays(job_dir / "recon.npz", ["extrinsic", "intrinsic", "confidence"])
    boxes = {
        o["index"]: (lift[f"obj{o['index']}_points"].min(axis=0), lift[f"obj{o['index']}_points"].max(axis=0))
        for o in lift_info["objects"]
    }
    scale = json.loads((job_dir / "physics.json").read_text())["scale_m_per_unit"]
    antialiased = json.loads((job_dir / "splats.json").read_text()).get("antialiased", False)
    options = cleanup_options(lift_info["voxel"], scale) | (overrides or {})
    grow = options.pop("grow", CUT_MARGIN_M / scale)
    owner = assign(trained, boxes, **options)
    from scipy.spatial import ConvexHull

    footprints = []
    for o in lift_info["objects"]:
        pts = lift[f"obj{o['index']}_points"]
        low = pts[pts[:, 1] <= pts[:, 1].min() + 0.5 * np.ptp(pts[:, 1])][:, [0, 2]]
        footprints.append(low[ConvexHull(low).vertices])
    scenery = fill_under(
        subset(trained, owner == 0),
        footprints,
        spacing=lift_info["voxel"],
        table_band=2 * lift_info["plane_threshold"],
        grow=grow,
    )
    frames = sorted((job_dir / "frames").glob("*.jpg"))
    frame = len(frames) // 2 if frame is None else frame
    image = np.asarray(Image.open(frames[frame]).convert("RGB"))
    height, width = image.shape[:2]
    viewmats, Ks = aligned_cameras(recon, lift["transform"], (width, height))
    t = {k: torch.tensor(v, device="cuda") for k, v in scenery.items() if k != "labels"}
    with torch.no_grad():
        rendered, _, _ = rasterization(
            t["means"],
            t["quats"],
            torch.exp(t["log_scales"]),
            torch.sigmoid(t["opacity_logits"]),
            torch.cat([t["sh0"][:, None, :], t["shN"]], 1),
            torch.tensor(viewmats[frame : frame + 1], device="cuda"),
            torch.tensor(Ks[frame : frame + 1], device="cuda"),
            width,
            height,
            sh_degree=SH_DEGREE,
            packed=False,
            rasterize_mode="antialiased" if antialiased else "classic",
        )
    out = job_dir / "background_preview.jpg"
    preview = (rendered[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(np.concatenate([image, preview], axis=1)).save(out, quality=90)
    return out
