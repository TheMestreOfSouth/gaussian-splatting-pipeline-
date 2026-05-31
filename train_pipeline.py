"""
train_pipeline.py — Core Optimization Engine for 3D Gaussian Splatting
=======================================================================
Implements the full differentiable rendering loop:
  1. Gaussian parameter initialization from a sparse .ply point cloud
  2. Differentiable rasterization via gsplat
  3. Combined L1 + SSIM loss
  4. Adam optimization with per-parameter learning rates
  5. Adaptive Density Control (clone + split)

Mathematical conventions:
  - World-to-camera:  P_cam = R @ P_world + T      (R: [3,3], T: [3])
  - Intrinsics:       K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  - Quaternions:      [w, x, y, z] (scalar-first), always L2-normalized
  - Scales:           stored as log(scale); activated via exp() → strictly > 0
  - Opacity:          stored as logit; activated via sigmoid() → (0, 1)
  - SH coefficients:  [N, (deg+1)^2, 3]  (RGB per band)

Dependencies:
  pip install torch torchvision gsplat plyfile scipy pytorch-msssim tqdm
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from pytorch_msssim import ssim as compute_ssim
from torch import Tensor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("3dgs.train")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """All hyper-parameters for the optimization loop."""

    # ── Data ──────────────────────────────────────────────────────────────
    ply_path: str = "sparse/0/points3D.ply"
    cameras_path: str = "sparse/0/cameras_processed.pt"  # output of preprocess.py
    output_dir: str = "output"

    # ── Training ──────────────────────────────────────────────────────────
    num_iterations: int = 30_000
    warmup_iterations: int = 500          # iterations before densification starts
    densify_interval: int = 100           # run densification every N iters
    densify_until: int = 15_000           # stop densification after this iter
    opacity_reset_interval: int = 3_000   # periodically reset near-zero opacities

    # ── Loss ──────────────────────────────────────────────────────────────
    lambda_ssim: float = 0.2              # λ in L = (1-λ)·L1 + λ·SSIM

    # ── Learning rates ────────────────────────────────────────────────────
    lr_means3d: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quaternions: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh: float = 2.5e-3
    lr_decay_means3d_end: float = 1.6e-6  # exponential decay target for positions

    # ── Densification thresholds ──────────────────────────────────────────
    grad_threshold_clone: float = 2e-4    # ||∇μ|| > τ  → clone Gaussian
    scale_threshold_split: float = 0.01   # max(scale) > τ  → split instead of clone
    min_opacity_prune: float = 0.005      # prune Gaussians below this opacity
    max_gaussians: int = 1_000_000        # hard cap

    # ── Spherical Harmonics ───────────────────────────────────────────────
    sh_degree: int = 3                    # degree 0-3; bands = (deg+1)²

    # ── Rendering ─────────────────────────────────────────────────────────
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)  # black bg

    # ── Hardware ──────────────────────────────────────────────────────────
    device: str = "cuda"
    log_every: int = 100


# ---------------------------------------------------------------------------
# Camera / Dataset
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    """
    Single camera view.

    Coordinate convention (OpenCV / COLMAP):
      - X right, Y down, Z into scene
      - P_cam = R @ P_world + T
    """
    image: Tensor          # [3, H, W] float32 in [0, 1]
    R: Tensor              # [3, 3]  rotation  world→camera
    T: Tensor              # [3]     translation world→camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> Tensor:
        """Intrinsic matrix [3, 3]."""
        return torch.tensor(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.image.device,
        )

    @property
    def fov_x(self) -> float:
        """Horizontal field of view in radians."""
        return 2.0 * math.atan(self.width / (2.0 * self.fx))

    @property
    def fov_y(self) -> float:
        """Vertical field of view in radians."""
        return 2.0 * math.atan(self.height / (2.0 * self.fy))


def load_cameras(path: str, device: str) -> list[Camera]:
    """
    Load camera data saved by preprocess.py.
    Expected dict keys per camera: image, R, T, fx, fy, cx, cy, W, H.
    """
    # Adicionado weights_only=False para permitir que o PyTorch 2.6 leia os dicionários/matrizes do NumPy
    data: list[dict] = torch.load(path, map_location=device, weights_only=False)
    cameras = []
    for d in data:
        cameras.append(Camera(
            image=d["image"].to(device),
            R=d["R"].to(device),
            T=d["T"].to(device),
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            width=int(d["W"]),
            height=int(d["H"]),
        ))
    log.info("Loaded %d cameras from %s", len(cameras), path)
    return cameras


# ---------------------------------------------------------------------------
# Gaussian Model
# ---------------------------------------------------------------------------
class GaussianModel:
    """
    Holds all learnable parameters that describe a cloud of 3D Gaussians.

    Storage vs. activated quantities
    ─────────────────────────────────
      _scales      (stored) → exp(_scales)          (activated, always > 0)
      _quaternions (stored) → normalize(_quaternions)(activated, unit quaternion)
      _opacities   (stored) → sigmoid(_opacities)   (activated, ∈ (0,1))
    """

    def __init__(self, means3d: Tensor, colors_init: Tensor, sh_degree: int = 3):
        """
        Args:
            means3d:     [N, 3]  initial Gaussian centers (from SfM point cloud)
            colors_init: [N, 3]  initial RGB colors (from SfM point colors), float32 [0,1]
            sh_degree:   maximum SH degree (0–3)
        """
        self.sh_degree = sh_degree
        self.sh_bands: int = (sh_degree + 1) ** 2
        N = means3d.shape[0]
        dev = means3d.device

        # ── Learnable parameters ──────────────────────────────────────────
        # μ  — Gaussian centers in world space [N, 3]
        self.means3d = torch.nn.Parameter(means3d.clone().float())

        # log(σ) — isotropic initialization; shape [N, 3]
        # Initial scale: mean nearest-neighbor distance (heuristic)
        init_scale = self._estimate_init_scale(means3d)                  # scalar
        self._scales = torch.nn.Parameter(
            torch.full((N, 3), math.log(init_scale), device=dev)
        )

        # q = [w, x, y, z]  — unit quaternion representing rotation [N, 4]
        # Initialize to identity rotation: q = (1, 0, 0, 0)
        quats = torch.zeros(N, 4, device=dev)
        quats[:, 0] = 1.0                                                # w = 1
        self._quaternions = torch.nn.Parameter(quats)

        # logit(α) — opacity [N, 1]
        # Initialize to ~0.1 opacity: logit(0.1) ≈ −2.2
        self._opacities = torch.nn.Parameter(
            torch.full((N, 1), -2.197, device=dev)
        )

        # SH coefficients [N, bands, 3]
        # Band-0 (DC term) initialized from RGB; higher bands = 0
        sh_coeff = torch.zeros(N, self.sh_bands, 3, device=dev)
        # SH band-0 maps to (color - 0.5) / C0  where C0 = 1/(2√π)
        C0 = 0.28209479177387814
        sh_coeff[:, 0, :] = (colors_init.float() - 0.5) / C0
        self._sh = torch.nn.Parameter(sh_coeff)

        # ── Non-parameter buffers for Adaptive Density Control ────────────
        # Accumulated ||∇μ|| (position gradient magnitude) per Gaussian
        self.grad_accum: Tensor = torch.zeros(N, device=dev)
        self.grad_accum_count: Tensor = torch.zeros(N, device=dev, dtype=torch.long)

    # ── Activated accessors ───────────────────────────────────────────────

    @property
    def scales(self) -> Tensor:
        """Activated scales [N, 3], strictly positive."""
        return torch.exp(self._scales)

    @property
    def quaternions(self) -> Tensor:
        """Normalized quaternions [N, 4] ∈ S³."""
        return F.normalize(self._quaternions, dim=-1)

    @property
    def opacities(self) -> Tensor:
        """Activated opacities [N, 1] ∈ (0, 1)."""
        return torch.sigmoid(self._opacities)

    @property
    def sh_coefficients(self) -> Tensor:
        """SH coefficients [N, bands, 3]."""
        return self._sh

    def num_gaussians(self) -> int:
        return self.means3d.shape[0]

    # ── Optimizer parameter groups ────────────────────────────────────────

    def get_param_groups(self, cfg: TrainConfig) -> list[dict]:
        """Return Adam parameter groups with per-tensor learning rates."""
        return [
            {"params": [self.means3d],    "lr": cfg.lr_means3d,    "name": "means3d"},
            {"params": [self._scales],    "lr": cfg.lr_scales,     "name": "scales"},
            {"params": [self._quaternions],"lr": cfg.lr_quaternions,"name": "quaternions"},
            {"params": [self._opacities], "lr": cfg.lr_opacities,  "name": "opacities"},
            {"params": [self._sh],        "lr": cfg.lr_sh,         "name": "sh"},
        ]

    # ── Adaptive Density Control ──────────────────────────────────────────

    def accumulate_gradients(self) -> None:
        """Record position gradient norms for densification decision."""
        if self.means3d.grad is not None:
            grad_norm = self.means3d.grad.norm(dim=-1)               # [N]
            self.grad_accum += grad_norm
            self.grad_accum_count += 1

    def densify_and_prune(
        self,
        optimizer: torch.optim.Optimizer,
        grad_threshold: float,
        scale_threshold: float,
        min_opacity: float,
        max_gaussians: int,
    ) -> int:
        """
        Adaptive Density Control:
          • CLONE  if ||∇μ|| > τ  AND  max(scale) ≤ scale_threshold
          • SPLIT  if ||∇μ|| > τ  AND  max(scale) >  scale_threshold
          • PRUNE  if opacity  < min_opacity  OR  scale too large (floaters)

        Returns the new total number of Gaussians.
        """
        avg_grad = self.grad_accum / (self.grad_accum_count.float().clamp(min=1))
        over_thresh = avg_grad > grad_threshold                       # [N] bool

        max_scale = self.scales.detach().max(dim=-1).values           # [N]
        clone_mask = over_thresh & (max_scale <= scale_threshold)
        split_mask = over_thresh & (max_scale >  scale_threshold)

        # --- Clone --------------------------------------------------------
        if clone_mask.any():
            new_means   = self.means3d.detach()[clone_mask]
            new_scales  = self._scales.detach()[clone_mask]
            new_quats   = self._quaternions.detach()[clone_mask]
            new_opac    = self._opacities.detach()[clone_mask]
            new_sh      = self._sh.detach()[clone_mask]
            self._concat_gaussians(optimizer, new_means, new_scales, new_quats, new_opac, new_sh)

        # --- Split --------------------------------------------------------
        if split_mask.any():
            # Sample 2 child centers along the principal axis (largest scale dim)
            s = self.scales.detach()[split_mask]                      # [M, 3]
            q = self.quaternions.detach()[split_mask]                 # [M, 4]
            R_mats = self._quat_to_rotmat(q)                          # [M, 3, 3]
            principal = R_mats[:, :, s.argmax(dim=-1)[0]]             # [M, 3] ← approx
            offset = s.max(dim=-1, keepdim=True).values * principal   # [M, 3]
            child_a = self.means3d.detach()[split_mask] + offset
            child_b = self.means3d.detach()[split_mask] - offset
            # Reduce scale of children by 1/φ  (φ = 1.6 heuristic)
            new_scales = self._scales.detach()[split_mask] - math.log(1.6)
            for child_centers in [child_a, child_b]:
                self._concat_gaussians(
                    optimizer, child_centers, new_scales,
                    self._quaternions.detach()[split_mask],
                    self._opacities.detach()[split_mask],
                    self._sh.detach()[split_mask],
                )
            # Remove original split Gaussians
            keep = ~split_mask
            self._prune(optimizer, keep)

        # --- Prune --------------------------------------------------------
        low_opacity = self.opacities.detach().squeeze(-1) < min_opacity
        large_scale = self.scales.detach().max(dim=-1).values > 0.5   # scene-dependent
        prune_mask = low_opacity | large_scale
        if prune_mask.any():
            self._prune(optimizer, ~prune_mask)

        # --- Hard cap -----------------------------------------------------
        if self.num_gaussians() > max_gaussians:
            keep_idx = torch.randperm(self.num_gaussians(), device=self.means3d.device)[:max_gaussians]
            keep_mask = torch.zeros(self.num_gaussians(), dtype=torch.bool, device=self.means3d.device)
            keep_mask[keep_idx] = True
            self._prune(optimizer, keep_mask)

        # Reset accumulators
        N = self.num_gaussians()
        self.grad_accum = torch.zeros(N, device=self.means3d.device)
        self.grad_accum_count = torch.zeros(N, device=self.means3d.device, dtype=torch.long)

        return self.num_gaussians()

    def reset_opacities(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Periodically push all opacities below 0.01 threshold.
        Prevents opaque floaters from persisting after early training.
        New value: sigmoid^{-1}(0.01) ≈ -4.6
        """
        with torch.no_grad():
            self._opacities.clamp_(max=-4.595)                        # logit(0.01)
        # Zero optimizer state for opacities so momentum doesn't pull them back
        for group in optimizer.param_groups:
            if group["name"] == "opacities":
                for p in group["params"]:
                    if p in optimizer.state:
                        optimizer.state[p] = {}

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _estimate_init_scale(pts: Tensor, k: int = 4) -> float:
        """
        Heuristic: mean k-NN distance among a random subset of points.
        This sets Gaussians to roughly cover their local neighborhood.
        """
        pts_np = pts.detach().cpu().numpy()
        from scipy.spatial import cKDTree  # lazy import
        subset = pts_np[np.random.choice(len(pts_np), min(1000, len(pts_np)), replace=False)]
        tree = cKDTree(subset)
        dists, _ = tree.query(subset, k=k + 1)
        return float(np.mean(dists[:, 1:]))                           # exclude self

    def _concat_gaussians(
        self,
        optimizer: torch.optim.Optimizer,
        means: Tensor, scales: Tensor, quats: Tensor,
        opacities: Tensor, sh: Tensor,
    ) -> None:
        """Append new Gaussians and update optimizer state tensors."""
        param_map = {
            "means3d":     (self.means3d,    means),
            "scales":      (self._scales,    scales),
            "quaternions": (self._quaternions, quats),
            "opacities":   (self._opacities,  opacities),
            "sh":          (self._sh,         sh),
        }
        for group in optimizer.param_groups:
            name = group["name"]
            old_param, extension = param_map[name]
            new_data = torch.cat([old_param.detach(), extension], dim=0)
            new_param = torch.nn.Parameter(new_data)
            # Replace tensor in optimizer state
            if old_param in optimizer.state:
                old_state = optimizer.state.pop(old_param)
                new_state: dict = {}
                for k, v in old_state.items():
                    if isinstance(v, Tensor) and v.shape[0] == old_param.shape[0]:
                        pad = torch.zeros(extension.shape[0], *v.shape[1:], device=v.device)
                        new_state[k] = torch.cat([v, pad], dim=0)
                    else:
                        new_state[k] = v
                optimizer.state[new_param] = new_state
            group["params"] = [new_param]
            # Mutate the model attribute in-place
            setattr(self, {"means3d": "means3d", "scales": "_scales",
                           "quaternions": "_quaternions", "opacities": "_opacities",
                           "sh": "_sh"}[name], new_param)

    def _prune(self, optimizer: torch.optim.Optimizer, keep_mask: Tensor) -> None:
        """Remove Gaussians not in keep_mask, update optimizer state."""
        attr_map = {
            "means3d":     "means3d",
            "scales":      "_scales",
            "quaternions": "_quaternions",
            "opacities":   "_opacities",
            "sh":          "_sh",
        }
        for group in optimizer.param_groups:
            name = group["name"]
            old_param: Tensor = group["params"][0]
            new_data = old_param.detach()[keep_mask]
            new_param = torch.nn.Parameter(new_data)
            if old_param in optimizer.state:
                old_state = optimizer.state.pop(old_param)
                new_state = {}
                for k, v in old_state.items():
                    new_state[k] = v[keep_mask] if (isinstance(v, Tensor) and v.shape[0] == old_param.shape[0]) else v
                optimizer.state[new_param] = new_state
            group["params"] = [new_param]
            setattr(self, attr_map[name], new_param)

    @staticmethod
    def _quat_to_rotmat(q: Tensor) -> Tensor:
        """
        Convert [N, 4] unit quaternions [w, x, y, z] to [N, 3, 3] rotation matrices.
        Formula: R_ij from standard quaternion-to-matrix expansion.
        """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
            2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
            2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)
        return R


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def combined_loss(rendered: Tensor, target: Tensor, lambda_ssim: float = 0.2) -> tuple[Tensor, Tensor, Tensor]:
    """
    Photometric loss:
        L = (1 - λ) · L1 + λ · (1 - SSIM)

    Args:
        rendered: [1, 3, H, W]  rendered image from rasterizer
        target:   [1, 3, H, W]  ground-truth image
        lambda_ssim: weight for SSIM term (default 0.2 per original 3DGS paper)

    Returns:
        total_loss, l1_loss, ssim_loss
    """
    l1 = F.l1_loss(rendered, target)
    # SSIM returns similarity ∈ [0,1]; loss = 1 - SSIM
    ssim_val = compute_ssim(rendered, target, data_range=1.0, size_average=True)
    ssim_loss = 1.0 - ssim_val
    total = (1.0 - lambda_ssim) * l1 + lambda_ssim * ssim_loss
    return total, l1, ssim_loss


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------
def exponential_lr_lambda(iteration: int, start_lr: float, end_lr: float, max_steps: int) -> float:
    """
    Computes the LR multiplier at `iteration` for exponential decay.
        lr(t) = start_lr · (end_lr / start_lr)^(t / max_steps)
    Returns a scalar multiplier relative to the base lr in the optimizer group.
    """
    t = min(iteration, max_steps) / max_steps
    return (end_lr / start_lr) ** t


# ---------------------------------------------------------------------------
# Point cloud loader
# ---------------------------------------------------------------------------
def load_ply_points(ply_path: str, device: str) -> tuple[Tensor, Tensor]:
    """
    Read a COLMAP-style .ply file and return:
        means3d:  [N, 3]  float32 coordinates
        colors:   [N, 3]  float32 RGB in [0, 1]
    """
    ply = PlyData.read(ply_path)
    verts = ply["vertex"]
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=-1).astype(np.float32)
    # COLMAP stores uint8 colors; normalize to [0, 1]
    rgb = np.stack([verts["red"], verts["green"], verts["blue"]], axis=-1).astype(np.float32) / 255.0
    means3d = torch.from_numpy(xyz).to(device)
    colors  = torch.from_numpy(rgb).to(device)
    log.info("Loaded %d points from %s", len(means3d), ply_path)
    return means3d, colors


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig) -> GaussianModel:
    """
    Full 3DGS optimization loop.

    Pipeline per iteration:
      1.  Sample a camera view (cyclic or random)
      2.  Project & rasterize Gaussians  →  rendered image  [1, 3, H, W]
      3.  Compute L1 + SSIM loss against ground-truth image
      4.  Backward pass; accumulate position gradients for densification
      5.  Optimizer step
      6.  (Every densify_interval)  Adaptive Density Control
      7.  (Every opacity_reset_interval)  Reset near-zero opacities
    """
    import gsplat  # imported here so the module is usable even without gsplat installed for inspection

    device = cfg.device
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    means3d_init, colors_init = load_ply_points(cfg.ply_path, device)
    cameras: list[Camera] = load_cameras(cfg.cameras_path, device)
    num_cameras = len(cameras)

    # ── Initialize model ──────────────────────────────────────────────────
    model = GaussianModel(means3d_init, colors_init, sh_degree=cfg.sh_degree)
    log.info("Initialized %d Gaussians  (SH degree %d)", model.num_gaussians(), cfg.sh_degree)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.get_param_groups(cfg),
        lr=0.0,     # overridden per group
        eps=1e-15,
    )

    # ── Background color tensor ───────────────────────────────────────────
    bg = torch.tensor(cfg.background_color, dtype=torch.float32, device=device)

    # ── Training loop ─────────────────────────────────────────────────────
    for step in tqdm(range(1, cfg.num_iterations + 1), desc="Training 3DGS"):

        # Cycle through cameras
        cam: Camera = cameras[(step - 1) % num_cameras]
        H, W = cam.height, cam.width

        # ── Learning rate schedule (exponential decay for positions) ──────
        for group in optimizer.param_groups:
            if group["name"] == "means3d":
                lr_mult = exponential_lr_lambda(
                    step, cfg.lr_means3d, cfg.lr_decay_means3d_end, cfg.num_iterations
                )
                group["lr"] = cfg.lr_means3d * lr_mult

        # ── Differentiable rasterization ──────────────────────────────────
        #
        # gsplat API (v0.1+):
        #   project_gaussians(means3d, scales, quats, viewmat, K, W, H, ...)
        #     → xys      [N, 2]   projected 2D centers (NDC)
        #       depths   [N]      depth per Gaussian
        #       radii    [N]      2D bounding radius in pixels
        #       conics   [N, 3]   upper triangle of 2D covariance inverse
        #       num_tiles [int]   tile occupancy count
        #
        #   rasterize_gaussians(xys, depths, radii, conics, opacities, colors, ...)
        #     → rendered [H, W, 3]
        #       alphas   [H, W, 1]

        # View matrix: [4, 4]  world→camera  (homogeneous)
        # [ R | T ]
        # [ 0 | 1 ]
        viewmat = torch.eye(4, device=device)
        viewmat[:3, :3] = cam.R
        viewmat[:3,  3] = cam.T

        (xys, depths, radii, conics,
         comp, num_tiles_hit, cov3d) = gsplat.project_gaussians(
            means3d=model.means3d,
            scales=model.scales,
            glob_scale=1.0,
            quats=model.quaternions,
            viewmat=viewmat,
            projmat=viewmat,           # for simple pinhole, projmat = viewmat
            fx=cam.fx, fy=cam.fy,
            cx=cam.cx, cy=cam.cy,
            img_height=H, img_width=W,
            block_width=16,
            clip_thresh=0.01,
        )

        # Evaluate SH → view-dependent RGB colors
        # Directions: unit vector from Gaussian center to camera origin
        cam_pos = (-cam.R.T @ cam.T)                                  # [3] camera in world
        dirs = cam_pos.unsqueeze(0) - model.means3d.detach()          # [N, 3]
        dirs = F.normalize(dirs, dim=-1)                              # [N, 3] unit dirs

        rgb_colors = gsplat.spherical_harmonics(
            degrees_to_use=min(step // 1000, cfg.sh_degree),          # progressive SH
            dirs=dirs,
            coeffs=model.sh_coefficients,
        )                                                              # [N, 3] ∈ [0,1]
        rgb_colors = torch.clamp(rgb_colors + 0.5, 0.0, 1.0)

        rendered_img, alphas = gsplat.rasterize_gaussians(
            xys=xys,
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=rgb_colors,
            opacity=model.opacities,
            img_height=H,
            img_width=W,
            block_width=16,
            background=bg,
        )                                                              # [H, W, 3]

        # Rearrange to [1, 3, H, W] for loss computation
        rendered_img = rendered_img.permute(2, 0, 1).unsqueeze(0)     # [1, 3, H, W]
        gt_img = cam.image.unsqueeze(0)                               # [1, 3, H, W]

        # ── Loss + backward ───────────────────────────────────────────────
        loss, l1_val, ssim_val = combined_loss(rendered_img, gt_img, cfg.lambda_ssim)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Accumulate position gradients for ADC decision
        model.accumulate_gradients()

        optimizer.step()

        # ── Adaptive Density Control ──────────────────────────────────────
        if (step >= cfg.warmup_iterations and
                step <= cfg.densify_until and
                step % cfg.densify_interval == 0):
            n = model.densify_and_prune(
                optimizer,
                grad_threshold=cfg.grad_threshold_clone,
                scale_threshold=cfg.scale_threshold_split,
                min_opacity=cfg.min_opacity_prune,
                max_gaussians=cfg.max_gaussians,
            )
            log.info("  step %6d | ADC → %d Gaussians", step, n)

        # ── Periodic opacity reset ─────────────────────────────────────────
        if step % cfg.opacity_reset_interval == 0:
            model.reset_opacities(optimizer)

        # ── Logging ───────────────────────────────────────────────────────
        if step % cfg.log_every == 0:
            log.info(
                "  step %6d | loss %.5f  (L1 %.5f | SSIM %.5f) | N=%d",
                step, loss.item(), l1_val.item(), ssim_val.item(), model.num_gaussians(),
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % 5000 == 0 or step == cfg.num_iterations:
            ckpt_path = Path(cfg.output_dir) / f"gaussians_{step:06d}.pt"
            torch.save({
                "step": step,
                "means3d":     model.means3d.detach(),
                "scales":      model._scales.detach(),
                "quaternions": model._quaternions.detach(),
                "opacities":   model._opacities.detach(),
                "sh":          model._sh.detach(),
                "sh_degree":   model.sh_degree,
            }, ckpt_path)
            log.info("  Checkpoint saved → %s", ckpt_path)

    log.info("Training complete. Final: %d Gaussians.", model.num_gaussians())
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="3D Gaussian Splatting — Core Optimization Engine")
    parser.add_argument("--ply",      type=str, default="sparse/0/points3D.ply")
    parser.add_argument("--cameras",  type=str, default="sparse/0/cameras_processed.pt")
    parser.add_argument("--output",   type=str, default="output")
    parser.add_argument("--iters",    type=int, default=30_000)
    parser.add_argument("--device",   type=str, default="cuda")
    args = parser.parse_args()

    cfg = TrainConfig(
        ply_path=args.ply,
        cameras_path=args.cameras,
        output_dir=args.output,
        num_iterations=args.iters,
        device=args.device,
    )

    trained_model = train(cfg)