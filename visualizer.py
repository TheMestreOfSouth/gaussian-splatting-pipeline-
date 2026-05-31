"""
visualizer.py — Dual-Mode 3DGS Renderer
========================================
Loads a trained Gaussian checkpoint (.pt) and renders in two modes:

  Mode A — Photorealistic:  full splatted render via gsplat rasterizer
  Mode B — Cyberpunk LiDAR: renders only Gaussian centers as neon emissive
                             points, sized by depth and colored by opacity

Controls (keyboard):
  M      — toggle between Photorealistic / LiDAR mode
  S      — save screenshot to output/screenshot_NNNN.png
  Q/ESC  — quit

Usage:
  python visualizer.py --checkpoint output/gaussians_030000.pt
  python visualizer.py --checkpoint output/gaussians_030000.pt --mode lidar
  python visualizer.py --checkpoint output/gaussians_030000.pt --save-only
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("3dgs.visualizer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint(path: str, device: str = "cpu") -> dict:
    """Load a Gaussian checkpoint saved by train_pipeline.py."""
    ckpt = torch.load(path, map_location=device)
    required = ["means3d", "scales", "quaternions", "opacities", "sh"]
    for k in required:
        if k not in ckpt:
            raise KeyError(f"Checkpoint missing key '{k}'. Is this a valid 3DGS checkpoint?")
    log.info(
        "Loaded checkpoint: %d Gaussians  (step %d)",
        ckpt["means3d"].shape[0],
        ckpt.get("step", -1),
    )
    return ckpt


# ---------------------------------------------------------------------------
# Camera helper
# ---------------------------------------------------------------------------
class OrbitCamera:
    """
    Simple orbit camera for interactive visualization.
    Orbits around the scene centroid.
    """

    def __init__(self, centroid: np.ndarray, radius: float = 3.0):
        self.centroid = centroid
        self.radius = radius
        self.theta = 0.0       # azimuth  (radians)
        self.phi = 0.3         # elevation (radians)
        self.fov_y = math.radians(60.0)
        self.W = 1280
        self.H = 720

    @property
    def fx(self) -> float:
        return self.H / (2.0 * math.tan(self.fov_y / 2.0))

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.W / 2.0

    @property
    def cy(self) -> float:
        return self.H / 2.0

    def get_RT(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute R, T for current orbit position.
        Camera looks at centroid from spherical coordinates (theta, phi, radius).
        """
        # Camera position in world space
        x = self.radius * math.cos(self.phi) * math.sin(self.theta)
        y = self.radius * math.sin(self.phi)
        z = self.radius * math.cos(self.phi) * math.cos(self.theta)
        cam_pos = self.centroid + np.array([x, y, z], dtype=np.float32)

        # Look-at matrix
        forward = self.centroid - cam_pos
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # Y-down (OpenCV)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        # R: world→camera  (rows are camera axes expressed in world)
        R = np.stack([right, -up, forward], axis=0).astype(np.float32)
        T = (-R @ cam_pos).astype(np.float32)

        return R, T

    def orbit(self, dtheta: float, dphi: float) -> None:
        self.theta += dtheta
        self.phi = np.clip(self.phi + dphi, -1.4, 1.4)

    def zoom(self, delta: float) -> None:
        self.radius = max(0.5, self.radius - delta)


# ---------------------------------------------------------------------------
# Neon LiDAR renderer (pure numpy/OpenCV — no gsplat needed)
# ---------------------------------------------------------------------------
NEON_PALETTE = np.array([
    [0,   255, 255],   # cyan
    [255,  0,  255],   # magenta
    [0,   255,  80],   # green
    [255, 160,   0],   # amber
    [80,  120, 255],   # blue
], dtype=np.float32)


def render_lidar(
    means3d: torch.Tensor,
    opacities: torch.Tensor,
    R: np.ndarray,
    T: np.ndarray,
    fx: float, fy: float,
    cx: float, cy: float,
    W: int, H: int,
) -> np.ndarray:
    """
    Render Gaussian centers as neon emissive points.

    Point size  ∝  1 / depth  (closer = bigger)
    Point color = neon palette index by opacity quintile
    Point alpha = opacity value

    Returns BGR image [H, W, 3] uint8.
    """
    import cv2

    pts = means3d.numpy()        # [N, 3]
    alpha = opacities.numpy().squeeze(-1)  # [N]

    # World → camera
    pts_cam = (R @ pts.T).T + T[None]  # [N, 3]

    # Keep points in front of camera
    mask = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[mask]
    alpha_f = alpha[mask]

    if len(pts_cam) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # Project to 2D
    x2d = (pts_cam[:, 0] * fx / pts_cam[:, 2] + cx).astype(np.float32)
    y2d = (pts_cam[:, 1] * fy / pts_cam[:, 2] + cy).astype(np.float32)
    depth = pts_cam[:, 2]

    # Clip to image bounds
    in_frame = (x2d >= 0) & (x2d < W) & (y2d >= 0) & (y2d < H)
    x2d, y2d = x2d[in_frame], y2d[in_frame]
    depth_f = depth[in_frame]
    alpha_f = alpha_f[in_frame]

    # Sort back-to-front so closer points draw on top
    order = np.argsort(-depth_f)
    x2d, y2d = x2d[order], y2d[order]
    depth_f = depth_f[order]
    alpha_f = alpha_f[order]

    # Normalize depth for point sizing: min size 1, max size 8
    d_min, d_max = depth_f.min(), depth_f.max() + 1e-6
    norm_depth = (depth_f - d_min) / (d_max - d_min)   # 0=close, 1=far
    radii = (1 + (1.0 - norm_depth) * 6).astype(np.int32)

    # Assign neon colors by opacity quintile
    quintile = (alpha_f * 5).astype(np.int32).clip(0, 4)

    # Render on black canvas
    canvas = np.zeros((H, W, 3), dtype=np.float32)

    for i in range(len(x2d)):
        cx_i, cy_i = int(x2d[i]), int(y2d[i])
        r = int(radii[i])
        color = NEON_PALETTE[quintile[i]] * alpha_f[i]
        cv2.circle(canvas, (cx_i, cy_i), r, color.tolist(), -1, cv2.LINE_AA)

    # Bloom: slight blur then add back for glow effect
    blur = cv2.GaussianBlur(canvas, (9, 9), 3)
    canvas = np.clip(canvas * 0.7 + blur * 0.6, 0, 255).astype(np.uint8)

    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Photorealistic renderer (via gsplat)
# ---------------------------------------------------------------------------
def render_photorealistic(
    ckpt: dict,
    R: np.ndarray,
    T: np.ndarray,
    fx: float, fy: float,
    cx: float, cy: float,
    W: int, H: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Render full splatted image using gsplat rasterizer.
    Returns BGR image [H, W, 3] uint8.
    """
    import cv2
    import gsplat

    means3d = ckpt["means3d"].to(device)
    scales = torch.exp(ckpt["scales"].to(device))
    quats = F.normalize(ckpt["quaternions"].to(device), dim=-1)
    opacities = torch.sigmoid(ckpt["opacities"].to(device))
    sh = ckpt["sh"].to(device)
    sh_degree = ckpt.get("sh_degree", 3)

    R_t = torch.from_numpy(R).to(device)
    T_t = torch.from_numpy(T).to(device)

    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = R_t
    viewmat[:3,  3] = T_t

    try:
        (xys, depths, radii, conics,
         comp, num_tiles_hit, _) = gsplat.project_gaussians(
            means3d=means3d,
            scales=scales,
            glob_scale=1.0,
            quats=quats,
            viewmat=viewmat,
            projmat=viewmat,
            fx=fx, fy=fy, cx=cx, cy=cy,
            img_height=H, img_width=W,
            block_width=16,
            clip_thresh=0.01,
        )

        cam_pos = (-R_t.T @ T_t)
        dirs = F.normalize(cam_pos.unsqueeze(0) - means3d, dim=-1)
        rgb = gsplat.spherical_harmonics(
            degrees_to_use=sh_degree,
            dirs=dirs,
            coeffs=sh,
        )
        rgb = torch.clamp(rgb + 0.5, 0.0, 1.0)

        bg = torch.zeros(3, device=device)
        rendered, _ = gsplat.rasterize_gaussians(
            xys=xys, depths=depths, radii=radii,
            conics=conics, num_tiles_hit=num_tiles_hit,
            colors=rgb, opacity=opacities,
            img_height=H, img_width=W,
            block_width=16, background=bg,
        )

        img_np = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    except Exception as e:
        log.error("gsplat render failed: %s", e)
        # Fallback: show LiDAR mode if gsplat fails
        return render_lidar(
            means3d.cpu(), opacities.cpu(),
            R, T, fx, fy, cx, cy, W, H
        )


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------
def run_viewer(ckpt: dict, start_mode: str = "lidar", device: str = "cpu") -> None:
    """
    Interactive OpenCV window with orbit controls.

    Mouse drag  — orbit
    Scroll      — zoom
    M           — toggle mode
    S           — screenshot
    Q / ESC     — quit
    """
    import cv2

    means3d = ckpt["means3d"]
    opacities = torch.sigmoid(ckpt["opacities"])

    # Compute scene centroid and scale
    centroid = means3d.mean(dim=0).numpy()
    pts_np = means3d.numpy()
    extent = np.linalg.norm(pts_np - centroid, axis=1).mean()
    radius = float(extent) * 2.5

    camera = OrbitCamera(centroid, radius=radius)
    mode = start_mode   # "lidar" or "photo"
    screenshot_idx = 0
    Path("output").mkdir(exist_ok=True)

    # Mouse state
    mouse = {"dragging": False, "last_x": 0, "last_y": 0}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["dragging"] = True
            mouse["last_x"] = x
            mouse["last_y"] = y
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["dragging"] = False
        elif event == cv2.EVENT_MOUSEMOVE and mouse["dragging"]:
            dx = x - mouse["last_x"]
            dy = y - mouse["last_y"]
            camera.orbit(dx * 0.005, dy * 0.005)
            mouse["last_x"] = x
            mouse["last_y"] = y
        elif event == cv2.EVENT_MOUSEWHEEL:
            camera.zoom(0.3 if flags > 0 else -0.3)

    win = "3DGS Viewer  |  M=toggle mode  S=screenshot  Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, camera.W, camera.H)
    cv2.setMouseCallback(win, on_mouse)

    log.info("Viewer open. Controls: drag=orbit, scroll=zoom, M=mode, S=screenshot, Q=quit")
    log.info("Starting in %s mode", mode.upper())

    while True:
        R, T = camera.get_RT()

        if mode == "lidar":
            frame = render_lidar(
                means3d, opacities,
                R, T,
                camera.fx, camera.fy,
                camera.cx, camera.cy,
                camera.W, camera.H,
            )
            label = "MODE: CYBERPUNK LiDAR"
            color = (0, 255, 200)
        else:
            frame = render_photorealistic(
                ckpt, R, T,
                camera.fx, camera.fy,
                camera.cx, camera.cy,
                camera.W, camera.H,
                device=device,
            )
            label = "MODE: PHOTOREALISTIC"
            color = (200, 200, 200)

        # HUD overlay
        cv2.putText(frame, label, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        n_gauss = f"{means3d.shape[0]:,} Gaussians"
        cv2.putText(frame, n_gauss, (20, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # Q or ESC
            break
        elif key == ord("m"):
            mode = "photo" if mode == "lidar" else "lidar"
            log.info("Switched to %s mode", mode.upper())
        elif key == ord("s"):
            path = f"output/screenshot_{screenshot_idx:04d}.png"
            cv2.imwrite(path, frame)
            log.info("Screenshot saved → %s", path)
            screenshot_idx += 1

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Save-only mode (no window — for Colab)
# ---------------------------------------------------------------------------
def save_renders(ckpt: dict, output_dir: str, device: str = "cpu") -> None:
    """
    Save one LiDAR and one photorealistic render without opening a window.
    Useful for running in Google Colab.
    """
    import cv2

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    means3d = ckpt["means3d"]
    opacities = torch.sigmoid(ckpt["opacities"])
    centroid = means3d.mean(dim=0).numpy()
    extent = float(np.linalg.norm(means3d.numpy() - centroid, axis=1).mean())

    camera = OrbitCamera(centroid, radius=extent * 2.5)

    angles = [0.0, math.pi / 4, math.pi / 2, math.pi]

    for i, theta in enumerate(angles):
        camera.theta = theta
        R, T = camera.get_RT()

        # LiDAR
        lidar = render_lidar(
            means3d, opacities,
            R, T,
            camera.fx, camera.fy,
            camera.cx, camera.cy,
            camera.W, camera.H,
        )
        path = out / f"lidar_{i:02d}.png"
        cv2.imwrite(str(path), lidar)
        log.info("Saved %s", path)

        # Photorealistic
        photo = render_photorealistic(
            ckpt, R, T,
            camera.fx, camera.fy,
            camera.cx, camera.cy,
            camera.W, camera.H,
            device=device,
        )
        path = out / f"photo_{i:02d}.png"
        cv2.imwrite(str(path), photo)
        log.info("Saved %s", path)

    log.info("All renders saved to %s", out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGS Dual-Mode Visualizer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to gaussians_XXXXXX.pt checkpoint")
    parser.add_argument("--mode", type=str, default="lidar",
                        choices=["lidar", "photo"],
                        help="Starting render mode (default: lidar)")
    parser.add_argument("--save-only", action="store_true",
                        help="Save renders to output/ without opening a window (for Colab)")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory for screenshots")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for photorealistic render (cpu or cuda)")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint, device=args.device)

    if args.save_only:
        save_renders(ckpt, args.output, device=args.device)
    else:
        run_viewer(ckpt, start_mode=args.mode, device=args.device)