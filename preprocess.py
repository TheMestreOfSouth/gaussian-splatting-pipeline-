"""
preprocess.py — Data Ingestion & Structure-from-Motion Pipeline
===============================================================
Automates the full COLMAP reconstruction pipeline:
  1. Extract frames from input video using ffmpeg
  2. Run COLMAP feature extraction, matching, and sparse reconstruction
  3. Parse COLMAP output → camera intrinsics (K) + extrinsics (R, T)
  4. Save processed cameras as cameras_processed.pt (list of dicts)
  5. Copy points3D.ply for Gaussian initialization

Output structure:
  data/
    frames/          ← extracted video frames
    sparse/0/
      cameras.bin    ← COLMAP camera intrinsics
      images.bin     ← COLMAP camera extrinsics per frame
      points3D.bin   ← sparse point cloud
      points3D.ply   ← converted for Gaussian init
      cameras_processed.pt  ← ready for train_pipeline.py

Usage:
  python preprocess.py --video data/museum.mp4 --output data/
  python preprocess.py --frames data/frames/ --output data/   # skip extraction
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("3dgs.preprocess")


# ---------------------------------------------------------------------------
# Step 1: Extract frames from video
# ---------------------------------------------------------------------------
def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 2.0,
    max_dimension: int = 1280,
) -> int:
    """
    Extract frames from video at given FPS using ffmpeg.
    Resizes so the longest side <= max_dimension.

    Returns number of extracted frames.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scale_filter = (
        f"fps={fps},"
        f"scale='if(gt(iw,ih),min(iw,{max_dimension}),-2)'"
        f":'if(gt(iw,ih),-2,min(ih,{max_dimension}))'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", scale_filter,
        "-qscale:v", "1",
        "-start_number", "0",
        str(out / "%04d.jpg"),
    ]

    log.info("Extracting frames: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("ffmpeg failed:\n%s", result.stderr)
        raise RuntimeError("Frame extraction failed. Is ffmpeg installed?")

    frames = list(out.glob("*.jpg"))
    log.info("Extracted %d frames → %s", len(frames), out)
    return len(frames)


# ---------------------------------------------------------------------------
# Step 2: Run COLMAP
# ---------------------------------------------------------------------------
def run_colmap(
    image_dir: str,
    workspace_dir: str,
    use_gpu: bool = True,
) -> Path:
    """
    Run COLMAP automatic reconstruction pipeline:
      feature_extractor → exhaustive_matcher → mapper → model_converter

    Returns path to sparse/0/ directory.
    """
    ws = Path(workspace_dir)
    sparse_dir = ws / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    db_path = ws / "colmap.db"
    gpu_flag = "1" if use_gpu else "0"

    def run(cmd: list[str], step: str) -> None:
        log.info("[COLMAP] %s ...", step)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("COLMAP %s failed:\n%s", step, result.stderr[-3000:])
            raise RuntimeError(f"COLMAP {step} failed.")
        log.info("[COLMAP] %s done.", step)

    # Feature extraction
    run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", gpu_flag,
        "--SiftExtraction.max_image_size", "1600",
    ], "feature_extractor")

    # Exhaustive matching
    run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", gpu_flag,
    ], "exhaustive_matcher")

    # Sparse reconstruction (mapper)
    run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.num_threads", "8",
        "--Mapper.init_min_tri_angle", "4",
    ], "mapper")

    # Check output
    sparse_0 = sparse_dir / "0"
    if not sparse_0.exists():
        raise RuntimeError(
            "COLMAP mapper produced no output in sparse/0/. "
            "This usually means the video has too much motion blur, "
            "cuts, or a static camera. Try a different video."
        )

    # Convert binary model → text for easier parsing
    run([
        "colmap", "model_converter",
        "--input_path", str(sparse_0),
        "--output_path", str(sparse_0),
        "--output_type", "TXT",
    ], "model_converter")

    # Convert to PLY point cloud
    run([
        "colmap", "model_converter",
        "--input_path", str(sparse_0),
        "--output_path", str(sparse_0 / "points3D.ply"),
        "--output_type", "PLY",
    ], "ply_export")

    log.info("COLMAP reconstruction complete → %s", sparse_0)
    return sparse_0


# ---------------------------------------------------------------------------
# Step 3: Parse COLMAP text output
# ---------------------------------------------------------------------------
def parse_colmap_cameras(cameras_txt: str) -> dict[int, dict]:
    """
    Parse cameras.txt → dict of camera intrinsics keyed by camera_id.

    Supports PINHOLE and SIMPLE_PINHOLE models.
    cameras.txt format:
      # Camera list with one line of data per camera:
      #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
      1 PINHOLE 1920 1080 1200.0 1200.0 960.0 540.0
    """
    cameras: dict[int, dict] = {}
    with open(cameras_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            W, H = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))

            if model == "SIMPLE_PINHOLE":
                # params: f, cx, cy
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "PINHOLE":
                # params: fx, fy, cx, cy
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model in ("SIMPLE_RADIAL", "RADIAL"):
                # params: f, cx, cy, k  (ignore distortion k)
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                log.warning("Unsupported camera model %s — treating as PINHOLE", model)
                fx = fy = params[0]
                cx = W / 2.0
                cy = H / 2.0

            cameras[cam_id] = dict(fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H)

    log.info("Parsed %d cameras from cameras.txt", len(cameras))
    return cameras


def parse_colmap_images(images_txt: str) -> list[dict]:
    """
    Parse images.txt → list of per-image pose dicts.

    images.txt format (2 lines per image):
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      POINTS2D[] as (X, Y, POINT3D_ID)

    Quaternion convention: COLMAP uses [qw, qx, qy, qz] world→camera.
    Rotation matrix R = quat_to_rotmat(q)  (world→camera)
    Translation T = [TX, TY, TZ]           (world→camera)
    """
    images = []
    with open(images_txt) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # Lines come in pairs: pose line, then points line
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]

        # Quaternion [w, x, y, z] → rotation matrix (world→camera)
        R = quat_to_rotmat(np.array([qw, qx, qy, qz]))
        T = np.array([tx, ty, tz], dtype=np.float32)

        images.append(dict(
            name=name,
            cam_id=cam_id,
            R=R.astype(np.float32),
            T=T,
        ))

    log.info("Parsed %d image poses from images.txt", len(images))
    return images


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    Standard formula from unit quaternion to SO(3).
    """
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 4: Load images + assemble camera dicts
# ---------------------------------------------------------------------------
def build_camera_list(
    sparse_dir: str,
    image_dir: str,
    resize_to: Optional[int] = 800,
) -> list[dict]:
    """
    Combine intrinsics + extrinsics + actual images into a list of dicts
    ready to be saved as cameras_processed.pt.

    Each dict:
      image: Tensor [3, H, W] float32 [0, 1]
      R:     Tensor [3, 3]
      T:     Tensor [3]
      fx, fy, cx, cy: float
      W, H: int
    """
    sp = Path(sparse_dir)
    intrinsics = parse_colmap_cameras(str(sp / "cameras.txt"))
    poses = parse_colmap_images(str(sp / "images.txt"))

    camera_list = []
    missing = 0

    for pose in poses:
        img_path = Path(image_dir) / pose["name"]
        if not img_path.exists():
            missing += 1
            continue

        # Load and convert image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            missing += 1
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cam = intrinsics[pose["cam_id"]]
        orig_H, orig_W = img_rgb.shape[:2]

        # Resize if requested
        if resize_to is not None:
            scale = resize_to / max(orig_H, orig_W)
            new_W = int(orig_W * scale)
            new_H = int(orig_H * scale)
            img_rgb = cv2.resize(img_rgb, (new_W, new_H), interpolation=cv2.INTER_AREA)

            # Scale intrinsics proportionally
            fx = cam["fx"] * scale
            fy = cam["fy"] * scale
            cx = cam["cx"] * scale
            cy = cam["cy"] * scale
            W, H = new_W, new_H
        else:
            fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
            W, H = orig_W, orig_H

        # HWC uint8 → CHW float32 [0, 1]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        camera_list.append({
            "image": img_tensor,
            "R": torch.from_numpy(pose["R"]),
            "T": torch.from_numpy(pose["T"]),
            "fx": fx, "fy": fy,
            "cx": cx, "cy": cy,
            "W": W, "H": H,
            "name": pose["name"],
        })

    if missing > 0:
        log.warning("Skipped %d images (not found on disk)", missing)

    log.info("Built %d camera entries", len(camera_list))
    return camera_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="3DGS Preprocessor — COLMAP automation")
    parser.add_argument("--video",    type=str, default=None,
                        help="Path to input video file (mp4, mov, etc.)")
    parser.add_argument("--frames",   type=str, default=None,
                        help="Path to existing frames directory (skip extraction)")
    parser.add_argument("--output",   type=str, default="data",
                        help="Output workspace directory")
    parser.add_argument("--fps",      type=float, default=2.0,
                        help="Frames per second to extract (default: 2)")
    parser.add_argument("--resize",   type=int, default=800,
                        help="Resize longest side to this (default: 800)")
    parser.add_argument("--no-gpu",   action="store_true",
                        help="Disable GPU for COLMAP (slower but works without NVIDIA)")
    args = parser.parse_args()

    if args.video is None and args.frames is None:
        parser.error("Provide either --video or --frames")

    ws = Path(args.output)
    frames_dir = ws / "frames"
    sparse_dir = ws / "sparse" / "0"

    # Step 1: extract frames
    if args.video:
        log.info("=== Step 1: Extracting frames from video ===")
        n = extract_frames(args.video, str(frames_dir), fps=args.fps)
        if n < 30:
            log.warning(
                "Only %d frames extracted. COLMAP needs at least 30. "
                "Try lowering --fps or using a longer video.", n
            )
    else:
        frames_dir = Path(args.frames)
        log.info("=== Step 1: Using existing frames from %s ===", frames_dir)

    # Step 2: run COLMAP
    log.info("=== Step 2: Running COLMAP (this takes 20-40 min) ===")
    sparse_0 = run_colmap(
        image_dir=str(frames_dir),
        workspace_dir=str(ws),
        use_gpu=not args.no_gpu,
    )

    # Step 3+4: parse output and build camera list
    log.info("=== Step 3: Parsing COLMAP output ===")
    camera_list = build_camera_list(
        sparse_dir=str(sparse_0),
        image_dir=str(frames_dir),
        resize_to=args.resize,
    )

    if len(camera_list) == 0:
        raise RuntimeError(
            "No cameras were processed. "
            "Check that COLMAP ran successfully and image names match."
        )

    # Save cameras_processed.pt
    out_pt = sparse_0 / "cameras_processed.pt"
    torch.save(camera_list, str(out_pt))
    log.info("Saved %d cameras → %s", len(camera_list), out_pt)

    # Summary
    ply_path = sparse_0 / "points3D.ply"
    log.info("=== Preprocessing complete ===")
    log.info("  Point cloud : %s", ply_path)
    log.info("  Cameras     : %s", out_pt)
    log.info("")
    log.info("Next step:")
    log.info("  python train_pipeline.py \\")
    log.info("    --ply %s \\", ply_path)
    log.info("    --cameras %s \\", out_pt)
    log.info("    --output output/")


if __name__ == "__main__":
    main()