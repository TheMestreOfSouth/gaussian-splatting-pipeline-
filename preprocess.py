"""
preprocess.py — Data Ingestion & Structure-from-Motion Pipeline
===============================================================
Automates the full COLMAP reconstruction pipeline:
  1. Extract frames from input video using ffmpeg
  2. Run COLMAP feature extraction, matching, and sparse reconstruction
  3. Parse COLMAP output → camera intrinsics (K) + extrinsics (R, T)
  4. Save processed cameras as cameras_processed.pt (list of dicts)
  5. Copy points3D.ply for Gaussian initialization
"""

from __future__ import annotations
import argparse
import logging
import os
import shutil
import subprocess
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

def extract_frames(video_path: str, output_dir: str, fps: float = 2.0, max_dimension: int = 1280) -> int:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scale_filter = f"fps={fps},scale='if(gt(iw,ih),min(iw,{max_dimension}),-2)':'if(gt(iw,ih),-2,min(ih,{max_dimension}))'"
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", scale_filter, "-qscale:v", "1", "-start_number", "0", str(out / "%04d.jpg")]
    log.info("Extracting frames: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg failed:\n%s", result.stderr)
        raise RuntimeError("Frame extraction failed. Is ffmpeg installed?")
    frames = list(out.glob("*.jpg"))
    log.info("Extracted %d frames → %s", len(frames), out)
    return len(frames)

def run_colmap(image_dir: str, workspace_dir: str, use_gpu: bool = True) -> Path:
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

    run(["colmap", "feature_extractor", "--database_path", str(db_path), "--image_path", str(image_dir), "--ImageReader.single_camera", "1", "--SiftExtraction.use_gpu", gpu_flag, "--SiftExtraction.max_image_size", "1600"], "feature_extractor")
    run(["colmap", "exhaustive_matcher", "--database_path", str(db_path), "--SiftMatching.use_gpu", gpu_flag], "exhaustive_matcher")
    run(["colmap", "mapper", "--database_path", str(db_path), "--image_path", str(image_dir), "--output_path", str(sparse_dir), "--Mapper.num_threads", "8", "--Mapper.init_min_tri_angle", "4"], "mapper")

    sparse_0 = sparse_dir / "0"
    if not sparse_0.exists():
        raise RuntimeError("COLMAP mapper produced no output. Video might have blur or insufficient frames.")

    run(["colmap", "model_converter", "--input_path", str(sparse_0), "--output_path", str(sparse_0), "--output_type", "TXT"], "model_converter")
    run(["colmap", "model_converter", "--input_path", str(sparse_0), "--output_path", str(sparse_0 / "points3D.ply"), "--output_type", "PLY"], "ply_export")
    return sparse_0

def parse_colmap_cameras(cameras_txt: str) -> dict[int, dict]:
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
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                fx = fy = params[0]
                cx, cy = W / 2.0, H / 2.0
            cameras[cam_id] = dict(fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H)
    return cameras

def parse_colmap_images(images_txt: str) -> list[dict]:
    images = []
    with open(images_txt) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]
        R = quat_to_rotmat(np.array([qw, qx, qy, qz]))
        T = np.array([tx, ty, tz], dtype=np.float32)
        images.append(dict(name=name, cam_id=cam_id, R=R.astype(np.float32), T=T))
    return images

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)

def build_camera_list(sparse_dir: str, image_dir: str, resize_to: Optional[int] = 800) -> list[dict]:
    sp = Path(sparse_dir)
    intrinsics = parse_colmap_cameras(str(sp / "cameras.txt"))
    poses = parse_colmap_images(str(sp / "images.txt"))
    camera_list = []
    for pose in poses:
        img_path = Path(image_dir) / pose["name"]
        if not img_path.exists(): continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cam = intrinsics[pose["cam_id"]]
        orig_H, orig_W = img_rgb.shape[:2]
        if resize_to is not None:
            scale = resize_to / max(orig_H, orig_W)
            new_W, new_H = int(orig_W * scale), int(orig_H * scale)
            img_rgb = cv2.resize(img_rgb, (new_W, new_H), interpolation=cv2.INTER_AREA)
            fx, fy, cx, cy = cam["fx"] * scale, cam["fy"] * scale, cam["cx"] * scale, cam["cy"] * scale
            W, H = new_W, new_H
        else:
            fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
            W, H = orig_W, orig_H
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        camera_list.append({
            "image": img_tensor, "R": torch.from_numpy(pose["R"]), "T": torch.from_numpy(pose["T"]),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "W": W, "H": H, "name": pose["name"],
        })
    return camera_list

def main() -> None:
    parser = argparse.ArgumentParser(description="3DGS Preprocessor — COLMAP automation")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--frames", type=str, default=None)
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--resize", type=int, default=800)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    if args.video is None and args.frames is None:
        parser.error("Provide either --video or --frames")

    ws = Path(args.output)
    frames_dir = ws / "frames"
    
    if args.video:
        n = extract_frames(args.video, str(frames_dir), fps=args.fps)
    else:
        frames_dir = Path(args.frames)

    sparse_0 = run_colmap(image_dir=str(frames_dir), workspace_dir=str(ws), use_gpu=not args.no_gpu)
    camera_list = build_camera_list(sparse_dir=str(sparse_0), image_dir=str(frames_dir), resize_to=args.resize)

    out_pt = sparse_0 / "cameras_processed.pt"
    torch.save(camera_list, str(out_pt))
    log.info("Preprocessing complete. Points and real cameras saved.")

if __name__ == "__main__":
    main()