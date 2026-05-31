# 3D Gaussian Splatting Optimization Pipeline

This repository implements a complete, lightweight pipeline for 3D Gaussian Splatting (3DGS). It handles data preprocessing, structure-from-motion (SfM) camera data integration, camera parameter parsing, scene optimization, and interactive/headless point-cloud rendering.

The codebase is structured to allow development inside standard IDEs (such as VS Code) while supporting seamless computation partitioning to high-performance headless GPU clusters or cloud environments (such as Google Colab).

## Repository Architecture

The project structure is organized as follows:
* `data/` : Contains input media, extracted video frames, and sparse reconstruction artifacts.
* `output/` : Directory where trained point clouds, optimized Gaussian parameters, and rendered viewpoints are saved.
* `preprocess.py` : Handles frame extraction, camera intrinsic calibration, and parsing of initial geometric constraints.
* `train_pipeline.py` : Core training loop responsible for initializing Gaussians, computing loss functions (L1/SSIM), and optimizing spatial attributes (position, covariance, opacity, and spherical harmonics).
* `visualizer.py` : Renders the trained scene, supporting both interactive dual-mode display and headless offline snapshot generation.
* `requirements.txt` : Python package dependencies required to run the pipeline.
* `.gitignore` : Configured to exclude heavy media files, raw video sequences, and volatile cache directories from version control.

## Engineering Challenges and Core Bug Fixes

During deployment and infrastructure validation on virtualized Linux server environments, two critical architectural bottlenecks were identified and resolved within this repository:

### 1. Headless Server Environment Emulation (COLMAP Qt/XCB Display Crash)
* **Problem:** Standard Structure-from-Motion engines like COLMAP are natively compiled with Qt platform dependencies. When executed over virtualized or headless cloud backends without a physical display server, the process raises a fatal segmentation fault (`qt.qpa.xcb: could not connect to display`).
* **Solution:** The pipeline was updated to accept a decoupled initialization layout. By running spatial matrix factories inside Python, we can generate compliant camera matrices (`cameras_processed.pt`) and initial 3D anchoring point coordinates directly into the tracking pipeline, bypassing XCB dependency failures entirely while keeping the down-stream training loop intact.

### 2. PyTorch 2.6+ Security Protocol Integration (Serialization Restrictions)
* **Problem:** In recent PyTorch releases, the default security behavior of the serialization module (`torch.load`) changed from `weights_only=False` to `weights_only=True`. This restriction completely blocks the unpickling of complex nested dictionary structures, Python lists, and underlying NumPy multiarray configurations generated during camera tracking.
* **Solution:** The ingestion function `load_cameras` in `train_pipeline.py` was structurally patched to enforce `weights_only=False` during internal load routines. This allows safe, controlled object reconstruction of matrix lists directly into GPU memory without breaking execution state.

## Getting Started

### Prerequisites
Ensure your Python environment meets the required specifications:
```bash
pip install -r requirements.txt