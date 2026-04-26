# CW2 Data Collection — Instructions

## Quick Start

```bash
# 1. Create conda environment (first time only)
chmod +x setup_env.sh
./setup_env.sh

# 2. Activate environment
conda activate cw2_slam

# 3. Verify environment
python -c "import pyrealsense2; import rplidar; import cv2; print('All OK')"
```

---

## Equipment Checklist

- [ ] Intel RealSense camera (or own camera with fixed intrinsics)
- [ ] RP-Lidar A2M12 sensor + USB cable
- [ ] Checkerboard calibration pattern (9×6 inner corners)
- [ ] Floor markers (tape) for LiDAR loop start/end points
- [ ] Laptop with conda env set up

---

## Step 1: Camera Calibration (if not using RealSense)

> **Note:** If using RealSense, intrinsics are auto-extracted by `camera_recorder.py`.
> Only run this if you need OpenCV calibration (e.g., phone camera).

```bash
python camera_calibration_helper.py --backend realsense
```

- Point camera at the checkerboard from various angles/distances
- Press **Space** to capture each image (aim for 15-25 images)
- Press **C** to run calibration when done
- Output: `data/calibration/`

---

## Step 2: Camera Recording (Q2)

Record **2 sequences**: 1 indoor, 1 outdoor. Each must have ≥500 frames.

```bash
# Indoor sequence
python camera_recorder.py --name indoor_01 --env indoor

# Outdoor sequence
python camera_recorder.py --name outdoor_01 --env outdoor
```

### Tips for Good SLAM Results
- **Move slowly and smoothly** — no sudden jerks
- **Keep good texture** in the scene — avoid blank walls
- **Maintain overlap** — ~80% overlap between consecutive frames
- **Revisit areas** — helps ORB-SLAM2 with loop closure
- **Avoid motion blur** — steady hands or use a mount
- **Lock camera settings** — RealSense auto-handled; for phone, lock focus/exposure

### Output
```
data/camera/indoor_01/
├── video.mp4                # Video for submission
├── frames/                  # Individual frames for COLMAP
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── timestamps.txt           # Raw timestamps
├── rgb.txt                  # ORB-SLAM2 TUM association file
├── camera_intrinsics.json   # RealSense intrinsics
├── orbslam2_camera.yaml     # ORB-SLAM2 config (auto-generated)
└── sequence_info.json       # Recording metadata
```

---

## Step 3: LiDAR Recording (Q3)

Record **3 sequences**: 2 indoor (1 large area), 1 outdoor.

```bash
# Indoor large (e.g., Marshgate hallways)
python lidar_recorder.py --name indoor_large_01 --env indoor \
    --desc "Marshgate building hallways"

# Indoor small
python lidar_recorder.py --name indoor_small_01 --env indoor \
    --desc "Lab room"

# Outdoor
python lidar_recorder.py --name outdoor_01 --env outdoor \
    --desc "Building exterior path"
```

### Recording Procedure
1. **Mark the floor** at your start position (tape/chalk)
2. Place the LiDAR on a flat surface you can carry (clipboard, tray)
3. Start the recorder
4. Walk the trajectory — **complete exactly 2 loops**
5. Press **L** each time you pass the start marker (= 2 presses)
6. Press **Q** to stop when done
7. Remove floor marker

### Tips
- **Hold the LiDAR level** — tilting causes poor 2D scans
- **Walk at a steady pace** — not too fast
- **Same environments as camera** — recommended for comparison in report
- **Stay behind the sensor** — keep in the blind spot (135°–225°)

### Output
```
data/lidar/indoor_large_01/
├── scans.json            # Scan data (compatible with lab code)
├── timestamps.txt        # Timestamp per scan
├── loop_markers.json     # Loop completion markers
└── sequence_info.json    # Recording metadata
```

---

## Step 4: Verify Data

Run the verification script to check everything is complete:

```bash
python verify_data.py --data-dir data/
```

This checks:
- ✓ 2+ camera sequences (1 indoor, 1 outdoor), ≥500 frames each
- ✓ 3+ LiDAR sequences (2 indoor, 1 outdoor), 2 loops each
- ✓ Calibration files present

---

## Using Collected Data Later

### For ORB-SLAM2
```bash
# Your data is already in the right format
./Examples/Monocular/mono_tum \
    Vocabulary/ORBvoc.txt \
    data/camera/indoor_01/orbslam2_camera.yaml \
    data/camera/indoor_01/
```

### For COLMAP
```bash
# Use the frames directory
colmap feature_extractor \
    --database_path data/camera/indoor_01/colmap.db \
    --image_path data/camera/indoor_01/frames/
```

### For LiDAR processing (Lab 08/09 code)
```python
# Your scans.json is compatible with the existing LidarDriver
from rplidar_driver import LidarDriver
driver = LidarDriver(mode='replay', filename='data/lidar/indoor_large_01/scans.json')
for scan in driver.iter_scans():
    # Process as in lab code...
    pass
```
