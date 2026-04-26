#!/usr/bin/env python3
"""
CW2 Data Collection — Camera Calibration Helper
=================================================
Capture checkerboard calibration images and compute camera intrinsics.

This is useful if:
- You used a non-RealSense camera (phone, webcam)
- You want to double-check RealSense intrinsics with OpenCV calibration

Outputs:
  - calibration_images/  : Captured checkerboard images
  - calibration.json     : Calibration results (fx, fy, cx, cy, distortion)
  - calibration.yaml     : ORB-SLAM2 format config
  - camera.txt           : COLMAP camera model file

Usage:
  # With RealSense
  python camera_calibration_helper.py --backend realsense

  # With webcam
  python camera_calibration_helper.py --backend opencv --camera 0

  # Custom checkerboard (inner corners, e.g., 10x7 board has 9x6 inner corners)
  python camera_calibration_helper.py --rows 6 --cols 9 --square-size 25.0

Controls:
  Space : Capture image (when checkerboard is detected)
  c     : Run calibration with captured images
  q     : Quit
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2

REALSENSE_AVAILABLE = False
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    pass


def run_calibration(image_dir, board_rows, board_cols, square_size_mm):
    """
    Run OpenCV camera calibration on captured checkerboard images.

    Returns: (camera_matrix, dist_coeffs, image_size)
    """
    board_size = (board_cols, board_rows)  # (cols, rows) for OpenCV

    # 3D points in real-world coordinates
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []  # 3D points
    img_points = []  # 2D image points

    images = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith((".png", ".jpg"))
    ])

    if len(images) < 5:
        print(f"[ERROR] Need at least 5 calibration images, found {len(images)}")
        return None, None, None

    img_size = None
    used = 0

    for fname in images:
        img = cv2.imread(os.path.join(image_dir, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = gray.shape[::-1]  # (width, height)

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            # Refine corner locations
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.001
            )
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners2)
            used += 1

    print(f"[INFO] Used {used}/{len(images)} images for calibration")

    if used < 5:
        print("[ERROR] Not enough valid calibration images")
        return None, None, None

    # ─── Calibrate ───
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    print(f"\n[INFO] Calibration RMS error: {ret:.4f}")
    print(f"[INFO] Camera Matrix:\n{camera_matrix}")
    print(f"[INFO] Distortion Coeffs: {dist_coeffs.ravel()}")

    return camera_matrix, dist_coeffs, img_size


def save_results(output_dir, camera_matrix, dist_coeffs, image_size):
    """Save calibration results in multiple formats."""
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = dist_coeffs.ravel()[:5]
    w, h = image_size

    # ─── JSON format ───
    calib = {
        "fx": float(fx), "fy": float(fy),
        "cx": float(cx), "cy": float(cy),
        "width": w, "height": h,
        "distortion_model": "plumb_bob",
        "distortion_coeffs": [float(k1), float(k2), float(p1), float(p2), float(k3)],
        "camera_matrix": camera_matrix.tolist(),
    }
    json_path = os.path.join(output_dir, "calibration.json")
    with open(json_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"[SAVED] {json_path}")

    # ─── ORB-SLAM2 YAML format ───
    yaml_path = os.path.join(output_dir, "calibration.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""%YAML:1.0

# Camera Parameters (from OpenCV calibration)
Camera.fx: {fx:.6f}
Camera.fy: {fy:.6f}
Camera.cx: {cx:.6f}
Camera.cy: {cy:.6f}

Camera.k1: {k1:.6f}
Camera.k2: {k2:.6f}
Camera.p1: {p1:.6f}
Camera.p2: {p2:.6f}
Camera.k3: {k3:.6f}

Camera.width: {w}
Camera.height: {h}

Camera.fps: 30.0
Camera.RGB: 1

# ORB Parameters
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

# Viewer Parameters
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
""")
    print(f"[SAVED] {yaml_path}")

    # ─── COLMAP camera.txt format ───
    # PINHOLE model: fx, fy, cx, cy
    colmap_path = os.path.join(output_dir, "camera_colmap.txt")
    with open(colmap_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    print(f"[SAVED] {colmap_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CW2 Camera Calibration Helper"
    )
    parser.add_argument(
        "--backend", default="realsense", choices=["realsense", "opencv"],
        help="Camera backend (default: realsense)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index for OpenCV backend"
    )
    parser.add_argument(
        "--rows", type=int, default=6,
        help="Checkerboard inner corner rows (default: 6)"
    )
    parser.add_argument(
        "--cols", type=int, default=9,
        help="Checkerboard inner corner columns (default: 9)"
    )
    parser.add_argument(
        "--square-size", type=float, default=25.0,
        help="Checkerboard square size in mm (default: 25.0)"
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Camera resolution width"
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Camera resolution height"
    )
    parser.add_argument(
        "--output-dir", default="data/calibration",
        help="Output directory (default: data/calibration)"
    )
    parser.add_argument(
        "--calibrate-only", action="store_true",
        help="Skip capture, run calibration on existing images"
    )

    args = parser.parse_args()

    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    board_size = (args.cols, args.rows)

    if not args.calibrate_only:
        # ─── Capture mode ───
        if args.backend == "realsense":
            if not REALSENSE_AVAILABLE:
                print("[ERROR] pyrealsense2 not installed.")
                sys.exit(1)
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(
                rs.stream.color, args.width, args.height, rs.format.bgr8, 30
            )
            pipeline.start(config)
            # Settle auto-exposure
            for _ in range(30):
                pipeline.wait_for_frames()

            def get_frame():
                frames = pipeline.wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    return None
                return np.asanyarray(color.get_data())

            def cleanup():
                pipeline.stop()
        else:
            cap = cv2.VideoCapture(args.camera)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

            def get_frame():
                ret, frame = cap.read()
                return frame if ret else None

            def cleanup():
                cap.release()

        count = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
        print("\n" + "=" * 50)
        print("  CAMERA CALIBRATION CAPTURE")
        print("=" * 50)
        print(f"  Board: {args.cols}x{args.rows} inner corners")
        print(f"  Square size: {args.square_size}mm")
        print(f"  Aim for 15-25 images from varied angles")
        print(f"  Images so far: {count}")
        print("  Controls: SPACE=capture, C=calibrate, Q=quit")
        print("=" * 50 + "\n")

        try:
            while True:
                frame = get_frame()
                if frame is None:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, board_size, None)

                display = frame.copy()

                if ret:
                    # Draw detected corners
                    cv2.drawChessboardCorners(display, board_size, corners, ret)
                    cv2.putText(display, "CHECKERBOARD DETECTED - Press SPACE",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                else:
                    cv2.putText(display, "No checkerboard found",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

                cv2.putText(display, f"Captured: {count} images",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

                cv2.imshow("Calibration Capture", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and ret:
                    fname = os.path.join(img_dir, f"calib_{count:03d}.png")
                    cv2.imwrite(fname, frame)
                    count += 1
                    print(f"[CAPTURED] Image {count}: {fname}")
                elif key == ord("c"):
                    break
                elif key == ord("q"):
                    print("Quitting without calibrating.")
                    cleanup()
                    cv2.destroyAllWindows()
                    return

        except KeyboardInterrupt:
            pass
        finally:
            cleanup()
            cv2.destroyAllWindows()

    # ─── Run calibration ───
    print("\n[INFO] Running calibration...")
    camera_matrix, dist_coeffs, image_size = run_calibration(
        img_dir, args.rows, args.cols, args.square_size
    )

    if camera_matrix is not None:
        save_results(args.output_dir, camera_matrix, dist_coeffs, image_size)
        print("\n[SUCCESS] Calibration complete!")
    else:
        print("\n[FAILED] Calibration failed. Capture more images.")


if __name__ == "__main__":
    main()
