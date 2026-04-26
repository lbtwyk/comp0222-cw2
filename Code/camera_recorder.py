#!/usr/bin/env python3
"""
CW2 Data Collection — Camera Recorder
======================================
Record monocular RGB video sequences using an Intel RealSense camera
(with OpenCV webcam fallback).

Outputs:
  - video.mp4          : H.264 video file (for submission & ORB-SLAM2)
  - frames/XXXXXX.png  : Individual frames (for COLMAP)
  - timestamps.txt     : Timestamp per frame in TUM format
  - rgb.txt            : Association file for ORB-SLAM2 TUM format
  - sequence_info.json : Metadata about the recording

Usage:
  # RealSense camera (default)
  python camera_recorder.py --name indoor_01 --env indoor

  # Fallback to webcam (e.g., if RealSense unavailable)
  python camera_recorder.py --name outdoor_01 --env outdoor --backend opencv --camera 0

  # Custom resolution
  python camera_recorder.py --name indoor_01 --env indoor --width 1280 --height 720

Controls:
  q     : Stop recording
  Ctrl+C: Stop recording (safe shutdown)
"""

import os
import sys
import time
import json
import argparse
import numpy as np

# ─── Try to import RealSense, fall back to OpenCV ───
REALSENSE_AVAILABLE = False
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    pass

import cv2


def create_output_dir(base_dir, name):
    """Create output directory structure."""
    out_dir = os.path.join(base_dir, "camera", name)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    return out_dir, frames_dir


class RealSenseCapture:
    """Intel RealSense camera capture (monocular RGB only)."""

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable only the color stream (monocular)
        self.config.enable_stream(
            rs.stream.color, width, height, rs.format.bgr8, fps
        )

        self.profile = None

    def start(self):
        self.profile = self.pipeline.start(self.config)
        # Allow auto-exposure to settle
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("[RealSense] Camera started. Auto-exposure settled.")

    def read(self):
        """Returns (success, frame) like OpenCV VideoCapture."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return False, None
            frame = np.asanyarray(color_frame.get_data())
            return True, frame
        except Exception as e:
            print(f"[RealSense] Frame error: {e}")
            return False, None

    def get_intrinsics(self):
        """Get camera intrinsic parameters from RealSense."""
        if self.profile is None:
            return None
        stream = self.profile.get_stream(rs.stream.color)
        intrinsics = stream.as_video_stream_profile().get_intrinsics()
        return {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
            "width": intrinsics.width,
            "height": intrinsics.height,
            "distortion_model": str(intrinsics.model),
            "distortion_coeffs": list(intrinsics.coeffs),
        }

    def release(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class OpenCVCapture:
    """OpenCV webcam fallback capture."""

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        print(f"[OpenCV] Camera opened: {self.width}x{self.height}")

    def read(self):
        return self.cap.read()

    def get_intrinsics(self):
        return None  # Must calibrate separately

    def release(self):
        self.cap.release()


def record_sequence(capture, out_dir, frames_dir, env_type, min_frames=500):
    """
    Main recording loop.
    Saves video, individual frames, timestamps, and metadata.
    """
    # ─── Setup video writer ───
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(out_dir, "video.mp4")
    video_writer = None

    # ─── Setup timestamp / association files ───
    ts_path = os.path.join(out_dir, "timestamps.txt")
    rgb_path = os.path.join(out_dir, "rgb.txt")
    ts_file = open(ts_path, "w")
    rgb_file = open(rgb_path, "w")

    # Write TUM-format headers
    rgb_file.write("# timestamp filename\n")

    frame_count = 0
    start_time = time.time()
    timestamps = []

    print("\n" + "=" * 50)
    print("  RECORDING — Press 'q' to stop")
    print(f"  Environment: {env_type}")
    print(f"  Minimum frames: {min_frames}")
    print("=" * 50 + "\n")

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("[WARN] Failed to read frame, retrying...")
                continue

            timestamp = time.time()
            elapsed = timestamp - start_time

            # Initialize video writer on first frame (get actual resolution)
            if video_writer is None:
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    video_path, fourcc, 30.0, (w, h)
                )
                print(f"[INFO] Recording at {w}x{h}")

            # ─── Save frame ───
            frame_name = f"{frame_count:06d}.png"
            frame_path = os.path.join(frames_dir, frame_name)
            cv2.imwrite(frame_path, frame)

            # ─── Write to video ───
            video_writer.write(frame)

            # ─── Write timestamps ───
            ts_str = f"{timestamp:.6f}"
            ts_file.write(f"{ts_str}\n")
            rgb_file.write(f"{ts_str} frames/{frame_name}\n")
            timestamps.append(timestamp)

            frame_count += 1

            # ─── Display preview ───
            display = frame.copy()
            # Status bar
            color = (0, 255, 0) if frame_count >= min_frames else (0, 165, 255)
            status = f"Frames: {frame_count}/{min_frames}  |  Time: {elapsed:.1f}s"
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if frame_count >= min_frames:
                cv2.putText(display, "MINIMUM REACHED - Press 'q' to stop",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
            else:
                remaining = min_frames - frame_count
                cv2.putText(display, f"{remaining} more frames needed",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 165, 255), 2)

            # Recording indicator (blinking red dot)
            if int(elapsed * 2) % 2 == 0:
                cv2.circle(display, (w - 30, 30), 10, (0, 0, 255), -1)

            cv2.imshow("CW2 Camera Recorder", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                if frame_count < min_frames:
                    print(f"\n[WARN] Only {frame_count}/{min_frames} frames "
                          f"recorded. Continue? (press 'q' again to force stop, "
                          f"any other key to continue)")
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 != ord("q"):
                        continue
                break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C — stopping recording...")

    finally:
        # ─── Cleanup ───
        ts_file.close()
        rgb_file.close()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    end_time = time.time()
    duration = end_time - start_time

    # ─── Save camera intrinsics if available ───
    intrinsics = capture.get_intrinsics()
    if intrinsics is not None:
        intrinsics_path = os.path.join(out_dir, "camera_intrinsics.json")
        with open(intrinsics_path, "w") as f:
            json.dump(intrinsics, f, indent=2)
        print(f"[INFO] Camera intrinsics saved to {intrinsics_path}")

        # Also save in ORB-SLAM2 YAML format
        orbslam_cfg_path = os.path.join(out_dir, "orbslam2_camera.yaml")
        save_orbslam2_config(intrinsics, orbslam_cfg_path)
        print(f"[INFO] ORB-SLAM2 config saved to {orbslam_cfg_path}")

    # ─── Save sequence metadata ───
    info = {
        "name": os.path.basename(out_dir),
        "environment": env_type,
        "sensor": "Intel RealSense" if intrinsics else "OpenCV Camera",
        "resolution": {"width": w, "height": h},
        "frame_count": frame_count,
        "duration_seconds": round(duration, 2),
        "avg_fps": round(frame_count / duration, 2) if duration > 0 else 0,
        "min_frames_requirement": min_frames,
        "meets_requirement": frame_count >= min_frames,
        "timestamp_start": timestamps[0] if timestamps else None,
        "timestamp_end": timestamps[-1] if timestamps else None,
    }

    info_path = os.path.join(out_dir, "sequence_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # ─── Summary ───
    print("\n" + "=" * 50)
    print("  RECORDING COMPLETE")
    print("=" * 50)
    print(f"  Frames:    {frame_count}")
    print(f"  Duration:  {duration:.1f}s")
    print(f"  Avg FPS:   {info['avg_fps']}")
    print(f"  Output:    {out_dir}")
    print(f"  Video:     {video_path}")
    print(f"  Frames:    {frames_dir}/ ({frame_count} files)")
    print(f"  Timestamps:{ts_path}")
    status = "✓ PASS" if frame_count >= min_frames else "✗ FAIL"
    print(f"  Min frames:{status} ({frame_count}/{min_frames})")
    print("=" * 50)

    return info


def save_orbslam2_config(intrinsics, path):
    """Save camera parameters in ORB-SLAM2 YAML format."""
    content = f"""%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters (auto-generated from RealSense intrinsics)
#--------------------------------------------------------------------------------------------

Camera.fx: {intrinsics['fx']:.6f}
Camera.fy: {intrinsics['fy']:.6f}
Camera.cx: {intrinsics['cx']:.6f}
Camera.cy: {intrinsics['cy']:.6f}

Camera.k1: {intrinsics['distortion_coeffs'][0]:.6f}
Camera.k2: {intrinsics['distortion_coeffs'][1]:.6f}
Camera.p1: {intrinsics['distortion_coeffs'][2]:.6f}
Camera.p2: {intrinsics['distortion_coeffs'][3]:.6f}
Camera.k3: {intrinsics['distortion_coeffs'][4] if len(intrinsics['distortion_coeffs']) > 4 else 0.0:.6f}

Camera.width: {intrinsics['width']}
Camera.height: {intrinsics['height']}

# Camera frames per second
Camera.fps: 30.0

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1000

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
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
"""
    with open(path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="CW2 Camera Recorder — Record monocular sequences for Visual SLAM"
    )
    parser.add_argument(
        "--name", required=True,
        help="Sequence name (e.g., 'indoor_01', 'outdoor_01')"
    )
    parser.add_argument(
        "--env", required=True, choices=["indoor", "outdoor"],
        help="Environment type"
    )
    parser.add_argument(
        "--backend", default="realsense", choices=["realsense", "opencv"],
        help="Camera backend (default: realsense)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index for OpenCV backend (default: 0)"
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Frame width (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Frame height (default: 480)"
    )
    parser.add_argument(
        "--min-frames", type=int, default=500,
        help="Minimum frame count requirement (default: 500)"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Base data directory (default: data/)"
    )

    args = parser.parse_args()

    # ─── Select camera backend ───
    if args.backend == "realsense":
        if not REALSENSE_AVAILABLE:
            print("[ERROR] pyrealsense2 not installed. Use --backend opencv")
            print("        or install with: pip install pyrealsense2")
            sys.exit(1)
        capture = RealSenseCapture(args.width, args.height, fps=30)
    else:
        capture = OpenCVCapture(args.camera, args.width, args.height, fps=30)

    # ─── Create output directory ───
    out_dir, frames_dir = create_output_dir(args.data_dir, args.name)

    # Check if directory already has data
    if os.path.exists(os.path.join(out_dir, "video.mp4")):
        print(f"[WARN] Data already exists in {out_dir}")
        resp = input("Overwrite? (y/n): ").strip().lower()
        if resp != "y":
            print("Aborted.")
            sys.exit(0)

    # ─── Start recording ───
    try:
        capture.start()
        record_sequence(capture, out_dir, frames_dir, args.env, args.min_frames)
    finally:
        capture.release()


if __name__ == "__main__":
    main()
