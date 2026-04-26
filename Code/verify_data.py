#!/usr/bin/env python3
"""
CW2 Data Collection — Verification Script
===========================================
Verify that all collected data meets CW2 requirements.

Usage:
  python verify_data.py --data-dir data/

Checks:
  Camera: 2+ sequences (1 indoor, 1 outdoor), 500+ frames each
  LiDAR:  3+ sequences (2 indoor, 1 outdoor), loop markers present
  Calibration: calibration files exist with valid parameters
"""

import os
import sys
import json
import argparse


class bcolors:
    OK = "\033[92m"
    WARN = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def check_pass(msg):
    print(f"  {bcolors.OK}✓ PASS{bcolors.END}  {msg}")


def check_fail(msg):
    print(f"  {bcolors.FAIL}✗ FAIL{bcolors.END}  {msg}")


def check_warn(msg):
    print(f"  {bcolors.WARN}! WARN{bcolors.END}  {msg}")


def verify_camera_sequence(seq_dir):
    """Verify a single camera sequence."""
    issues = []
    name = os.path.basename(seq_dir)

    # Check video
    video_path = os.path.join(seq_dir, "video.mp4")
    if not os.path.exists(video_path):
        issues.append("video.mp4 missing")

    # Check frames
    frames_dir = os.path.join(seq_dir, "frames")
    if os.path.isdir(frames_dir):
        frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        if frame_count < 500:
            issues.append(f"Only {frame_count}/500 frames")
    else:
        issues.append("frames/ directory missing")
        frame_count = 0

    # Check timestamps
    ts_path = os.path.join(seq_dir, "timestamps.txt")
    if not os.path.exists(ts_path):
        issues.append("timestamps.txt missing")

    # Check rgb.txt (ORB-SLAM2 association)
    rgb_path = os.path.join(seq_dir, "rgb.txt")
    if not os.path.exists(rgb_path):
        issues.append("rgb.txt missing")

    # Check sequence info
    info_path = os.path.join(seq_dir, "sequence_info.json")
    info = None
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    else:
        issues.append("sequence_info.json missing")

    env = info.get("environment", "unknown") if info else "unknown"

    return {
        "name": name,
        "path": seq_dir,
        "environment": env,
        "frame_count": frame_count,
        "issues": issues,
        "ok": len(issues) == 0 and frame_count >= 500,
    }


def verify_lidar_sequence(seq_dir):
    """Verify a single LiDAR sequence."""
    issues = []
    name = os.path.basename(seq_dir)

    # Check scans
    scans_path = os.path.join(seq_dir, "scans.json")
    scan_count = 0
    if os.path.exists(scans_path):
        with open(scans_path) as f:
            scan_count = sum(1 for line in f if line.strip())
        if scan_count < 50:
            issues.append(f"Only {scan_count} scans (seems too few)")
    else:
        issues.append("scans.json missing")

    # Check timestamps
    ts_path = os.path.join(seq_dir, "timestamps.txt")
    if not os.path.exists(ts_path):
        issues.append("timestamps.txt missing")

    # Check loop markers
    markers_path = os.path.join(seq_dir, "loop_markers.json")
    loop_count = 0
    if os.path.exists(markers_path):
        with open(markers_path) as f:
            markers = json.load(f)
            loop_count = len(markers)
        if loop_count < 2:
            issues.append(f"Only {loop_count}/2 loops marked")
    else:
        issues.append("loop_markers.json missing")

    # Check sequence info
    info_path = os.path.join(seq_dir, "sequence_info.json")
    info = None
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    else:
        issues.append("sequence_info.json missing")

    env = info.get("environment", "unknown") if info else "unknown"

    return {
        "name": name,
        "path": seq_dir,
        "environment": env,
        "scan_count": scan_count,
        "loop_count": loop_count,
        "issues": issues,
        "ok": len(issues) == 0,
    }


def verify_calibration(calib_dir):
    """Verify calibration data."""
    issues = []

    images_dir = os.path.join(calib_dir, "images")
    if os.path.isdir(images_dir):
        img_count = len([f for f in os.listdir(images_dir) if f.endswith(".png")])
        if img_count < 10:
            issues.append(f"Only {img_count} calibration images (recommend 15-25)")
    else:
        issues.append("calibration images/ directory missing")

    for fname in ["calibration.json", "calibration.yaml", "camera_colmap.txt"]:
        path = os.path.join(calib_dir, fname)
        if not os.path.exists(path):
            issues.append(f"{fname} missing")

    return {"issues": issues, "ok": len(issues) == 0}


def main():
    parser = argparse.ArgumentParser(description="CW2 Data Verification")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()

    print(f"\n{bcolors.BOLD}{'=' * 60}")
    print("  CW2 DATA COLLECTION — VERIFICATION REPORT")
    print(f"{'=' * 60}{bcolors.END}\n")

    all_ok = True

    # ─── 1. Camera sequences ───
    print(f"{bcolors.BOLD}1. CAMERA SEQUENCES (Q2){bcolors.END}")
    print(f"   Required: 2+ sequences (1 indoor, 1 outdoor), 500+ frames each\n")

    camera_dir = os.path.join(args.data_dir, "camera")
    camera_results = []

    if os.path.isdir(camera_dir):
        for name in sorted(os.listdir(camera_dir)):
            seq_dir = os.path.join(camera_dir, name)
            if os.path.isdir(seq_dir) and not name.startswith("."):
                result = verify_camera_sequence(seq_dir)
                camera_results.append(result)
                if result["ok"]:
                    check_pass(f"{name}: {result['frame_count']} frames, {result['environment']}")
                else:
                    for issue in result["issues"]:
                        check_fail(f"{name}: {issue}")
                    all_ok = False
    else:
        check_fail("camera/ directory not found")
        all_ok = False

    # Check environment coverage
    envs = {r["environment"] for r in camera_results if r["ok"]}
    if "indoor" not in envs:
        check_fail("Missing indoor camera sequence")
        all_ok = False
    if "outdoor" not in envs:
        check_fail("Missing outdoor camera sequence")
        all_ok = False
    if len(camera_results) < 2:
        check_fail(f"Only {len(camera_results)}/2 sequences")
        all_ok = False

    # ─── 2. LiDAR sequences ───
    print(f"\n{bcolors.BOLD}2. LIDAR SEQUENCES (Q3){bcolors.END}")
    print(f"   Required: 3+ sequences (2 indoor incl. 1 large, 1 outdoor), 2 loops each\n")

    lidar_dir = os.path.join(args.data_dir, "lidar")
    lidar_results = []

    if os.path.isdir(lidar_dir):
        for name in sorted(os.listdir(lidar_dir)):
            seq_dir = os.path.join(lidar_dir, name)
            if os.path.isdir(seq_dir) and not name.startswith("."):
                result = verify_lidar_sequence(seq_dir)
                lidar_results.append(result)
                if result["ok"]:
                    check_pass(f"{name}: {result['scan_count']} scans, "
                               f"{result['loop_count']} loops, {result['environment']}")
                else:
                    for issue in result["issues"]:
                        check_fail(f"{name}: {issue}")
                    all_ok = False
    else:
        check_fail("lidar/ directory not found")
        all_ok = False

    # Check environment coverage
    lidar_envs = [r["environment"] for r in lidar_results if r["ok"]]
    indoor_count = lidar_envs.count("indoor")
    outdoor_count = lidar_envs.count("outdoor")

    if indoor_count < 2:
        check_fail(f"Only {indoor_count}/2 indoor LiDAR sequences")
        all_ok = False
    if outdoor_count < 1:
        check_fail("Missing outdoor LiDAR sequence")
        all_ok = False

    # ─── 3. Calibration ───
    print(f"\n{bcolors.BOLD}3. CAMERA CALIBRATION{bcolors.END}")
    print(f"   Required: Camera intrinsic parameters for ORB-SLAM2 and COLMAP\n")

    calib_dir = os.path.join(args.data_dir, "calibration")
    # Also check if intrinsics were saved by camera_recorder (RealSense)
    has_rs_intrinsics = False
    if os.path.isdir(camera_dir):
        for name in os.listdir(camera_dir):
            intrinsics_path = os.path.join(camera_dir, name, "camera_intrinsics.json")
            if os.path.exists(intrinsics_path):
                check_pass(f"RealSense intrinsics found in {name}/")
                has_rs_intrinsics = True
                break

    if os.path.isdir(calib_dir):
        result = verify_calibration(calib_dir)
        if result["ok"]:
            check_pass("OpenCV calibration files present")
        else:
            for issue in result["issues"]:
                if has_rs_intrinsics:
                    check_warn(f"(optional with RealSense) {issue}")
                else:
                    check_fail(issue)
                    all_ok = False
    elif not has_rs_intrinsics:
        check_fail("No calibration data found")
        all_ok = False
    else:
        check_warn("No OpenCV calibration (RealSense intrinsics available instead)")

    # ─── Summary ───
    print(f"\n{bcolors.BOLD}{'=' * 60}")
    if all_ok:
        print(f"  {bcolors.OK}ALL CHECKS PASSED ✓{bcolors.END}")
    else:
        print(f"  {bcolors.FAIL}SOME CHECKS FAILED — see above{bcolors.END}")
    print(f"{'=' * 60}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
