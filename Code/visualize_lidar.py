#!/usr/bin/env python3
"""
CW2 Data Validation — LiDAR Visualizer
========================================
Replay and visualize recorded LiDAR data to validate quality.

Shows:
  - Live polar scan replay
  - Accumulated 2D point cloud map (via simple ICP odometry)
  - Robot trajectory
  - Loop marker positions

Usage:
  python visualize_lidar.py data/lidar/indoor_large_01/scans.json
  python visualize_lidar.py data/lidar/indoor_small_01/scans.json --speed 5
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ─── Constants ───
MAX_RANGE_MM = 6000.0
BLIND_SPOT_MIN = 135.0
BLIND_SPOT_MAX = 225.0


def process_scan(scan_data, max_range=MAX_RANGE_MM):
    """Convert raw scan [(q, angle_deg, dist_mm), ...] to XY in meters."""
    raw = np.array(scan_data)
    if len(raw) == 0:
        return None

    distances = raw[:, 2]
    angles = raw[:, 1]

    # Distance filter
    dist_mask = (distances > 10) & (distances < max_range)
    # Blind spot filter
    angle_mask = (angles < BLIND_SPOT_MIN) | (angles > BLIND_SPOT_MAX)
    mask = dist_mask & angle_mask

    if np.sum(mask) < 10:
        return None

    angles_rad = np.radians(raw[mask, 1])
    dists_m = raw[mask, 2] / 1000.0

    x = dists_m * np.cos(angles_rad)
    y = dists_m * np.sin(angles_rad)

    return np.column_stack((x, y))

def estimate_normals_pca(points, k=5):
    """Estimate surface normals via PCA on local neighborhoods (from Lab 08)."""
    if len(points) < k + 1:
        return np.zeros((len(points), 2))

    neigh = NearestNeighbors(n_neighbors=k + 1)
    neigh.fit(points)
    _, indices_all = neigh.kneighbors(points)

    normals = np.zeros((points.shape[0], 2))
    for i in range(points.shape[0]):
        neighbor_points = points[indices_all[i]]
        centered = neighbor_points - np.mean(neighbor_points, axis=0)
        cov = centered.T @ centered / k
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        normal = eig_vecs[:, 0]  # smallest eigenvalue = normal direction
        if np.dot(normal, points[i]) < 0:
            normal = -normal
        normals[i] = normal
    return normals


def solve_point_to_plane(src, dst, dst_normals):
    """Solve point-to-plane ICP step via linearized least-squares (from Lab 08)."""
    A = []
    b = []
    for i in range(len(src)):
        s = src[i]
        d = dst[i]
        n = dst_normals[i]
        # Cross term: s_x * n_y - s_y * n_x
        cross_term = s[0] * n[1] - s[1] * n[0]
        A.append([cross_term, n[0], n[1]])
        b.append(np.dot(d - s, n))

    if not A:
        return np.identity(3)

    A = np.array(A)
    b = np.array(b)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    c, s = np.cos(x[0]), np.sin(x[0])
    R = np.array([[c, -s], [s, c]])
    T = np.identity(3)
    T[:2, :2] = R
    T[:2, 2] = [x[1], x[2]]
    return T


ICP_MAX_ITER = 20
CORRESPONDENCE_THRESH = 0.5
KEYFRAME_DIST_THRESH = 0.2
KEYFRAME_ANGLE_THRESH = 0.2
LOCAL_MAP_SIZE = 20


def icp_scan_to_map(src_points, map_points, map_normals, init_pose_guess):
    """Point-to-plane ICP: align scan to local map (from Lab 08)."""
    m = src_points.shape[1]
    src_h = np.ones((m + 1, src_points.shape[0]))
    src_h[:m, :] = np.copy(src_points.T)
    current_global_pose = np.copy(init_pose_guess)

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(map_points)

    for _ in range(ICP_MAX_ITER):
        src_global_h = current_global_pose @ src_h
        src_global = src_global_h[:2, :].T

        distances, indices = neigh.kneighbors(src_global, return_distance=True)
        distances = distances.ravel()
        indices = indices.ravel()

        mask = distances < CORRESPONDENCE_THRESH
        if np.sum(mask) < 10:
            break

        src_valid = src_global[mask]
        dst_valid = map_points[indices[mask]]
        normals_valid = map_normals[indices[mask]]

        T_delta = solve_point_to_plane(src_valid, dst_valid, normals_valid)
        current_global_pose = T_delta @ current_global_pose

        if (np.linalg.norm(T_delta[:2, 2]) < 0.001 and
                abs(np.arctan2(T_delta[1, 0], T_delta[0, 0])) < 0.001):
            break

    return current_global_pose


def load_scans(scan_file):
    """Load scans from JSON-lines file."""
    scans = []
    with open(scan_file, 'r') as f:
        for line in f:
            if line.strip():
                scans.append(json.loads(line))
    return scans


def load_loop_markers(data_dir):
    """Load loop markers if available."""
    markers_file = os.path.join(data_dir, "loop_markers.json")
    if os.path.exists(markers_file):
        with open(markers_file) as f:
            return json.load(f)
    return []


def visualize(scan_file, speed=1, max_range=MAX_RANGE_MM):
    """Main visualization loop."""
    data_dir = os.path.dirname(scan_file)
    seq_name = os.path.basename(data_dir)

    print(f"\n[INFO] Loading: {scan_file}")
    scans = load_scans(scan_file)
    markers = load_loop_markers(data_dir)
    marker_indices = {m["scan_index"] for m in markers}

    print(f"[INFO] Loaded {len(scans)} scans")
    print(f"[INFO] Loop markers at scan indices: {sorted(marker_indices)}")
    print(f"[INFO] Using Point-to-Plane ICP with keyframe buffer (Lab 08 method)")

    # ─── Setup figure ───
    fig = plt.figure(figsize=(16, 8))
    fig.canvas.manager.set_window_title(f"LiDAR Validation — {seq_name}")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5])

    # Left: current scan (polar)
    ax_scan = fig.add_subplot(gs[0], projection='polar')
    ax_scan.set_rmax(max_range)
    ax_scan.set_title("Current Scan", pad=15)

    # Right: accumulated map (Cartesian)
    ax_map = fig.add_subplot(gs[1])
    ax_map.set_aspect('equal')
    ax_map.set_title("Accumulated Map & Trajectory")
    ax_map.set_xlabel("X (meters)")
    ax_map.set_ylabel("Y (meters)")
    ax_map.grid(True, alpha=0.3)

    # Status text
    status_text = fig.text(0.02, 0.02, "", fontsize=9, fontfamily="monospace",
                           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.ion()
    plt.tight_layout()

    # ─── Point-to-Plane ICP with Keyframe Buffer ───
    current_pose = np.identity(3)
    last_keyframe_pose = np.identity(3)
    keyframe_buffer = []       # List of (points, normals) in global frame
    trajectory = [[0.0, 0.0]]
    all_map_points = []
    scan_scatter = None
    first_scan_done = False

    is_running = True

    def on_key(event):
        nonlocal is_running
        if event.key in ("q", "escape"):
            is_running = False

    fig.canvas.mpl_connect("key_press_event", on_key)

    print(f"\n[INFO] Replaying at {speed}x speed. Press Q to stop.\n")

    for i, scan in enumerate(scans):
        if not is_running or not plt.fignum_exists(fig.number):
            break

        scan_xy = process_scan(scan, max_range)
        if scan_xy is None:
            continue

        # ─── Point-to-Plane ICP Odometry ───
        if not first_scan_done:
            normals = estimate_normals_pca(scan_xy)
            keyframe_buffer.append((scan_xy, normals))
            all_map_points.append(scan_xy)
            first_scan_done = True
        elif HAS_SKLEARN:
            # Build local map from keyframe buffer
            active_points = np.vstack([k[0] for k in keyframe_buffer])
            active_normals = np.vstack([k[1] for k in keyframe_buffer])

            # Run point-to-plane ICP against local map
            new_pose = icp_scan_to_map(scan_xy, active_points, active_normals, current_pose)
            current_pose = new_pose

            # Check if we should add a new keyframe
            delta_T = np.linalg.inv(last_keyframe_pose) @ current_pose
            dx, dy = delta_T[0, 2], delta_T[1, 2]
            dtheta = np.arctan2(delta_T[1, 0], delta_T[0, 0])
            dist_moved = np.sqrt(dx**2 + dy**2)

            if dist_moved > KEYFRAME_DIST_THRESH or abs(dtheta) > KEYFRAME_ANGLE_THRESH:
                # Transform current scan to global frame
                curr_h = np.ones((3, scan_xy.shape[0]))
                curr_h[:2, :] = scan_xy.T
                curr_global = (current_pose @ curr_h)[:2, :].T

                curr_normals = estimate_normals_pca(curr_global)
                keyframe_buffer.append((curr_global, curr_normals))
                all_map_points.append(curr_global)

                last_keyframe_pose = np.copy(current_pose)
                if len(keyframe_buffer) > LOCAL_MAP_SIZE:
                    keyframe_buffer.pop(0)

        cx, cy = current_pose[0, 2], current_pose[1, 2]
        trajectory.append([cx, cy])

        # ─── Update plots (every frame or based on speed) ───
        if i % speed == 0:
            # Left: polar scan
            if scan_scatter is not None:
                scan_scatter.remove()
            raw = np.array(scan)
            valid = raw[raw[:, 2] > 0]
            if len(valid) > 0:
                angles_rad = np.radians(valid[:, 1])
                dists = valid[:, 2]
                scan_scatter = ax_scan.scatter(angles_rad, dists, s=2, c="blue", alpha=0.6)

            # Right: map
            ax_map.clear()
            ax_map.set_aspect('equal')
            ax_map.grid(True, alpha=0.3)

            # Draw map points
            if all_map_points:
                all_pts = np.vstack(all_map_points)
                # Subsample for rendering performance
                step = max(1, len(all_pts) // 5000)
                ax_map.scatter(all_pts[::step, 0], all_pts[::step, 1],
                               s=0.5, c="gray", alpha=0.4, label="Map")

            # Draw trajectory
            traj = np.array(trajectory)
            ax_map.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.7, label="Trajectory")

            # Draw current position
            ax_map.plot(cx, cy, 'ro', markersize=6)

            # Draw loop markers on trajectory
            for m in markers:
                idx = m["scan_index"]
                if idx < len(trajectory):
                    mx, my = trajectory[idx]
                    ax_map.plot(mx, my, 'g*', markersize=15)
                    ax_map.annotate(f"Loop {m['loop_number']}", (mx, my),
                                    fontsize=8, fontweight='bold', color='green',
                                    xytext=(5, 5), textcoords='offset points')

            # Start position
            ax_map.plot(0, 0, 'ks', markersize=8, label="Start")

            ax_map.legend(loc='upper right', fontsize=8)
            ax_map.set_title(f"Accumulated Map & Trajectory — {seq_name}")

            # Loop marker indicator
            loop_info = ""
            if i in marker_indices:
                loop_info = " ★ LOOP MARKER ★"

            status_text.set_text(
                f"Scan: {i+1}/{len(scans)}  |  "
                f"Points: {len(scan_xy):4d}  |  "
                f"Pos: ({cx:.2f}, {cy:.2f})"
                f"{loop_info}"
            )

            plt.draw()
            plt.pause(0.001)

    # ─── Final summary ───
    print("\n" + "=" * 50)
    print("  VALIDATION SUMMARY")
    print("=" * 50)
    print(f"  Sequence:    {seq_name}")
    print(f"  Total scans: {len(scans)}")
    print(f"  Loop markers:{len(markers)}")

    if len(trajectory) > 1:
        traj = np.array(trajectory)
        total_dist = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        print(f"  Total distance: {total_dist:.2f} m")

        # Loop closure error (distance from end to start)
        closure_error = np.linalg.norm(traj[-1] - traj[0])
        print(f"  Closure error:  {closure_error:.3f} m (end-to-start distance)")
        if closure_error > 1.0:
            print(f"  [WARN] Large closure error — expect drift correction needed in Q3d")
        else:
            print(f"  [OK] Reasonable closure error")

    print("=" * 50)

    # Keep plot open
    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="CW2 LiDAR Data Visualizer")
    parser.add_argument("scan_file", help="Path to scans.json file")
    parser.add_argument("--speed", type=int, default=1,
                        help="Playback speed multiplier (default: 1)")
    parser.add_argument("--max-range", type=float, default=MAX_RANGE_MM,
                        help="Max display range in mm (default: 6000)")
    args = parser.parse_args()

    if not os.path.exists(args.scan_file):
        print(f"[ERROR] File not found: {args.scan_file}")
        sys.exit(1)

    visualize(args.scan_file, args.speed, args.max_range)


if __name__ == "__main__":
    main()
