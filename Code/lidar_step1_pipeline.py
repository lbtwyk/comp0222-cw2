#!/usr/bin/env python3
"""
CW2 Q3 Step 1 LiDAR baseline pipeline.

This module runs replay-only LiDAR mapping on the selected coursework
sequences. It produces:
  - scan-to-map ICP odometry
  - a merged point-cloud map
  - an occupancy grid generated in a second pass
  - report-friendly plots and summary metrics
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


BLIND_SPOT_MIN_DEG = 135.0
BLIND_SPOT_MAX_DEG = 225.0
MIN_VALID_RANGE_MM = 10.0

ICP_MAX_ITER = 20
CORRESPONDENCE_THRESH_M = 0.5
KEYFRAME_DIST_THRESH_M = 0.2
KEYFRAME_ANGLE_THRESH_RAD = 0.2
LOCAL_MAP_SIZE = 20

OCCUPANCY_CELL_SIZE_M = 0.05
OCCUPANCY_PADDING_M = 2.0
LOG_ODDS_OCCUPIED = 0.85
LOG_ODDS_FREE = -0.4
LOG_ODDS_MIN = -2.0
LOG_ODDS_MAX = 3.5

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "lidar"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output" / "lidar_step1"

SEQUENCE_REGISTRY = {
    "indoor_large_03": {
        "label": "Marshgate large-classroom",
        "path": DATA_DIR / "indoor_large_03",
    },
    "indoor_small_01": {
        "label": "Marshgate small-room",
        "path": DATA_DIR / "indoor_small_01",
    },
    "outdoor_02": {
        "label": "Building exterior",
        "path": DATA_DIR / "outdoor_02",
    },
}


@dataclass(frozen=True)
class PipelineConfig:
    max_range_mm: float = 6000.0
    beam_step: int = 1
    voxel_size_m: float = 0.05
    scan_stride: int = 1
    occupancy_cell_size_m: float = OCCUPANCY_CELL_SIZE_M
    occupancy_padding_m: float = OCCUPANCY_PADDING_M
    icp_max_iter: int = ICP_MAX_ITER
    correspondence_thresh_m: float = CORRESPONDENCE_THRESH_M
    keyframe_dist_thresh_m: float = KEYFRAME_DIST_THRESH_M
    keyframe_angle_thresh_rad: float = KEYFRAME_ANGLE_THRESH_RAD
    local_map_size: int = LOCAL_MAP_SIZE


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scans(path: Path) -> list[list[list[float]]]:
    scans = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                scans.append(json.loads(line))
    return scans


def load_timestamps(path: Path) -> list[float]:
    timestamps: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            _, ts = stripped.split()
            timestamps.append(float(ts))
    return timestamps


def preprocess_scan(
    scan_data: list[list[float]],
    *,
    max_range_mm: float,
    beam_step: int,
) -> np.ndarray | None:
    raw = np.asarray(scan_data, dtype=float)
    if raw.size == 0:
        return None

    distances = raw[:, 2]
    angles = raw[:, 1]

    dist_mask = (distances > MIN_VALID_RANGE_MM) & (distances < max_range_mm)
    angle_mask = (angles < BLIND_SPOT_MIN_DEG) | (angles > BLIND_SPOT_MAX_DEG)
    filtered = raw[dist_mask & angle_mask]

    if filtered.shape[0] < 10:
        return None

    filtered = filtered[::beam_step]
    if filtered.shape[0] < 10:
        return None

    angles_rad = np.radians(filtered[:, 1])
    distances_m = filtered[:, 2] / 1000.0

    x = distances_m * np.cos(angles_rad)
    y = distances_m * np.sin(angles_rad)
    return np.column_stack((x, y))


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    homogeneous = np.ones((3, points.shape[0]), dtype=float)
    homogeneous[:2, :] = points.T
    transformed = (pose @ homogeneous)[:2, :].T
    return transformed


def pose_to_xytheta(pose: np.ndarray) -> tuple[float, float, float]:
    return (
        float(pose[0, 2]),
        float(pose[1, 2]),
        float(math.atan2(pose[1, 0], pose[0, 0])),
    )


def xytheta_to_pose(x: float, y: float, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    pose = np.identity(3)
    pose[:2, :2] = np.array([[c, -s], [s, c]], dtype=float)
    pose[:2, 2] = [x, y]
    return pose


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if points.size == 0 or voxel_size_m <= 0:
        return points

    buckets = np.floor(points / voxel_size_m).astype(np.int64)
    _, unique_idx = np.unique(buckets, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def estimate_normals_pca(points: np.ndarray, k: int = 5) -> np.ndarray:
    if len(points) < k + 1:
        return np.zeros((len(points), 2), dtype=float)

    neigh = NearestNeighbors(n_neighbors=k + 1)
    neigh.fit(points)
    _, indices_all = neigh.kneighbors(points)

    normals = np.zeros((points.shape[0], 2), dtype=float)
    for i in range(points.shape[0]):
        neighbor_points = points[indices_all[i]]
        centered = neighbor_points - np.mean(neighbor_points, axis=0)
        cov = centered.T @ centered / k
        _, eig_vecs = np.linalg.eigh(cov)
        normal = eig_vecs[:, 0]
        if np.dot(normal, points[i]) < 0:
            normal = -normal
        normals[i] = normal
    return normals


def solve_point_to_plane(
    src: np.ndarray,
    dst: np.ndarray,
    dst_normals: np.ndarray,
) -> np.ndarray:
    a_rows = []
    b_rows = []
    for i in range(len(src)):
        s = src[i]
        d = dst[i]
        n = dst_normals[i]
        cross_term = s[0] * n[1] - s[1] * n[0]
        a_rows.append([cross_term, n[0], n[1]])
        b_rows.append(np.dot(d - s, n))

    if not a_rows:
        return np.identity(3)

    a = np.asarray(a_rows, dtype=float)
    b = np.asarray(b_rows, dtype=float)
    x, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

    c = math.cos(x[0])
    s = math.sin(x[0])
    transform = np.identity(3)
    transform[:2, :2] = np.array([[c, -s], [s, c]])
    transform[:2, 2] = [x[1], x[2]]
    return transform


def icp_scan_to_map(
    src_points: np.ndarray,
    map_points: np.ndarray,
    map_normals: np.ndarray,
    init_pose_guess: np.ndarray,
    *,
    max_iter: int,
    correspondence_thresh_m: float,
) -> np.ndarray:
    dim = src_points.shape[1]
    src_h = np.ones((dim + 1, src_points.shape[0]), dtype=float)
    src_h[:dim, :] = src_points.T
    current_global_pose = np.copy(init_pose_guess)

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(map_points)

    for _ in range(max_iter):
        src_global_h = current_global_pose @ src_h
        src_global = src_global_h[:2, :].T

        distances, indices = neigh.kneighbors(src_global, return_distance=True)
        distances = distances.ravel()
        indices = indices.ravel()

        mask = distances < correspondence_thresh_m
        if np.sum(mask) < 10:
            break

        src_valid = src_global[mask]
        dst_valid = map_points[indices[mask]]
        normals_valid = map_normals[indices[mask]]

        delta = solve_point_to_plane(src_valid, dst_valid, normals_valid)
        current_global_pose = delta @ current_global_pose

        dxy = np.linalg.norm(delta[:2, 2])
        dtheta = abs(math.atan2(delta[1, 0], delta[0, 0]))
        if dxy < 1e-3 and dtheta < 1e-3:
            break

    return current_global_pose


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> Iterable[tuple[int, int]]:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        err2 = 2 * err
        if err2 > -dy:
            err -= dy
            x += sx
        if err2 < dx:
            err += dx
            y += sy


def world_to_grid(points: np.ndarray, origin_xy: np.ndarray, cell_size_m: float) -> np.ndarray:
    return np.floor((points - origin_xy) / cell_size_m).astype(int)


def compute_total_distance(trajectory_xy: np.ndarray) -> float:
    if trajectory_xy.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(trajectory_xy, axis=0), axis=1)))


def build_output_tag(config: PipelineConfig) -> str:
    voxel_str = f"{config.voxel_size_m:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return (
        f"mr{int(config.max_range_mm)}"
        f"_bs{config.beam_step}"
        f"_vs{voxel_str}"
        f"_ss{config.scan_stride}"
    )


def load_sequence_assets(sequence_name: str, *, include_markers: bool = True) -> dict:
    if sequence_name not in SEQUENCE_REGISTRY:
        raise KeyError(f"Unknown sequence: {sequence_name}")

    sequence_meta = SEQUENCE_REGISTRY[sequence_name]
    sequence_dir = sequence_meta["path"]

    scans = load_scans(sequence_dir / "scans.json")
    timestamps = load_timestamps(sequence_dir / "timestamps.txt")
    info = load_json(sequence_dir / "sequence_info.json")
    if include_markers is False:
        info = dict(info)
        info.pop("loop_markers", None)
        info.pop("loops_marked", None)

    if timestamps and len(timestamps) != len(scans):
        raise RuntimeError(
            f"Timestamp count mismatch for {sequence_name}: {len(timestamps)} timestamps vs {len(scans)} scans."
        )

    bundle = {
        "sequence": sequence_name,
        "label": sequence_meta["label"],
        "sequence_dir": sequence_dir,
        "scans": scans,
        "timestamps": timestamps,
        "source_metadata": info,
    }

    if include_markers:
        bundle["markers"] = load_json(sequence_dir / "loop_markers.json")

    return bundle


def process_replay_sequence(scans: list[list[list[float]]], timestamps: list[float], config: PipelineConfig) -> dict:
    processed_entries: list[dict] = []
    keyframes: list[dict] = []
    keyframe_buffer: list[tuple[np.ndarray, np.ndarray]] = []
    point_cloud_chunks: list[np.ndarray] = []

    current_pose = np.identity(3)
    last_keyframe_pose = np.identity(3)
    first_scan_initialized = False
    cumulative_path_length_m = 0.0

    started = time.perf_counter()

    for scan_index in range(0, len(scans), config.scan_stride):
        scan_xy = preprocess_scan(
            scans[scan_index],
            max_range_mm=config.max_range_mm,
            beam_step=config.beam_step,
        )
        if scan_xy is None:
            continue

        timestamp = timestamps[scan_index] if timestamps else float(scan_index)
        previous_pose = np.copy(current_pose)
        keyframe_added = False

        if not first_scan_initialized:
            world_points = voxel_downsample(scan_xy, config.voxel_size_m)
            world_normals = estimate_normals_pca(world_points)
            keyframe_buffer.append((world_points, world_normals))
            point_cloud_chunks.append(world_points)
            first_scan_initialized = True
            keyframe_added = True
        else:
            map_points = np.vstack([entry[0] for entry in keyframe_buffer])
            map_normals = np.vstack([entry[1] for entry in keyframe_buffer])
            current_pose = icp_scan_to_map(
                scan_xy,
                map_points,
                map_normals,
                current_pose,
                max_iter=config.icp_max_iter,
                correspondence_thresh_m=config.correspondence_thresh_m,
            )

            cumulative_path_length_m += float(
                np.linalg.norm(current_pose[:2, 2] - previous_pose[:2, 2])
            )

            delta = np.linalg.inv(last_keyframe_pose) @ current_pose
            dx = float(delta[0, 2])
            dy = float(delta[1, 2])
            dtheta = float(math.atan2(delta[1, 0], delta[0, 0]))
            dist_moved = math.hypot(dx, dy)

            if dist_moved > config.keyframe_dist_thresh_m or abs(dtheta) > config.keyframe_angle_thresh_rad:
                world_points = transform_points(scan_xy, current_pose)
                world_points = voxel_downsample(world_points, config.voxel_size_m)
                world_normals = estimate_normals_pca(world_points)
                keyframe_buffer.append((world_points, world_normals))
                point_cloud_chunks.append(world_points)
                last_keyframe_pose = np.copy(current_pose)
                if len(keyframe_buffer) > config.local_map_size:
                    keyframe_buffer.pop(0)
                keyframe_added = True

        processed_entry = {
            "processed_index": len(processed_entries),
            "scan_index": scan_index,
            "timestamp": timestamp,
            "pose": np.copy(current_pose),
            "scan_xy": np.copy(scan_xy),
            "cumulative_path_length_m": cumulative_path_length_m,
        }
        processed_entries.append(processed_entry)

        if keyframe_added:
            keyframes.append(
                {
                    "keyframe_id": len(keyframes),
                    "processed_index": processed_entry["processed_index"],
                    "scan_index": scan_index,
                    "timestamp": timestamp,
                    "pose": np.copy(current_pose),
                    "scan_xy_local": np.copy(scan_xy),
                    "cumulative_path_length_m": cumulative_path_length_m,
                }
            )

    if not processed_entries:
        raise RuntimeError("No usable scans found for replay processing.")

    runtime_seconds = time.perf_counter() - started
    all_map_points = np.vstack(point_cloud_chunks) if point_cloud_chunks else np.empty((0, 2), dtype=float)
    keyframe_poses = np.asarray([entry["pose"] for entry in keyframes], dtype=float) if keyframes else np.empty((0, 3, 3))

    return {
        "processed_entries": processed_entries,
        "keyframes": keyframes,
        "keyframe_poses": keyframe_poses,
        "all_map_points": all_map_points,
        "runtime_seconds": runtime_seconds,
    }


def build_sequence_payload(
    sequence_name: str,
    config: PipelineConfig,
    *,
    include_markers: bool = True,
) -> dict:
    bundle = load_sequence_assets(sequence_name, include_markers=include_markers)
    replay = process_replay_sequence(bundle["scans"], bundle["timestamps"], config)
    payload = {**bundle, **replay}
    return payload


def build_occupancy_grid(
    processed_entries: list[dict],
    *,
    cell_size_m: float,
    padding_m: float,
) -> dict:
    min_xy = np.array([np.inf, np.inf], dtype=float)
    max_xy = np.array([-np.inf, -np.inf], dtype=float)

    for entry in processed_entries:
        pose = entry["pose"]
        robot_xy = np.array(pose_to_xytheta(pose)[:2], dtype=float)
        min_xy = np.minimum(min_xy, robot_xy)
        max_xy = np.maximum(max_xy, robot_xy)

        world_points = transform_points(entry["scan_xy"], pose)
        if world_points.size:
            min_xy = np.minimum(min_xy, world_points.min(axis=0))
            max_xy = np.maximum(max_xy, world_points.max(axis=0))

    if not np.isfinite(min_xy).all():
        raise RuntimeError("Cannot build occupancy grid without valid processed scans.")

    min_xy -= padding_m
    max_xy += padding_m

    width = int(math.ceil((max_xy[0] - min_xy[0]) / cell_size_m)) + 1
    height = int(math.ceil((max_xy[1] - min_xy[1]) / cell_size_m)) + 1

    log_odds = np.zeros((height, width), dtype=np.float32)
    origin_xy = min_xy

    for entry in processed_entries:
        pose = entry["pose"]
        scan_xy = entry["scan_xy"]
        if scan_xy.size == 0:
            continue

        robot_xy = np.array(pose_to_xytheta(pose)[:2], dtype=float)
        robot_rc = world_to_grid(robot_xy.reshape(1, 2), origin_xy, cell_size_m)[0]
        world_hits = transform_points(scan_xy, pose)
        hit_rc = world_to_grid(world_hits, origin_xy, cell_size_m)

        for col_row in hit_rc:
            end_x, end_y = int(col_row[0]), int(col_row[1])
            start_x, start_y = int(robot_rc[0]), int(robot_rc[1])
            if not (0 <= end_x < width and 0 <= end_y < height):
                continue
            if not (0 <= start_x < width and 0 <= start_y < height):
                continue

            cells = list(bresenham_line(start_x, start_y, end_x, end_y))
            if len(cells) > 1:
                free_cells = np.asarray(cells[:-1], dtype=int)
                log_odds[free_cells[:, 1], free_cells[:, 0]] += LOG_ODDS_FREE
            log_odds[end_y, end_x] += LOG_ODDS_OCCUPIED

    np.clip(log_odds, LOG_ODDS_MIN, LOG_ODDS_MAX, out=log_odds)
    probabilities = 1.0 / (1.0 + np.exp(-log_odds))

    return {
        "log_odds": log_odds,
        "probabilities": probabilities.astype(np.float32),
        "origin_xy": origin_xy,
        "bounds_min_xy": min_xy,
        "bounds_max_xy": max_xy,
        "cell_size_m": cell_size_m,
    }


def rebuild_point_cloud_map(
    processed_entries: list[dict],
    *,
    voxel_size_m: float,
) -> np.ndarray:
    point_cloud_chunks: list[np.ndarray] = []

    for entry in processed_entries:
        scan_xy = entry["scan_xy"]
        if scan_xy.size == 0:
            continue
        world_points = transform_points(scan_xy, entry["pose"])
        world_points = voxel_downsample(world_points, voxel_size_m)
        if world_points.size:
            point_cloud_chunks.append(world_points)

    if not point_cloud_chunks:
        return np.empty((0, 2), dtype=float)

    map_points = np.vstack(point_cloud_chunks)
    return voxel_downsample(map_points, voxel_size_m)


def clone_processed_entries_with_poses(
    processed_entries: list[dict],
    poses: list[np.ndarray] | np.ndarray,
) -> list[dict]:
    cloned_entries: list[dict] = []
    for entry, pose in zip(processed_entries, poses):
        cloned_entry = dict(entry)
        cloned_entry["pose"] = np.asarray(pose, dtype=float)
        cloned_entries.append(cloned_entry)
    return cloned_entries


def save_trajectory_csv(path: Path, processed_entries: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scan_index", "timestamp", "x", "y", "theta"])
        for entry in processed_entries:
            x, y, theta = pose_to_xytheta(entry["pose"])
            writer.writerow([entry["scan_index"], f"{entry['timestamp']:.6f}", x, y, theta])


def save_summary(path: Path, summary: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def save_required_plots(
    output_dir: Path,
    sequence_name: str,
    label: str,
    processed_entries: list[dict],
    markers: list[dict],
    all_map_points: np.ndarray,
    occupancy: dict,
    *,
    save_debug_plots: bool,
) -> None:
    trajectory_xy = np.array([pose_to_xytheta(entry["pose"])[:2] for entry in processed_entries], dtype=float)
    marker_positions = []
    processed_scan_indices = np.array([entry["scan_index"] for entry in processed_entries], dtype=int)
    for marker in markers:
        nearest = int(np.argmin(np.abs(processed_scan_indices - marker["scan_index"])))
        marker_positions.append((marker, trajectory_xy[nearest]))

    title = f"{sequence_name} — {label}"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], linewidth=1.5, color="tab:blue")
    ax.scatter(trajectory_xy[0, 0], trajectory_xy[0, 1], color="black", marker="s", s=40, label="Start")
    ax.scatter(trajectory_xy[-1, 0], trajectory_xy[-1, 1], color="tab:red", s=30, label="End")
    for marker, position in marker_positions:
        ax.scatter(position[0], position[1], color="tab:green", marker="*", s=120)
        ax.annotate(f"Loop {marker['loop_number']}", position, xytext=(4, 4), textcoords="offset points")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Trajectory\n{title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    if all_map_points.size:
        render_step = max(1, len(all_map_points) // 40000)
        ax.scatter(
            all_map_points[::render_step, 0],
            all_map_points[::render_step, 1],
            s=0.5,
            c="0.35",
            alpha=0.45,
        )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(f"Point-Cloud Map\n{title}")
    fig.tight_layout()
    fig.savefig(output_dir / "point_cloud_map.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    bounds = occupancy["bounds_min_xy"], occupancy["bounds_max_xy"]
    extent = [
        bounds[0][0],
        bounds[1][0],
        bounds[0][1],
        bounds[1][1],
    ]
    ax.imshow(
        occupancy["probabilities"],
        origin="lower",
        cmap="gray",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_aspect("equal")
    ax.set_title(f"Occupancy Grid\n{title}")
    fig.tight_layout()
    fig.savefig(output_dir / "occupancy_grid.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(
        occupancy["probabilities"],
        origin="lower",
        cmap="gray",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
    )
    ax.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], linewidth=1.3, color="tab:cyan")
    ax.scatter(trajectory_xy[0, 0], trajectory_xy[0, 1], color="yellow", marker="s", s=35)
    ax.scatter(trajectory_xy[-1, 0], trajectory_xy[-1, 1], color="tab:red", s=25)
    for marker, position in marker_positions:
        ax.scatter(position[0], position[1], color="lime", marker="*", s=110)
    ax.set_aspect("equal")
    ax.set_title(f"Occupancy Grid With Trajectory\n{title}")
    fig.tight_layout()
    fig.savefig(output_dir / "map_with_trajectory.png", dpi=180)
    plt.close(fig)

    if save_debug_plots:
        fig, ax = plt.subplots(figsize=(8, 8))
        if all_map_points.size:
            render_step = max(1, len(all_map_points) // 30000)
            ax.scatter(
                all_map_points[::render_step, 0],
                all_map_points[::render_step, 1],
                s=0.5,
                c="0.55",
                alpha=0.35,
            )
        ax.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], linewidth=1.0, color="tab:blue")
        for marker, position in marker_positions:
            ax.scatter(position[0], position[1], color="tab:green", marker="*", s=140)
            ax.annotate(
                f"Loop {marker['loop_number']}\nscan {marker['scan_index']}",
                position,
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"Debug Marker Overlay\n{title}")
        fig.tight_layout()
        fig.savefig(output_dir / "debug_closure_markers.png", dpi=180)
        plt.close(fig)


def nearest_processed_entry(processed_entries: list[dict], target_scan_index: int) -> dict:
    processed_scan_indices = np.array([entry["scan_index"] for entry in processed_entries], dtype=int)
    nearest_idx = int(np.argmin(np.abs(processed_scan_indices - target_scan_index)))
    return processed_entries[nearest_idx]


def run_sequence(
    sequence_name: str,
    config: PipelineConfig,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output_tag: str | None = None,
    save_debug_plots: bool = False,
) -> dict:
    payload = build_sequence_payload(sequence_name, config, include_markers=True)
    sequence_dir = payload["sequence_dir"]
    markers = payload["markers"]
    info = payload["source_metadata"]

    output_tag = output_tag or build_output_tag(config)
    output_dir = output_root / sequence_name / output_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_entries = payload["processed_entries"]
    keyframes = payload["keyframes"]
    keyframe_poses = payload["keyframe_poses"]
    all_map_points = payload["all_map_points"]
    runtime_seconds = payload["runtime_seconds"]

    trajectory_xytheta = np.array([pose_to_xytheta(entry["pose"]) for entry in processed_entries], dtype=float)
    trajectory_xy = trajectory_xytheta[:, :2]

    occupancy = build_occupancy_grid(
        processed_entries,
        cell_size_m=config.occupancy_cell_size_m,
        padding_m=config.occupancy_padding_m,
    )

    save_trajectory_csv(output_dir / "trajectory.csv", processed_entries)
    np.savez_compressed(
        output_dir / "point_cloud_map.npz",
        points=all_map_points,
        keyframe_poses=np.asarray(keyframe_poses, dtype=float),
    )
    np.savez_compressed(
        output_dir / "occupancy_grid.npz",
        grid=occupancy["probabilities"],
        log_odds=occupancy["log_odds"],
        resolution_m=occupancy["cell_size_m"],
        origin_xy=occupancy["origin_xy"],
        bounds_min_xy=occupancy["bounds_min_xy"],
        bounds_max_xy=occupancy["bounds_max_xy"],
    )

    marker_metrics = []
    for marker in markers:
        nearest_entry = nearest_processed_entry(processed_entries, int(marker["scan_index"]))
        marker_pose = nearest_entry["pose"]
        closure_error = float(np.linalg.norm(marker_pose[:2, 2] - processed_entries[0]["pose"][:2, 2]))
        marker_metrics.append(
            {
                "loop_number": marker["loop_number"],
                "requested_scan_index": marker["scan_index"],
                "matched_processed_scan_index": nearest_entry["scan_index"],
                "closure_error_m": closure_error,
            }
        )

    summary = {
        "sequence": sequence_name,
        "label": payload["label"],
        "sequence_dir": str(sequence_dir),
        "source_metadata": info,
        "config": asdict(config),
        "raw_scan_count": len(payload["scans"]),
        "processed_scan_count": len(processed_entries),
        "keyframe_count": len(keyframes),
        "runtime_seconds": round(runtime_seconds, 3),
        "total_distance_m": round(compute_total_distance(trajectory_xy), 3),
        "final_start_to_end_drift_m": round(float(np.linalg.norm(trajectory_xy[-1] - trajectory_xy[0])), 3),
        "loop_marker_metrics": marker_metrics,
        "output_dir": str(output_dir),
    }
    save_summary(output_dir / "summary.json", summary)

    save_required_plots(
        output_dir,
        sequence_name,
        payload["label"],
        processed_entries,
        markers,
        all_map_points,
        occupancy,
        save_debug_plots=save_debug_plots,
    )

    return summary
