#!/usr/bin/env python3
"""Run CW2 Q3c loop-closure detection on the selected LiDAR sequences."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from lidar_step1_pipeline import (
    BASE_DIR,
    PipelineConfig,
    build_sequence_payload,
    compute_total_distance,
    estimate_normals_pca,
    icp_scan_to_map,
    load_json,
    pose_to_xytheta,
    preprocess_scan,
    SEQUENCE_REGISTRY,
    transform_points,
    xytheta_to_pose,
)


Q3C_OUTPUT_ROOT = BASE_DIR / "output" / "lidar_q3c"
SELECTED_SEQUENCES = ["indoor_large_03", "indoor_small_01", "outdoor_02"]

BASE_CONFIGS = {
    "indoor_large_03": PipelineConfig(
        max_range_mm=12000.0,
        beam_step=2,
        voxel_size_m=0.05,
        scan_stride=3,
    ),
    "indoor_small_01": PipelineConfig(
        max_range_mm=12000.0,
        beam_step=3,
        voxel_size_m=0.05,
        scan_stride=2,
    ),
    "outdoor_02": PipelineConfig(
        max_range_mm=6000.0,
        beam_step=1,
        voxel_size_m=0.05,
        scan_stride=2,
    ),
}

DETECTION_PREPROCESS = {
    "indoor_large_03": {"max_range_mm": 12000.0, "beam_step": 1},
    "indoor_small_01": {"max_range_mm": 12000.0, "beam_step": 1},
    "outdoor_02": {"max_range_mm": 6000.0, "beam_step": 1},
}

EXCLUSION_TIME_S = 20.0
EXCLUSION_PATH_M = 8.0
POSE_GATE_ANGLE_RAD = math.radians(45.0)
VERIFICATION_TOP_K = 5
VERIFICATION_INLIER_THRESH_M = 0.25
MARKER_MATCH_WINDOW_S = 15.0
MARKER_TARGET_START_DIST_M = 3.0
ANCHOR_SOURCE_MIN_FRACTION = 0.70
ANCHOR_TARGET_MAX_FRACTION = 0.20
DESCRIPTOR_BINS = 72
DEDUP_SOURCE_WINDOW = 5
DEDUP_TARGET_WINDOW = 5
ANCHOR_CLUSTER_SCORE_TOLERANCE = 0.05

POSE_GATES = {
    "indoor": {"distance_m": 3.0},
    "outdoor": {"distance_m": 5.0},
}

LOCAL_ACCEPT_THRESHOLDS = {
    "indoor": {
        "inlier_ratio": 0.35,
        "mean_residual_m": 0.15,
        "translation_correction_m": 2.0,
        "rotation_correction_rad": math.radians(25.0),
    },
    "outdoor": {
        "inlier_ratio": 0.35,
        "mean_residual_m": 0.20,
        "translation_correction_m": 3.5,
        "rotation_correction_rad": math.radians(35.0),
    },
}

ANCHOR_ACCEPT_THRESHOLDS = {
    "indoor": {
        "inlier_ratio": 0.65,
        "mean_residual_m": 0.10,
        "translation_correction_m": 1.20,
        "rotation_correction_rad": 0.60,
    },
    "outdoor": {
        "inlier_ratio": 0.75,
        "mean_residual_m": 0.12,
        "translation_correction_m": 1.00,
        "rotation_correction_rad": 0.65,
    },
}

ANCHOR_DESCRIPTOR_THRESHOLDS = {
    "indoor": 0.20,
    "outdoor": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LiDAR loop-closure detection for CW2 Q3c."
    )
    parser.add_argument(
        "--sequence",
        default="all",
        choices=["all", *SELECTED_SEQUENCES],
        help="Sequence to process, or 'all' for the full Q3c batch.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Q3C_OUTPUT_ROOT,
        help="Root directory for Q3c outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute sequence outputs even if detection_summary.json already exists.",
    )
    return parser.parse_args()


def normalize_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def relative_pose_matrix(source_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(target_pose) @ source_pose


def pose_vector_from_matrix(pose: np.ndarray) -> list[float]:
    return [float(pose[0, 2]), float(pose[1, 2]), float(math.atan2(pose[1, 0], pose[0, 0]))]


def pose_distance_and_angle(source_pose: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
    dist = float(np.linalg.norm(source_pose[:2, 2] - target_pose[:2, 2]))
    source_theta = math.atan2(source_pose[1, 0], source_pose[0, 0])
    target_theta = math.atan2(target_pose[1, 0], target_pose[0, 0])
    angle = abs(normalize_angle(source_theta - target_theta))
    return dist, angle


def get_environment(sequence_payload: dict) -> str:
    return str(sequence_payload["source_metadata"]["environment"])


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def build_sector_descriptor(points: np.ndarray, max_range_m: float) -> np.ndarray:
    descriptor = np.full(DESCRIPTOR_BINS, max_range_m, dtype=float)
    if points.size == 0:
        return descriptor

    ranges = np.linalg.norm(points, axis=1)
    angles = (np.arctan2(points[:, 1], points[:, 0]) + 2.0 * math.pi) % (2.0 * math.pi)
    bin_indices = np.floor((angles / (2.0 * math.pi)) * DESCRIPTOR_BINS).astype(int) % DESCRIPTOR_BINS

    for bin_idx, point_range in zip(bin_indices, ranges, strict=True):
        if point_range < descriptor[bin_idx]:
            descriptor[bin_idx] = point_range
    return descriptor


def shift_to_angle_rad(shift_bins: int) -> float:
    signed_shift = shift_bins if shift_bins <= DESCRIPTOR_BINS // 2 else shift_bins - DESCRIPTOR_BINS
    return float(signed_shift * (2.0 * math.pi / DESCRIPTOR_BINS))


def descriptor_match(
    source_descriptor: np.ndarray,
    target_descriptor: np.ndarray,
    max_range_m: float,
) -> tuple[float, float]:
    best_error = float("inf")
    best_shift = 0
    for shift in range(DESCRIPTOR_BINS):
        shifted_source = np.roll(source_descriptor, shift)
        error = float(np.mean(np.abs(shifted_source - target_descriptor)) / max_range_m)
        if error < best_error:
            best_error = error
            best_shift = shift
    return best_error, shift_to_angle_rad(best_shift)


def build_anchor_init_pose(
    source_points: np.ndarray,
    target_points: np.ndarray,
    yaw_init_rad: float,
) -> np.ndarray:
    rotation_pose = xytheta_to_pose(0.0, 0.0, yaw_init_rad)
    rotated_source = transform_points(source_points, rotation_pose)
    if rotated_source.size == 0 or target_points.size == 0:
        return rotation_pose

    source_centroid = rotated_source.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    translation = target_centroid - source_centroid
    return xytheta_to_pose(float(translation[0]), float(translation[1]), yaw_init_rad)


def attach_detection_data(payload: dict, sequence_name: str) -> dict:
    preprocess_config = DETECTION_PREPROCESS[sequence_name]
    max_range_m = preprocess_config["max_range_mm"] / 1000.0
    keyframe_count = len(payload["keyframes"])
    denominator = max(1, keyframe_count - 1)

    for keyframe in payload["keyframes"]:
        raw_scan = payload["scans"][keyframe["scan_index"]]
        detection_scan = preprocess_scan(
            raw_scan,
            max_range_mm=preprocess_config["max_range_mm"],
            beam_step=preprocess_config["beam_step"],
        )
        if detection_scan is None:
            detection_scan = np.copy(keyframe["scan_xy_local"])

        detection_descriptor = build_sector_descriptor(detection_scan, max_range_m)
        keyframe["detection_scan_xy_local"] = detection_scan
        keyframe["detection_descriptor"] = detection_descriptor
        keyframe["source_fraction"] = float(keyframe["keyframe_id"] / denominator)
        keyframe["target_fraction"] = keyframe["source_fraction"]

    payload["detection_preprocess"] = {
        "max_range_mm": preprocess_config["max_range_mm"],
        "beam_step": preprocess_config["beam_step"],
        "max_range_m": max_range_m,
    }
    return payload


def make_target_cache(keyframes: list[dict]) -> dict[int, dict]:
    cache = {}
    for keyframe in keyframes:
        points = keyframe["detection_scan_xy_local"]
        normals = estimate_normals_pca(points)
        neighbor = NearestNeighbors(n_neighbors=1)
        neighbor.fit(points)
        cache[keyframe["keyframe_id"]] = {
            "normals": normals,
            "neighbor": neighbor,
        }
    return cache


def rejection_reason(metrics: dict, thresholds: dict) -> str | None:
    if metrics["inlier_ratio"] < thresholds["inlier_ratio"]:
        return "below_inlier_ratio"
    if metrics["mean_residual_m"] > thresholds["mean_residual_m"]:
        return "above_mean_residual"
    if metrics["translation_correction_m"] > thresholds["translation_correction_m"]:
        return "above_translation_correction"
    if metrics["rotation_correction_rad"] > thresholds["rotation_correction_rad"]:
        return "above_rotation_correction"
    return None


def verify_candidate(
    source_points: np.ndarray,
    target_points: np.ndarray,
    target_cache_entry: dict,
    init_pose: np.ndarray,
    *,
    correspondence_thresh_m: float,
    icp_max_iter: int,
) -> dict:
    icp_pose = icp_scan_to_map(
        source_points,
        target_points,
        target_cache_entry["normals"],
        init_pose,
        max_iter=icp_max_iter,
        correspondence_thresh_m=correspondence_thresh_m,
    )

    transformed_points = transform_points(source_points, icp_pose)
    distances, _ = target_cache_entry["neighbor"].kneighbors(transformed_points, return_distance=True)
    distances = distances.ravel()
    inlier_mask = distances < VERIFICATION_INLIER_THRESH_M
    inlier_ratio = float(np.mean(inlier_mask)) if len(distances) else 0.0
    mean_residual = float(np.mean(distances[inlier_mask])) if np.any(inlier_mask) else float("inf")

    init_vector = pose_vector_from_matrix(init_pose)
    icp_vector = pose_vector_from_matrix(icp_pose)
    translation_correction = float(np.linalg.norm(icp_pose[:2, 2] - init_pose[:2, 2]))
    rotation_correction = float(abs(normalize_angle(icp_vector[2] - init_vector[2])))
    score = float(
        inlier_ratio
        - mean_residual
        - 0.1 * translation_correction
        - 0.05 * rotation_correction
    )

    return {
        "relative_pose_init": init_vector,
        "relative_pose_icp": icp_vector,
        "mean_residual_m": mean_residual,
        "inlier_ratio": inlier_ratio,
        "translation_correction_m": translation_correction,
        "rotation_correction_rad": rotation_correction,
        "score": score,
    }


def build_candidate_record(
    *,
    candidate_id: int,
    branch_type: str,
    current_keyframe: dict,
    target_keyframe: dict,
    pose_distance_m: float,
    pose_angle_diff_rad: float,
    path_gap_m: float,
    metrics: dict,
    descriptor_error: float | None,
    descriptor_shift_rad: float | None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "branch_type": branch_type,
        "source_keyframe_id": current_keyframe["keyframe_id"],
        "target_keyframe_id": target_keyframe["keyframe_id"],
        "source_scan_index": current_keyframe["scan_index"],
        "target_scan_index": target_keyframe["scan_index"],
        "source_processed_index": current_keyframe["processed_index"],
        "target_processed_index": target_keyframe["processed_index"],
        "source_timestamp": float(current_keyframe["timestamp"]),
        "target_timestamp": float(target_keyframe["timestamp"]),
        "source_pose_xytheta": list(pose_to_xytheta(current_keyframe["pose"])),
        "target_pose_xytheta": list(pose_to_xytheta(target_keyframe["pose"])),
        "pose_distance_m": float(pose_distance_m),
        "pose_angle_diff_rad": float(pose_angle_diff_rad),
        "relative_pose_init": metrics["relative_pose_init"],
        "relative_pose_icp": metrics["relative_pose_icp"],
        "mean_residual_m": metrics["mean_residual_m"],
        "inlier_ratio": metrics["inlier_ratio"],
        "translation_correction_m": metrics["translation_correction_m"],
        "rotation_correction_rad": metrics["rotation_correction_rad"],
        "score": metrics["score"],
        "descriptor_error": descriptor_error,
        "descriptor_shift_rad": descriptor_shift_rad,
        "source_fraction": float(current_keyframe["source_fraction"]),
        "target_fraction": float(target_keyframe["target_fraction"]),
        "keyframe_gap": int(current_keyframe["keyframe_id"] - target_keyframe["keyframe_id"]),
        "path_gap_m": float(path_gap_m),
        "accepted_before_dedup": False,
        "accepted_final": False,
        "accepted_loop_id": None,
        "rejection_reason": "",
    }


def cluster_records(records: list[dict]) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    visited: set[int] = set()

    for idx in range(len(records)):
        if idx in visited:
            continue
        queue = [idx]
        visited.add(idx)
        cluster_indices = []
        while queue:
            current = queue.pop()
            cluster_indices.append(current)
            current_record = records[current]
            for other in range(len(records)):
                if other in visited:
                    continue
                other_record = records[other]
                if (
                    abs(current_record["source_keyframe_id"] - other_record["source_keyframe_id"])
                    <= DEDUP_SOURCE_WINDOW
                    and abs(current_record["target_keyframe_id"] - other_record["target_keyframe_id"])
                    <= DEDUP_TARGET_WINDOW
                ):
                    visited.add(other)
                    queue.append(other)
        clusters.append([records[cluster_idx] for cluster_idx in cluster_indices])

    return clusters


def deduplicate_loops(accepted_pre: list[dict]) -> list[dict]:
    final = []
    for cluster in cluster_records(accepted_pre):
        local_records = [record for record in cluster if record["branch_type"] == "local"]
        anchor_records = [record for record in cluster if record["branch_type"] == "anchor"]

        best_local = max(local_records, key=lambda row: row["score"]) if local_records else None
        best_anchor = max(anchor_records, key=lambda row: row["score"]) if anchor_records else None

        if best_local is not None and best_anchor is not None:
            if best_anchor["score"] >= best_local["score"] - ANCHOR_CLUSTER_SCORE_TOLERANCE:
                chosen = best_anchor
            else:
                chosen = max(cluster, key=lambda row: row["score"])
        else:
            chosen = max(cluster, key=lambda row: row["score"])

        chosen["accepted_final"] = True
        final.append(chosen)
        for record in cluster:
            if record is not chosen:
                record["rejection_reason"] = "suppressed_by_dedup_cluster"

    final.sort(key=lambda row: (row["source_keyframe_id"], row["target_keyframe_id"]))
    for accepted_loop_id, record in enumerate(final):
        record["accepted_loop_id"] = accepted_loop_id
    return final


def evaluate_markers(accepted_final: list[dict], markers: list[dict], start_pose: np.ndarray) -> tuple[list[dict], list[int]]:
    matched_loop_ids = set()
    results = []

    for marker in markers:
        candidates = []
        for record in accepted_final:
            if abs(record["source_timestamp"] - marker["timestamp"]) > MARKER_MATCH_WINDOW_S:
                continue
            target_start_dist = float(
                np.linalg.norm(np.asarray(record["target_pose_xytheta"][:2]) - start_pose[:2, 2])
            )
            if target_start_dist > MARKER_TARGET_START_DIST_M:
                continue
            candidates.append((abs(record["source_timestamp"] - marker["timestamp"]), -record["score"], record))

        if candidates:
            _, _, best = min(candidates)
            matched_loop_ids.add(best["accepted_loop_id"])
            results.append(
                {
                    "loop_number": marker["loop_number"],
                    "marker_timestamp": marker["timestamp"],
                    "hit": True,
                    "matched_loop_id": best["accepted_loop_id"],
                    "time_delta_s": round(best["source_timestamp"] - marker["timestamp"], 3),
                    "score": round(best["score"], 6),
                    "branch_type": best["branch_type"],
                    "source_keyframe_id": best["source_keyframe_id"],
                    "target_keyframe_id": best["target_keyframe_id"],
                }
            )
        else:
            results.append(
                {
                    "loop_number": marker["loop_number"],
                    "marker_timestamp": marker["timestamp"],
                    "hit": False,
                    "matched_loop_id": None,
                    "time_delta_s": None,
                    "score": None,
                    "branch_type": None,
                    "source_keyframe_id": None,
                    "target_keyframe_id": None,
                }
            )

    return results, sorted(matched_loop_ids)


def create_trajectory_plot(output_dir: Path, payload: dict, markers: list[dict], accepted_final: list[dict]) -> None:
    processed_entries = payload["processed_entries"]
    trajectory_xy = np.array([pose_to_xytheta(entry["pose"])[:2] for entry in processed_entries], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], color="tab:blue", linewidth=1.4, label="Trajectory")
    ax.scatter(trajectory_xy[0, 0], trajectory_xy[0, 1], color="black", marker="s", s=40, label="Start")
    ax.scatter(trajectory_xy[-1, 0], trajectory_xy[-1, 1], color="tab:red", s=30, label="End")

    processed_timestamps = np.array([entry["timestamp"] for entry in processed_entries], dtype=float)
    for marker in markers:
        nearest = int(np.argmin(np.abs(processed_timestamps - marker["timestamp"])))
        marker_xy = trajectory_xy[nearest]
        ax.scatter(marker_xy[0], marker_xy[1], color="tab:green", marker="*", s=120)
        ax.annotate(f"Loop {marker['loop_number']}", marker_xy, xytext=(4, 4), textcoords="offset points")

    for record in accepted_final:
        source_xy = np.asarray(record["source_pose_xytheta"][:2], dtype=float)
        target_xy = np.asarray(record["target_pose_xytheta"][:2], dtype=float)
        edge_color = "tab:orange" if record["branch_type"] == "anchor" else "tab:purple"
        ax.plot([source_xy[0], target_xy[0]], [source_xy[1], target_xy[1]], color=edge_color, linewidth=1.4, alpha=0.8)
        ax.scatter(source_xy[0], source_xy[1], color=edge_color, s=18)
        ax.scatter(target_xy[0], target_xy[1], color="tab:brown", s=18)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Detected Loop Closures\n{payload['sequence']} — {payload['label']}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_with_detected_loops.png", dpi=180)
    plt.close(fig)


def create_score_timeline(output_dir: Path, payload: dict, markers: list[dict], accepted_final: list[dict]) -> None:
    sequence_start = payload["processed_entries"][0]["timestamp"]
    fig, ax = plt.subplots(figsize=(10, 4.5))

    anchor_records = [record for record in accepted_final if record["branch_type"] == "anchor"]
    local_records = [record for record in accepted_final if record["branch_type"] == "local"]

    if local_records:
        x_values = [record["source_timestamp"] - sequence_start for record in local_records]
        y_values = [record["score"] for record in local_records]
        ax.scatter(x_values, y_values, color="tab:purple", s=30, label="Accepted local loops")

    if anchor_records:
        x_values = [record["source_timestamp"] - sequence_start for record in anchor_records]
        y_values = [record["score"] for record in anchor_records]
        ax.scatter(x_values, y_values, color="tab:orange", s=38, label="Accepted anchor loops")

    for marker in markers:
        marker_x = marker["timestamp"] - sequence_start
        ax.axvline(marker_x, color="tab:green", linestyle="--", linewidth=1.0)
        ax.text(marker_x, 0.98, f"Loop {marker['loop_number']}", rotation=90, va="top", ha="right", transform=ax.get_xaxis_transform())

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Accepted loop score")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Accepted Loop Scores Over Time\n{payload['sequence']} — {payload['label']}")
    if accepted_final:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "score_timeline.png", dpi=180)
    plt.close(fig)


def candidate_fieldnames() -> list[str]:
    return [
        "candidate_id",
        "branch_type",
        "source_keyframe_id",
        "target_keyframe_id",
        "source_scan_index",
        "target_scan_index",
        "source_processed_index",
        "target_processed_index",
        "source_timestamp",
        "target_timestamp",
        "source_fraction",
        "target_fraction",
        "keyframe_gap",
        "path_gap_m",
        "pose_distance_m",
        "pose_angle_diff_rad",
        "descriptor_error",
        "descriptor_shift_rad",
        "relative_pose_init_dx",
        "relative_pose_init_dy",
        "relative_pose_init_dtheta",
        "relative_pose_icp_dx",
        "relative_pose_icp_dy",
        "relative_pose_icp_dtheta",
        "mean_residual_m",
        "inlier_ratio",
        "translation_correction_m",
        "rotation_correction_rad",
        "score",
        "accepted_before_dedup",
        "accepted_final",
        "accepted_loop_id",
        "rejection_reason",
    ]


def accepted_fieldnames() -> list[str]:
    return [
        "accepted_loop_id",
        "branch_type",
        "source_keyframe_id",
        "target_keyframe_id",
        "source_scan_index",
        "target_scan_index",
        "source_processed_index",
        "target_processed_index",
        "source_timestamp",
        "target_timestamp",
        "source_fraction",
        "target_fraction",
        "keyframe_gap",
        "path_gap_m",
        "descriptor_error",
        "descriptor_shift_rad",
        "relative_pose_init_dx",
        "relative_pose_init_dy",
        "relative_pose_init_dtheta",
        "relative_pose_icp_dx",
        "relative_pose_icp_dy",
        "relative_pose_icp_dtheta",
        "score",
        "mean_residual_m",
        "inlier_ratio",
        "translation_correction_m",
        "rotation_correction_rad",
    ]


def flatten_candidate_record(record: dict) -> dict:
    return {
        "candidate_id": record["candidate_id"],
        "branch_type": record["branch_type"],
        "source_keyframe_id": record["source_keyframe_id"],
        "target_keyframe_id": record["target_keyframe_id"],
        "source_scan_index": record["source_scan_index"],
        "target_scan_index": record["target_scan_index"],
        "source_processed_index": record["source_processed_index"],
        "target_processed_index": record["target_processed_index"],
        "source_timestamp": record["source_timestamp"],
        "target_timestamp": record["target_timestamp"],
        "source_fraction": record["source_fraction"],
        "target_fraction": record["target_fraction"],
        "keyframe_gap": record["keyframe_gap"],
        "path_gap_m": record["path_gap_m"],
        "pose_distance_m": record["pose_distance_m"],
        "pose_angle_diff_rad": record["pose_angle_diff_rad"],
        "descriptor_error": record["descriptor_error"],
        "descriptor_shift_rad": record["descriptor_shift_rad"],
        "relative_pose_init_dx": record["relative_pose_init"][0],
        "relative_pose_init_dy": record["relative_pose_init"][1],
        "relative_pose_init_dtheta": record["relative_pose_init"][2],
        "relative_pose_icp_dx": record["relative_pose_icp"][0],
        "relative_pose_icp_dy": record["relative_pose_icp"][1],
        "relative_pose_icp_dtheta": record["relative_pose_icp"][2],
        "mean_residual_m": record["mean_residual_m"],
        "inlier_ratio": record["inlier_ratio"],
        "translation_correction_m": record["translation_correction_m"],
        "rotation_correction_rad": record["rotation_correction_rad"],
        "score": record["score"],
        "accepted_before_dedup": record["accepted_before_dedup"],
        "accepted_final": record["accepted_final"],
        "accepted_loop_id": record["accepted_loop_id"],
        "rejection_reason": record["rejection_reason"],
    }


def serializable_loop_record(record: dict) -> dict:
    return {
        "accepted_loop_id": record["accepted_loop_id"],
        "branch_type": record["branch_type"],
        "source_keyframe_id": record["source_keyframe_id"],
        "target_keyframe_id": record["target_keyframe_id"],
        "source_scan_index": record["source_scan_index"],
        "target_scan_index": record["target_scan_index"],
        "source_processed_index": record["source_processed_index"],
        "target_processed_index": record["target_processed_index"],
        "source_timestamp": record["source_timestamp"],
        "target_timestamp": record["target_timestamp"],
        "source_fraction": record["source_fraction"],
        "target_fraction": record["target_fraction"],
        "keyframe_gap": record["keyframe_gap"],
        "path_gap_m": record["path_gap_m"],
        "descriptor_error": record["descriptor_error"],
        "descriptor_shift_rad": record["descriptor_shift_rad"],
        "relative_pose_init": record["relative_pose_init"],
        "relative_pose_icp": record["relative_pose_icp"],
        "score": record["score"],
        "mean_residual_m": record["mean_residual_m"],
        "inlier_ratio": record["inlier_ratio"],
        "translation_correction_m": record["translation_correction_m"],
        "rotation_correction_rad": record["rotation_correction_rad"],
    }


def flatten_accepted_record(record: dict) -> dict:
    return {
        "accepted_loop_id": record["accepted_loop_id"],
        "branch_type": record["branch_type"],
        "source_keyframe_id": record["source_keyframe_id"],
        "target_keyframe_id": record["target_keyframe_id"],
        "source_scan_index": record["source_scan_index"],
        "target_scan_index": record["target_scan_index"],
        "source_processed_index": record["source_processed_index"],
        "target_processed_index": record["target_processed_index"],
        "source_timestamp": record["source_timestamp"],
        "target_timestamp": record["target_timestamp"],
        "source_fraction": record["source_fraction"],
        "target_fraction": record["target_fraction"],
        "keyframe_gap": record["keyframe_gap"],
        "path_gap_m": record["path_gap_m"],
        "descriptor_error": record["descriptor_error"],
        "descriptor_shift_rad": record["descriptor_shift_rad"],
        "relative_pose_init_dx": record["relative_pose_init"][0],
        "relative_pose_init_dy": record["relative_pose_init"][1],
        "relative_pose_init_dtheta": record["relative_pose_init"][2],
        "relative_pose_icp_dx": record["relative_pose_icp"][0],
        "relative_pose_icp_dy": record["relative_pose_icp"][1],
        "relative_pose_icp_dtheta": record["relative_pose_icp"][2],
        "score": record["score"],
        "mean_residual_m": record["mean_residual_m"],
        "inlier_ratio": record["inlier_ratio"],
        "translation_correction_m": record["translation_correction_m"],
        "rotation_correction_rad": record["rotation_correction_rad"],
    }


def detect_sequence(sequence_name: str, *, output_root: Path, force: bool) -> dict:
    output_dir = output_root / sequence_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "detection_summary.json"
    if summary_path.exists() and not force:
        return load_json(summary_path)

    config = BASE_CONFIGS[sequence_name]
    payload = build_sequence_payload(sequence_name, config, include_markers=False)
    payload = attach_detection_data(payload, sequence_name)
    keyframes = payload["keyframes"]
    environment = get_environment(payload)
    pose_gate_distance = POSE_GATES[environment]["distance_m"]
    detection_max_range_m = payload["detection_preprocess"]["max_range_m"]
    local_thresholds = LOCAL_ACCEPT_THRESHOLDS[environment]
    anchor_thresholds = ANCHOR_ACCEPT_THRESHOLDS[environment]
    anchor_descriptor_threshold = ANCHOR_DESCRIPTOR_THRESHOLDS[environment]
    target_cache = make_target_cache(keyframes)

    candidate_records: list[dict] = []
    accepted_pre: list[dict] = []
    candidate_id = 0
    late_to_early_candidates_evaluated = 0

    for current_keyframe in keyframes:
        local_eligible = []
        anchor_eligible = []

        for target_keyframe in keyframes[: current_keyframe["keyframe_id"]]:
            age_s = current_keyframe["timestamp"] - target_keyframe["timestamp"]
            path_delta = current_keyframe["cumulative_path_length_m"] - target_keyframe["cumulative_path_length_m"]
            if age_s < EXCLUSION_TIME_S or path_delta < EXCLUSION_PATH_M:
                continue

            pose_distance_m, pose_angle_diff_rad = pose_distance_and_angle(
                current_keyframe["pose"],
                target_keyframe["pose"],
            )

            if (
                pose_distance_m <= pose_gate_distance
                and pose_angle_diff_rad <= POSE_GATE_ANGLE_RAD
            ):
                local_eligible.append((pose_distance_m, target_keyframe, pose_angle_diff_rad, path_delta))

            if (
                current_keyframe["source_fraction"] >= ANCHOR_SOURCE_MIN_FRACTION
                and target_keyframe["target_fraction"] <= ANCHOR_TARGET_MAX_FRACTION
            ):
                descriptor_error, descriptor_shift_rad = descriptor_match(
                    current_keyframe["detection_descriptor"],
                    target_keyframe["detection_descriptor"],
                    detection_max_range_m,
                )
                if descriptor_error <= anchor_descriptor_threshold:
                    anchor_eligible.append(
                        (
                            descriptor_error,
                            target_keyframe,
                            descriptor_shift_rad,
                            pose_distance_m,
                            pose_angle_diff_rad,
                            path_delta,
                        )
                    )

        local_eligible.sort(key=lambda item: item[0])
        anchor_eligible.sort(key=lambda item: item[0])

        local_verified_records = []
        for pose_distance_m, target_keyframe, pose_angle_diff_rad, path_delta in local_eligible[:VERIFICATION_TOP_K]:
            init_pose = relative_pose_matrix(current_keyframe["pose"], target_keyframe["pose"])
            metrics = verify_candidate(
                current_keyframe["detection_scan_xy_local"],
                target_keyframe["detection_scan_xy_local"],
                target_cache[target_keyframe["keyframe_id"]],
                init_pose,
                correspondence_thresh_m=config.correspondence_thresh_m,
                icp_max_iter=config.icp_max_iter,
            )
            record = build_candidate_record(
                candidate_id=candidate_id,
                branch_type="local",
                current_keyframe=current_keyframe,
                target_keyframe=target_keyframe,
                pose_distance_m=pose_distance_m,
                pose_angle_diff_rad=pose_angle_diff_rad,
                path_gap_m=path_delta,
                metrics=metrics,
                descriptor_error=None,
                descriptor_shift_rad=None,
            )
            candidate_id += 1

            reason = rejection_reason(metrics, local_thresholds)
            if reason is not None:
                record["rejection_reason"] = reason
            else:
                local_verified_records.append(record)
            candidate_records.append(record)

        if local_verified_records:
            best_local = max(local_verified_records, key=lambda row: row["score"])
            best_local["accepted_before_dedup"] = True
            accepted_pre.append(best_local)
            for record in local_verified_records:
                if record is not best_local:
                    record["rejection_reason"] = "lower_score_than_best_current"

        anchor_verified_records = []
        for (
            descriptor_error,
            target_keyframe,
            descriptor_shift_rad,
            pose_distance_m,
            pose_angle_diff_rad,
            path_delta,
        ) in anchor_eligible[:VERIFICATION_TOP_K]:
            init_pose = build_anchor_init_pose(
                current_keyframe["detection_scan_xy_local"],
                target_keyframe["detection_scan_xy_local"],
                descriptor_shift_rad,
            )
            metrics = verify_candidate(
                current_keyframe["detection_scan_xy_local"],
                target_keyframe["detection_scan_xy_local"],
                target_cache[target_keyframe["keyframe_id"]],
                init_pose,
                correspondence_thresh_m=config.correspondence_thresh_m,
                icp_max_iter=config.icp_max_iter,
            )
            record = build_candidate_record(
                candidate_id=candidate_id,
                branch_type="anchor",
                current_keyframe=current_keyframe,
                target_keyframe=target_keyframe,
                pose_distance_m=pose_distance_m,
                pose_angle_diff_rad=pose_angle_diff_rad,
                path_gap_m=path_delta,
                metrics=metrics,
                descriptor_error=descriptor_error,
                descriptor_shift_rad=descriptor_shift_rad,
            )
            candidate_id += 1
            late_to_early_candidates_evaluated += 1

            reason = rejection_reason(metrics, anchor_thresholds)
            if reason is not None:
                record["rejection_reason"] = reason
            else:
                anchor_verified_records.append(record)
            candidate_records.append(record)

        if anchor_verified_records:
            best_anchor = max(anchor_verified_records, key=lambda row: row["score"])
            best_anchor["accepted_before_dedup"] = True
            accepted_pre.append(best_anchor)
            for record in anchor_verified_records:
                if record is not best_anchor:
                    record["rejection_reason"] = "lower_score_than_best_current"

    accepted_final = deduplicate_loops(accepted_pre)

    markers = load_json(payload["sequence_dir"] / "loop_markers.json")
    marker_matches, matched_loop_ids = evaluate_markers(accepted_final, markers, payload["processed_entries"][0]["pose"])
    extra_revisits = [record["accepted_loop_id"] for record in accepted_final if record["accepted_loop_id"] not in matched_loop_ids]

    accepted_serializable = [serializable_loop_record(record) for record in accepted_final]
    save_json(output_dir / "accepted_loops.json", accepted_serializable)
    save_csv(
        output_dir / "accepted_loops.csv",
        [flatten_accepted_record(record) for record in accepted_final],
        accepted_fieldnames(),
    )
    save_csv(
        output_dir / "candidate_scores.csv",
        [flatten_candidate_record(record) for record in candidate_records],
        candidate_fieldnames(),
    )

    create_trajectory_plot(output_dir, payload, markers, accepted_final)
    create_score_timeline(output_dir, payload, markers, accepted_final)

    trajectory_xy = np.array([pose_to_xytheta(entry["pose"])[:2] for entry in payload["processed_entries"]], dtype=float)
    accepted_anchor_loops = sum(1 for record in accepted_final if record["branch_type"] == "anchor")
    accepted_local_loops = sum(1 for record in accepted_final if record["branch_type"] == "local")
    summary = {
        "sequence": sequence_name,
        "label": payload["label"],
        "sequence_dir": str(payload["sequence_dir"]),
        "source_metadata": payload["source_metadata"],
        "config": config.__dict__,
        "detection_preprocess": payload["detection_preprocess"],
        "output_dir": str(output_dir),
        "processed_scan_count": len(payload["processed_entries"]),
        "keyframe_count": len(keyframes),
        "total_distance_m": round(compute_total_distance(trajectory_xy), 3),
        "final_start_to_end_drift_m": round(float(np.linalg.norm(trajectory_xy[-1] - trajectory_xy[0])), 3),
        "total_candidates_evaluated": len(candidate_records),
        "accepted_loops_before_dedup": len(accepted_pre),
        "accepted_loops_after_dedup": len(accepted_final),
        "accepted_anchor_loops": accepted_anchor_loops,
        "accepted_local_loops": accepted_local_loops,
        "late_to_early_candidates_evaluated": late_to_early_candidates_evaluated,
        "late_to_early_candidates_accepted": accepted_anchor_loops,
        "accepted_loop_ids_for_q3d": [record["accepted_loop_id"] for record in accepted_final],
        "marker_matches": marker_matches,
        "marker_hits": sum(1 for match in marker_matches if match["hit"]),
        "marker_misses": sum(1 for match in marker_matches if not match["hit"]),
        "extra_accepted_revisits": extra_revisits,
    }
    save_json(summary_path, summary)
    return summary


def write_rollup(root: Path, summaries: list[dict]) -> None:
    rows = []
    for summary in summaries:
        rows.append(
            {
                "sequence": summary["sequence"],
                "label": summary["label"],
                "processed_scan_count": summary["processed_scan_count"],
                "keyframe_count": summary["keyframe_count"],
                "total_candidates_evaluated": summary["total_candidates_evaluated"],
                "accepted_loops_before_dedup": summary["accepted_loops_before_dedup"],
                "accepted_loops_after_dedup": summary["accepted_loops_after_dedup"],
                "accepted_anchor_loops": summary["accepted_anchor_loops"],
                "accepted_local_loops": summary["accepted_local_loops"],
                "late_to_early_candidates_evaluated": summary["late_to_early_candidates_evaluated"],
                "late_to_early_candidates_accepted": summary["late_to_early_candidates_accepted"],
                "marker_hits": summary["marker_hits"],
                "marker_misses": summary["marker_misses"],
                "extra_accepted_revisits": len(summary["extra_accepted_revisits"]),
                "final_start_to_end_drift_m": summary["final_start_to_end_drift_m"],
                "output_dir": summary["output_dir"],
            }
        )

    save_csv(
        root / "all_detection_metrics.csv",
        rows,
        list(rows[0].keys()) if rows else [],
    )

    lines = [
        "# Q3c Results Summary",
        "",
        "This file summarises the loop-closure detection runs for the selected LiDAR sequences.",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"## {summary['label']}",
                "",
                f"- Sequence: `{summary['sequence']}`",
                f"- Accepted loops after dedup: `{summary['accepted_loops_after_dedup']}`",
                f"- Accepted anchor / local loops: `{summary['accepted_anchor_loops']}` / `{summary['accepted_local_loops']}`",
                f"- Late-to-early candidates evaluated / accepted: `{summary['late_to_early_candidates_evaluated']}` / `{summary['late_to_early_candidates_accepted']}`",
                f"- Marker hits / misses: `{summary['marker_hits']}` / `{summary['marker_misses']}`",
                f"- Output folder: `{summary['output_dir']}`",
                "",
                "| Marker | Hit | Matched Loop | Branch | Time Delta (s) | Score |",
                "| --- | --- | ---: | --- | ---: | ---: |",
            ]
        )
        for match in summary["marker_matches"]:
            loop_id = "" if match["matched_loop_id"] is None else match["matched_loop_id"]
            delta = "" if match["time_delta_s"] is None else f"{match['time_delta_s']:.3f}"
            score = "" if match["score"] is None else f"{match['score']:.6f}"
            branch = "" if match["branch_type"] is None else match["branch_type"]
            lines.append(
                f"| Loop {match['loop_number']} | {'yes' if match['hit'] else 'no'} | {loop_id} | {branch} | {delta} | {score} |"
            )
        lines.append("")

    (root / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sequence_names = SELECTED_SEQUENCES if args.sequence == "all" else [args.sequence]

    summaries = []
    for sequence_name in sequence_names:
        summaries.append(detect_sequence(sequence_name, output_root=args.output_root, force=args.force))

    write_rollup(args.output_root, summaries)
    print(json.dumps({"output_root": str(args.output_root), "sequences": sequence_names}, indent=2))


if __name__ == "__main__":
    main()
