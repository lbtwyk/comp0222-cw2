#!/usr/bin/env python3
"""Prepare and finalize CW2 Q3d LiDAR pose-graph optimization outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

from lidar_step1_pipeline import (
    BASE_DIR,
    PipelineConfig,
    SEQUENCE_REGISTRY,
    build_occupancy_grid,
    build_sequence_payload,
    clone_processed_entries_with_poses,
    compute_total_distance,
    load_json,
    pose_to_xytheta,
    rebuild_point_cloud_map,
    xytheta_to_pose,
)


Q3C_OUTPUT_ROOT = BASE_DIR / "output" / "lidar_q3c"
Q3D_OUTPUT_ROOT = BASE_DIR / "output" / "lidar_q3d"
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

LOCAL_LOOP_FILTER_THRESHOLDS = {
    "indoor": {
        "score": 0.40,
        "translation_correction_m": 1.0,
        "rotation_correction_rad": 0.45,
    },
    "outdoor": {
        "score": 0.45,
        "translation_correction_m": 2.0,
        "rotation_correction_rad": 0.50,
    },
}

ANCHOR_LOOP_FILTER_THRESHOLDS = {
    "indoor": {
        "score": 0.50,
        "translation_correction_m": 1.20,
        "rotation_correction_rad": 0.60,
    },
    "outdoor": {
        "score": 0.60,
        "translation_correction_m": 1.00,
        "rotation_correction_rad": 0.65,
        "inlier_ratio": 0.75,
    },
}

ODOM_INFORMATION = np.diag([80.0, 80.0, 120.0])
LOOP_INFORMATION_BASE = np.diag([40.0, 40.0, 80.0])

MIN_ABSOLUTE_KEYFRAME_GAP = 20
MIN_KEYFRAME_GAP_FRACTION = 0.10
LONG_BASELINE_FRACTION = 0.25
TOUCHES_START_FRACTION = 0.20
TOUCHES_END_FRACTION = 0.70
NON_OVERLAP_WINDOW = 15
MIN_SELECTION_IMPROVEMENT_M = 0.10
NEAR_BEST_CLOSURE_TOLERANCE_M = 0.02
MAX_TRANSLATION_CORRECTION_WEIGHT = 0.25
TRANSLATION_SMOOTHNESS_WEIGHT = 10.0
ROTATION_SMOOTHNESS_WEIGHT = 2.0
VARIANT_ORDER = [
    "odom_only",
    "anchor_only",
    "anchor_plus_support",
    "top2_anchor_or_long",
]

CANONICAL_ARTIFACTS = [
    "graph_input.mat",
    "graph_optimized.mat",
    "loop_edges_used.json",
    "optimized_keyframes.csv",
    "optimized_scan_poses.csv",
    "trajectory_before_after.png",
    "occupancy_before_after.png",
    "map_with_trajectory_before_after.png",
    "occupancy_before.npz",
    "occupancy_after.npz",
    "point_cloud_before.npz",
    "point_cloud_after.npz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or finalize CW2 Q3d pose-graph optimization outputs."
    )
    parser.add_argument(
        "--sequence",
        default="all",
        choices=["all", *SELECTED_SEQUENCES],
        help="Sequence to process, or 'all' for the full Q3d batch.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["prepare", "finalize"],
        help="Q3d stage to run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Q3D_OUTPUT_ROOT,
        help="Root directory for Q3d outputs.",
    )
    parser.add_argument(
        "--q3c-root",
        type=Path,
        default=Q3C_OUTPUT_ROOT,
        help="Root directory for the prerequisite Q3c outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs even if the expected summary file already exists.",
    )
    return parser.parse_args()


def normalize_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def relative_pose_xytheta(source_pose: np.ndarray, target_pose: np.ndarray) -> list[float]:
    relative = np.linalg.inv(source_pose) @ target_pose
    return list(pose_to_xytheta(relative))


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def repeated_information_matrices(matrix: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((3, 3, 0), dtype=float)
    return np.repeat(matrix[:, :, np.newaxis], count, axis=2)


def get_environment(payload: dict) -> str:
    return str(payload["source_metadata"]["environment"])


def load_q3c_loops(sequence_name: str, q3c_root: Path) -> list[dict]:
    loops_path = q3c_root / sequence_name / "accepted_loops.json"
    if not loops_path.exists():
        raise FileNotFoundError(
            f"Missing Q3c loop file for {sequence_name}: {loops_path}"
        )
    return load_json(loops_path)


def build_odometry_edges(keyframes: list[dict]) -> list[dict]:
    edges = []
    for idx in range(len(keyframes) - 1):
        source = keyframes[idx]
        target = keyframes[idx + 1]
        edges.append(
            {
                "from_keyframe_id": int(source["keyframe_id"]),
                "to_keyframe_id": int(target["keyframe_id"]),
                "from_row": idx + 1,
                "to_row": idx + 2,
                "measurement_xytheta": [
                    float(value)
                    for value in relative_pose_xytheta(source["pose"], target["pose"])
                ],
                "information": ODOM_INFORMATION.tolist(),
                "delta_timestamp_s": float(target["timestamp"] - source["timestamp"]),
            }
        )
    return edges


def prepare_graph_input_mat(
    output_path: Path,
    payload: dict,
    odom_edges: list[dict],
    loop_edges: list[dict],
    *,
    variant_name: str,
) -> None:
    keyframe_poses_xytheta = np.asarray(
        [pose_to_xytheta(keyframe["pose"]) for keyframe in payload["keyframes"]],
        dtype=float,
    )

    mat_dict = {
        "sequence_name": payload["sequence"],
        "sequence_label": payload["label"],
        "variant_name": variant_name,
        "keyframe_ids": np.asarray(
            [keyframe["keyframe_id"] for keyframe in payload["keyframes"]],
            dtype=np.int32,
        ).reshape(-1, 1),
        "keyframe_scan_indices": np.asarray(
            [keyframe["scan_index"] for keyframe in payload["keyframes"]],
            dtype=np.int32,
        ).reshape(-1, 1),
        "keyframe_processed_indices": np.asarray(
            [keyframe["processed_index"] for keyframe in payload["keyframes"]],
            dtype=np.int32,
        ).reshape(-1, 1),
        "keyframe_timestamps": np.asarray(
            [keyframe["timestamp"] for keyframe in payload["keyframes"]],
            dtype=float,
        ).reshape(-1, 1),
        "keyframe_poses_xytheta": keyframe_poses_xytheta,
        "fixed_keyframe_row": np.array([[1]], dtype=np.int32),
        "odom_edge_vertex_rows": np.asarray(
            [[edge["from_row"], edge["to_row"]] for edge in odom_edges],
            dtype=np.int32,
        ).reshape(-1, 2),
        "odom_edge_measurements": np.asarray(
            [edge["measurement_xytheta"] for edge in odom_edges],
            dtype=float,
        ),
        "odom_edge_information_matrices": repeated_information_matrices(
            ODOM_INFORMATION, len(odom_edges)
        ),
        "loop_edge_vertex_rows": np.asarray(
            [[edge["from_row"], edge["to_row"]] for edge in loop_edges],
            dtype=np.int32,
        ).reshape(-1, 2),
        "loop_edge_measurements": np.asarray(
            [edge["measurement_xytheta"] for edge in loop_edges],
            dtype=float,
        ).reshape(-1, 3),
        "loop_edge_information_matrices": (
            np.stack(
                [np.asarray(edge["information"], dtype=float) for edge in loop_edges],
                axis=2,
            )
            if loop_edges
            else np.zeros((3, 3, 0), dtype=float)
        ),
        "loop_edge_ids": np.asarray(
            [edge["accepted_loop_id"] for edge in loop_edges],
            dtype=np.int32,
        ).reshape(-1, 1),
        "loop_edge_scores": np.asarray(
            [edge["score"] for edge in loop_edges],
            dtype=float,
        ).reshape(-1, 1),
        "loop_edge_inlier_ratios": np.asarray(
            [edge["inlier_ratio"] for edge in loop_edges],
            dtype=float,
        ).reshape(-1, 1),
    }
    savemat(output_path, mat_dict, do_compression=True)


def build_loop_candidate_pool(
    sequence_name: str,
    payload: dict,
    q3c_root: Path,
) -> tuple[list[dict], dict]:
    keyframes = payload["keyframes"]
    keyframe_count = len(keyframes)
    keyframe_denominator = max(1, keyframe_count - 1)
    min_keyframe_gap = max(
        MIN_ABSOLUTE_KEYFRAME_GAP,
        round(MIN_KEYFRAME_GAP_FRACTION * keyframe_denominator),
    )
    long_baseline_gap = max(
        MIN_ABSOLUTE_KEYFRAME_GAP,
        round(LONG_BASELINE_FRACTION * keyframe_denominator),
    )

    keyframes_by_id = {int(keyframe["keyframe_id"]): keyframe for keyframe in keyframes}
    keyframe_rows = {int(keyframe["keyframe_id"]): idx + 1 for idx, keyframe in enumerate(keyframes)}
    environment = get_environment(payload)
    local_thresholds = LOCAL_LOOP_FILTER_THRESHOLDS[environment]
    anchor_thresholds = ANCHOR_LOOP_FILTER_THRESHOLDS[environment]

    candidates = []
    for record in load_q3c_loops(sequence_name, q3c_root):
        source_keyframe_id = int(record["source_keyframe_id"])
        target_keyframe_id = int(record["target_keyframe_id"])
        if source_keyframe_id not in keyframes_by_id or target_keyframe_id not in keyframes_by_id:
            continue

        source_keyframe = keyframes_by_id[source_keyframe_id]
        target_keyframe = keyframes_by_id[target_keyframe_id]
        keyframe_gap = source_keyframe_id - target_keyframe_id
        path_gap_m = float(
            source_keyframe["cumulative_path_length_m"]
            - target_keyframe["cumulative_path_length_m"]
        )
        source_fraction = float(record.get("source_fraction", source_keyframe_id / keyframe_denominator))
        target_fraction = float(record.get("target_fraction", target_keyframe_id / keyframe_denominator))
        touches_start = target_fraction <= TOUCHES_START_FRACTION
        touches_end = source_fraction >= TOUCHES_END_FRACTION
        is_long_baseline = keyframe_gap >= long_baseline_gap
        branch_type = str(record.get("branch_type", "local"))
        thresholds = anchor_thresholds if branch_type == "anchor" else local_thresholds

        rejection_reasons = []
        if float(record["score"]) < thresholds["score"]:
            rejection_reasons.append("below_score")
        if float(record["translation_correction_m"]) > thresholds["translation_correction_m"]:
            rejection_reasons.append("above_translation_correction")
        if float(record["rotation_correction_rad"]) > thresholds["rotation_correction_rad"]:
            rejection_reasons.append("above_rotation_correction")
        if "inlier_ratio" in thresholds and float(record["inlier_ratio"]) < thresholds["inlier_ratio"]:
            rejection_reasons.append("below_inlier_ratio")
        if keyframe_gap < min_keyframe_gap:
            rejection_reasons.append("below_min_keyframe_gap")

        anchor_rank = float(
            float(record["score"])
            + 0.20 * source_fraction
            + 0.20 * (1.0 - target_fraction)
            + 0.10 * min(1.0, keyframe_gap / max(1.0, 0.5 * keyframe_count))
        )

        scale = max(0.5, float(record["inlier_ratio"]))
        information = (LOOP_INFORMATION_BASE * scale).astype(float)
        candidates.append(
            {
                "accepted_loop_id": int(record["accepted_loop_id"]),
                "from_keyframe_id": target_keyframe_id,
                "to_keyframe_id": source_keyframe_id,
                "from_row": keyframe_rows[target_keyframe_id],
                "to_row": keyframe_rows[source_keyframe_id],
                "measurement_xytheta": [float(value) for value in record["relative_pose_icp"]],
                "information": information.tolist(),
                "score": float(record["score"]),
                "mean_residual_m": float(record["mean_residual_m"]),
                "inlier_ratio": float(record["inlier_ratio"]),
                "translation_correction_m": float(record["translation_correction_m"]),
                "rotation_correction_rad": float(record["rotation_correction_rad"]),
                "branch_type": branch_type,
                "descriptor_error": (
                    None
                    if record.get("descriptor_error") is None
                    else float(record["descriptor_error"])
                ),
                "descriptor_shift_rad": (
                    None
                    if record.get("descriptor_shift_rad") is None
                    else float(record["descriptor_shift_rad"])
                ),
                "source_keyframe_id": source_keyframe_id,
                "target_keyframe_id": target_keyframe_id,
                "source_scan_index": int(record["source_scan_index"]),
                "target_scan_index": int(record["target_scan_index"]),
                "source_timestamp": float(record["source_timestamp"]),
                "target_timestamp": float(record["target_timestamp"]),
                "source_fraction": float(source_fraction),
                "target_fraction": float(target_fraction),
                "keyframe_gap": int(keyframe_gap),
                "path_gap_m": path_gap_m,
                "touches_start": bool(touches_start),
                "touches_end": bool(touches_end),
                "is_long_baseline": bool(is_long_baseline),
                "anchor_rank": anchor_rank,
                "eligible": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
                "used_in_variants": [],
            }
        )

    candidates.sort(
        key=lambda row: (row["anchor_rank"], row["score"], row["keyframe_gap"]),
        reverse=True,
    )
    metadata = {
        "keyframe_count": keyframe_count,
        "min_keyframe_gap": min_keyframe_gap,
        "long_baseline_gap": long_baseline_gap,
        "selection_rules": {
            "touches_start_fraction": TOUCHES_START_FRACTION,
            "touches_end_fraction": TOUCHES_END_FRACTION,
            "non_overlap_window": NON_OVERLAP_WINDOW,
        },
        "local_thresholds": local_thresholds,
        "anchor_thresholds": anchor_thresholds,
    }
    return candidates, metadata


def overlaps_selected(candidate: dict, selected: list[dict]) -> bool:
    return any(
        abs(candidate["source_keyframe_id"] - existing["source_keyframe_id"]) <= NON_OVERLAP_WINDOW
        or abs(candidate["target_keyframe_id"] - existing["target_keyframe_id"]) <= NON_OVERLAP_WINDOW
        for existing in selected
    )


def build_variant_specs(candidates: list[dict]) -> list[dict]:
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    anchor_candidates = [candidate for candidate in eligible if candidate["branch_type"] == "anchor"]
    local_long_candidates = [
        candidate
        for candidate in eligible
        if candidate["branch_type"] == "local" and candidate["is_long_baseline"]
    ]
    anchor_or_long_candidates = anchor_candidates + [
        candidate
        for candidate in local_long_candidates
        if candidate["accepted_loop_id"] not in {row["accepted_loop_id"] for row in anchor_candidates}
    ]

    variant_specs = []
    seen_signatures: set[tuple[int, ...]] = set()

    def register_variant(variant_name: str, loops: list[dict]) -> None:
        if not loops and variant_name != "odom_only":
            return
        signature = tuple(sorted(loop["accepted_loop_id"] for loop in loops))
        if not loops:
            signature = tuple()
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        variant_specs.append(
            {
                "variant_name": variant_name,
                "loops": loops,
                "loop_edge_ids": [loop["accepted_loop_id"] for loop in loops],
            }
        )

    register_variant("odom_only", [])

    anchor_loop = next(iter(anchor_candidates), None)

    if anchor_loop is not None:
        register_variant("anchor_only", [anchor_loop])

        support_loop = next(
            (
                candidate
                for candidate in anchor_or_long_candidates
                if candidate["accepted_loop_id"] != anchor_loop["accepted_loop_id"]
                and not overlaps_selected(candidate, [anchor_loop])
            ),
            None,
        )
        anchor_plus_support = [anchor_loop]
        if support_loop is not None:
            anchor_plus_support.append(support_loop)
        register_variant("anchor_plus_support", anchor_plus_support)

    top2_anchor_or_long = []
    for candidate in anchor_or_long_candidates:
        if overlaps_selected(candidate, top2_anchor_or_long):
            continue
        top2_anchor_or_long.append(candidate)
        if len(top2_anchor_or_long) == 2:
            break
    register_variant("top2_anchor_or_long", top2_anchor_or_long)

    ordered = []
    for variant_name in VARIANT_ORDER:
        ordered.extend(
            spec for spec in variant_specs if spec["variant_name"] == variant_name
        )
    return ordered


def annotate_candidate_selection(candidates: list[dict], variant_specs: list[dict]) -> None:
    variant_names_by_loop_id: dict[int, list[str]] = {}
    for spec in variant_specs:
        for loop in spec["loops"]:
            variant_names_by_loop_id.setdefault(loop["accepted_loop_id"], []).append(
                spec["variant_name"]
            )

    for candidate in candidates:
        used_in = variant_names_by_loop_id.get(candidate["accepted_loop_id"], [])
        candidate["used_in_variants"] = used_in
        if used_in:
            candidate["selection_note"] = "exported_to_variants"
        elif candidate["eligible"]:
            candidate["selection_note"] = "eligible_but_not_selected_by_variant_builder"
        else:
            candidate["selection_note"] = "; ".join(candidate["rejection_reasons"])


def serialize_candidate_for_manifest(candidate: dict) -> dict:
    return {
        "accepted_loop_id": candidate["accepted_loop_id"],
        "branch_type": candidate["branch_type"],
        "source_keyframe_id": candidate["source_keyframe_id"],
        "target_keyframe_id": candidate["target_keyframe_id"],
        "source_scan_index": candidate["source_scan_index"],
        "target_scan_index": candidate["target_scan_index"],
        "score": candidate["score"],
        "inlier_ratio": candidate["inlier_ratio"],
        "translation_correction_m": candidate["translation_correction_m"],
        "rotation_correction_rad": candidate["rotation_correction_rad"],
        "descriptor_error": candidate["descriptor_error"],
        "descriptor_shift_rad": candidate["descriptor_shift_rad"],
        "source_fraction": candidate["source_fraction"],
        "target_fraction": candidate["target_fraction"],
        "keyframe_gap": candidate["keyframe_gap"],
        "path_gap_m": candidate["path_gap_m"],
        "touches_start": candidate["touches_start"],
        "touches_end": candidate["touches_end"],
        "is_long_baseline": candidate["is_long_baseline"],
        "anchor_rank": candidate["anchor_rank"],
        "eligible": candidate["eligible"],
        "rejection_reasons": candidate["rejection_reasons"],
        "used_in_variants": candidate["used_in_variants"],
        "selection_note": candidate["selection_note"],
    }


def prepare_sequence(
    sequence_name: str,
    *,
    output_root: Path,
    q3c_root: Path,
    force: bool,
) -> dict:
    output_dir = output_root / sequence_name
    variants_dir = output_dir / "variants"
    summary_path = output_dir / "prepare_summary.json"
    manifest_path = output_dir / "variant_manifest.json"

    if summary_path.exists() and manifest_path.exists() and not force:
        return load_json(summary_path)

    if force and variants_dir.exists():
        shutil.rmtree(variants_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)

    config = BASE_CONFIGS[sequence_name]
    payload = build_sequence_payload(sequence_name, config, include_markers=False)
    odom_edges = build_odometry_edges(payload["keyframes"])
    candidates, pool_metadata = build_loop_candidate_pool(sequence_name, payload, q3c_root)
    variant_specs = build_variant_specs(candidates)
    annotate_candidate_selection(candidates, variant_specs)

    variant_summaries = []
    for spec in variant_specs:
        variant_dir = variants_dir / spec["variant_name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        prepare_graph_input_mat(
            variant_dir / "graph_input.mat",
            payload,
            odom_edges,
            spec["loops"],
            variant_name=spec["variant_name"],
        )
        save_json(variant_dir / "loop_edges_used.json", spec["loops"])

        variant_summary = {
            "sequence": sequence_name,
            "label": payload["label"],
            "variant_name": spec["variant_name"],
            "config": config.__dict__,
            "environment": get_environment(payload),
            "output_dir": str(variant_dir),
            "graph_input_path": str(variant_dir / "graph_input.mat"),
            "keyframe_count": len(payload["keyframes"]),
            "processed_scan_count": len(payload["processed_entries"]),
            "odometry_edge_count": len(odom_edges),
            "loop_edge_count": len(spec["loops"]),
            "odometry_only": len(spec["loops"]) == 0,
            "loop_edge_ids": spec["loop_edge_ids"],
            "q3c_source_dir": str(q3c_root / sequence_name),
        }
        save_json(variant_dir / "prepare_summary.json", variant_summary)
        variant_summaries.append(variant_summary)

    manifest = {
        "sequence": sequence_name,
        "label": payload["label"],
        "config": config.__dict__,
        "candidate_pool_count": len(candidates),
        "eligible_candidate_count": sum(1 for candidate in candidates if candidate["eligible"]),
        "pool_metadata": pool_metadata,
        "variants": [
            {
                "variant_name": summary["variant_name"],
                "loop_edge_ids": summary["loop_edge_ids"],
                "loop_edge_count": summary["loop_edge_count"],
                "odometry_only": summary["odometry_only"],
                "output_dir": summary["output_dir"],
            }
            for summary in variant_summaries
        ],
        "candidates": [serialize_candidate_for_manifest(candidate) for candidate in candidates],
    }
    save_json(manifest_path, manifest)

    summary = {
        "sequence": sequence_name,
        "label": payload["label"],
        "config": config.__dict__,
        "environment": get_environment(payload),
        "output_dir": str(output_dir),
        "keyframe_count": len(payload["keyframes"]),
        "processed_scan_count": len(payload["processed_entries"]),
        "odometry_edge_count": len(odom_edges),
        "candidate_pool_count": len(candidates),
        "eligible_candidate_count": sum(1 for candidate in candidates if candidate["eligible"]),
        "variant_names_exported": [summary["variant_name"] for summary in variant_summaries],
        "variant_count": len(variant_summaries),
        "q3c_source_dir": str(q3c_root / sequence_name),
    }
    save_json(summary_path, summary)
    return summary


def as_pose_rows(data) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    if array.ndim == 1:
        if array.shape[0] != 3:
            raise ValueError(
                f"Expected a single xytheta row with 3 elements, got shape {array.shape}."
            )
        return array.reshape(1, 3)
    if array.shape[1] != 3:
        raise ValueError(f"Expected pose rows with 3 columns, got shape {array.shape}.")
    return array


def load_optimized_keyframes(path: Path) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MATLAB optimization output: {path}\n"
            "Run `setup` in MATLAB and then `q3d.run_all('<absolute output/lidar_q3d path>')` first."
        )

    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "optimized_keyframe_poses_xytheta" not in data:
        raise KeyError(
            f"{path} does not contain 'optimized_keyframe_poses_xytheta'."
        )

    metadata = {}
    for key in ("sequence_name", "sequence_label", "chi2_initial", "chi2_final", "iterations"):
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, np.ndarray) and value.shape == ():
            metadata[key] = float(value)
        elif np.isscalar(value):
            if key in {"sequence_name", "sequence_label"}:
                metadata[key] = str(value)
            else:
                metadata[key] = float(value)
        else:
            metadata[key] = value

    return as_pose_rows(data["optimized_keyframe_poses_xytheta"]), metadata


def keyframe_corrections(
    original_keyframe_poses: np.ndarray,
    optimized_keyframe_poses: np.ndarray,
) -> np.ndarray:
    corrections = np.zeros((len(original_keyframe_poses), 3), dtype=float)
    for idx, (orig_xytheta, opt_xytheta) in enumerate(
        zip(original_keyframe_poses, optimized_keyframe_poses, strict=True)
    ):
        orig_pose = xytheta_to_pose(*orig_xytheta)
        opt_pose = xytheta_to_pose(*opt_xytheta)
        correction = opt_pose @ np.linalg.inv(orig_pose)
        corrections[idx] = np.asarray(pose_to_xytheta(correction), dtype=float)
    return corrections


def interpolate_scan_poses(
    processed_entries: list[dict],
    keyframes: list[dict],
    optimized_keyframe_poses: np.ndarray,
) -> list[np.ndarray]:
    original_keyframe_poses = np.asarray(
        [pose_to_xytheta(keyframe["pose"]) for keyframe in keyframes],
        dtype=float,
    )
    corrections = keyframe_corrections(original_keyframe_poses, optimized_keyframe_poses)
    keyframe_processed_indices = np.asarray(
        [keyframe["processed_index"] for keyframe in keyframes],
        dtype=int,
    )

    optimized_scan_poses = []
    for entry in processed_entries:
        processed_index = int(entry["processed_index"])
        if processed_index <= keyframe_processed_indices[0]:
            correction_xytheta = corrections[0]
        elif processed_index >= keyframe_processed_indices[-1]:
            correction_xytheta = corrections[-1]
        else:
            right = int(np.searchsorted(keyframe_processed_indices, processed_index, side="right"))
            left = right - 1
            left_index = keyframe_processed_indices[left]
            right_index = keyframe_processed_indices[right]
            alpha = (processed_index - left_index) / float(right_index - left_index)

            left_correction = corrections[left]
            right_correction = corrections[right]
            interp_theta_delta = normalize_angle(right_correction[2] - left_correction[2])
            correction_xytheta = np.array(
                [
                    (1.0 - alpha) * left_correction[0] + alpha * right_correction[0],
                    (1.0 - alpha) * left_correction[1] + alpha * right_correction[1],
                    normalize_angle(left_correction[2] + alpha * interp_theta_delta),
                ],
                dtype=float,
            )

        correction_pose = xytheta_to_pose(*correction_xytheta)
        optimized_scan_poses.append(correction_pose @ entry["pose"])

    return optimized_scan_poses


def correction_metrics(
    keyframes: list[dict],
    optimized_keyframe_poses: np.ndarray,
) -> dict:
    original_keyframe_poses = np.asarray(
        [pose_to_xytheta(keyframe["pose"]) for keyframe in keyframes],
        dtype=float,
    )
    corrections = keyframe_corrections(original_keyframe_poses, optimized_keyframe_poses)
    translation_magnitudes = np.linalg.norm(corrections[:, :2], axis=1)

    if len(corrections) > 1:
        translation_steps = np.linalg.norm(
            np.diff(corrections[:, :2], axis=0),
            axis=1,
        )
        rotation_steps = np.asarray(
            [
                abs(normalize_angle(current - previous))
                for previous, current in zip(
                    corrections[:-1, 2],
                    corrections[1:, 2],
                    strict=True,
                )
            ],
            dtype=float,
        )
    else:
        translation_steps = np.zeros(1, dtype=float)
        rotation_steps = np.zeros(1, dtype=float)

    mean_translation = float(np.mean(translation_magnitudes))
    max_translation = float(np.max(translation_magnitudes))
    mean_translation_step = float(np.mean(translation_steps))
    mean_rotation_step = float(np.mean(rotation_steps))
    deformation_score = float(
        mean_translation
        + MAX_TRANSLATION_CORRECTION_WEIGHT * max_translation
        + TRANSLATION_SMOOTHNESS_WEIGHT * mean_translation_step
        + ROTATION_SMOOTHNESS_WEIGHT * mean_rotation_step
    )

    return {
        "mean_keyframe_translation_correction_m": mean_translation,
        "max_keyframe_translation_correction_m": max_translation,
        "mean_keyframe_translation_step_m": mean_translation_step,
        "mean_keyframe_rotation_step_rad": mean_rotation_step,
        "deformation_score": deformation_score,
    }


def occupancy_extent(occupancy: dict) -> list[float]:
    return [
        float(occupancy["bounds_min_xy"][0]),
        float(occupancy["bounds_max_xy"][0]),
        float(occupancy["bounds_min_xy"][1]),
        float(occupancy["bounds_max_xy"][1]),
    ]


def create_before_after_plots(
    output_dir: Path,
    label: str,
    original_entries: list[dict],
    optimized_entries: list[dict],
    occupancy_before: dict,
    occupancy_after: dict,
) -> None:
    original_xy = np.asarray(
        [pose_to_xytheta(entry["pose"])[:2] for entry in original_entries],
        dtype=float,
    )
    optimized_xy = np.asarray(
        [pose_to_xytheta(entry["pose"])[:2] for entry in optimized_entries],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, xy, title in (
        (axes[0], original_xy, "Before Optimization"),
        (axes[1], optimized_xy, "After Optimization"),
    ):
        ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", linewidth=1.4)
        ax.scatter(xy[0, 0], xy[0, 1], color="black", marker="s", s=40, label="Start")
        ax.scatter(xy[-1, 0], xy[-1, 1], color="tab:red", s=28, label="End")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_title(title)
    axes[0].legend(loc="best")
    fig.suptitle(f"Trajectory Before/After\n{label}")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_before_after.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, occupancy, title in (
        (axes[0], occupancy_before, "Before Optimization"),
        (axes[1], occupancy_after, "After Optimization"),
    ):
        ax.imshow(
            occupancy["probabilities"],
            origin="lower",
            cmap="gray",
            extent=occupancy_extent(occupancy),
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.suptitle(f"Occupancy Grid Before/After\n{label}")
    fig.tight_layout()
    fig.savefig(output_dir / "occupancy_before_after.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, occupancy, xy, title in (
        (axes[0], occupancy_before, original_xy, "Before Optimization"),
        (axes[1], occupancy_after, optimized_xy, "After Optimization"),
    ):
        ax.imshow(
            occupancy["probabilities"],
            origin="lower",
            cmap="gray",
            extent=occupancy_extent(occupancy),
            vmin=0.0,
            vmax=1.0,
        )
        ax.plot(xy[:, 0], xy[:, 1], color="tab:cyan", linewidth=1.3)
        ax.scatter(xy[0, 0], xy[0, 1], color="yellow", marker="s", s=35)
        ax.scatter(xy[-1, 0], xy[-1, 1], color="tab:red", s=25)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.suptitle(f"Map With Trajectory Before/After\n{label}")
    fig.tight_layout()
    fig.savefig(output_dir / "map_with_trajectory_before_after.png", dpi=180)
    plt.close(fig)


def save_optimized_keyframes_csv(
    path: Path,
    keyframes: list[dict],
    optimized_keyframe_poses: np.ndarray,
) -> None:
    original_keyframe_poses = np.asarray(
        [pose_to_xytheta(keyframe["pose"]) for keyframe in keyframes],
        dtype=float,
    )
    corrections = keyframe_corrections(original_keyframe_poses, optimized_keyframe_poses)

    rows = []
    for keyframe, orig_xytheta, opt_xytheta, correction in zip(
        keyframes,
        original_keyframe_poses,
        optimized_keyframe_poses,
        corrections,
        strict=True,
    ):
        rows.append(
            {
                "keyframe_id": int(keyframe["keyframe_id"]),
                "processed_index": int(keyframe["processed_index"]),
                "scan_index": int(keyframe["scan_index"]),
                "timestamp": float(keyframe["timestamp"]),
                "original_x": float(orig_xytheta[0]),
                "original_y": float(orig_xytheta[1]),
                "original_theta": float(orig_xytheta[2]),
                "optimized_x": float(opt_xytheta[0]),
                "optimized_y": float(opt_xytheta[1]),
                "optimized_theta": float(opt_xytheta[2]),
                "correction_dx": float(correction[0]),
                "correction_dy": float(correction[1]),
                "correction_dtheta": float(correction[2]),
            }
        )

    save_csv(path, rows, list(rows[0].keys()) if rows else [])


def save_optimized_scan_poses_csv(
    path: Path,
    original_entries: list[dict],
    optimized_entries: list[dict],
) -> None:
    rows = []
    for original, optimized in zip(original_entries, optimized_entries, strict=True):
        orig_xytheta = pose_to_xytheta(original["pose"])
        opt_xytheta = pose_to_xytheta(optimized["pose"])
        rows.append(
            {
                "processed_index": int(original["processed_index"]),
                "scan_index": int(original["scan_index"]),
                "timestamp": float(original["timestamp"]),
                "original_x": float(orig_xytheta[0]),
                "original_y": float(orig_xytheta[1]),
                "original_theta": float(orig_xytheta[2]),
                "optimized_x": float(opt_xytheta[0]),
                "optimized_y": float(opt_xytheta[1]),
                "optimized_theta": float(opt_xytheta[2]),
            }
        )

    save_csv(path, rows, list(rows[0].keys()) if rows else [])


def finalize_variant(
    payload: dict,
    config: PipelineConfig,
    variant_dir: Path,
    *,
    occupancy_before: dict,
    before_map: np.ndarray,
    original_entries: list[dict],
    original_xy: np.ndarray,
    total_distance_before_m: float,
    closure_error_before_m: float,
    force: bool,
) -> dict:
    summary_path = variant_dir / "q3d_summary.json"
    if summary_path.exists() and not force:
        return load_json(summary_path)

    prepare_summary = load_json(variant_dir / "prepare_summary.json")
    optimized_keyframe_poses, optimization_meta = load_optimized_keyframes(
        variant_dir / "graph_optimized.mat"
    )
    if optimized_keyframe_poses.shape[0] != len(payload["keyframes"]):
        raise ValueError(
            f"Optimized keyframe count mismatch for {payload['sequence']} / {prepare_summary['variant_name']}: "
            f"expected {len(payload['keyframes'])}, got {optimized_keyframe_poses.shape[0]}."
        )

    optimized_scan_poses = interpolate_scan_poses(
        payload["processed_entries"],
        payload["keyframes"],
        optimized_keyframe_poses,
    )
    correction_stats = correction_metrics(payload["keyframes"], optimized_keyframe_poses)
    optimized_entries = clone_processed_entries_with_poses(
        payload["processed_entries"],
        optimized_scan_poses,
    )
    optimized_xy = np.asarray(
        [pose_to_xytheta(entry["pose"])[:2] for entry in optimized_entries],
        dtype=float,
    )

    after_map = rebuild_point_cloud_map(
        optimized_entries,
        voxel_size_m=config.voxel_size_m,
    )
    occupancy_after = build_occupancy_grid(
        optimized_entries,
        cell_size_m=config.occupancy_cell_size_m,
        padding_m=config.occupancy_padding_m,
    )

    np.savez_compressed(
        variant_dir / "occupancy_before.npz",
        probabilities=occupancy_before["probabilities"],
        log_odds=occupancy_before["log_odds"],
        origin_xy=occupancy_before["origin_xy"],
        bounds_min_xy=occupancy_before["bounds_min_xy"],
        bounds_max_xy=occupancy_before["bounds_max_xy"],
        cell_size_m=occupancy_before["cell_size_m"],
    )
    np.savez_compressed(
        variant_dir / "occupancy_after.npz",
        probabilities=occupancy_after["probabilities"],
        log_odds=occupancy_after["log_odds"],
        origin_xy=occupancy_after["origin_xy"],
        bounds_min_xy=occupancy_after["bounds_min_xy"],
        bounds_max_xy=occupancy_after["bounds_max_xy"],
        cell_size_m=occupancy_after["cell_size_m"],
    )
    np.savez_compressed(
        variant_dir / "point_cloud_before.npz",
        points=before_map,
    )
    np.savez_compressed(
        variant_dir / "point_cloud_after.npz",
        points=after_map,
    )

    save_optimized_keyframes_csv(
        variant_dir / "optimized_keyframes.csv",
        payload["keyframes"],
        optimized_keyframe_poses,
    )
    save_optimized_scan_poses_csv(
        variant_dir / "optimized_scan_poses.csv",
        payload["processed_entries"],
        optimized_entries,
    )
    create_before_after_plots(
        variant_dir,
        f"{payload['label']} — {prepare_summary['variant_name']}",
        original_entries,
        optimized_entries,
        occupancy_before,
        occupancy_after,
    )

    total_distance_after_m = float(compute_total_distance(optimized_xy))
    closure_error_after_m = float(np.linalg.norm(optimized_xy[-1] - optimized_xy[0]))

    summary = {
        "sequence": payload["sequence"],
        "label": payload["label"],
        "variant_name": prepare_summary["variant_name"],
        "config": config.__dict__,
        "output_dir": str(variant_dir),
        "graph_input_path": str(variant_dir / "graph_input.mat"),
        "graph_optimized_path": str(variant_dir / "graph_optimized.mat"),
        "processed_scan_count": len(payload["processed_entries"]),
        "keyframe_count": len(payload["keyframes"]),
        "loop_edge_count": int(prepare_summary["loop_edge_count"]),
        "loop_edge_ids_used": list(prepare_summary["loop_edge_ids"]),
        "odometry_only": bool(prepare_summary["odometry_only"]),
        "total_distance_before_m": total_distance_before_m,
        "total_distance_after_m": total_distance_after_m,
        "total_distance_change_abs_m": abs(total_distance_after_m - total_distance_before_m),
        "closure_error_before_m": closure_error_before_m,
        "closure_error_after_m": closure_error_after_m,
        "closure_error_improvement_m": closure_error_before_m - closure_error_after_m,
        "optimization_meta": optimization_meta,
        **correction_stats,
    }
    if summary["odometry_only"]:
        summary["note"] = "Odometry-only baseline variant."
    else:
        summary["note"] = "Pose-graph optimization used filtered automatic Q3c loop edges only."

    save_json(summary_path, summary)
    return summary


def select_best_variant(variant_summaries: list[dict]) -> tuple[dict, str]:
    odom_only = next(
        (summary for summary in variant_summaries if summary["variant_name"] == "odom_only"),
        None,
    )
    improving_pose_graph = [
        summary
        for summary in variant_summaries
        if not summary["odometry_only"]
        and summary["closure_error_after_m"]
        <= summary["closure_error_before_m"] - MIN_SELECTION_IMPROVEMENT_M
    ]

    if improving_pose_graph:
        best_closure_after = min(
            summary["closure_error_after_m"] for summary in improving_pose_graph
        )
        near_best = [
            summary
            for summary in improving_pose_graph
            if summary["closure_error_after_m"]
            <= best_closure_after + NEAR_BEST_CLOSURE_TOLERANCE_M
        ]
        chosen = min(
            near_best,
            key=lambda summary: (
                summary["deformation_score"],
                summary["closure_error_after_m"],
                summary["loop_edge_count"],
                summary["variant_name"],
            ),
        )
        reason = (
            f"{chosen['variant_name']} was selected because it was within "
            f"{NEAR_BEST_CLOSURE_TOLERANCE_M:.2f} m of the best closure result and had the lowest "
            "deformation score, favoring smaller and smoother keyframe corrections."
        )
        return chosen, reason

    if odom_only is not None:
        reason = (
            f"No pose-graph variant improved closure by at least "
            f"{MIN_SELECTION_IMPROVEMENT_M:.2f} m, so odom_only was selected."
        )
        return odom_only, reason

    chosen = min(
        variant_summaries,
        key=lambda summary: (
            summary["closure_error_after_m"],
            summary["total_distance_change_abs_m"],
            summary["loop_edge_count"],
            summary["variant_name"],
        ),
    )
    reason = "No odom_only variant was available, so the lowest closure error result was selected."
    return chosen, reason


def promote_variant_outputs(sequence_output_dir: Path, chosen_variant: dict) -> None:
    variant_dir = Path(chosen_variant["output_dir"])
    for artifact_name in CANONICAL_ARTIFACTS:
        source_path = variant_dir / artifact_name
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing artifact in selected variant {chosen_variant['variant_name']}: {source_path}"
            )
        shutil.copy2(source_path, sequence_output_dir / artifact_name)


def finalize_sequence(
    sequence_name: str,
    *,
    output_root: Path,
    force: bool,
) -> dict:
    output_dir = output_root / sequence_name
    summary_path = output_dir / "q3d_summary.json"
    selected_variant_path = output_dir / "selected_variant.json"
    if summary_path.exists() and selected_variant_path.exists() and not force:
        return load_json(summary_path)

    prepare_summary_path = output_dir / "prepare_summary.json"
    variants_dir = output_dir / "variants"
    if not prepare_summary_path.exists() or not variants_dir.exists():
        raise FileNotFoundError(
            f"Missing prepare outputs for {sequence_name}: {prepare_summary_path}\n"
            f"Run `python3 {BASE_DIR / 'run_lidar_q3d.py'} --stage prepare --sequence {sequence_name}` first."
        )

    config = BASE_CONFIGS[sequence_name]
    payload = build_sequence_payload(sequence_name, config, include_markers=False)
    original_entries = payload["processed_entries"]
    original_xy = np.asarray(
        [pose_to_xytheta(entry["pose"])[:2] for entry in original_entries],
        dtype=float,
    )
    total_distance_before_m = float(compute_total_distance(original_xy))
    closure_error_before_m = float(np.linalg.norm(original_xy[-1] - original_xy[0]))
    before_map = rebuild_point_cloud_map(
        original_entries,
        voxel_size_m=config.voxel_size_m,
    )
    occupancy_before = build_occupancy_grid(
        original_entries,
        cell_size_m=config.occupancy_cell_size_m,
        padding_m=config.occupancy_padding_m,
    )

    variant_summaries = []
    for variant_dir in sorted(path for path in variants_dir.iterdir() if path.is_dir()):
        if not (variant_dir / "prepare_summary.json").exists():
            continue
        if not (variant_dir / "graph_optimized.mat").exists():
            continue
        variant_summaries.append(
            finalize_variant(
                payload,
                config,
                variant_dir,
                occupancy_before=occupancy_before,
                before_map=before_map,
                original_entries=original_entries,
                original_xy=original_xy,
                total_distance_before_m=total_distance_before_m,
                closure_error_before_m=closure_error_before_m,
                force=force,
            )
        )

    if not variant_summaries:
        raise FileNotFoundError(
            f"No optimized variant outputs found for {sequence_name} under {variants_dir}.\n"
            "Run `setup` in MATLAB and then "
            f"`q3d.run_all('{output_root.resolve()}')` before finalizing."
        )

    chosen_variant, selection_reason = select_best_variant(variant_summaries)

    comparison_rows = []
    for summary in sorted(
        variant_summaries,
        key=lambda row: (
            row["variant_name"] != "odom_only",
            row["closure_error_after_m"],
            row["total_distance_change_abs_m"],
        ),
    ):
        comparison_rows.append(
            {
                "variant_name": summary["variant_name"],
                "selected": summary["variant_name"] == chosen_variant["variant_name"],
                "loop_edge_count": summary["loop_edge_count"],
                "loop_edge_ids_used": ",".join(str(loop_id) for loop_id in summary["loop_edge_ids_used"]),
                "odometry_only": summary["odometry_only"],
                "closure_error_before_m": round(summary["closure_error_before_m"], 6),
                "closure_error_after_m": round(summary["closure_error_after_m"], 6),
                "closure_error_improvement_m": round(summary["closure_error_improvement_m"], 6),
                "total_distance_change_abs_m": round(summary["total_distance_change_abs_m"], 6),
                "mean_keyframe_translation_correction_m": round(
                    summary["mean_keyframe_translation_correction_m"], 6
                ),
                "max_keyframe_translation_correction_m": round(
                    summary["max_keyframe_translation_correction_m"], 6
                ),
                "mean_keyframe_translation_step_m": round(
                    summary["mean_keyframe_translation_step_m"], 6
                ),
                "mean_keyframe_rotation_step_rad": round(
                    summary["mean_keyframe_rotation_step_rad"], 6
                ),
                "deformation_score": round(summary["deformation_score"], 6),
                "output_dir": summary["output_dir"],
            }
        )

    save_csv(
        output_dir / "variant_comparison.csv",
        comparison_rows,
        list(comparison_rows[0].keys()),
    )
    save_json(output_dir / "variant_comparison.json", comparison_rows)

    selected_variant = {
        "sequence": sequence_name,
        "label": payload["label"],
        "selected_variant": chosen_variant["variant_name"],
        "selection_reason": selection_reason,
        "variant_count_evaluated": len(variant_summaries),
        "loop_edge_ids_used": chosen_variant["loop_edge_ids_used"],
        "chosen_metrics": {
            "closure_error_before_m": chosen_variant["closure_error_before_m"],
            "closure_error_after_m": chosen_variant["closure_error_after_m"],
            "closure_error_improvement_m": chosen_variant["closure_error_improvement_m"],
            "total_distance_change_abs_m": chosen_variant["total_distance_change_abs_m"],
            "loop_edge_count": chosen_variant["loop_edge_count"],
            "deformation_score": chosen_variant["deformation_score"],
        },
    }
    save_json(selected_variant_path, selected_variant)

    promote_variant_outputs(output_dir, chosen_variant)

    canonical_summary = dict(chosen_variant)
    canonical_summary.update(
        {
            "selected_variant": chosen_variant["variant_name"],
            "variant_count_evaluated": len(variant_summaries),
            "selection_reason": selection_reason,
            "loop_edge_ids_used": chosen_variant["loop_edge_ids_used"],
            "output_dir": str(output_dir),
            "selected_variant_output_dir": chosen_variant["output_dir"],
            "graph_input_path": str(output_dir / "graph_input.mat"),
            "graph_optimized_path": str(output_dir / "graph_optimized.mat"),
        }
    )
    save_json(summary_path, canonical_summary)
    return canonical_summary


def write_rollup(output_root: Path, summaries: list[dict]) -> None:
    if not summaries:
        return

    rows = []
    for summary in summaries:
        rows.append(
            {
                "sequence": summary["sequence"],
                "label": summary["label"],
                "selected_variant": summary["selected_variant"],
                "processed_scan_count": summary["processed_scan_count"],
                "keyframe_count": summary["keyframe_count"],
                "loop_edge_count": summary["loop_edge_count"],
                "odometry_only": summary["odometry_only"],
                "closure_error_before_m": round(summary["closure_error_before_m"], 6),
                "closure_error_after_m": round(summary["closure_error_after_m"], 6),
                "closure_error_improvement_m": round(summary["closure_error_improvement_m"], 6),
                "output_dir": summary["output_dir"],
            }
        )

    save_csv(
        output_root / "all_optimization_metrics.csv",
        rows,
        list(rows[0].keys()),
    )

    lines = [
        "# Q3d Results Summary",
        "",
        "This file summarises the keyframe pose-graph optimization outputs for the selected LiDAR sequences.",
        "",
        "Selected datasets:",
        "- `indoor_large_03`: Marshgate large-classroom",
        "- `indoor_small_01`: Marshgate small-room",
        "- `outdoor_02`: Building exterior",
        "",
        "| Sequence | Selected Variant | Loop Edges | Mode | Closure Before (m) | Closure After (m) | Improvement (m) |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]

    for summary in summaries:
        mode = "odometry-only" if summary["odometry_only"] else "pose-graph"
        lines.append(
            "| "
            + f"{summary['label']} | {summary['selected_variant']} | {summary['loop_edge_count']} | {mode} | "
            + f"{summary['closure_error_before_m']:.3f} | {summary['closure_error_after_m']:.3f} | "
            + f"{summary['closure_error_improvement_m']:.3f} |"
        )
        lines.append("")
        lines.append(f"Selection note for **{summary['label']}**: {summary['selection_reason']}")
        lines.append("")

    (output_root / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sequence_names = SELECTED_SEQUENCES if args.sequence == "all" else [args.sequence]

    summaries = []
    for sequence_name in sequence_names:
        if args.stage == "prepare":
            summary = prepare_sequence(
                sequence_name,
                output_root=args.output_root,
                q3c_root=args.q3c_root,
                force=args.force,
            )
        else:
            summary = finalize_sequence(
                sequence_name,
                output_root=args.output_root,
                force=args.force,
            )
        summaries.append(summary)

    if args.stage == "finalize":
        write_rollup(args.output_root, summaries)

    print(
        json.dumps(
            {
                "stage": args.stage,
                "output_root": str(args.output_root),
                "runs": len(summaries),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
