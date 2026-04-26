#!/usr/bin/env python3
"""Run the full CW2 Q3b LiDAR experiment set and generate report-ready outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lidar_step1_pipeline import (
    BASE_DIR,
    PipelineConfig,
    SEQUENCE_REGISTRY,
    run_sequence,
)


Q3B_OUTPUT_ROOT = BASE_DIR / "output" / "lidar_q3b"
SELECTED_SEQUENCES = ["indoor_large_03", "indoor_small_01", "outdoor_02"]
BASELINE_CONFIG = PipelineConfig(
    max_range_mm=6000.0,
    beam_step=1,
    voxel_size_m=0.05,
    scan_stride=1,
)

EXPERIMENT_FAMILIES = {
    "max_range": {
        "title": "Maximum Range",
        "parameter": "max_range_mm",
        "description": "Compare a practical 6 m cap against the RP-Lidar A2M12 rated 12 m range.",
        "variants": [
            ("6 m", {"max_range_mm": 6000.0}),
            ("12 m", {"max_range_mm": 12000.0}),
        ],
    },
    "angular_resolution": {
        "title": "Angular Resolution",
        "parameter": "beam_step",
        "description": "Compare the full scan against beam downsampling to every 2nd and 3rd beam.",
        "variants": [
            ("full", {"beam_step": 1}),
            ("n=2", {"beam_step": 2}),
            ("n=3", {"beam_step": 3}),
        ],
    },
    "voxel_grid": {
        "title": "Voxel Grid Downsampling",
        "parameter": "voxel_size_m",
        "description": "Compare fine and coarse point-cloud map voxel filters.",
        "variants": [
            ("0.05 m", {"voxel_size_m": 0.05}),
            ("0.10 m", {"voxel_size_m": 0.10}),
        ],
    },
    "scan_rate": {
        "title": "Scan Rate",
        "parameter": "scan_stride",
        "description": "Compare all scans against 50% reduction and 2-of-3 skipped scans.",
        "variants": [
            ("all", {"scan_stride": 1}),
            ("1/2", {"scan_stride": 2}),
            ("1/3", {"scan_stride": 3}),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Q3b LiDAR experiment set and generate summary figures/tables."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Q3B_OUTPUT_ROOT,
        help="Root folder for Q3b outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run experiments even if a summary.json already exists.",
    )
    return parser.parse_args()


def config_with_override(**kwargs) -> PipelineConfig:
    base = BASELINE_CONFIG.__dict__.copy()
    base.update(kwargs)
    return PipelineConfig(**base)


def row_from_summary(family_key: str, setting_label: str, summary: dict) -> dict:
    loop_metrics = summary.get("loop_marker_metrics", [])
    loop1 = next((entry["closure_error_m"] for entry in loop_metrics if entry["loop_number"] == 1), None)
    loop2 = next((entry["closure_error_m"] for entry in loop_metrics if entry["loop_number"] == 2), None)

    return {
        "experiment_family": family_key,
        "setting_label": setting_label,
        "sequence": summary["sequence"],
        "label": summary["label"],
        "drift_m": summary["final_start_to_end_drift_m"],
        "loop1_error_m": loop1,
        "loop2_error_m": loop2,
        "total_distance_m": summary["total_distance_m"],
        "runtime_seconds": summary["runtime_seconds"],
        "keyframe_count": summary["keyframe_count"],
        "processed_scan_count": summary["processed_scan_count"],
        "output_dir": summary["output_dir"],
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def create_metric_plots(family_dir: Path, family_key: str, family_meta: dict, rows: list[dict]) -> None:
    sequence_names = SELECTED_SEQUENCES
    setting_labels = [label for label, _ in family_meta["variants"]]

    fig, axes = plt.subplots(1, len(sequence_names), figsize=(5 * len(sequence_names), 4), squeeze=False)
    for idx, sequence_name in enumerate(sequence_names):
        seq_rows = [row for row in rows if row["sequence"] == sequence_name]
        seq_rows.sort(key=lambda row: setting_labels.index(row["setting_label"]))
        ax = axes[0, idx]
        ax.bar(
            np.arange(len(seq_rows)),
            [row["drift_m"] for row in seq_rows],
            color="tab:blue",
            alpha=0.85,
        )
        ax.set_xticks(np.arange(len(seq_rows)), [row["setting_label"] for row in seq_rows], rotation=20)
        ax.set_ylabel("Final drift (m)")
        ax.set_title(SEQUENCE_REGISTRY[sequence_name]["label"])
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"{family_meta['title']} — Final Drift Comparison")
    fig.tight_layout()
    fig.savefig(family_dir / "drift_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(sequence_names), figsize=(5 * len(sequence_names), 4), squeeze=False)
    for idx, sequence_name in enumerate(sequence_names):
        seq_rows = [row for row in rows if row["sequence"] == sequence_name]
        seq_rows.sort(key=lambda row: setting_labels.index(row["setting_label"]))
        x = np.arange(len(seq_rows))
        ax = axes[0, idx]
        ax.plot(x, [row["loop1_error_m"] for row in seq_rows], marker="o", label="Loop 1")
        ax.plot(x, [row["loop2_error_m"] for row in seq_rows], marker="s", label="Loop 2")
        ax.set_xticks(x, [row["setting_label"] for row in seq_rows], rotation=20)
        ax.set_ylabel("Closure error (m)")
        ax.set_title(SEQUENCE_REGISTRY[sequence_name]["label"])
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(f"{family_meta['title']} — Loop Closure Metrics")
    fig.tight_layout()
    fig.savefig(family_dir / "loop_errors_comparison.png", dpi=180)
    plt.close(fig)


def create_image_grid(sequence_name: str, family_dir: Path, family_meta: dict, rows: list[dict]) -> None:
    seq_rows = [row for row in rows if row["sequence"] == sequence_name]
    setting_labels = [label for label, _ in family_meta["variants"]]
    seq_rows.sort(key=lambda row: setting_labels.index(row["setting_label"]))

    images = []
    for row in seq_rows:
        image_path = Path(row["output_dir"]) / "map_with_trajectory.png"
        images.append((row["setting_label"], Image.open(image_path)))

    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 5), squeeze=False)
    for idx, (setting_label, image) in enumerate(images):
        axes[0, idx].imshow(np.asarray(image))
        axes[0, idx].set_title(setting_label)
        axes[0, idx].axis("off")

    fig.suptitle(f"{family_meta['title']} — {SEQUENCE_REGISTRY[sequence_name]['label']}")
    fig.tight_layout()
    fig.savefig(family_dir / f"{sequence_name}_map_grid.png", dpi=180)
    plt.close(fig)

    for _, image in images:
        image.close()


def write_markdown_report(output_root: Path, all_rows: list[dict]) -> None:
    lines = [
        "# Q3b Results Summary",
        "",
        "This file summarises the automated Q3b LiDAR experiments for the selected sequences.",
        "",
        "Selected datasets:",
        "- `indoor_large_03`: Marshgate large-classroom",
        "- `indoor_small_01`: Marshgate small-room",
        "- `outdoor_02`: Building exterior",
        "",
        "Notes:",
        "- The maximum-range experiment uses `12 m` as the RP-Lidar A2M12 rated range.",
        "- Larger closure error does not always mean a bad map alone; in your case some collected second loops were not perfectly aligned to the first.",
        "",
    ]

    for family_key, family_meta in EXPERIMENT_FAMILIES.items():
        rows = [row for row in all_rows if row["experiment_family"] == family_key]
        lines.extend(
            [
                f"## {family_meta['title']}",
                "",
                family_meta["description"],
                "",
                "| Sequence | Setting | Drift (m) | Loop 1 (m) | Loop 2 (m) | Distance (m) | Keyframes |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for sequence_name in SELECTED_SEQUENCES:
            seq_rows = [row for row in rows if row["sequence"] == sequence_name]
            setting_labels = [label for label, _ in family_meta["variants"]]
            seq_rows.sort(key=lambda row: setting_labels.index(row["setting_label"]))
            for row in seq_rows:
                lines.append(
                    "| "
                    + f"{row['label']} | {row['setting_label']} | "
                    + f"{row['drift_m']:.3f} | {row['loop1_error_m']:.3f} | {row['loop2_error_m']:.3f} | "
                    + f"{row['total_distance_m']:.3f} | {row['keyframe_count']} |"
                )

            best_row = min(seq_rows, key=lambda row: row["drift_m"])
            lines.append("")
            lines.append(
                f"Best drift for **{SEQUENCE_REGISTRY[sequence_name]['label']}**: "
                f"`{best_row['setting_label']}` with `{best_row['drift_m']:.3f} m` final drift."
            )
            lines.append("")

    (output_root / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_summaries = []

    for family_key, family_meta in EXPERIMENT_FAMILIES.items():
        family_dir = args.output_root / family_key
        family_dir.mkdir(parents=True, exist_ok=True)

        family_rows = []
        family_summaries = []

        for setting_label, overrides in family_meta["variants"]:
            config = config_with_override(**overrides)
            output_tag = (
                f"{family_key}_"
                f"{setting_label.replace(' ', '').replace('/', '_').replace('=', '').replace('.', 'p')}"
            )

            for sequence_name in SELECTED_SEQUENCES:
                summary_path = family_dir / sequence_name / output_tag / "summary.json"
                if summary_path.exists() and not args.force:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                else:
                    summary = run_sequence(
                        sequence_name,
                        config,
                        output_root=family_dir,
                        output_tag=output_tag,
                        save_debug_plots=False,
                    )
                family_summaries.append(summary)
                row = row_from_summary(family_key, setting_label, summary)
                family_rows.append(row)
                all_rows.append(row)
                all_summaries.append(summary)

        save_csv(family_dir / "summary_metrics.csv", family_rows)
        save_json(family_dir / "summary_metrics.json", family_rows)
        create_metric_plots(family_dir, family_key, family_meta, family_rows)
        for sequence_name in SELECTED_SEQUENCES:
            create_image_grid(sequence_name, family_dir, family_meta, family_rows)

    save_csv(args.output_root / "all_metrics.csv", all_rows)
    save_json(args.output_root / "all_metrics.json", all_rows)
    write_markdown_report(args.output_root, all_rows)

    print(json.dumps({"output_root": str(args.output_root), "runs": len(all_summaries)}, indent=2))


if __name__ == "__main__":
    main()
