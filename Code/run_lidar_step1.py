#!/usr/bin/env python3
"""CLI entrypoint for the CW2 Q3 Step 1 LiDAR baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lidar_step1_pipeline import (
    DEFAULT_OUTPUT_ROOT,
    PipelineConfig,
    SEQUENCE_REGISTRY,
    build_output_tag,
    run_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CW2 Q3 step-1 LiDAR baseline on the selected replay sequences."
    )
    parser.add_argument(
        "--sequence",
        default="all",
        choices=["all", *SEQUENCE_REGISTRY.keys()],
        help="Sequence to run. Use 'all' to process all selected coursework sequences.",
    )
    parser.add_argument(
        "--max-range-mm",
        type=float,
        default=6000.0,
        help="Maximum accepted LiDAR range in millimetres.",
    )
    parser.add_argument(
        "--beam-step",
        type=int,
        default=1,
        help="Keep every n-th valid beam after filtering.",
    )
    parser.add_argument(
        "--voxel-size-m",
        type=float,
        default=0.05,
        help="Voxel size used to downsample keyframe world points.",
    )
    parser.add_argument(
        "--scan-stride",
        type=int,
        default=1,
        help="Keep every n-th scan from the replay sequence.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Folder tag for outputs. If omitted, a deterministic tag is generated from the config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for generated outputs.",
    )
    parser.add_argument(
        "--save-debug-plots",
        action="store_true",
        help="Save an extra debug overlay plot with manual loop markers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beam_step < 1:
        raise ValueError("--beam-step must be >= 1")
    if args.scan_stride < 1:
        raise ValueError("--scan-stride must be >= 1")
    if args.voxel_size_m <= 0:
        raise ValueError("--voxel-size-m must be > 0")
    if args.max_range_mm <= 0:
        raise ValueError("--max-range-mm must be > 0")

    config = PipelineConfig(
        max_range_mm=args.max_range_mm,
        beam_step=args.beam_step,
        voxel_size_m=args.voxel_size_m,
        scan_stride=args.scan_stride,
    )
    output_tag = args.output_tag or build_output_tag(config)

    sequence_names = list(SEQUENCE_REGISTRY.keys()) if args.sequence == "all" else [args.sequence]

    summaries = []
    for sequence_name in sequence_names:
        summary = run_sequence(
            sequence_name,
            config,
            output_root=args.output_root,
            output_tag=output_tag,
            save_debug_plots=args.save_debug_plots,
        )
        summaries.append(summary)

    print(json.dumps({"runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
