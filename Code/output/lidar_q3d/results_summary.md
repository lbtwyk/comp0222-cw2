# Q3d Results Summary

This file summarises the keyframe pose-graph optimization outputs for the selected LiDAR sequences.

Selected datasets:
- `indoor_large_03`: Marshgate large-classroom
- `indoor_small_01`: Marshgate small-room
- `outdoor_02`: Building exterior

| Sequence | Selected Variant | Loop Edges | Mode | Closure Before (m) | Closure After (m) | Improvement (m) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Marshgate large-classroom | anchor_only | 1 | pose-graph | 2.443 | 0.898 | 1.546 |

Selection note for **Marshgate large-classroom**: anchor_only was selected because it was within 0.02 m of the best closure result and had the lowest deformation score, favoring smaller and smoother keyframe corrections.

| Marshgate small-room | anchor_only | 1 | pose-graph | 2.395 | 0.546 | 1.848 |

Selection note for **Marshgate small-room**: anchor_only was selected because it was within 0.02 m of the best closure result and had the lowest deformation score, favoring smaller and smoother keyframe corrections.

| Building exterior | anchor_plus_support | 2 | pose-graph | 2.820 | 0.817 | 2.003 |

Selection note for **Building exterior**: anchor_plus_support was selected because it was within 0.02 m of the best closure result and had the lowest deformation score, favoring smaller and smoother keyframe corrections.
