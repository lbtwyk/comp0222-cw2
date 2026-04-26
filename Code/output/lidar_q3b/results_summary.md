# Q3b Results Summary

This file summarises the automated Q3b LiDAR experiments for the selected sequences.

Selected datasets:
- `indoor_large_03`: Marshgate large-classroom
- `indoor_small_01`: Marshgate small-room
- `outdoor_02`: Building exterior

Notes:
- The maximum-range experiment uses `12 m` as the RP-Lidar A2M12 rated range.
- Larger closure error does not always mean a bad map alone; in your case some collected second loops were not perfectly aligned to the first.

## Maximum Range

Compare a practical 6 m cap against the RP-Lidar A2M12 rated 12 m range.

| Sequence | Setting | Drift (m) | Loop 1 (m) | Loop 2 (m) | Distance (m) | Keyframes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Marshgate large-classroom | 6 m | 0.886 | 1.234 | 0.905 | 52.320 | 220 |
| Marshgate large-classroom | 12 m | 0.494 | 0.989 | 0.478 | 53.773 | 222 |

Best drift for **Marshgate large-classroom**: `12 m` with `0.494 m` final drift.

| Marshgate small-room | 6 m | 3.586 | 1.879 | 3.586 | 21.632 | 97 |
| Marshgate small-room | 12 m | 2.301 | 1.864 | 2.301 | 21.804 | 98 |

Best drift for **Marshgate small-room**: `12 m` with `2.301 m` final drift.

| Building exterior | 6 m | 7.000 | 13.477 | 7.027 | 221.280 | 436 |
| Building exterior | 12 m | 18.183 | 23.898 | 18.095 | 320.233 | 560 |

Best drift for **Building exterior**: `6 m` with `7.000 m` final drift.

## Angular Resolution

Compare the full scan against beam downsampling to every 2nd and 3rd beam.

| Sequence | Setting | Drift (m) | Loop 1 (m) | Loop 2 (m) | Distance (m) | Keyframes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Marshgate large-classroom | full | 0.886 | 1.234 | 0.905 | 52.320 | 220 |
| Marshgate large-classroom | n=2 | 0.451 | 0.588 | 0.442 | 60.498 | 221 |
| Marshgate large-classroom | n=3 | 0.852 | 0.621 | 0.841 | 65.341 | 215 |

Best drift for **Marshgate large-classroom**: `n=2` with `0.451 m` final drift.

| Marshgate small-room | full | 3.586 | 1.879 | 3.586 | 21.632 | 97 |
| Marshgate small-room | n=2 | 4.782 | 4.358 | 4.780 | 22.643 | 96 |
| Marshgate small-room | n=3 | 2.258 | 2.369 | 2.234 | 24.135 | 95 |

Best drift for **Marshgate small-room**: `n=3` with `2.258 m` final drift.

| Building exterior | full | 7.000 | 13.477 | 7.027 | 221.280 | 436 |
| Building exterior | n=2 | 9.203 | 15.335 | 9.212 | 250.627 | 458 |
| Building exterior | n=3 | 7.875 | 6.980 | 7.839 | 480.366 | 534 |

Best drift for **Building exterior**: `full` with `7.000 m` final drift.

## Voxel Grid Downsampling

Compare fine and coarse point-cloud map voxel filters.

| Sequence | Setting | Drift (m) | Loop 1 (m) | Loop 2 (m) | Distance (m) | Keyframes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Marshgate large-classroom | 0.05 m | 0.886 | 1.234 | 0.905 | 52.320 | 220 |
| Marshgate large-classroom | 0.10 m | 0.886 | 1.245 | 0.937 | 53.530 | 221 |

Best drift for **Marshgate large-classroom**: `0.05 m` with `0.886 m` final drift.

| Marshgate small-room | 0.05 m | 3.586 | 1.879 | 3.586 | 21.632 | 97 |
| Marshgate small-room | 0.10 m | 3.674 | 2.577 | 3.669 | 21.477 | 97 |

Best drift for **Marshgate small-room**: `0.05 m` with `3.586 m` final drift.

| Building exterior | 0.05 m | 7.000 | 13.477 | 7.027 | 221.280 | 436 |
| Building exterior | 0.10 m | 2251.157 | 2251.157 | 2251.157 | 3850.896 | 216 |

Best drift for **Building exterior**: `0.05 m` with `7.000 m` final drift.

## Scan Rate

Compare all scans against 50% reduction and 2-of-3 skipped scans.

| Sequence | Setting | Drift (m) | Loop 1 (m) | Loop 2 (m) | Distance (m) | Keyframes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Marshgate large-classroom | all | 0.886 | 1.234 | 0.905 | 52.320 | 220 |
| Marshgate large-classroom | 1/2 | 0.915 | 1.143 | 0.944 | 50.496 | 198 |
| Marshgate large-classroom | 1/3 | 0.687 | 1.067 | 0.674 | 50.090 | 184 |

Best drift for **Marshgate large-classroom**: `1/3` with `0.687 m` final drift.

| Marshgate small-room | all | 3.586 | 1.879 | 3.586 | 21.632 | 97 |
| Marshgate small-room | 1/2 | 0.608 | 0.735 | 0.600 | 21.192 | 89 |
| Marshgate small-room | 1/3 | 1.978 | 0.647 | 1.979 | 22.123 | 84 |

Best drift for **Marshgate small-room**: `1/2` with `0.608 m` final drift.

| Building exterior | all | 7.000 | 13.477 | 7.027 | 221.280 | 436 |
| Building exterior | 1/2 | 2.820 | 8.047 | 2.737 | 173.291 | 355 |
| Building exterior | 1/3 | 3.708 | 5.311 | 3.651 | 149.886 | 307 |

Best drift for **Building exterior**: `1/2` with `2.820 m` final drift.
