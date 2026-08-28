# Hesiod Apple Metal Phase 7 Baseline

## Reproducibility point

Phase 7 was started from the published feature branches after refreshing both
upstreams:

| Repository | Upstream `dev` | Feature baseline | Feature commits | Feature-range merges |
|---|---|---|---:|---:|
| Hesiod | `324c3ed51202ec6aa43d89730da39c913060d392` | `2d406ceb51863c07af354ae7e4e39e42481b82b5` | 18 | 0 |
| HighMap | `e0d1279fced75cf7556de233128abfb5650e25b5` | `f8b91e1f12c7e1ab9cb796c8aae822c880525734` | 15 | 0 |

Hesiod was rebased directly onto the refreshed `upstream/dev`; the upstream
change was the upstream `ColorizeBivariate` sharpness change and replayed
without a conflict. HighMap was already based on the current upstream. The
Hesiod submodule points at the HighMap feature baseline above.

Both worktrees had only pre-existing untracked build/output artifacts. Those
artifacts are intentionally preserved and are not part of this phase.

## Phase 6 gates reproduced before implementation

### HighMap

* Metal-focused: 31/31 passed.
* Metal-enabled full suite: 352 passed and the pre-existing
  `PathSplines.PreservePathShape` failure remained.
* No-Metal: 321 passed, 31 Metal tests skipped, and the same pre-existing
  `PathSplines.PreservePathShape` failure remained.

The failure is unchanged from the published Phase 6 baseline and is not
attributed to Phase 7.

### Hesiod Graph A

`data/examples/SpectralEqualizer.hsd`, single tile, cache off:

| Shape | Fallback wall | Resident wall | Resident nodes | Host nodes | Uploads | Readbacks | Parity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512² | 87.048 ms | 22.215 ms | 4 | 0 | 0 | 1 | 0.00901616 PASS |
| 1024² | 260.140 ms | 119.660 ms | 4 | 0 | 0 | 1 | 0.00459123 PASS |
| 2048² | 1234.066 ms | 852.748 ms | 4 | 0 | 0 | 1 | 0.00232887 PASS |
| 4096² | 8557.370 ms | 6363.101 ms | 4 | 0 | 0 | 1 | 0.00117421 PASS |

The historical Phase 5/6 repeated medians remain the comparison anchor:
fallback/resident were 94.925/21.308 ms, 250.236/122.364 ms,
1270.759/854.844 ms, and 8707.482/6593.112 ms respectively.

### Tiled Graphs B and C

* `MakePeriodic.hsd`, 512², 2×2, overlap 0.25: fallback 27.645 ms,
  resident 11.495 ms, one host node, zero uploads, one readback, parity
  `0.00000030`, seam `0.00000024`, PASS.
* `MakePeriodicStitching.hsd`, 512², 4×4, overlap 0.25: fallback 93.301 ms,
  resident 76.935 ms, one host node, zero uploads, one readback, parity
  `0.00000030`, seam `0.00000024`, PASS.

### Cache and no-Metal

The cache-on Graph A/Phase 5 edit matrix was reproduced with
`HESIOD_METAL_PERSISTENT_CACHE=1`: CoherentNoise 19.910 ms, SpectralEqualizer
10.660 ms, Thermal 5.591 ms, and Blend 2.510 ms at 512²; all four edits had
zero uploads and one terminal readback. Cache remains opt-in.

The no-Metal tiled Graph A path passed at 512² 2×2 overlap 0.25 with parity
`0.00162351`, seam `0.00023705`, and no resident activity.

All six bootstraps loaded successfully with the executable launched from the
build directory. Launching from the repository root still emits the known
relative `data/color_gradients` lookup diagnostic; it is not a fatal load
failure and is excluded from the bootstrap gate.

## Baseline decision

The existing resident Graph A and tiled boundary contracts are healthy. Phase
7 therefore proceeds with one targeted family only and keeps cache behavior,
fallback defaults, and all Phase 6 graph contracts unchanged.
