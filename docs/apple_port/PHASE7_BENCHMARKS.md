# Phase 7 Benchmarks

## Method

HighMap measurements were run on the Apple M3 host with three Google
Benchmark repetitions per point, a 50 ms minimum benchmark time, real-time
reporting, and a fixed morphological radius of four pixels. The table reports
minimum/median/maximum wall time in milliseconds. The Metal counters were
recorded from the same resident synchronous wrapper; the direct operation
uses one upload, one neighborhood dispatch, and one readback.

## HighMap operation

| Shape | CPU min/median/max | OpenCL min/median/max | Metal min/median/max | CPU/Metal median |
|---:|---:|---:|---:|---:|
| 512² | 2.861 / 2.873 / 2.900 | 6.334 / 6.668 / 6.676 | 0.701 / 0.713 / 0.781 | 4.03× |
| 1024² | 11.004 / 11.032 / 11.045 | 12.960 / 12.988 / 13.360 | 2.413 / 2.421 / 2.450 | 4.56× |
| 2048² | 50.762 / 51.330 / 51.351 | 39.773 / 39.834 / 40.208 | 8.823 / 9.137 / 10.564 | 5.62× |
| 4096² | 214.221 / 219.736 / 221.030 | 139.423 / 139.815 / 147.571 | 35.103 / 37.541 / 45.335 | 5.85× |

Metal peak resident bytes were 2,097,152, 8,388,608, 33,554,432, and
134,217,728 for the four shapes. These are direct-operation numbers, not a
claim that the whole Hesiod graph has the same scaling.

## Graph D: `texturing_101.hsd`

The selected node is `MorphologicalGradient` id 26. The host fallback is the
pre-Phase-7 equivalent; the resident run adds only the selected single-tile
Metal core. Each point has three samples, whole-output parity passed for every
sample, and the resident path reported one resident node, 22 host nodes, one
input upload, and one terminal readback.

| Shape | Host fallback min/median/max | Phase 7 resident min/median/max | Median speedup |
|---:|---:|---:|---:|
| 512² | 207.138 / 227.725 / 231.444 ms | 171.042 / 174.080 / 175.191 ms | 1.31× |
| 1024² | 1163.717 / 1170.064 / 1183.983 ms | 1081.449 / 1103.133 / 1131.029 ms | 1.06× |

At 2048² the host-heavy graph completed its fallback spot run in 13,554.375
ms; a resident full-graph sample did not complete within the practical
measurement window, and 4096² was not attempted for this graph. The direct
HighMap operation remains qualified through 4096² above. This keeps the full
graph claim limited to reproducible points instead of extrapolating a kernel
speedup to unrelated host branches.

## Tiling and seam checks

At 512² with overlap 0.25, `texturing_101.hsd` classified the selected node as
`MorphologicalGradient: multi-tile halo fallback` for 2×1, 1×2, 2×2, and 4×4.
Whole-output parity and seam/off-seam comparisons passed at zero measured
error in all four configurations. The established Graph B/C checks retained
their maximum seam error of `0.00000024` and off-seam error of `0.00000030`.

The separate `meander.hsd` lifecycle check also passed at 512² in 1×1 and 2×2
configurations. Its selected node was explicitly reported as a closed-session
fallback after an earlier host materialization, with no session-finished node
error and whole-output parity `0.00000030`.

## Graph A regression

The primary `SpectralEqualizer.hsd` Graph A remains on the existing four-node
resident path with one terminal readback and no host uploads. Its final
post-Phase-7 512², 1024², 2048², and 4096² measurements and parity values are
recorded in `PHASE7_REVIEW.md`; the required 4096² parity gate remains PASS.
