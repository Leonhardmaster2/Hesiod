# Phase 5: Benchmarks

Machine: Apple M3. Primary graph:
`data/examples/SpectralEqualizer.hsd`. Final runs used one tile and zero
overlap. The fallback and resident legs are evaluated in the same process;
one 4096² sample used resident-first order and the others fallback-first.

## Final primary-graph wall time

The table reports the mean of two fresh samples for 512²–2048² and the median
of three fresh samples for 4096². These samples are not mixed with the
historical Phase 4 measurements.

| Shape | Fallback ms | Final resident ms | Approx. resident speedup | Parity max abs |
|---:|---:|---:|---:|---:|
| 512² | 94.925 | 21.308 | 4.45× | 0.00901616 |
| 1024² | 250.236 | 122.364 | 2.05× | 0.00459123 |
| 2048² | 1,270.759 | 854.844 | 1.49× | 0.00232887 |
| 4096² | 8,707.482 | 6,593.112 | 1.32× | 0.00117421 |

Every row passed the existing maximum absolute-error tolerance of `1e-2`.

## Final resident transfer and memory shape

| Shape | Resident nodes | Host compute nodes | Uploads | Readbacks | Peak resident bytes | GPU ms |
|---:|---:|---:|---:|---:|---:|---:|
| 512² | 4 | 0 | 0 | 1 | 16,777,228 | 16.16–16.04 |
| 1024² | 4 | 0 | 0 | 1 | 67,108,876 | 108.99–99.64 |
| 2048² | 4 | 0 | 0 | 1 | 268,435,468 | 790.20–793.84 |
| 4096² | 4 | 0 | 0 | 1 | 1,073,741,836 | 6,303.17–6,338.84 |

The resident 4096² run used 21 buffer allocations and 17 pool reuses; peak
resident bytes were about 1.0 GiB. The Metal recommended working set was
5,726,633,984 bytes.

## HighMap and fallback checks

- Metal-focused HighMap suite: 29/29 passed.
- Full Metal-enabled HighMap suite: 350 passed, with the pre-existing
  `PathSplines.PreservePathShape` failure (`0.15608564 > 0.15`).
- No-Metal HighMap suite: 321 passed, 29 Metal tests skipped, and the same
  pre-existing spline failure.
- Hesiod no-Metal SpectralEqualizer graph at 512²: parity `0.00108373`, pass.
- All six built-in Hesiod bootstrap projects batch-loaded at 128² in the
  no-Metal build.
- `hesiod --help` exits successfully in the Metal-enabled build. A normal
  `--file` launch under `QT_QPA_PLATFORM=offscreen` reaches application startup
  and graph update but exits with the same segmentation fault in the Phase 4
  baseline, the Phase 5 Metal build, and the Phase 5 no-Metal build; this is an
  existing headless Qt/UI limitation rather than a Phase 5-only failure.

## Incremental edit matrix

At 512², the opt-in `HESIOD_PHASE5_EDIT_MATRIX=1` run changed one attribute
per node in the same graph: seed, spectral `rmax`, Thermal duration, and Blend
input-1 weight. Dirty propagation was observed as follows:

| Edited node | Reevaluated nodes | Resident | Host | Uploads | Readbacks | Wall ms |
|---|---:|---:|---:|---:|---:|---:|
| CoherentNoise | 4 | 4 | 0 | 0 | 1 | 21.407 |
| SpectralEqualizer | 2 | 2 | 0 | 2 | 1 | 16.946 |
| Thermal | 2 | 2 | 0 | 2 | 1 | 9.615 |
| Blend | 1 | 1 | 0 | 2 | 1 | 6.360 |

Downstream edits upload unchanged host inputs because Phase 5 intentionally
does not persist DeviceArrays across graph updates. The changed source edit
rebuilds the complete resident chain without uploads. This is the measured
trade-off behind deferring persistent DeviceArray caching.

## Supporting resident HighMap benchmarks

Single samples from the published HighMap feature revision were also run at
1024². The benchmark process was shared with other local work, so these are
sanity measurements rather than a new performance gate:

| Benchmark | Measured total | Upload bytes | Readback bytes | Peak resident | Sync / command buffers |
|---|---:|---:|---:|---:|---:|
| `BM_Phase3_DeviceArrayShared_ChainA/1024` | 4.4975 ms | 0 | 4.1943 MB | 12.5829 MB | 3.5646 ms / 1 |
| `BM_Apple_Metal_HydraulicVPipes/1024/100` | 238.775 ms | 8.3886 MB | 4.1943 MB | 75.5302 MB | 223.037 ms / 1 |

The CTest configurations report no registered tests; executable GTest runs
are the authoritative suite results for this checkout.
