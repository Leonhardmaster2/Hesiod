# Hesiod Apple Metal Phase 6 Benchmarks

## Measurement policy

The Phase 5 repeated medians remain the primary regression anchor. Phase 6
expansion and tiling rows below are spot measurements from the headless CLI
after refreshing the current upstream revisions. Each row records wall time,
Metal GPU time, residency boundary, transfer counts, synchronization count,
and peak resident bytes. The fallback wall time is the established CPU/OpenCL
execution path; its Metal counters are zero by design.

## Graph A: SpectralEqualizer primary regression

`data/examples/SpectralEqualizer.hsd` remains the permanent four-node resident
chain. Current cache-off spot checks were:

| Shape | Fallback wall ms | Resident wall ms | Resident GPU ms | Resident nodes | Host nodes | Uploads | Readbacks | Command buffers | Syncs | Peak resident bytes | Max abs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512² | 98.559 | 22.896 | 17.311 | 4 | 0 | 0 | 1 | 2 | 2 | 16,777,228 | 0.00901616 |
| 1024² | 260.424 | 125.653 | 105.794 | 4 | 0 | 0 | 1 | 2 | 2 | 67,108,876 | 0.00459123 |
| 2048² | 1,226.790 | 883.981 | 797.837 | 4 | 0 | 0 | 1 | 2 | 2 | 268,435,468 | 0.00232887 |
| 4096² | 8,671.567 | 6,458.723 | 6,245.903 | 4 | 0 | 0 | 1 | 2 | 2 | 1,073,741,836 | 0.00117421 |

All rows passed the established `1e-2` primary parity gate. The 4096²
resident result is within approximately two percent of the Phase 5 repeated
median of 6,593.112 ms, with no source-level behavior change.

## Graph B: MakePeriodic

Graph B is a configured built-in GaborWaveFbm source followed by a real
periodic boundary consumer. Phase 5-style coverage had no resident Gabor
operation; Phase 6 therefore compares the existing fallback behavior with the
new Gabor resident island.

| Shape | Fallback wall ms | Resident wall ms | Resident GPU ms | Resident/host nodes | Uploads | Readbacks | Peak resident bytes | Max abs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512² | 34.269 | 13.312 | 3.083 | 1 / 1 | 0 | 1 | 2,097,168 | 3.0e-7 |
| 1024² | 62.367 | 29.335 | 9.498 | 1 / 1 | 0 | 1 | 8,388,624 | 3.6e-7 |
| 2048² | 174.633 | 103.746 | 36.678 | 1 / 1 | 0 | 1 | 33,554,448 | 3.0e-7 |
| 4096² | 669.544 | 419.446 | 150.797 | 1 / 1 | 0 | 1 | 134,217,744 | 3.6e-7 |

## Graph C: tiled boundary qualification

The full layout and seam table is in `PHASE6_TILING_BENCHMARKS.md`. Graph C
qualified 1×1, 2×1, 1×2, 2×2, and 4×4 at 512², plus 4×4 at 1024², all with
overlap `0.25`. The maximum seam error was `3.0e-7`; the maximum off-seam
error was `4.2e-7`. Resident runs used zero uploads and one terminal readback.

## Supporting execution counters

The Graph A 4096² resident run allocated and reused buffers as follows:

```text
buffer_allocations=21
buffer_reuses=17
bytes_allocated=1,074,790,412
bytes_reused=1,140,856,688
peak_resident_bytes=1,073,741,836
command_buffers=2
synchronizations=2
```

The cache matrix, including 512², 1024², 2048², and 4096² incremental edits,
is recorded in `PHASE6_CACHE_BENCHMARKS.md`. No HSD format or renderer change
was needed for any measurement.
